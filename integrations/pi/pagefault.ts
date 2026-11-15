/**
 * pi-pagefault: replace pi's lossy summary compaction with pagefault.
 *
 *   tool_result            -> stream every tool call to the pagefault API (graph capture)
 *   tool_call (bash)       -> warn when an identical command already failed and nothing changed
 *   session_before_compact -> ask pagefault which evicted steps to page back in
 *   recall tool            -> the model pages in anything else (a page fault the graph learns from)
 *   session_shutdown       -> the graph learns from this session
 *
 * Env: PAGEFAULT_URL (default http://127.0.0.1:8080), PAGEFAULT_API_KEY,
 *      PAGEFAULT_KEEP_RECENT (default 20000), PAGEFAULT_BUDGET (default 6000)
 * Fails open: if the API is down, pi's default compaction runs.
 */
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { Type } from "typebox";
import { basename } from "path";

const URL_ = (process.env.PAGEFAULT_URL ?? "http://127.0.0.1:8080").replace(/\/$/, "");
const KEY = process.env.PAGEFAULT_API_KEY;
const KEEP_RECENT = Number(process.env.PAGEFAULT_KEEP_RECENT ?? 20000);
const BUDGET = Number(process.env.PAGEFAULT_BUDGET ?? 6000);

async function api(path: string, body?: unknown, signal?: AbortSignal): Promise<any> {
	const res = await fetch(URL_ + path, {
		method: body === undefined ? "GET" : "POST",
		headers: { "content-type": "application/json", ...(KEY ? { authorization: `Bearer ${KEY}` } : {}) },
		body: body === undefined ? undefined : JSON.stringify(body),
		signal: signal ?? AbortSignal.timeout(15000),
	});
	const text = await res.text();
	let json: any;
	try {
		json = JSON.parse(text);
	} catch {
		throw new Error(`${path}: HTTP ${res.status}, not JSON: ${text.slice(0, 80)}`);
	}
	if (!res.ok || json.error) throw new Error(`${path}: ${json.error ?? "HTTP " + res.status}`);
	return json;
}

const DEBUG = !!process.env.PAGEFAULT_DEBUG;
const log = (...a: unknown[]) => {
	if (DEBUG) console.error("[pagefault]", ...a);
};

const textOf = (content: any): string =>
	Array.isArray(content) ? content.filter((c) => c?.type === "text").map((c) => c.text).join("\n") : String(content ?? "");

export default function (pi: ExtensionAPI) {
	let sessionId: string | undefined;
	const warnings = new Map<string, string>();

	async function ensureSession(ctx: any) {
		if (sessionId) return sessionId;
		const cwd: string = ctx?.cwd ?? process.cwd();
		const r = await api("/v1/sessions", { repo: basename(cwd), harness: "pi", model: ctx?.model?.id ?? "", root: cwd });
		sessionId = r.session_id;
		return sessionId;
	}

	pi.on("session_start", async (_event, ctx) => {
		sessionId = undefined;
		try {
			await ensureSession(ctx);
			ctx.ui?.notify?.(`pagefault: memory graph on (${URL_})`, "info");
		} catch (e) {
			ctx.ui?.notify?.(`pagefault unavailable, default compaction will run: ${e}`, "warning");
		}
	});

	pi.on("tool_call", async (event: any, ctx) => {
		if (event.toolName !== "bash" || !event.input?.command) return;
		try {
			const id = await ensureSession(ctx);
			const r = await api(`/v1/sessions/${id}/check`, { command: event.input.command });
			if (r.warning) {
				warnings.set(event.toolCallId, r.warning);
				log("check warning:", r.warning);
				ctx.ui?.notify?.(`pagefault: ${r.warning}`, "warning");
			}
		} catch (e) {
			log("call failed:", String(e));
		}
	});

	pi.on("tool_result", async (event: any, ctx) => {
		const output = textOf(event.content);
		try {
			const id = await ensureSession(ctx);
			await api(`/v1/sessions/${id}/events`, {
				events: [{ tool: event.toolName, args: event.input ?? {}, output: output.slice(0, 200_000), is_error: !!event.isError }],
			});
		} catch (e) {
			log("call failed:", String(e));
		}
		const w = warnings.get(event.toolCallId);
		if (w) {
			warnings.delete(event.toolCallId);
			return { content: [...(event.content ?? []), { type: "text", text: `\n[pagefault] ${w}` }] };
		}
	});

	pi.on("session_before_compact", async (event: any, ctx) => {
		const { preparation, signal } = event;
		try {
			const id = await ensureSession(ctx);
			const r = await api(`/v1/sessions/${id}/compact`, { keep_recent_tokens: KEEP_RECENT, budget_tokens: BUDGET }, signal);
			const prev = preparation.previousSummary ? `\n\n## Earlier summary\n${preparation.previousSummary}` : "";
			ctx.ui?.notify?.(`pagefault: paged out ${r.evicted_steps} steps, restored ${r.kept.length} (${r.kept_tokens} tokens)`, "info");
			log(`compaction: paged out ${r.evicted_steps} steps (${r.evicted_tokens} tokens), restored ${r.kept.length} (${r.kept_tokens} tokens)`);
			for (const k of r.kept) log(`  kept step ${k.step} ${k.action} | why: ${k.why}`);
			return {
				compaction: {
					summary: r.summary_md + prev,
					firstKeptEntryId: preparation.firstKeptEntryId,
					tokensBefore: preparation.tokensBefore,
				},
			};
		} catch (e) {
			ctx.ui?.notify?.(`pagefault compaction failed, using default: ${e}`, "warning");
			return;
		}
	});

	pi.registerTool({
		name: "recall",
		label: "Recall",
		description:
			"Page back in earlier session context that compaction moved to the memory graph: a file you viewed, a command's output, a search result or an error. Query with a path, symbol, command or error text.",
		promptSnippet: "Recall earlier context (files viewed, command output, errors) after compaction",
		promptGuidelines: ["Use recall before re-reading a file or re-running a command you already ran earlier in this session."],
		parameters: Type.Object({
			query: Type.String({ description: "Path, symbol, command or error text to recall" }),
			k: Type.Optional(Type.Number({ description: "How many steps to return (default 3)" })),
		}),
		async execute(_id: string, params: any, signal: AbortSignal, _onUpdate: any, ctx: any) {
			try {
				const id = await ensureSession(ctx);
				const r = await api(`/v1/sessions/${id}/recall?q=${encodeURIComponent(params.query)}&k=${params.k ?? 3}`, undefined, signal);
				log(`recall "${params.query}" -> ${r.items.length} items (page fault recorded)`);
				const text = r.items.length
					? r.items.map((i: any) => `### step ${i.step}: ${i.action}\n${i.text}`).join("\n\n")
					: "Nothing in memory matches. Fetch it directly.";
				return { content: [{ type: "text", text }], details: { items: r.items.length } };
			} catch (e) {
				return { content: [{ type: "text", text: `recall unavailable: ${e}` }], details: { items: 0 }, isError: true };
			}
		},
	});

	pi.on("session_shutdown", async () => {
		if (!sessionId) return;
		try {
			await api(`/v1/sessions/${sessionId}/end`, { resolved: process.env.PAGEFAULT_RESOLVED === "1" });
		} catch (e) {
			log("call failed:", String(e));
		}
		sessionId = undefined;
	});
}
