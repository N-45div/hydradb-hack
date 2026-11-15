"""LLM-summary baseline: does an OpenCode-style summary beat pagefault at the same budget?

Reads compaction points from `dump_points`, asks an OpenAI model to summarize the
evicted history with OpenCode's compaction template (sst/opencode
packages/core/src/session/compaction.ts), and scores it with the same ground
truth as every other arm: what the agent re-fetched within 20 steps.

Scoring is deliberately generous to the summary. A need counts as covered if the
summary merely *mentions* it (the file path or basename, the search term, the
command or the file it runs), even though a mention does not show the content
the agent re-fetched. pagefault only gets credit when a restored step shows that
content.

Arms, all with the same evicted-history budget (2k tokens unless --budget):
  recency          newest evicted steps (truncation)
  summary          LLM summary, capped at the budget
  pagefault        learned-graph restore
  summary+pf       summary at half budget + pagefault at half budget

usage: OPENAI_API_KEY=... python scripts/llm_summary_arm.py results/points_pydantic.jsonl \
          --model gpt-5.4-mini --out results/llm_summary_pydantic.json
"""
import argparse
import gzip
import json
import os
import re
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

TEMPLATE = """You MUST use this format for your response (you may omit sections that aren't applicable). Do not include the <template> tags in your response.
<template>
## Objective
- [one or two brief sentences describing what the user is trying to accomplish]

## Requirements
- [constraints, preferences, requirements, and scope boundaries stated by the user, or "(none)"]

## Decisions
- [decisions already made and why, or "(none)"]

## Work State
### Completed
- [goals that have been completed; otherwise "(none)"]

### Active
- [goals currently being worked on; otherwise "(none)"]

### Blocked
- [anything blocking progress, and why; otherwise "(none)"]

## Next Move
1. [ordered list of next actions, or "(none)"]

## Relevant Files
List the files and directories that another agent would need to open to continue this work. Include at most 15, most important first.
- `[file or directory path]`: [brief reason it matters]

## Important Context
- [facts the next agent cannot continue without and cannot easily find on its own; or "(none)"]
</template>"""

RULES = """Rules:
- Keep each section concise. Use terse, single-line bullets, not prose paragraphs or nested lists.
- Prefer short references over detailed restatement. It is fine to leave out information the next agent can recover from the code or the files listed above.
- Preserve exact file paths, symbols, commands, error strings, URLs, and identifiers.
- Do not mention the summary process or that context was compacted."""

PROMPT = "\n\n".join([
    "You MUST summarize the conversation above into a structured summary that will be given to another agent to resume the work.",
    TEMPLATE,
    RULES,
    "Do not continue the task or call tools.",
    "Return only the structured summary in the requested format.",
])


def call(model, transcript, budget_tokens, key, effort):
    body = {
        "model": model,
        "messages": [
            {"role": "user", "content": "<conversation>\n" + transcript + "\n</conversation>\n\n" + PROMPT
             + f"\n\nThe summary must fit in about {budget_tokens} tokens."},
        ],
        "max_completion_tokens": budget_tokens + 6000,
        "reasoning_effort": effort,
    }
    req = urllib.request.Request("https://api.openai.com/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"content-type": "application/json", "authorization": f"Bearer {key}"})
    for attempt in range(5):
        try:
            t = time.perf_counter()
            with urllib.request.urlopen(req, timeout=180) as r:
                d = json.loads(r.read())
            text = d["choices"][0]["message"]["content"] or ""
            # Cap the visible summary at the budget (~4 chars per token), like every other arm.
            return text[: budget_tokens * 4], time.perf_counter() - t, d.get("usage", {})
        except Exception as e:  # noqa: BLE001
            if attempt == 4:
                raise
            time.sleep(2 ** attempt * 2)


def mention_terms(need):
    kind, key = need["kind"], need["key"]
    if kind == "File":
        terms = [key]
        base = key.rsplit("/", 1)[-1]
        if len(base) >= 6 and "." in base:
            terms.append(base)
        return terms
    if kind == "Search":
        q = re.findall(r'"([^"]+)"|\'([^\']+)\'', key)
        flat = [a or b for a, b in q]
        if flat:
            return flat
        parts = key.split()
        return [parts[1]] if len(parts) > 1 else [key]
    # Cmd: the full command, or a file it runs
    terms = [key]
    terms += [t for t in key.split() if "." in t and len(t) >= 5]
    return terms


def covered_by_summary(need, summary):
    low = summary.lower()
    return any(t and t.lower() in low for t in mention_terms(need))


def score(points, summaries, budget):
    arms = {k: {"compactions": 0, "needed": 0, "hits": 0, "rework": 0} for k in
            ["recency", "summary", "pagefault", "summary+pf"]}
    for p in points:
        key = (p["session"], p["t"])
        s_full = summaries.get((key, budget))
        s_half = summaries.get((key, budget // 2))
        if s_full is None or s_half is None:
            continue
        for arm in arms:
            a = arms[arm]
            a["compactions"] += 1
            for n in p["needed"]:
                a["needed"] += 1
                if arm == "recency":
                    hit = n["label"] in p["cover"]["recency"]
                elif arm == "pagefault":
                    hit = n["label"] in p["cover"]["pagefault"]
                elif arm == "summary":
                    hit = covered_by_summary(n, s_full)
                else:
                    hit = n["label"] in p["cover"]["pagefault_half"] or covered_by_summary(n, s_half)
                if hit:
                    a["hits"] += 1
                else:
                    a["rework"] += n["refetch_tokens"]
    out = {}
    for arm, a in arms.items():
        c = max(a["compactions"], 1)
        out[arm] = {
            "compactions": a["compactions"],
            "needed": a["needed"],
            "recall": a["hits"] / max(a["needed"], 1),
            "faults_per_compaction": (a["needed"] - a["hits"]) / c,
            "rework_tokens_per_compaction": a["rework"] / c,
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("points")
    ap.add_argument("--model", default="gpt-5.4-mini")
    ap.add_argument("--effort", default="low")
    ap.add_argument("--budget", type=int, default=2000)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--cache", default="")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    key = os.environ["OPENAI_API_KEY"]
    opener = gzip.open if a.points.endswith(".gz") else open
    points = [json.loads(l) for l in opener(a.points, "rt", encoding="utf-8")]
    if a.limit:
        points = points[: a.limit]
    cache_path = a.cache or a.out + ".cache.jsonl"
    summaries, latency, usage = {}, [], {"prompt_tokens": 0, "completion_tokens": 0}
    if os.path.exists(cache_path):
        for l in open(cache_path, encoding="utf-8"):
            r = json.loads(l)
            summaries[((r["session"], r["t"]), r["budget"])] = r["summary"]
            latency.append(r["seconds"])
            for k in usage:
                usage[k] += r["usage"].get(k, 0)
    jobs = [(p, b) for p in points for b in (a.budget, a.budget // 2) if ((p["session"], p["t"]), b) not in summaries]
    print(f"{len(points)} points, {len(jobs)} summaries to write with {a.model}", file=sys.stderr)
    cache = open(cache_path, "a", encoding="utf-8")

    def run(job):
        p, b = job
        text, secs, u = call(a.model, p["transcript"], b, key, a.effort)
        return p, b, text, secs, u

    done = 0
    with ThreadPoolExecutor(a.workers) as ex:
        for p, b, text, secs, u in ex.map(run, jobs):
            summaries[((p["session"], p["t"]), b)] = text
            latency.append(secs)
            for k in usage:
                usage[k] += u.get(k, 0)
            cache.write(json.dumps({"session": p["session"], "t": p["t"], "budget": b, "summary": text, "seconds": secs, "usage": u}) + "\n")
            cache.flush()
            done += 1
            if done % 25 == 0:
                print(f"  {done}/{len(jobs)}", file=sys.stderr)
    cache.close()
    res = score(points, summaries, a.budget)
    latency.sort()
    report = {
        "model": a.model, "reasoning_effort": a.effort, "budget_tokens": a.budget,
        "scoring": "summary credited when it mentions the needed file/search/command (generous); pagefault only when a restored step shows it",
        "arms": res,
        "summary_latency_seconds": {"p50": latency[len(latency) // 2], "p95": latency[int(len(latency) * 0.95)]} if latency else {},
        "llm_tokens": usage,
    }
    json.dump(report, open(a.out, "w", encoding="utf-8"), indent=2)
    for arm, r in res.items():
        print(f"{arm:12} faults/compaction={r['faults_per_compaction']:.3f}  recall={r['recall']:.3f}  rework={r['rework_tokens_per_compaction']:.0f}  (n={r['compactions']})")
    print(json.dumps(report["summary_latency_seconds"]), json.dumps(usage))


if __name__ == "__main__":
    main()
