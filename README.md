# pagefault

**Compaction that learns.** A Rust compaction API for coding agents, built on the open-source [HydraDB](https://github.com/hydra-db/hydradb) graph engine.

When an agent's context window fills, harnesses summarize the history and throw the rest away. The summary drops exact paths, error text and the approach that already failed. The agent then re-opens files, re-runs commands and repeats searches it already did.

pagefault treats the context window as a cache:

- **Page out.** Old steps leave the window but stay in a HydraDB graph of files, commands, searches, errors and symbols.
- **Page in.** At compaction, it traverses the graph from what the agent is working on now and restores the evicted steps the next steps will most likely need.
- **Learn.** Every miss (a *page fault*: the agent had to fetch again something compaction dropped) teaches the repo's graph. The next compaction is better.

> Track 2 (agent context layer), HydraDB Hackathon. Extraction, retrieval and learning run in deterministic Rust, with **zero LLM calls**.

## Why not just summarize?

Today's compaction rewrites the window once and hopes the rewrite holds what the next steps need:

| | How it compacts | What it gives up |
|---|---|---|
| **OpenCode** ([`compaction.ts`](https://github.com/sst/opencode/blob/d432a06b9c07bab0be4e8d642173b1cb14137f11/packages/core/src/session/compaction.ts)) | An LLM fills a fixed template (Objective, Decisions, Work State, Relevant Files…). The newest 15k tokens are kept verbatim, and the summary is precomputed in the background because writing one "takes 30–60 s". | Its own prompt says it is "fine to leave out information the next agent can recover from the code or the files listed". It plans for the agent to fetch things again. |
| **OpenAI Responses** ([`/responses/compact`](https://developers.openai.com/api/reference/resources/responses/methods/compact), or `context_management.compact_threshold`) | The server replaces history with an encrypted compaction item, "opaque and not intended to be human-interpretable". | You can't see what was kept, it only works with OpenAI models, and there's no way to ask for a dropped detail back. |
| **pi default** | An LLM summary plus the newest 20k tokens. | Same as OpenCode. |

All three are one-shot: nothing brings a dropped detail back when the agent needs it ten steps later, nothing learns which details a repo's agents keep needing, and nothing measures what was lost. Every re-opened file and re-run command is a cost these designs accept without counting it. **We count it (page faults), and we cut it.**

pagefault complements a summary rather than replacing it:
- **The summary** keeps *intent*: the objective, decisions and next move.
- **pagefault** keeps *evidence*: the exact file view, command output and error that the next step will reach for, chosen by graph traversal.

It's also:
- **Learned** per repo.
- **Inspectable:** every kept step comes with its graph path.
- **Model-agnostic**, local models included.
- **Fast:** zero LLM calls, about 0.2 s versus 5 s for a summary from a small model (OpenCode budgets 30–60 s for its own).

We measured it: on the same held-out compactions, an OpenCode-template summary written by `gpt-5.4-mini` leaves **twice as many page faults** as pagefault, even when the summary gets credit for merely mentioning a file. **Summary and pagefault together cut faults by 80%** compared with the summary alone (see [Against summary compaction](#against-summary-compaction)).

## Results

We replay real SWE-agent runs ([nebius/SWE-agent-trajectories](https://huggingface.co/datasets/nebius/SWE-agent-trajectories), CC-BY-4.0) with an 8k-token context limit, the regime local models live in.
- At every compaction, each policy keeps the same 2k-token budget of evicted history.
- A **page fault** is the agent, in the next 20 steps, re-opening a file, repeating a search, or re-running an *unchanged* command that an evicted step had already shown. Re-running after an edit is not counted.
- **Held out:** test sessions are different GitHub issues from anything the graph learned from.
- **Not a one-off:** 5 different training orders; ± is the std.

**Main table.** Page faults per compaction (lower is better), before learning → after ingesting the repo's training sessions. This test set keeps the held-out sessions where at least one re-fetch happened after a compaction:

| Policy | pydantic (253 compactions) | sqlglot (232) | dvc (276) |
|---|---|---|---|
| Keep most recent (truncation, the usual default) | 0.395 | 0.547 | 0.616 |
| Flat store + keyword search (BM25) | 0.577 | 0.961 | 0.877 |
| Session graph only, no learning | 0.360 | 0.603 | 0.460 |
| Repo graph in HydraDB, no learning | 0.360 → 0.352 | 0.603 → 0.569 | 0.460 → 0.449 |
| Plain key-value fault counter + learning (no graph) | 0.336 → 0.188 ±0.017 | 0.608 → 0.345 ±0.015 | 0.504 → 0.420 ±0.028 |
| Learning, session graph, no HydraDB repo graph | 0.360 → 0.178 ±0.006 | 0.603 → 0.353 ±0.015 | 0.460 → 0.412 ±0.026 |
| HydraDB repo graph + learning | 0.360 → 0.170 ±0.007 | 0.603 → 0.331 ±0.016 | 0.460 → 0.404 ±0.024 |
| **pagefault: HydraDB repo graph + KV counter + learning** | **0.360 → 0.163 ±0.016** | **0.603 → 0.335 ±0.016** | **0.460 → 0.392 ±0.024** |
| Oracle (knows the future) | 0.016 | 0.065 | 0.112 |

**Unfiltered.** All held-out sessions with a compaction, including ones where nothing was ever re-fetched (`results/unfiltered/`). Every rate drops because most compactions cost nothing, but the ranking holds:

| Policy | pydantic (725 compactions) | sqlglot (363) | dvc (906) |
|---|---|---|---|
| Keep most recent | 0.110 | 0.193 | 0.137 |
| Plain key-value fault counter + learning | 0.050 | 0.117 | 0.086 |
| **pagefault** | **0.040** | **0.118** | **0.079** |

What this shows, plainly:
- **Learning from page faults is the main lever.** pagefault has 59%, 39% and 36% fewer faults than truncation, winning on all 5 training orders in every repo. Accumulating graph structure without learning barely moves (0.360 → 0.352).
- **The HydraDB graph adds a real, mostly consistent gain on top.** The fair test is pagefault against the same learner on a plain key-value fault counter. The graph cuts faults by a further **13% on pydantic and 7% on dvc (5/5 orders each)**, and 2.5% on sqlglot (3/5). Unfiltered, it's 20% and 8% (5/5), and −1% on sqlglot. Most of the gain comes from two-hop semantic chains that HydraDB's `MSpaths` finds, such as command → error → the file that fixed it; a counter can't express these. Every restore also shows the path that pulled it in. A flat keyword store is the worst policy everywhere.
- **It's real continual learning.** Faults fall as sessions are learned, on held-out issues, across 5 training orders. See the live chart at `/`.

The pydantic run was repeated with HydraDB doing all traversal (`results/pydantic_hydradb.json`, produced in CI by the `experiments` workflow against the open-source engine). pagefault's faults per compaction match the in-memory run at every checkpoint. The no-KV graph arm differs at one checkpoint (0.158 vs 0.154).

### Against summary compaction

Every held-out pydantic compaction (253), at the same 2k-token budget. An LLM (`gpt-5.4-mini`, low reasoning effort) wrote an OpenCode-template summary of the evicted history ([`scripts/llm_summary_arm.py`](scripts/llm_summary_arm.py)). The scoring favours the summary: it gets credit when it merely **mentions** a needed file, search term or command. pagefault gets credit only when a restored step **shows** it.

| Arm | Page faults per compaction | Recall | Time per compaction | LLM tokens |
|---|---|---|---|---|
| Keep most recent | 0.395 | 59.5% | none | 0 |
| LLM summary (OpenCode template) | 0.324 | 66.8% | 5.2 s median | ~10k in, ~800 out |
| **pagefault** | **0.154** | **84.2%** | ~0.2 s | 0 |
| **Summary (1k) + pagefault (1k)** | **0.063** | **93.5%** | 5.2 s | ~10k |

The summary keeps intent and pagefault keeps evidence, and the two stack. Results are in `results/llm_summary_pydantic.json`.

Reproduce:
```bash
python scripts/prepare_data.py --shards 0 1 2 3 4 5 6 --max-per-repo 2000   # ~630 MB download
cargo run --release --bin replay -- --repos pydantic__pydantic --seeds 5 --store hydra   # or --store memory; add --keep-empty-tests
cargo run --release --bin dump_points -- --data data/sessions/pydantic__pydantic.jsonl --out points.jsonl
OPENAI_API_KEY=... python scripts/llm_summary_arm.py points.jsonl --out llm.json
python scripts/summarize.py results/*.json
```
The `experiments` GitHub Actions workflow runs the replay and the point export from scratch and commits the results.

## How it works

```
any harness ── tool calls ──▶  pagefault (Rust, axum)  ── HTTP JSON ──▶  HydraDB (open-source engine)
(pi, Claude Code, Codex,        extract · page out ·                    repo memory graph, learned weights
 Rust pi ports, MCP)            page in · learn
```

1. **Capture** (`extract.rs`, `adapter.rs`): each tool call becomes entities with no LLM: `File`, `Cmd`, `Search`, `Error` (normalized signature) and `Symbol` (from traceback frames).
2. **Repo graph in HydraDB** (`graph.rs`, `hydra.rs`). Relationship types:
   - `CO`: seen in the same step
   - `MENTIONS`: command→error, error→file, symbol→file
   - `NEXT`: learned, "when working on A, you soon need B again"
   - `FIXED_BY`: learned, error → the file whose edit resolved it

   Every edge carries a weight `w` and a count `n`.
3. **Page in** (`policy.rs`). The traversal runs inside HydraDB, with two native path calls per compaction:
   - **One `algo.MSpaths` call** from every focus entity to every evicted candidate, one hop, over all four relationship types.
   - **One `algo.MSpaths` call** for two-hop *semantic* chains over `MENTIONS`/`NEXT`/`FIXED_BY`, such as `Cmd:pytest -MENTIONS-> Error:KeyError -FIXED_BY-> File:auth.py`.
   - **`algo.degree`** gives how central each candidate is across past sessions, cached until the next write.

   Every restored step comes with the path that pulled it in, as its "why". Each evicted step gets features from those paths plus the session graph:
   - **From paths:** repo `CO`/`MENTIONS`/`NEXT`/`FIXED_BY` strength, the 2-hop chain, and repo degree.
   - **From the session and simple counts:** in focus, session co-occurrence, session degree, edited, eviction recency, and fault count.

   A per-repo linear model scores them, and a greedy budgeted cover picks the steps.
4. **Learn** (`learn.rs`), after each session:
   - **Fault** (needed, not restored): create or strengthen `NEXT(focus → needed)`.
   - **Waste** (restored, never used): weaken the `NEXT` edges that pulled it in.
   - **Weights:** fit the repo's feature weights with online logistic regression.
   - **Fault counter:** count how often each entity was a page fault. It's a plain per-repo counter and one more feature; the benchmark reports it alone as the no-graph ablation.

   Everything learned lives in HydraDB:
   - edges and their weights;
   - fault counts as a vertex property;
   - the per-repo feature weights on a metadata vertex.

   On startup the service rebuilds every repo's learner from HydraDB (`load_from_store`), so a restart resumes learning instead of starting over. This was verified on Render: a repo's learned edges reloaded after a restart and after a redeploy.

   **Durable on S3.** The live deployment runs HydraDB in its native object-store mode:
   - **Storage:** a private S3 bucket in `ap-southeast-1`, accessed through an IAM user that can touch only that bucket.
   - **Verified on Render:** a repo's learned graph came back identical after a full redeploy (new container) and after a restart, and the first write afterwards succeeded within 2 s.
   - **Why not local disk:** the local-folder store can't do the conditional updates (`PutMode::Update`) that a restarted writer needs. So `start.sh` gives local mode a fresh store on each boot, and durability means S3.
   - **Configuration:** `CLOUD_PROVIDER=aws`, `AWS_BUCKET_NAME`, `AWS_DEFAULT_REGION` and credentials.
   - **Found along the way:** after a restart on S3, HydraDB's own write idempotency keys (`http-query-N`, a counter that restarts from zero) collide with keys stored by the previous process. pagefault avoids this by sending a globally unique `query_id` with every request.

HydraDB specifics we built around:
- Node ids are non-negative integers, so we hash entities to 56 bits plus a 7-bit space for tenants and experiment arms.
- Batched vertex upserts take exactly one label, and batched edge `MERGE … SET` needs single-label endpoints.
- `algo.MSpaths` selects by a string property, so every vertex carries a repo-scoped selector (`sel = "repo|File:path"`). Lists are inlined as escaped Cypher literals because list parameters are rejected.
- A `MATCH` takes one relationship type; the path procedures are the only way to cross types in one request, which is why traversal uses them.
- `scripts/hydra_probe.py` (run in CI by `hydra-probe.yml`) checks every query shape against the real engine. Its latest output is in `experiments/hydra_probe_output.txt`.

## API

| Endpoint | Purpose |
|---|---|
| `POST /v1/sessions` `{repo, harness, model, root}` | start a session |
| `POST /v1/sessions/:id/events` `{events:[{tool, args, output, is_error}]}` | stream tool calls (or `steps`) |
| `POST /v1/sessions/:id/compact` `{keep_recent_tokens, budget_tokens}` | `summary_md`, kept steps with `why` (the graph path), what stayed in memory |
| `GET /v1/sessions/:id/recall?q=` | page something back in; logged as a page fault and learned from |
| `POST /v1/sessions/:id/check` `{command}` | warns: "already failed at step 5 with KeyError, nothing edited since" |
| `POST /v1/sessions/:id/end` `{resolved}` | the repo graph learns from the session |
| `GET /v1/repos/:repo/memory` | learned weights, top learned edges, graph size |
| `GET /v1/results`, `GET /v1/demo`, `POST /v1/demo/learn` | benchmark data and the live learning demo |

Set `PAGEFAULT_API_KEY` to require a bearer token on session endpoints.

## pi extension

We ran pi for real, with `gpt-5.4-mini` against the deployed API: pi compacted twice through pagefault and fixed the bug, and the graph learned NEXT and FIXED_BY edges from that one session.

`integrations/pi/` replaces pi's summary compaction (via `session_before_compact`), streams tool results, adds a `recall` tool and warns before repeating a failed command. It works with any model pi can run, including local ones. See [integrations/pi/README.md](integrations/pi/README.md).

## Run it

```bash
# all-in-one container: HydraDB graph-node + pagefault API + dashboard on :8080
docker build -t pagefault . && docker run -p 8080:8080 -v pagefault-data:/data pagefault
```

Every push to `main` also builds and publishes this image to GHCR (`.github/workflows/image.yml`), so a host can deploy `ghcr.io/n-45div/hydradb-hack:latest` without compiling. The image builds the pinned open-source HydraDB engine from source. HydraDB listens only inside the container, and its token is generated at startup and never exposed.

[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy?repo=https://github.com/N-45div/hydradb-hack)

`render.yaml` deploys the prebuilt image as one web service. The free plan works, but it sleeps after 15 idle minutes and takes about a minute to wake. Its 0.1 CPU makes the live learning demo slow, so use the 1 CPU / 2 GB plan while it's being judged. Any host that runs a Docker image works the same way (Railway, Fly): `PORT` is respected, and a volume at `/data` keeps the graph across restarts.

Local development:
```bash
# HydraDB: see github.com/hydra-db/hydradb "Run a local server" (needs LAGraph: scripts/install_lagraph.sh /opt/lagraph)
cargo run --release --bin server     # HYDRADB_URL, HYDRADB_TOKEN, PAGEFAULT_STORE=hydra|memory
```

## Under load

`scripts/loadtest.py` replays held-out sessions as concurrent live agents against one HydraDB node. With 16 agents streaming events, compacting and recalling at once (282 calls in 6.3 s), compaction took p50 181 ms and p95 357 ms. That run shared the node with a running benchmark, and it's well under the latency of a single LLM call.

## Layout

```
crates/pagefault/src/
  extract.rs  adapter.rs   entities from agent steps / any harness's tool calls
  graph.rs    hydra.rs     repo graph, HydraDB store
  policy.rs   learn.rs     page-in scoring, online learning
  sim.rs      experiment.rs  compaction replay, page-fault ground truth, benchmark
  service.rs  bin/server.rs  HTTP API
  bin/replay.rs            benchmark CLI
integrations/pi/           pi extension
web/                       dashboard
```

## Limits and next steps

- **The benchmark measures re-fetches, not task success.** A live A/B with pi on local models is next.
- **Ground truth is what the recorded agent fetched again.** That's a proxy for need, not an oracle. All trajectories come from one family of agent models (SWE-agent with Llama).
- **A file counts as one item.** Restoring lines 1–100 counts as a hit even if the agent reopens it at line 600.
- **The graph's edge over a plain learned counter is real but uneven**: 7–20% on pydantic and dvc, about zero on sqlglot. Learning from page faults does most of the work.
- **The live service relabels its own wins.** Learning in `end()` treats only re-fetches as needed, so a restore that prevented a re-fetch can be scored as waste. The benchmark replays are unaffected.
- **Next:**
  - an MCP server, so Claude Code and Codex get `recall` and `compact` with no plugin
  - a shared team graph across developers and harnesses
  - S3-backed HydraDB for hosted multi-tenant memory

Apache-2.0.
