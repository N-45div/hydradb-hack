# pagefault

**Compaction that learns.** A Rust compaction API for coding agents, built on the open-source [HydraDB](https://github.com/hydra-db/hydradb) graph engine.

When an agent's context window fills, harnesses summarize the history and throw the rest away. The summary drops exact paths, error text and the approach that already failed. The agent then re-opens files, re-runs commands and repeats searches it already did.

pagefault treats the context window as a cache:

- **Page out.** Old steps leave the window but stay in a HydraDB graph of files, commands, searches, errors and symbols.
- **Page in.** At compaction, it traverses the graph from what the agent is working on now and restores the evicted steps the next steps will most likely need.
- **Learn.** Every miss (a *page fault*: the agent had to fetch again something compaction dropped) teaches the repo's graph. The next compaction is better.

> Track 2 (agent context layer), HydraDB Hackathon. Extraction, retrieval and learning run in deterministic Rust, with **zero LLM calls**.

## Results

We replay real SWE-agent runs ([nebius/SWE-agent-trajectories](https://huggingface.co/datasets/nebius/SWE-agent-trajectories), CC-BY-4.0) with an 8k-token context limit, the regime local models live in.
- At every compaction, each policy keeps the same 2k-token budget of evicted history.
- A **page fault** is the agent, in the next 20 steps, re-opening a file, repeating a search, or re-running an *unchanged* command that an evicted step had already shown. Re-running after an edit is not counted.
- **Held out:** test sessions are different GitHub issues from anything the graph learned from.
- **Not a one-off:** 5 different training orders; ± is the std.

Page faults per compaction (lower is better), before learning → after the graph has ingested the repo's training sessions:

| Policy | pydantic (253 compactions) | sqlglot (232) | dvc (276) |
|---|---|---|---|
| Keep most recent (truncation, the usual default) | 0.395 | 0.547 | 0.616 |
| Flat store + keyword search (BM25) | 0.577 | 0.961 | 0.877 |
| Session graph only | 0.360 | 0.603 | 0.460 |
| Repo graph in HydraDB, no learning | 0.360 → 0.344 | 0.603 → 0.582 | 0.460 → 0.453 |
| **pagefault: repo graph + learning** | **0.360 → 0.171** | **0.603 → 0.336** | **0.460 → 0.407** |
| Oracle (knows the future) | 0.016 | 0.065 | 0.112 |

Recall of needed context: **pydantic 63% → 83% (±1.1)**, **sqlglot 52% → 73% (±0.6)**, dvc 61% → 65% (±1.9).

What the baselines show:
- **The graph matters.** On the same events, a flat keyword store is the worst policy everywhere. It would fall apart on a plain store.
- **Learning matters.** Accumulating graph structure alone barely moves (0.360 → 0.344). Learning from page faults halves them on pydantic. On sqlglot the graph without learning *loses* to truncation; learning flips it.
- **It's real continual learning.** The curve rises with the number of sessions learned, on held-out issues, across 5 training orders. See the live chart at `/`.

The pydantic run was repeated with HydraDB as the store (`results/pydantic_hydradb.json`) and reproduces the in-memory run exactly.

Reproduce:
```bash
python scripts/prepare_data.py --shards 0 1 2 3 4 5 6 --max-per-repo 2000   # ~630 MB download
cargo run --release --bin replay -- --repos pydantic__pydantic --seeds 5 --store hydra   # or --store memory
python scripts/summarize.py results/*.json
```

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
3. **Page in** (`policy.rs`): traverse from the entities in focus over the session graph plus HydraDB (one hop, then two hops through the strongest intermediates). Each evicted step gets graph features: in focus, session co-occurrence, repo `CO`/`MENTIONS`/`NEXT`/`FIXED_BY`, 2-hop, session degree, edited, eviction recency and repo degree. A per-repo linear model scores them, and a greedy budgeted cover picks the steps.
4. **Learn** (`learn.rs`), after each session:
   - **Fault** (needed, not restored): create or strengthen `NEXT(focus → needed)`.
   - **Waste** (restored, never used): weaken the `NEXT` edges that pulled it in.
   - **Weights:** fit the repo's feature weights with online logistic regression.

   Weights and edges are written back to HydraDB in batches.

HydraDB specifics we built around:
- Node ids are non-negative integers, so we hash entities to 56 bits plus a 7-bit space for tenants and experiment arms.
- Batched vertex upserts take exactly one label, and batched edge `MERGE … SET` needs single-label endpoints.
- A `MATCH` takes one relationship type, so neighbor reads are one `UNION ALL` request per type across all focus ids.

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
docs/LAUNCH.md             go-to-market
```

## Limits and next steps

- **The benchmark measures re-fetches, not task success.** A live A/B with pi on local models is next.
- **Ground truth is what the recorded agent fetched again.** That's a proxy for need, not an oracle.
- **Next:**
  - an MCP server, so Claude Code and Codex get `recall` and `compact` with no plugin
  - a shared team graph across developers and harnesses
  - S3-backed HydraDB for hosted multi-tenant memory

Apache-2.0.
