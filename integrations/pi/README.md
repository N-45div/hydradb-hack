# pi-pagefault

pi extension that replaces pi's summary compaction with [pagefault](../../README.md).

```bash
# 1. run pagefault (see repo README) and point pi at it
export PAGEFAULT_URL=http://127.0.0.1:8080
# 2. install for all projects
mkdir -p ~/.pi/agent/extensions/pagefault && cp package.json pagefault.ts ~/.pi/agent/extensions/pagefault/
# or try once
pi -e ./pagefault.ts
```

| pi hook | what pagefault does |
|---|---|
| `tool_result` | streams the tool call into the HydraDB memory graph |
| `tool_call` (bash) | warns when an identical command already failed and nothing was edited since |
| `session_before_compact` | returns a compaction made of the evicted steps the graph predicts you need |
| `recall` tool | pages anything else back in; each call is a page fault the graph learns from |
| `session_shutdown` | the repo graph learns from the finished session |

If the API is unreachable the extension fails open and pi's default compaction runs.
