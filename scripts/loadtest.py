"""Latency of the compaction API under concurrent agents.

Replays N demo sessions as live agents: stream their steps, then call /compact
and /recall concurrently. Prints p50/p95/p99 per endpoint.
usage: python scripts/loadtest.py http://127.0.0.1:8080 [agents=16] [compactions_per_agent=5]
"""
import gzip, json, statistics, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor

BASE = sys.argv[1].rstrip("/")
AGENTS = int(sys.argv[2]) if len(sys.argv) > 2 else 16
PER = int(sys.argv[3]) if len(sys.argv) > 3 else 5


def call(path, body=None):
    req = urllib.request.Request(BASE + path, data=None if body is None else json.dumps(body).encode(),
                                 headers={"content-type": "application/json"}, method="GET" if body is None else "POST")
    t = time.perf_counter()
    with urllib.request.urlopen(req, timeout=120) as r:
        out = json.loads(r.read())
    return out, (time.perf_counter() - t) * 1000


sessions = [json.loads(l) for l in gzip.open("data/demo/pydantic__pydantic.test.jsonl.gz", "rt")][:AGENTS]
lat = {"events": [], "compact": [], "recall": []}


def agent(s):
    sid = call("/v1/sessions", {"repo": s["repo"], "harness": "loadtest", "model": s["model"]})[0]["session_id"]
    steps = s["steps"]
    chunk = max(1, len(steps) // PER)
    for i in range(0, len(steps), chunk):
        _, ms = call(f"/v1/sessions/{sid}/events", {"steps": steps[i:i + chunk]}); lat["events"].append(ms)
        _, ms = call(f"/v1/sessions/{sid}/compact", {"keep_recent_tokens": 2000, "budget_tokens": 2000}); lat["compact"].append(ms)
        _, ms = call(f"/v1/sessions/{sid}/recall?q=pydantic/main.py&k=2"); lat["recall"].append(ms)
    call(f"/v1/sessions/{sid}/end", {"resolved": s.get("resolved", False)})


t0 = time.perf_counter()
with ThreadPoolExecutor(AGENTS) as ex:
    list(ex.map(agent, sessions))
wall = time.perf_counter() - t0
for k, v in lat.items():
    v.sort()
    q = lambda p: v[min(len(v) - 1, int(p * len(v)))]
    print(f"{k:8} n={len(v):4}  p50={q(.5):7.1f}ms  p95={q(.95):7.1f}ms  p99={q(.99):7.1f}ms")
print(f"{AGENTS} concurrent agents, {sum(len(v) for v in lat.values())} calls in {wall:.1f}s")
