"""Loom demo: watch pagefault learn, live on HydraDB.

Two short agent sessions on the same repo, against the deployed API.
  Session 1: the agent reads auth.py, hits a test failure, then needs config.py.
             Compaction drops config.py, so the agent has to recall it: a page fault.
             The session ends and the graph learns from that fault.
  Session 2: the same situation again. This time compaction restores config.py,
             and shows the HydraDB path that justified it.

usage: python scripts/loom_demo.py [base_url]
"""
import base64, json, sys, time, urllib.request

B = (sys.argv[1] if len(sys.argv) > 1 else "https://pagefault.onrender.com").rstrip("/")
REPO = f"acme-api-{int(time.time())}"  # fresh repo each run, so "before" really is before


def call(path, body=None):
    req = urllib.request.Request(B + path, data=None if body is None else json.dumps(body).encode(),
                                 headers={"content-type": "application/json"}, method="GET" if body is None else "POST")
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read())


def file_view(name, n=70):
    return "\n".join(f"{i}: {name} line {i}  # configuration and helpers" for i in range(n))


def say(msg):
    print(f"\n\033[1m{msg}\033[0m")


def events(sid, evs):
    call(f"/v1/sessions/{sid}/events", {"events_b64": base64.b64encode(json.dumps(evs).encode()).decode()})


def history(sid):
    # The work before compaction: config.py, then session.py, then the failing test on auth.py.
    events(sid, [
        {"tool": "read", "args": {"path": "/app/src/config.py"}, "output": file_view("config.py")},
        {"tool": "read", "args": {"path": "/app/src/session.py"}, "output": file_view("session.py")},
        {"tool": "read", "args": {"path": "/app/src/auth.py"}, "output": file_view("auth.py", 18)},
        {"tool": "bash", "args": {"command": "pytest tests/test_auth.py"}, "is_error": True,
         "output": 'Traceback (most recent call last):\n  File "/app/src/auth.py", line 12, in login\n    token = settings["token"]\nKeyError: token'},
    ])


def compact(sid):
    r = call(f"/v1/sessions/{sid}/compact", {"keep_recent_tokens": 150, "budget_tokens": 1300})
    print(f"  paged out {r['evicted_steps']} steps ({r['evicted_tokens']} tokens), restored {len(r['kept'])} step(s):")
    for k in r["kept"]:
        print(f"    + {k['action']:<28} why: {k['why']}")
    return r


print(f"pagefault demo against {B}  (repo: {REPO})")

say("SESSION 1: a fresh repo. The graph knows nothing yet.")
s1 = call("/v1/sessions", {"repo": REPO, "harness": "loom-demo", "root": "/app"})["session_id"]
history(s1)
say("Context is full, so compaction runs:")
compact(s1)
say("The agent now needs config.py again. It was paged out, so it calls recall: a PAGE FAULT.")
rec = call(f"/v1/sessions/{s1}/recall?q=config.py&k=1")
print(f"  recall -> {rec['items'][0]['action'] if rec['items'] else 'nothing'}  (fault recorded)")
events(s1, [{"tool": "edit", "args": {"path": "/app/src/auth.py"}, "output": "fixed: read token from config"}])
say("Before re-running the failing test unchanged, pagefault checks it:")
print("  check:", call(f"/v1/sessions/{s1}/check", {"command": "pytest tests/test_auth.py"}).get("warning") or "ok (auth.py was edited since the failure)")
say("Session ends. The repo graph in HydraDB learns from the fault:")
print("  ", call(f"/v1/sessions/{s1}/end", {"resolved": True})["learned"])
m = call(f"/v1/repos/{REPO}/memory")
for src, dst, w in m["top_next"][:4]:
    print(f"    learned  {src}  -> {dst}   (weight {w:.2f})")

say("SESSION 2: same repo, same situation, a different day.")
s2 = call("/v1/sessions", {"repo": REPO, "harness": "loom-demo", "root": "/app"})["session_id"]
history(s2)
say("Compaction again. Now the graph knows what this work needs:")
r = compact(s2)
say("Before the agent re-runs the same failing command, pagefault warns it:")
print("  check:", call(f"/v1/sessions/{s2}/check", {"command": "pytest tests/test_auth.py"}).get("warning"))
call(f"/v1/sessions/{s2}/end", {"resolved": True})
print("\nDone. Same situation, but session 2 got config.py back without a page fault, and every restore names the graph path that justified it.")
