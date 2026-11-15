"""Write to HydraDB, report counts. Run before and after a restart to check
that a restarted graph-node can still write (it needs conditional updates).

usage: python scripts/restart_test.py <phase> http://127.0.0.1:8443 <token>
"""
import json, sys, urllib.request, urllib.error

phase, URL, TOKEN = sys.argv[1], sys.argv[2].rstrip("/"), sys.argv[3]


def q(query, params=None):
    body = {"cell_id": "cell-0", "query": query, "parameters": params or {}, "page_size": 4096}
    req = urllib.request.Request(f"{URL}/v1/graphs/default/query", data=json.dumps(body).encode(),
                                 headers={"content-type": "application/json", "authorization": f"Bearer {TOKEN}", "x-graph-namespace": "default"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        return {"http_error": e.code, "body": e.read().decode()[:400]}


base = 950000 if phase == "before" else 960000
rows = [{"id": base + i, "kind": "File", "key": f"{phase}/f{i}.py"} for i in range(5)]
print(phase, "upsert:", json.dumps(q("UNWIND $rows AS row MERGE (n {id: row.id}) SET n:S7, n.kind = row.kind, n.key = row.key", {"rows": rows}))[:300])
print(phase, "edge:", json.dumps(q("UNWIND $rows AS row MATCH (a:S7 {id: row.src}), (b:S7 {id: row.dst}) MERGE (a)-[r:CO {id: row.eid}]->(b) SET r.w = row.w",
                                  {"rows": [{"src": base, "dst": base + 1, "eid": base, "w": 0.5}]}))[:300])
c = q("MATCH (n:S7) RETURN count(*) AS c")
print(phase, "count:", c.get("rows", c))
