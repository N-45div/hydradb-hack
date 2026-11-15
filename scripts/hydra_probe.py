"""Probe the HydraDB query shapes pagefault relies on, against a running node.

usage: python scripts/hydra_probe.py http://127.0.0.1:8443 <token>
Prints each query's result or error. Used in CI (.github/workflows/hydra-probe.yml).
"""
import json, sys, urllib.request, urllib.error

URL, TOKEN = sys.argv[1].rstrip("/"), sys.argv[2]


def q(query, params=None, page_size=4096, query_id=None):
    body = {"cell_id": "cell-0", "query": query, "parameters": params or {}, "page_size": page_size}
    if query_id:
        body["query_id"] = query_id
    req = urllib.request.Request(f"{URL}/v1/graphs/default/query", data=json.dumps(body).encode(),
                                 headers={"content-type": "application/json", "authorization": f"Bearer {TOKEN}",
                                          "x-graph-namespace": "default"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        return {"http_error": e.code, "body": e.read().decode()[:600]}


def show(name, res):
    print(f"\n=== {name}")
    s = json.dumps(res)
    print(s[:3000] + (" …" if len(s) > 3000 else ""))


def lit(vals):
    return "[" + ", ".join("'" + v.replace("\\", "\\\\").replace("'", "\\'") + "'" for v in vals) + "]"


S = 9
ents = [
    (900001, "File", "shop/checkout.py"), (900002, "File", "shop/discounts.py"), (900003, "Cmd", "python -m unittest"),
    (900004, "Error", "AssertionError: N != N"), (900005, "File", "shop/pricing.py"), (900006, "Search", "search_dir \"it's\" q"),
]
show("clear", q(f"MATCH (n:S{S}) DETACH DELETE n"))
rows = [{"id": i, "kind": k, "key": key, "lbl": f"{k}:{key}", "repo": "probe", "faults": 2.0} for i, k, key in ents]
show("upsert vertices with lbl/repo/faults", q(f"UNWIND $rows AS row MERGE (n {{id: row.id}}) SET n:S{S}, n.kind = row.kind, n.key = row.key, n.lbl = row.lbl, n.repo = row.repo, n.faults = row.faults", {"rows": rows}))
edges = [
    ("CO", 900001, 900005, 0.6, 3), ("CO", 900001, 900002, 0.5, 2), ("NEXT", 900001, 900002, 1.2, 4),
    ("MENTIONS", 900003, 900004, 0.5, 2), ("FIXED_BY", 900004, 900002, 0.75, 3), ("MENTIONS", 900004, 900001, 0.5, 2),
]
for t, a, b, w, n in edges:
    show(f"edge {t}", q(f"UNWIND $rows AS row MATCH (a:S{S} {{id: row.src}}), (b:S{S} {{id: row.dst}}) MERGE (a)-[r:{t} {{id: row.eid}}]->(b) SET r.w = row.w, r.n = row.n, r.cost = row.cost",
                        {"rows": [{"src": a, "dst": b, "eid": a * 7 + b, "w": w, "n": n, "cost": 1.0 / (0.05 + min(w, 1.0))}]}))

focus = ["File:shop/checkout.py", "Cmd:python -m unittest"]
cands = ["File:shop/discounts.py", "File:shop/pricing.py", "Search:search_dir \"it's\" q"]
base = f"sourceLabel: 'S{S}', sourceProperty: 'lbl', sourceValues: {lit(focus)}, targetValues: {lit(cands)}, relDirection: 'outgoing', pairwise: false, resultLimit: 5000"
show("MSpaths 1-hop all types", q(f"CALL algo.MSpaths({{{base}, relTypes: ['CO','MENTIONS','NEXT','FIXED_BY'], maxLen: 1, pathCount: 8}}) YIELD path, pathWeight, pathCost RETURN path, pathWeight, pathCost"))
show("MSpaths 2-hop semantic", q(f"CALL algo.MSpaths({{{base}, relTypes: ['MENTIONS','NEXT','FIXED_BY'], maxLen: 2, pathCount: 16}}) YIELD path RETURN path"))
show("MSpaths 2-hop with costProp", q(f"CALL algo.MSpaths({{{base}, relTypes: ['MENTIONS','NEXT','FIXED_BY','CO'], maxLen: 2, pathCount: 2, costProp: 'cost'}}) YIELD path, pathWeight, pathCost RETURN path, pathWeight, pathCost"))
show("MSpaths 2-hop with weightProp", q(f"CALL algo.MSpaths({{{base}, relTypes: ['MENTIONS','NEXT','FIXED_BY','CO'], maxLen: 2, pathCount: 2, weightProp: 'w'}}) YIELD path, pathWeight, pathCost RETURN path, pathWeight, pathCost"))
show("degree CO outgoing", q(f"CALL algo.degree({{label: 'S{S}', relTypes: ['CO'], relDirection: 'outgoing', minDegree: 1, limit: 1000}}) YIELD node, degree RETURN node.id AS id, degree"))
show("degree big limit", q(f"CALL algo.degree({{label: 'S{S}', relTypes: ['CO'], relDirection: 'outgoing', minDegree: 1, limit: 1000000}}) YIELD node, degree RETURN node.id AS id, degree"))
show("load vertices", q(f"MATCH (n:S{S}) RETURN n.id AS id, n.kind AS kind, n.key AS key, n.repo AS repo, n.faults AS faults"))
show("load edges NEXT", q(f"MATCH (a:S{S})-[r:NEXT]->(b) RETURN a.id AS src, b.id AS dst, r.w AS w, r.n AS n"))
show("meta vertex write", q("UNWIND $rows AS row MERGE (n {id: row.id}) SET n:M9, n.repo = row.repo, n.weights = row.weights",
                            {"rows": [{"id": 909090, "repo": "probe", "weights": "[1.5,-3.0,2.0]"}]}))
show("meta vertex read", q("MATCH (m:M9) RETURN m.id AS id, m.repo AS repo, m.weights AS weights"))
show("faults update", q(f"UNWIND $rows AS row MERGE (n {{id: row.id}}) SET n:S{S}, n.faults = row.faults", {"rows": [{"id": 900002, "faults": 5.0}]}))
show("faults read", q(f"MATCH (n:S{S}) WHERE n.faults > 3.0 RETURN n.id AS id, n.faults AS f, n.kind AS kind"))
# Delete-then-recreate: batched DETACH DELETE of ids, then upsert the same ids again.
ids = [{"id": i} for i, _, _ in ents]
show("batched delete by id", q("UNWIND $rows AS row MATCH (n {id: row.id}) DETACH DELETE n", {"rows": ids}))
show("count after delete", q(f"MATCH (n:S{S}) RETURN count(*) AS c"))
show("re-upsert same ids", q(f"UNWIND $rows AS row MERGE (n {{id: row.id}}) SET n:S{S}, n.kind = row.kind, n.key = row.key, n.lbl = row.lbl, n.repo = row.repo, n.faults = row.faults", {"rows": rows}))
show("count after re-upsert", q(f"MATCH (n:S{S}) RETURN count(*) AS c"))
show("re-create edge on recreated ids", q(f"UNWIND $rows AS row MATCH (a:S{S} {{id: row.src}}), (b:S{S} {{id: row.dst}}) MERGE (a)-[r:CO {{id: row.eid}}]->(b) SET r.w = row.w, r.n = row.n", {"rows": [{"src": 900001, "dst": 900002, "eid": 77, "w": 0.5, "n": 1}]}))
show("re-upsert again (second time)", q(f"UNWIND $rows AS row MERGE (n {{id: row.id}}) SET n:S{S}, n.kind = row.kind, n.key = row.key, n.lbl = row.lbl, n.repo = row.repo, n.faults = row.faults", {"rows": rows}))
# Client-supplied query_id: does HydraDB accept it and key idempotency on it?
show("write with client query_id", q(f"UNWIND $rows AS row MERGE (n {{id: row.id}}) SET n:S{S}, n.kind = row.kind", {"rows": [{"id": 901001, "kind": "File"}]}, query_id="pf-probe-boot1-1"))
show("edge write with client query_id", q(f"UNWIND $rows AS row MATCH (a:S{S} {{id: row.src}}), (b:S{S} {{id: row.dst}}) MERGE (a)-[r:CO {{id: row.eid}}]->(b) SET r.w = row.w", {"rows": [{"src": 901001, "dst": 900002, "eid": 991, "w": 0.5}]}, query_id="pf-probe-boot1-2"))
show("same query_id, different payload (conflict expected?)", q(f"UNWIND $rows AS row MATCH (a:S{S} {{id: row.src}}), (b:S{S} {{id: row.dst}}) MERGE (a)-[r:CO {{id: row.eid}}]->(b) SET r.w = row.w", {"rows": [{"src": 901001, "dst": 900003, "eid": 992, "w": 0.7}]}, query_id="pf-probe-boot1-2"))
show("read with client query_id, small pages", q(f"MATCH (n:S{S}) RETURN n.id AS id", page_size=2, query_id="pf-probe-boot1-3"))
show("clear", q(f"MATCH (n:S{S}) DETACH DELETE n"))
show("clear meta", q("MATCH (n:M9) DETACH DELETE n"))
