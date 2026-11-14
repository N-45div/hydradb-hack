"""Print mean +/- std per arm per checkpoint from a replay results JSON."""
import json, statistics, sys

for path in sys.argv[1:]:
    d = json.load(open(path))
    for r in d["repos"]:
        print(f"\n{r['repo']}  train={r['train_sessions']} test_sessions={r['test_sessions']} test_compactions={r['test_compactions']} seeds={d['seeds']} store={d['store']}")
        arms = ["recency", "bm25", "graph_session", "graph_static", "graph_learned", "oracle"]
        n_cp = len(r["curves"]["graph_learned"][0])
        print("sessions  " + "  ".join(f"{a:>16}" for a in arms))
        for i in range(n_cp):
            row = []
            for a in arms:
                vals = [c[i]["recall"] for c in r["curves"][a] if i < len(c)]
                row.append(f"{statistics.mean(vals):.3f}±{(statistics.pstdev(vals) if len(vals)>1 else 0):.3f}".rjust(16))
            print(f"{r['curves']['graph_learned'][0][i]['sessions']:>8}  " + "  ".join(row))
