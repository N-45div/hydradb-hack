"""Convert nebius/SWE-agent-trajectories parquet shards into per-repo session JSONL.

Each session: {id, instance_id, repo, model, resolved, steps:[{thought, action, observation}]}
Usage: python scripts/prepare_data.py --shards 0 1 --repos iterative__dvc pydantic__pydantic --max-per-repo 400
"""
import argparse, json, os, re, urllib.request

import pyarrow.parquet as pq

URL = "https://huggingface.co/datasets/nebius/SWE-agent-trajectories/resolve/main/data/train-{:05d}-of-00012.parquet"
FENCE = re.compile(r"```[a-zA-Z]*\n?(.*?)```", re.S)


def split_ai(text):
    blocks = FENCE.findall(text)
    if not blocks:
        return text.strip(), ""
    action = blocks[-1].strip()
    idx = text.rfind("```" + blocks[-1]) if ("```" + blocks[-1]) in text else text.rfind(blocks[-1])
    thought = text[: max(0, idx)].strip()
    return thought, action


def to_steps(traj):
    steps, pending = [], None
    for m in traj:
        role = m["role"]
        if role == "ai":
            if pending is not None:
                steps.append({**pending, "observation": ""})
            thought, action = split_ai(m["text"] or "")
            pending = {"thought": thought, "action": action}
        elif role == "user" and pending is not None:
            steps.append({**pending, "observation": m["text"] or ""})
            pending = None
    if pending is not None:
        steps.append({**pending, "observation": ""})
    return steps


def issue_text(traj):
    for m in traj:
        if m["role"] == "user":
            return (m["text"] or "")[:4000]
    return ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", type=int, nargs="+", default=[0])
    ap.add_argument("--repos", nargs="*", default=[])
    ap.add_argument("--max-per-repo", type=int, default=400)
    ap.add_argument("--min-steps", type=int, default=12)
    ap.add_argument("--raw-dir", default="data/raw")
    ap.add_argument("--out-dir", default="data/sessions")
    a = ap.parse_args()
    os.makedirs(a.raw_dir, exist_ok=True)
    os.makedirs(a.out_dir, exist_ok=True)
    counts, files = {}, {}
    for s in a.shards:
        path = os.path.join(a.raw_dir, f"shard{s}.parquet")
        if not os.path.exists(path):
            print("downloading shard", s)
            urllib.request.urlretrieve(URL.format(s), path)
        pf = pq.ParquetFile(path)
        for batch in pf.iter_batches(batch_size=256):
            for r in batch.to_pylist():
                repo = r["instance_id"].rsplit("-", 1)[0]
                if a.repos and repo not in a.repos:
                    continue
                if counts.get(repo, 0) >= a.max_per_repo:
                    continue
                steps = to_steps(r["trajectory"])
                if len(steps) < a.min_steps:
                    continue
                n = counts.get(repo, 0)
                sess = {
                    "id": f"{r['instance_id']}#{s}-{n}",
                    "instance_id": r["instance_id"],
                    "repo": repo,
                    "model": r["model_name"],
                    "resolved": bool(r["target"]),
                    "issue": issue_text(r["trajectory"]),
                    "steps": steps,
                }
                if repo not in files:
                    files[repo] = open(os.path.join(a.out_dir, f"{repo}.jsonl"), "a")
                files[repo].write(json.dumps(sess) + "\n")
                counts[repo] = n + 1
    for f in files.values():
        f.close()
    for k, v in sorted(counts.items(), key=lambda x: -x[1])[:20]:
        print(k, v)


if __name__ == "__main__":
    main()
