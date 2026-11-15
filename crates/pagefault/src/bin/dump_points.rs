//! Export held-out compaction points for the LLM-summary comparison
//! (`scripts/llm_summary_arm.py`). For each point: the evicted history as a
//! transcript, what the agent re-fetched next, and what pagefault (learned
//! graph, trained on the repo's training stream) and truncation restore at the
//! full and half budget.
use clap::Parser;
use pagefault::experiment::{prepare, run_arm, split, ArmStores};
use pagefault::graph::MemStore;
use pagefault::learn::Learner;
use pagefault::model::load_sessions;
use pagefault::sim::SimCfg;
use serde_json::json;
use std::collections::{BTreeMap, BTreeSet};
use std::io::Write;
use std::path::PathBuf;

#[derive(Parser)]
struct Args {
    #[arg(long)]
    data: PathBuf,
    #[arg(long)]
    out: PathBuf,
    #[arg(long, default_value_t = 1500)]
    max_train: usize,
    #[arg(long, default_value_t = 300)]
    max_test: usize,
    /// Characters of evicted history sent to the summarizer (most recent kept).
    #[arg(long, default_value_t = 60000)]
    transcript_chars: usize,
}

fn clip_mid(s: &str, n: usize) -> String {
    if s.len() <= n {
        return s.to_string();
    }
    let mut a = n / 2;
    while !s.is_char_boundary(a) {
        a -= 1;
    }
    let mut b = s.len() - n / 2;
    while !s.is_char_boundary(b) {
        b += 1;
    }
    format!("{}\n…[{} chars cut]…\n{}", &s[..a], b - a, &s[b..])
}

fn main() -> anyhow::Result<()> {
    let a = Args::parse();
    let cfg = SimCfg::default();
    let sessions = load_sessions(&a.data)?;
    let repo = sessions.first().map(|s| s.repo.clone()).unwrap_or_default();
    let sp = split(&repo, prepare(sessions), &cfg, 4, a.max_test, a.max_train, 12, false);
    let mut store = MemStore::default();
    let mut learner = Learner::new(1, &repo, true);
    for p in &sp.train {
        learner.ingest(&p.infos, p.session.resolved, &cfg, &mut store)?;
    }
    eprintln!("[{repo}] trained on {} sessions, {} test sessions", sp.train.len(), sp.test.len());
    let half = SimCfg { budget: cfg.budget / 2, ..cfg.clone() };
    let mut out = std::io::BufWriter::new(std::fs::File::create(&a.out)?);
    let mut n = 0;
    for (p, cps) in &sp.test {
        for cp in cps {
            let mut cover = BTreeMap::new();
            for (arm, c, name) in [
                ("graph_learned", &cfg, "pagefault"),
                ("graph_learned", &half, "pagefault_half"),
                ("recency", &cfg, "recency"),
                ("recency", &half, "recency_half"),
            ] {
                let mut unused = MemStore::default();
                let mut stores = ArmStores {
                    stat: (&mut unused, 0),
                    learned: (&mut store, 1),
                    learned_params: learner.params.clone(),
                    extra: BTreeMap::new(),
                    graph_kv: None,
                };
                let sel = run_arm(arm, &repo, &p.infos, cp, c, &mut stores)?;
                let keys: BTreeSet<String> = sel.items.iter().flat_map(|&i| p.infos[i].provides.iter()).map(|e| e.label()).collect();
                cover.insert(name, keys);
            }
            let mut transcript = String::new();
            for (i, st) in p.session.steps[..cp.recent_start].iter().enumerate() {
                transcript.push_str(&format!(
                    "\n### step {i}\nASSISTANT: {}\n$ {}\nOUTPUT:\n{}\n",
                    clip_mid(&st.thought, 600),
                    clip_mid(&st.action, 600),
                    clip_mid(&st.observation, 2000)
                ));
            }
            if transcript.len() > a.transcript_chars {
                let mut cut = transcript.len() - a.transcript_chars;
                while !transcript.is_char_boundary(cut) {
                    cut += 1;
                }
                let issue = clip_mid(&p.session.issue, 3000);
                transcript = format!("ISSUE:\n{issue}\n…[earlier steps cut]…\n{}", &transcript[cut..]);
            }
            let needed: Vec<_> = cp
                .needed
                .iter()
                .map(|n| json!({"label": n.ent.label(), "kind": n.ent.kind.as_str(), "key": n.ent.key, "refetch_tokens": n.refetch_tokens}))
                .collect();
            let row = json!({
                "session": p.session.id, "t": cp.t, "recent_start": cp.recent_start,
                "budget": cfg.budget, "needed": needed, "cover": cover, "transcript": transcript,
            });
            writeln!(out, "{row}")?;
            n += 1;
        }
    }
    eprintln!("wrote {n} compaction points to {}", a.out.display());
    Ok(())
}
