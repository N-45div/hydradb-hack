//! Benchmark runner: learning curves of page faults per compaction.
use clap::Parser;
use pagefault::experiment::{learning_curve, prepare, split, RepoResult, ARMS};
use pagefault::graph::{GraphStore, MemStore};
use pagefault::hydra::HydraStore;
use pagefault::model::load_sessions;
use pagefault::sim::SimCfg;
use std::path::PathBuf;

#[derive(Parser)]
struct Args {
    #[arg(long, default_value = "data/sessions")]
    data_dir: PathBuf,
    #[arg(long, value_delimiter = ',', default_value = "iterative__dvc")]
    repos: Vec<String>,
    #[arg(long, value_delimiter = ',', default_value = "0,25,50,100,200,400,800,1500")]
    checkpoints: Vec<usize>,
    #[arg(long, default_value_t = 3)]
    seeds: u64,
    #[arg(long, default_value = "memory")]
    store: String,
    #[arg(long, env = "HYDRADB_URL", default_value = "http://127.0.0.1:8443")]
    hydra_url: String,
    #[arg(long, env = "HYDRADB_TOKEN", default_value = "local-development-token-32-bytes")]
    hydra_token: String,
    #[arg(long, default_value_t = 4)]
    test_every: usize,
    #[arg(long, default_value_t = 300)]
    max_test: usize,
    #[arg(long, default_value_t = 12)]
    per_instance: usize,
    #[arg(long, default_value_t = 1500)]
    max_train: usize,
    #[arg(long, default_value_t = 8000)]
    ctx_limit: usize,
    #[arg(long, default_value_t = 2000)]
    recent: usize,
    #[arg(long, default_value_t = 2000)]
    budget: usize,
    #[arg(long, default_value_t = 20)]
    window: usize,
    #[arg(long, default_value = "results/results.json")]
    out: PathBuf,
}

fn make_store(a: &Args) -> anyhow::Result<Box<dyn GraphStore>> {
    Ok(match a.store.as_str() {
        "memory" => Box::new(MemStore::default()),
        "hydra" => Box::new(HydraStore::new(&a.hydra_url, &a.hydra_token)?),
        s => anyhow::bail!("unknown store {s}"),
    })
}

fn main() -> anyhow::Result<()> {
    let a = Args::parse();
    let cfg = SimCfg { ctx_limit: a.ctx_limit, recent: a.recent, budget: a.budget, window: a.window, ..Default::default() };
    let mut results = Vec::new();
    for repo in &a.repos {
        let mut path = a.data_dir.join(format!("{repo}.jsonl"));
        if !path.exists() {
            path = a.data_dir.join(format!("{repo}.jsonl.gz"));
        }
        let sessions = load_sessions(&path)?;
        let sp = split(repo, prepare(sessions), &cfg, a.test_every, a.max_test, a.max_train, a.per_instance);
        let test_cps: usize = sp.test.iter().map(|t| t.1.len()).sum();
        eprintln!("[{repo}] train={} test={} test_compactions={} store={}", sp.train.len(), sp.test.len(), test_cps, a.store);
        let mut rr = RepoResult { repo: repo.clone(), train_sessions: sp.train.len(), test_sessions: sp.test.len(), test_compactions: test_cps, ..Default::default() };
        for seed in 0..a.seeds {
            let mut s1 = make_store(&a)?;
            let mut s2 = make_store(&a)?;
            let (sp1, sp2) = ((2 * seed + 1) as u8, (2 * seed + 2) as u8);
            s1.clear_space(sp1, &[])?;
            s2.clear_space(sp2, &[])?;
            let t0 = std::time::Instant::now();
            let (curves, learner) = learning_curve(&sp, &cfg, &a.checkpoints, seed, s1.as_mut(), sp1, s2.as_mut(), sp2, |n, res| {
                let row: Vec<String> = ARMS.iter().filter_map(|arm| res.get(*arm).map(|o| format!("{arm}={:.3}/{:.2}", o.recall(), o.faults_per_compaction()))).collect();
                eprintln!("  seed {seed} sessions {n:>4}: {}", row.join("  "));
            })?;
            eprintln!("  seed {seed} done in {:.1}s, learner {:?}", t0.elapsed().as_secs_f64(), learner.stats);
            for (arm, c) in curves {
                rr.curves.entry(arm).or_default().push(c);
            }
            rr.learn_stats.push(learner.stats.clone());
            rr.graph_summary.push(learner.summary());
        }
        results.push(rr);
    }
    if let Some(d) = a.out.parent() {
        std::fs::create_dir_all(d)?;
    }
    let out = serde_json::json!({
        "config": cfg,
        "store": a.store,
        "checkpoints": a.checkpoints,
        "seeds": a.seeds,
        "repos": results,
    });
    std::fs::write(&a.out, serde_json::to_string_pretty(&out)?)?;
    eprintln!("wrote {}", a.out.display());
    Ok(())
}
