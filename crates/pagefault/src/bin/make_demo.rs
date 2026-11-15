//! Write the bundled demo dataset: the first N training sessions (chronological)
//! and a held-out test set, using the same split as the benchmark.
use flate2::write::GzEncoder;
use pagefault::experiment::{prepare, split};
use pagefault::model::load_sessions;
use pagefault::sim::SimCfg;
use std::io::Write;
fn main() -> anyhow::Result<()> {
    let a: Vec<String> = std::env::args().collect();
    let (input, prefix) = (&a[1], &a[2]);
    let n_train: usize = a.get(3).and_then(|s| s.parse().ok()).unwrap_or(400);
    let n_test: usize = a.get(4).and_then(|s| s.parse().ok()).unwrap_or(40);
    let cfg = SimCfg::default();
    let sp = split("demo", prepare(load_sessions(std::path::Path::new(input))?), &cfg, 4, n_test, n_train, 12, false);
    for (name, sessions) in [("train", sp.train.iter().map(|p| &p.session).collect::<Vec<_>>()), ("test", sp.test.iter().map(|t| &t.0.session).collect())] {
        let f = std::fs::File::create(format!("{prefix}.{name}.jsonl.gz"))?;
        let mut gz = GzEncoder::new(f, flate2::Compression::best());
        for s in &sessions {
            writeln!(gz, "{}", serde_json::to_string(s)?)?;
        }
        gz.finish()?;
        eprintln!("{name}: {} sessions", sessions.len());
    }
    Ok(())
}
