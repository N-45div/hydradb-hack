//! Dataset diagnostics: how much re-fetch signal does a replay config produce?
use pagefault::extract::extract_session;
use pagefault::model::load_sessions;
use pagefault::sim::{compaction_points, SimCfg};
fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let path = &args[1];
    let ctx: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(12000);
    let recent: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(3000);
    let window: usize = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(15);
    let cfg = SimCfg { ctx_limit: ctx, recent, window, ..Default::default() };
    let ss = load_sessions(std::path::Path::new(path))?;
    let (mut toks, mut steps, mut cps, mut need, mut withneed, mut fetch) = (0usize, 0usize, 0usize, 0usize, 0usize, 0usize);
    let mut kinds = std::collections::HashMap::new();
    let mut inst = std::collections::HashSet::new();
    for s in &ss {
        inst.insert(s.instance_id.clone());
        let infos = extract_session(&s.steps);
        steps += infos.len();
        toks += infos.iter().map(|i| i.tokens).sum::<usize>();
        fetch += infos.iter().map(|i| i.fetches.len()).sum::<usize>();
        let c = compaction_points(&infos, &cfg);
        cps += c.len();
        let n: usize = c.iter().map(|c| c.needed.len()).sum();
        for cp in &c { for nd in &cp.needed { *kinds.entry(nd.ent.kind.as_str()).or_insert(0) += 1; } }
        need += n;
        if n > 0 { withneed += 1; }
    }
    let n = ss.len().max(1);
    println!("sessions={} instances={} avg_steps={:.1} avg_tokens={:.0} avg_fetches={:.1} compactions={} needed={} sessions_with_need={} kinds={:?}", ss.len(), inst.len(), steps as f64 / n as f64, toks as f64 / n as f64, fetch as f64 / n as f64, cps, need, withneed, kinds);
    Ok(())
}
