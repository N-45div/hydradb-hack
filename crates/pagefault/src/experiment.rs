//! The benchmark: replay a repository's recorded agent sessions in order, let
//! the graph learn from the training stream, and after every checkpoint measure
//! page faults on held-out sessions (different issues, never ingested).

use crate::extract::{extract_session, StepInfo};
use crate::graph::GraphStore;
use crate::learn::Learner;
use crate::model::Session;
use crate::policy::{bm25, graph_score, oracle, recency, select_scored, Ctx, GraphParams, Selection};
use crate::sim::{compaction_points, score_selection, CompactionPoint, Outcome, SimCfg};
use rand::seq::SliceRandom;
use rand::SeedableRng;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap};

pub struct Prepared {
    pub session: Session,
    pub infos: Vec<StepInfo>,
}

pub struct RepoSplit {
    pub repo: String,
    pub train: Vec<Prepared>,
    pub test: Vec<(Prepared, Vec<CompactionPoint>)>,
}

pub fn prepare(sessions: Vec<Session>) -> Vec<Prepared> {
    sessions
        .into_iter()
        .map(|s| {
            let infos = extract_session(&s.steps);
            Prepared { session: s, infos }
        })
        .collect()
}

/// Hold out every `test_every`-th issue (by instance number). Test sessions are
/// different issues from anything in the training stream.
pub fn split(repo: &str, mut all: Vec<Prepared>, cfg: &SimCfg, test_every: usize, max_test: usize, max_train: usize, per_instance: usize) -> RepoSplit {
    all.sort_by(|a, b| a.session.instance_number().cmp(&b.session.instance_number()).then(a.session.id.cmp(&b.session.id)));
    let mut instances: Vec<String> = all.iter().map(|p| p.session.instance_id.clone()).collect();
    instances.dedup();
    let test_inst: std::collections::HashSet<String> =
        instances.iter().enumerate().filter(|(i, _)| i % test_every == test_every - 1).map(|(_, s)| s.clone()).collect();
    let mut train = Vec::new();
    let mut test = Vec::new();
    let mut per_inst: HashMap<String, usize> = HashMap::new();
    for p in all {
        if test_inst.contains(&p.session.instance_id) {
            let c = per_inst.entry(p.session.instance_id.clone()).or_insert(0);
            if *c >= per_instance || test.len() >= max_test {
                continue;
            }
            let cps = compaction_points(&p.infos, cfg);
            if cps.iter().all(|c| c.needed.is_empty()) {
                continue;
            }
            *c += 1;
            test.push((p, cps));
        } else if train.len() < max_train {
            train.push(p);
        }
    }
    RepoSplit { repo: repo.to_string(), train, test }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Point {
    pub sessions: usize,
    pub recall: f64,
    pub faults_per_compaction: f64,
    pub rework_tokens_per_compaction: f64,
    pub waste_ratio: f64,
    pub compactions: usize,
    pub needed: usize,
}

impl Point {
    fn from(sessions: usize, o: &Outcome) -> Self {
        Point {
            sessions,
            recall: o.recall(),
            faults_per_compaction: o.faults_per_compaction(),
            rework_tokens_per_compaction: o.rework_per_compaction(),
            waste_ratio: if o.kept_tokens == 0 { 0.0 } else { o.waste_tokens as f64 / o.kept_tokens as f64 },
            compactions: o.compactions,
            needed: o.needed,
        }
    }
}

pub const ARMS: [&str; 6] = ["recency", "bm25", "graph_session", "graph_static", "graph_learned", "oracle"];

/// Fill whatever budget a policy left unused with the newest evicted steps, so
/// every arm spends the same number of tokens.
pub fn fill_recent(ctx: &Ctx, sel: &mut Selection) {
    let mut left = ctx.budget.saturating_sub(sel.items.iter().map(|&i| ctx.infos[i].tokens).sum::<usize>());
    for i in (0..ctx.cp.recent_start).rev() {
        if left < 50 {
            break;
        }
        if !sel.items.contains(&i) && ctx.infos[i].tokens <= left {
            sel.items.push(i);
            left -= ctx.infos[i].tokens;
        }
    }
}

pub struct ArmStores<'a> {
    pub stat: (&'a mut dyn GraphStore, u8),
    pub learned: (&'a mut dyn GraphStore, u8),
    pub learned_params: GraphParams,
}

pub fn run_arm(
    arm: &str,
    repo: &str,
    infos: &[StepInfo],
    cp: &CompactionPoint,
    cfg: &SimCfg,
    stores: &mut ArmStores,
) -> anyhow::Result<Selection> {
    let ctx = Ctx { repo, infos, cp, budget: cfg.budget, focus_steps: cfg.focus_steps };
    let mut sel = match arm {
        "recency" => return Ok(recency(&ctx)),
        "oracle" => return Ok(oracle(&ctx)),
        "bm25" => bm25(&ctx),
        "graph_session" => {
            let p = GraphParams { use_repo: false, ..Default::default() };
            let mut dummy = crate::graph::MemStore::default();
            let s = graph_score(&ctx, &mut dummy, 0, &p)?;
            select_scored(&ctx, &s, 0)
        }
        "graph_static" => {
            let (store, space) = (&mut *stores.stat.0, stores.stat.1);
            let s = graph_score(&ctx, store, space, &GraphParams::default())?;
            select_scored(&ctx, &s, space)
        }
        "graph_learned" => {
            let (store, space) = (&mut *stores.learned.0, stores.learned.1);
            let s = graph_score(&ctx, store, space, &stores.learned_params)?;
            select_scored(&ctx, &s, space)
        }
        other => anyhow::bail!("unknown arm {other}"),
    };
    fill_recent(&ctx, &mut sel);
    Ok(sel)
}

pub fn evaluate(
    split: &RepoSplit,
    cfg: &SimCfg,
    arms: &[&str],
    stores: &mut ArmStores,
) -> anyhow::Result<BTreeMap<String, Outcome>> {
    let mut out: BTreeMap<String, Outcome> = BTreeMap::new();
    for (p, cps) in &split.test {
        for cp in cps {
            for arm in arms {
                let sel = run_arm(arm, &split.repo, &p.infos, cp, cfg, stores)?;
                out.entry(arm.to_string()).or_default().add(&score_selection(&p.infos, cp, &sel.items));
            }
        }
    }
    Ok(out)
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct RepoResult {
    pub repo: String,
    pub train_sessions: usize,
    pub test_sessions: usize,
    pub test_compactions: usize,
    /// arm -> seed -> curve
    pub curves: BTreeMap<String, Vec<Vec<Point>>>,
    pub learn_stats: Vec<crate::learn::LearnStats>,
    pub graph_summary: Vec<HashMap<String, usize>>,
}

pub fn order_for_seed(n: usize, seed: u64) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..n).collect();
    if seed > 0 {
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        idx.shuffle(&mut rng);
    }
    idx
}

/// One seed of the learning curve for one repo. `stat` and `learned` must be
/// empty spaces; they are filled as training proceeds.
#[allow(clippy::too_many_arguments)]
pub fn learning_curve(
    split: &RepoSplit,
    cfg: &SimCfg,
    checkpoints: &[usize],
    seed: u64,
    stat_store: &mut dyn GraphStore,
    stat_space: u8,
    learned_store: &mut dyn GraphStore,
    learned_space: u8,
    mut on_point: impl FnMut(usize, &BTreeMap<String, Outcome>),
) -> anyhow::Result<(BTreeMap<String, Vec<Point>>, Learner)> {
    let order = order_for_seed(split.train.len(), seed);
    let mut stat = Learner::new(stat_space, &split.repo, false);
    let mut learned = Learner::new(learned_space, &split.repo, true);
    let mut curves: BTreeMap<String, Vec<Point>> = BTreeMap::new();
    let mut ingested = 0usize;
    // Arms that do not depend on training are evaluated once.
    let mut fixed: Option<BTreeMap<String, Outcome>> = None;
    for &cp_n in checkpoints {
        let target = cp_n.min(order.len());
        while ingested < target {
            let p = &split.train[order[ingested]];
            stat.ingest(&p.infos, p.session.resolved, cfg, stat_store)?;
            learned.ingest(&p.infos, p.session.resolved, cfg, learned_store)?;
            ingested += 1;
        }
        let mut stores = ArmStores { stat: (stat_store, stat_space), learned: (learned_store, learned_space), learned_params: learned.params.clone() };
        let mut res = evaluate(split, cfg, &["graph_static", "graph_learned"], &mut stores)?;
        if fixed.is_none() {
            fixed = Some(evaluate(split, cfg, &["recency", "bm25", "graph_session", "oracle"], &mut stores)?);
        }
        for (k, v) in fixed.as_ref().unwrap() {
            res.insert(k.clone(), v.clone());
        }
        on_point(ingested, &res);
        for (arm, o) in &res {
            curves.entry(arm.clone()).or_default().push(Point::from(ingested, o));
        }
        if target >= order.len() && cp_n > order.len() {
            break;
        }
    }
    Ok((curves, learned))
}
