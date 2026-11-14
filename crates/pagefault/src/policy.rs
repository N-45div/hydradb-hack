//! Compaction policies: what to keep from evicted history within a token budget.
//!
//! * `recency`  – keep the newest evicted steps (truncation; what most harnesses do)
//! * `bm25`     – plain store: keyword-match evicted steps against the current focus
//! * `graph`    – traverse the session graph plus the HydraDB repo graph from the
//!                entities in focus, score entities, keep the steps that show them
//! * `oracle`   – knows the future; upper bound for the chart

use crate::extract::{Ent, Kind, StepInfo};
use crate::graph::{entity_id, EdgeKind, GraphStore};
use crate::sim::CompactionPoint;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

pub struct Ctx<'a> {
    pub repo: &'a str,
    pub infos: &'a [StepInfo],
    pub cp: &'a CompactionPoint,
    pub budget: usize,
    pub focus_steps: usize,
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct Selection {
    pub items: Vec<usize>,
    /// Why each item was kept: the strongest graph path (`File:a -NEXT-> File:b`).
    pub why: HashMap<usize, String>,
}

// ---------------------------------------------------------------- recency

pub fn recency(ctx: &Ctx) -> Selection {
    let mut left = ctx.budget;
    let mut items = Vec::new();
    for i in (0..ctx.cp.recent_start).rev() {
        let tok = ctx.infos[i].tokens;
        if tok <= left {
            items.push(i);
            left -= tok;
        }
        if left < 50 {
            break;
        }
    }
    Selection { items, ..Default::default() }
}

// ---------------------------------------------------------------- bm25

fn tokenize(s: &str) -> Vec<String> {
    s.split(|c: char| !(c.is_alphanumeric() || c == '_'))
        .filter(|w| w.len() > 2)
        .map(|w| w.to_lowercase())
        .collect()
}

pub fn bm25(ctx: &Ctx) -> Selection {
    let docs: Vec<Vec<String>> = (0..ctx.cp.recent_start).map(|i| tokenize(&ctx.infos[i].text)).collect();
    let n = docs.len().max(1) as f64;
    let avgdl = docs.iter().map(|d| d.len()).sum::<usize>() as f64 / n;
    let mut df: HashMap<&str, usize> = HashMap::new();
    for d in &docs {
        let uniq: HashSet<&str> = d.iter().map(|s| s.as_str()).collect();
        for w in uniq {
            *df.entry(w).or_insert(0) += 1;
        }
    }
    let lo = ctx.cp.t.saturating_sub(ctx.focus_steps).max(ctx.cp.recent_start);
    let query: HashSet<String> = (lo..ctx.cp.t).flat_map(|i| tokenize(&ctx.infos[i].text)).collect();
    let (k1, b) = (1.2, 0.75);
    let mut scored: Vec<(f64, usize)> = docs
        .iter()
        .enumerate()
        .map(|(i, d)| {
            let mut tf: HashMap<&str, usize> = HashMap::new();
            for w in d {
                *tf.entry(w.as_str()).or_insert(0) += 1;
            }
            let dl = d.len() as f64;
            let s: f64 = query
                .iter()
                .filter_map(|q| {
                    let f = *tf.get(q.as_str())? as f64;
                    let dfq = *df.get(q.as_str()).unwrap_or(&0) as f64;
                    let idf = ((n - dfq + 0.5) / (dfq + 0.5) + 1.0).ln();
                    Some(idf * f * (k1 + 1.0) / (f + k1 * (1.0 - b + b * dl / avgdl.max(1.0))))
                })
                .sum();
            (s, i)
        })
        .collect();
    scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap().then(b.1.cmp(&a.1)));
    let mut left = ctx.budget;
    let mut items = Vec::new();
    for (s, i) in scored {
        if s <= 0.0 {
            break;
        }
        let tok = ctx.infos[i].tokens;
        if tok <= left {
            items.push(i);
            left -= tok;
        }
    }
    Selection { items, ..Default::default() }
}

// ---------------------------------------------------------------- oracle

pub fn oracle(ctx: &Ctx) -> Selection {
    let mut need: HashSet<&Ent> = ctx.cp.needed.iter().map(|n| &n.ent).collect();
    let mut left = ctx.budget;
    let mut items = Vec::new();
    loop {
        let best = (0..ctx.cp.recent_start)
            .filter(|i| !items.contains(i) && ctx.infos[*i].tokens <= left)
            .map(|i| {
                let gain = ctx.infos[i].provides.iter().filter(|e| need.contains(e)).count();
                (gain as f64 / ctx.infos[i].tokens.max(200) as f64, i)
            })
            .filter(|(g, _)| *g > 0.0)
            .max_by(|a, b| a.0.partial_cmp(&b.0).unwrap().then(a.1.cmp(&b.1)));
        let Some((_, i)) = best else { break };
        for e in &ctx.infos[i].provides {
            need.remove(e);
        }
        left -= ctx.infos[i].tokens;
        items.push(i);
    }
    Selection { items, ..Default::default() }
}

// ---------------------------------------------------------------- graph
//
// The graph policy turns traversal results into features for every entity an
// evicted step could bring back, then scores them with a per-repo linear model
// whose weights the learner fits online (see learn.rs). Every feature except
// the kind one-hots comes from a graph: the session graph (built from this
// session's steps) or the repo graph in HydraDB (built from past sessions).

pub const NF: usize = 15;
pub const FEATURES: [&str; NF] = [
    "bias",
    "in_focus",
    "session_co",
    "repo_co",
    "repo_mentions",
    "repo_next",
    "repo_fixed_by",
    "repo_2hop",
    "session_degree",
    "edited",
    "evict_recency",
    "repo_degree",
    "kind_file",
    "kind_search",
    "kind_cmd",
];

/// Hand-set prior. `graph_static` uses it forever; `graph_learned` starts here.
pub const PRIOR: [f64; NF] = [-3.0, 2.0, 1.0, 0.5, 0.5, 1.5, 1.0, 0.3, 1.0, 0.5, 2.0, 0.3, 0.0, 0.0, 0.0];

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GraphParams {
    pub weights: Vec<f64>,
    pub focus_decay: f64,
    pub use_repo: bool,
}

impl Default for GraphParams {
    fn default() -> Self {
        GraphParams { weights: PRIOR.to_vec(), focus_decay: 0.6, use_repo: true }
    }
}

pub fn edge_strength(kind: EdgeKind, w: f64, n: u64) -> f64 {
    match kind {
        EdgeKind::Co | EdgeKind::Mentions => n as f64 / (n as f64 + 2.0),
        EdgeKind::Next | EdgeKind::FixedBy => w,
    }
}

pub fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

/// Entity scores for one compaction point, plus the bookkeeping the learner needs.
#[derive(Clone, Debug, Default)]
pub struct Scored {
    /// Focus entities with their recency weight.
    pub focus: Vec<(u64, Ent, f64)>,
    /// Candidate entities: anything an evicted step shows.
    pub cand: HashMap<u64, Ent>,
    pub features: HashMap<u64, [f64; NF]>,
    /// Probability the next steps need this entity.
    pub scores: HashMap<u64, f64>,
    /// Strongest explanation per candidate.
    pub why: HashMap<u64, (f64, String)>,
    /// Repo edges that pulled each candidate in: (src, dst, kind).
    pub contrib: HashMap<u64, Vec<(u64, u64, EdgeKind)>>,
}

fn note(why: &mut HashMap<u64, (f64, String)>, id: u64, v: f64, s: impl FnOnce() -> String) {
    let w = why.entry(id).or_insert((0.0, String::new()));
    if v > w.0 {
        *w = (v, s());
    }
}

pub fn graph_score(ctx: &Ctx, store: &mut dyn GraphStore, space: u8, p: &GraphParams) -> anyhow::Result<Scored> {
    let infos = ctx.infos;
    let cp = ctx.cp;
    let id = |e: &Ent| entity_id(space, ctx.repo, e);
    let mut s = Scored::default();

    // Candidates and session-graph statistics.
    let mut last_provide: HashMap<u64, usize> = HashMap::new();
    for i in 0..cp.recent_start {
        for e in &infos[i].provides {
            let eid = id(e);
            s.cand.insert(eid, e.clone());
            last_provide.insert(eid, i);
        }
    }
    let mut degree: HashMap<u64, usize> = HashMap::new();
    let mut edited: HashSet<u64> = HashSet::new();
    for j in 0..cp.t {
        for e in infos[j].all_entities() {
            *degree.entry(id(e)).or_insert(0) += 1;
        }
        if let Some(e) = &infos[j].edited {
            edited.insert(id(e));
        }
    }
    let mut f: HashMap<u64, [f64; NF]> = HashMap::new();
    for (eid, e) in &s.cand {
        let mut v = [0.0; NF];
        v[0] = 1.0;
        v[8] = (1.0 + *degree.get(eid).unwrap_or(&0) as f64).ln() / 10f64.ln();
        v[9] = if edited.contains(eid) { 1.0 } else { 0.0 };
        let age = (cp.recent_start - 1 - last_provide[eid]) as f64;
        v[10] = (-age / 5.0).exp();
        match e.kind {
            Kind::File => v[12] = 1.0,
            Kind::Search => v[13] = 1.0,
            Kind::Cmd => v[14] = 1.0,
            _ => {}
        }
        f.insert(*eid, v);
    }

    // Focus: entities touched in the last few steps, decayed by age.
    let mut fw: HashMap<u64, (Ent, f64)> = HashMap::new();
    let lo = cp.t.saturating_sub(ctx.focus_steps.max(1)).max(cp.recent_start);
    for j in lo..cp.t {
        let w = p.focus_decay.powi((cp.t - 1 - j) as i32);
        for e in infos[j].all_entities() {
            let entry = fw.entry(id(e)).or_insert((e.clone(), 0.0));
            entry.1 = entry.1.max(w);
        }
    }
    let mut focus: Vec<(u64, Ent, f64)> = fw.into_iter().map(|(k, (e, w))| (k, e, w)).collect();
    focus.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap().then(a.0.cmp(&b.0)));
    focus.truncate(16);
    let focus_ids: HashMap<u64, (Ent, f64)> = focus.iter().map(|(k, e, w)| (*k, (e.clone(), *w))).collect();

    for (fid, (fe, w)) in &focus_ids {
        if let Some(v) = f.get_mut(fid) {
            v[1] = v[1].max(*w);
            note(&mut s.why, *fid, *w * 2.0, || format!("in focus: {}", fe.label()));
        }
    }
    // Session graph: co-occurrence with focus entities in earlier steps.
    let mut sess = HashMap::<u64, f64>::new();
    for j in 0..cp.t {
        let ents: Vec<&Ent> = infos[j].all_entities().into_iter().collect();
        if ents.len() < 2 {
            continue;
        }
        let norm = (ents.len() as f64).sqrt();
        for a in &ents {
            let Some((fe, w)) = focus_ids.get(&id(a)) else { continue };
            for e in &ents {
                let eid = id(e);
                if eid != id(a) && s.cand.contains_key(&eid) {
                    *sess.entry(eid).or_insert(0.0) += w / norm;
                    note(&mut s.why, eid, w / norm, || format!("{} ~session~ {}", fe.label(), e.label()));
                }
            }
        }
    }
    for (eid, x) in sess {
        if let Some(v) = f.get_mut(&eid) {
            v[2] = x.tanh();
        }
    }

    if p.use_repo {
        // Repo graph in HydraDB, one hop from focus, weighted by learned edges.
        let src_ids: Vec<u64> = focus.iter().map(|(k, _, _)| *k).collect();
        let edges = store.neighbors(&src_ids)?;
        let mut acc: HashMap<(u64, usize), f64> = HashMap::new();
        let mut mid: HashMap<u64, f64> = HashMap::new();
        for e in &edges {
            let Some((fe, w)) = focus_ids.get(&e.src) else { continue };
            let v = w * edge_strength(e.kind, e.w, e.n);
            let slot = match e.kind {
                EdgeKind::Co => 3,
                EdgeKind::Mentions => 4,
                EdgeKind::Next => 5,
                EdgeKind::FixedBy => 6,
            };
            if let Some(ce) = s.cand.get(&e.dst) {
                *acc.entry((e.dst, slot)).or_insert(0.0) += v;
                s.contrib.entry(e.dst).or_default().push((e.src, e.dst, e.kind));
                note(&mut s.why, e.dst, v * 1.5, || format!("{} -{}-> {}", fe.label(), e.kind.rel_type(), ce.label()));
            }
            if !focus_ids.contains_key(&e.dst) {
                *mid.entry(e.dst).or_insert(0.0) += v;
            }
        }
        // Two hops through the strongest intermediate entities.
        let mut mids: Vec<(u64, f64)> = mid.into_iter().collect();
        mids.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap().then(a.0.cmp(&b.0)));
        mids.truncate(8);
        let mid_w: HashMap<u64, f64> = mids.iter().cloned().collect();
        // Degree of each candidate in the repo graph (how central it is across sessions).
        let mut ids: Vec<u64> = mids.iter().map(|m| m.0).collect();
        ids.extend(s.cand.keys().copied().filter(|k| !mid_w.contains_key(k)));
        let mut deg: HashMap<u64, usize> = HashMap::new();
        for e in store.neighbors(&ids)? {
            if e.kind == EdgeKind::Co {
                *deg.entry(e.src).or_insert(0) += 1;
            }
            if let Some(mw) = mid_w.get(&e.src) {
                if s.cand.contains_key(&e.dst) {
                    *acc.entry((e.dst, 7)).or_insert(0.0) += mw * edge_strength(e.kind, e.w, e.n);
                }
            }
        }
        for ((eid, slot), x) in acc {
            if let Some(v) = f.get_mut(&eid) {
                v[slot] = x.tanh();
            }
        }
        for (eid, d) in deg {
            if let Some(v) = f.get_mut(&eid) {
                v[11] = (1.0 + d as f64).ln() / 50f64.ln();
            }
        }
    }

    for (eid, v) in &f {
        let z: f64 = v.iter().zip(&p.weights).map(|(a, b)| a * b).sum();
        s.scores.insert(*eid, sigmoid(z));
    }
    s.features = f;
    s.focus = focus;
    Ok(s)
}

/// Greedy budgeted selection of evicted steps, maximising expected coverage of
/// the entities the next steps will need.
pub fn select_scored(ctx: &Ctx, s: &Scored, space: u8) -> Selection {
    let infos = ctx.infos;
    let cp = ctx.cp;
    let id = |e: &Ent| entity_id(space, ctx.repo, e);
    let mut latest: HashMap<u64, usize> = HashMap::new();
    for i in 0..cp.recent_start {
        for e in &infos[i].provides {
            latest.insert(id(e), i);
        }
    }
    let mut covered: HashSet<u64> = HashSet::new();
    let mut left = ctx.budget;
    let mut sel = Selection::default();
    loop {
        let mut best: Option<(f64, usize, u64)> = None;
        for i in 0..cp.recent_start {
            if sel.items.contains(&i) || infos[i].tokens > left {
                continue;
            }
            let mut val = 0.0;
            let mut top: (f64, u64) = (0.0, 0);
            for e in &infos[i].provides {
                let eid = id(e);
                if covered.contains(&eid) {
                    continue;
                }
                let sc = s.scores.get(&eid).copied().unwrap_or(0.0);
                let fresh = if latest.get(&eid) == Some(&i) { 1.0 } else { 0.5 };
                val += sc * fresh;
                if sc * fresh > top.0 {
                    top = (sc * fresh, eid);
                }
            }
            if val <= 0.02 {
                continue;
            }
            let density = val / infos[i].tokens.max(300) as f64 + 1e-9 * i as f64;
            if best.map(|b| density > b.0).unwrap_or(true) {
                best = Some((density, i, top.1));
            }
        }
        let Some((_, i, top)) = best else { break };
        for e in &infos[i].provides {
            covered.insert(id(e));
        }
        left -= infos[i].tokens;
        sel.items.push(i);
        if let Some((_, w)) = s.why.get(&top) {
            sel.why.insert(i, w.clone());
        }
    }
    sel
}
