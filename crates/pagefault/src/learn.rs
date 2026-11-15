//! Continual learning on the repo graph.
//!
//! `ingest` adds what a finished session observed (co-occurrence and structural
//! edges). With learning on, the learner also replays the session's own
//! compaction points against the current graph and corrects itself:
//!   * page fault (needed, not kept) -> create/strengthen NEXT(focus -> needed)
//!   * waste (kept, not needed)      -> weaken the NEXT edges that pulled it in
//!   * error later fixed by an edit  -> FIXED_BY(error -> file)
//! Every change is written to the store (HydraDB) in batches.

use crate::extract::{Ent, Kind, StepInfo};
use crate::graph::{entity_id, Edge, EdgeKind, GraphStore, MemGraph};
use crate::policy::{graph_score, select_scored, Ctx, GraphParams, NF, PRIOR};
use crate::sim::{compaction_points, CompactionPoint, SimCfg};
use std::collections::{BTreeSet, HashMap, HashSet};

pub struct Learner {
    pub space: u8,
    pub repo: String,
    pub learn: bool,
    pub eta: f64,
    pub mirror: MemGraph,
    pub params: GraphParams,
    pub lr: f64,
    pub l2: f64,
    pub pos_weight: f64,
    pub updates: usize,
    pub sessions: usize,
    pub stats: LearnStats,
    /// Keep a plain per-entity page-fault counter in `params.kv` (ablation).
    pub kv: bool,
    pub need_count: HashMap<u64, f64>,
    /// Fault counts changed since the last `take_dirty_faults` (to persist).
    pub dirty_faults: HashSet<u64>,
}

#[derive(Clone, Debug, Default, serde::Serialize, serde::Deserialize)]
pub struct LearnStats {
    pub next_created: usize,
    pub next_strengthened: usize,
    pub next_weakened: usize,
    pub fixed_by: usize,
    pub faults_seen: usize,
}

impl Learner {
    pub fn new(space: u8, repo: &str, learn: bool) -> Self {
        Learner {
            space,
            repo: repo.to_string(),
            learn,
            eta: 0.25,
            mirror: MemGraph::new(),
            params: GraphParams::default(),
            lr: 0.05,
            l2: 0.01,
            pos_weight: 2.0,
            updates: 0,
            sessions: 0,
            stats: LearnStats::default(),
            kv: false,
            need_count: HashMap::new(),
            dirty_faults: HashSet::new(),
        }
    }

    /// Rebuild a learner from what the store persisted: entities, edges, fault
    /// counts and feature weights. A restarted service resumes learning here
    /// instead of starting from zero (and overwriting HydraDB's edge weights).
    pub fn from_state(space: u8, repo: &str, ents: &[(u64, Ent, f64)], edges: &[Edge], weights: Option<&[f64]>) -> Self {
        let mut l = Learner::with_params(space, repo, GraphParams::default(), true);
        for (id, e, faults) in ents {
            l.mirror.add_entity(*id, e);
            if *faults > 0.0 {
                l.need_count.insert(*id, *faults);
            }
        }
        for e in edges {
            l.mirror.upsert(e.clone());
        }
        if let Some(w) = weights {
            if w.len() == NF {
                l.params.weights = w.to_vec();
            }
        }
        l.params.kv = Some(l.need_count.clone());
        l
    }

    /// Fault counts changed since the last call, as (entity id, count).
    pub fn take_dirty_faults(&mut self) -> Vec<(u64, f64)> {
        let ids: Vec<u64> = self.dirty_faults.drain().collect();
        ids.into_iter().map(|id| (id, self.need_count.get(&id).copied().unwrap_or(0.0))).collect()
    }

    /// A learner for an ablation arm: same online learning, different features.
    pub fn with_params(space: u8, repo: &str, params: GraphParams, kv: bool) -> Self {
        let mut l = Learner::new(space, repo, true);
        l.params = params;
        l.kv = kv;
        l
    }

    fn id(&self, e: &Ent) -> u64 {
        entity_id(self.space, &self.repo, e)
    }

    fn bump(&mut self, dirty: &mut HashSet<(u64, u64, EdgeKind)>, src: u64, dst: u64, kind: EdgeKind) {
        if src == dst {
            return;
        }
        let mut e = self.mirror.get(src, dst, kind).cloned().unwrap_or(Edge { src, dst, kind, w: 0.0, n: 0 });
        e.n += 1;
        e.w = e.n as f64 / (e.n as f64 + 2.0);
        self.mirror.upsert(e);
        dirty.insert((src, dst, kind));
    }

    fn flush(&self, store: &mut dyn GraphStore, new_ents: Vec<(u64, Ent)>, dirty: HashSet<(u64, u64, EdgeKind)>) -> anyhow::Result<()> {
        if !new_ents.is_empty() {
            store.write_entities(&self.repo, &new_ents)?;
        }
        let mut edges: Vec<Edge> = dirty.into_iter().filter_map(|k| self.mirror.edges.get(&k).cloned()).collect();
        edges.sort_by_key(|e| (e.src, e.dst, e.kind));
        if !edges.is_empty() {
            store.write_edges(&edges)?;
        }
        Ok(())
    }

    /// Ingest a finished session. Returns the session's compaction points.
    pub fn ingest(&mut self, infos: &[StepInfo], resolved: bool, cfg: &SimCfg, store: &mut dyn GraphStore) -> anyhow::Result<Vec<CompactionPoint>> {
        let mut dirty = HashSet::new();
        let mut new_ents = Vec::new();
        let cps = compaction_points(infos, cfg);

        // Learning first, against the graph as it was before this session.
        if self.learn {
            for cp in &cps {
                self.learn_point(infos, cp, cfg, store, &mut dirty)?;
            }
            self.fixed_by(infos, resolved, &mut dirty);
        }
        if self.kv {
            for cp in &cps {
                for n in &cp.needed {
                    let id = self.id(&n.ent);
                    *self.need_count.entry(id).or_insert(0.0) += 1.0;
                    self.dirty_faults.insert(id);
                }
            }
            self.params.kv = Some(self.need_count.clone());
        }

        for s in infos {
            let ents: Vec<&Ent> = s.all_entities().into_iter().take(12).collect();
            for e in &ents {
                let id = self.id(e);
                if !self.mirror.ents.contains_key(&id) {
                    self.mirror.add_entity(id, e);
                    new_ents.push((id, (*e).clone()));
                }
            }
            for a in &ents {
                for b in &ents {
                    if a != b {
                        let (ia, ib) = (self.id(a), self.id(b));
                        self.bump(&mut dirty, ia, ib, EdgeKind::Co);
                    }
                }
            }
            let errors: Vec<&Ent> = s.mentions.iter().filter(|e| e.kind == Kind::Error).collect();
            for c in s.fetches.iter().filter(|e| e.kind == Kind::Cmd) {
                for er in &errors {
                    let (ic, ie) = (self.id(c), self.id(er));
                    self.bump(&mut dirty, ic, ie, EdgeKind::Mentions);
                }
            }
            for f in s.mentions.iter().filter(|e| e.kind == Kind::File) {
                for er in &errors {
                    let (ie, iff) = (self.id(er), self.id(f));
                    self.bump(&mut dirty, ie, iff, EdgeKind::Mentions);
                }
            }
            for sym in s.mentions.iter().filter(|e| e.kind == Kind::Symbol) {
                if let Some((_, file)) = sym.key.split_once('@') {
                    let f = Ent::new(Kind::File, file);
                    let (is, iff) = (self.id(sym), self.id(&f));
                    self.bump(&mut dirty, is, iff, EdgeKind::Mentions);
                    self.bump(&mut dirty, iff, is, EdgeKind::Mentions);
                }
            }
        }
        self.flush(store, new_ents, dirty)?;
        self.sessions += 1;
        Ok(cps)
    }

    fn learn_point(
        &mut self,
        infos: &[StepInfo],
        cp: &CompactionPoint,
        cfg: &SimCfg,
        store: &mut dyn GraphStore,
        dirty: &mut HashSet<(u64, u64, EdgeKind)>,
    ) -> anyhow::Result<()> {
        if cp.needed.is_empty() {
            return Ok(());
        }
        let ctx = Ctx { repo: &self.repo, infos, cp, budget: cfg.budget, focus_steps: cfg.focus_steps };
        let scored = graph_score(&ctx, store, self.space, &self.params)?;
        let sel = select_scored(&ctx, &scored, self.space);
        let kept: HashSet<u64> = sel.items.iter().flat_map(|&i| infos[i].provides.iter()).map(|e| self.id(e)).collect();
        let needed: HashSet<u64> = cp.needed.iter().map(|n| self.id(&n.ent)).collect();
        let focus: Vec<(u64, f64)> = scored.focus.iter().filter(|f| f.2 >= 0.2).take(8).map(|f| (f.0, f.2)).collect();

        // Fit the per-repo scoring weights: logistic regression, one SGD pass
        // over every candidate entity at this compaction point.
        for (eid, feats) in &scored.features {
            let y = if needed.contains(eid) { 1.0 } else { 0.0 };
            let p = scored.scores.get(eid).copied().unwrap_or(0.5);
            let g = (y - p) * if y > 0.5 { self.pos_weight } else { 1.0 };
            for k in 0..NF {
                let reg = if k == 0 { 0.0 } else { self.l2 * (self.params.weights[k] - PRIOR[k]) };
                self.params.weights[k] += self.lr * (g * feats[k] - reg);
            }
        }
        self.updates += 1;

        // Faults and hits: reinforce focus -> needed.
        for n in &needed {
            if !kept.contains(n) {
                self.stats.faults_seen += 1;
            }
            for &(f, fw) in &focus {
                if f == *n {
                    continue;
                }
                let key = (f, *n, EdgeKind::Next);
                let mut e = self.mirror.get(f, *n, EdgeKind::Next).cloned().unwrap_or_else(|| {
                    self.stats.next_created += 1;
                    Edge { src: f, dst: *n, kind: EdgeKind::Next, w: 0.0, n: 0 }
                });
                if e.n > 0 {
                    self.stats.next_strengthened += 1;
                }
                e.n += 1;
                e.w = (e.w + self.eta * fw * (1.0 - e.w / 3.0)).min(3.0);
                self.mirror.upsert(e);
                dirty.insert(key);
            }
        }
        // Waste: weaken NEXT edges that pulled in something not needed.
        for k in kept.difference(&needed) {
            if let Some(c) = scored.contrib.get(k) {
                for &(src, dst, kind) in c {
                    if kind != EdgeKind::Next {
                        continue;
                    }
                    if let Some(mut e) = self.mirror.get(src, dst, kind).cloned() {
                        e.w *= 1.0 - self.eta * 0.5;
                        self.mirror.upsert(e);
                        dirty.insert((src, dst, kind));
                        self.stats.next_weakened += 1;
                    }
                }
            }
        }
        Ok(())
    }

    /// Errors seen before a file was edited, in a session that ended resolved,
    /// point at that file.
    fn fixed_by(&mut self, infos: &[StepInfo], resolved: bool, dirty: &mut HashSet<(u64, u64, EdgeKind)>) {
        if !resolved {
            return;
        }
        let mut errors: BTreeSet<Ent> = BTreeSet::new();
        let mut pairs: Vec<(u64, u64)> = Vec::new();
        for s in infos {
            if let Some(f) = &s.edited {
                for e in &errors {
                    pairs.push((self.id(e), self.id(f)));
                }
            }
            for e in s.mentions.iter().filter(|e| e.kind == Kind::Error) {
                errors.insert(e.clone());
            }
        }
        let mut seen = HashSet::new();
        for (e, f) in pairs {
            if !seen.insert((e, f)) {
                continue;
            }
            let mut ed = self.mirror.get(e, f, EdgeKind::FixedBy).cloned().unwrap_or(Edge { src: e, dst: f, kind: EdgeKind::FixedBy, w: 0.0, n: 0 });
            ed.n += 1;
            ed.w = (ed.w + self.eta).min(2.0);
            self.mirror.upsert(ed);
            dirty.insert((e, f, EdgeKind::FixedBy));
            self.stats.fixed_by += 1;
        }
    }

    pub fn summary(&self) -> HashMap<String, usize> {
        let mut m = HashMap::new();
        for (k, v) in self.mirror.kind_counts() {
            m.insert(format!("edges_{}", k.rel_type()), v);
        }
        for (k, v) in self.mirror.entity_counts() {
            m.insert(format!("entities_{}", k.as_str()), v);
        }
        m.insert("sessions".into(), self.sessions);
        m
    }
}
