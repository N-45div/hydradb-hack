//! Repo memory graph: entities shared across sessions of one repository, with
//! learned edge weights. `MemGraph` is the in-process mirror; `HydraStore`
//! (see `hydra.rs`) persists and traverses the same graph in HydraDB.

use crate::extract::{Ent, Kind};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Relationship types. HydraDB `MATCH` patterns take exactly one type, so each
/// kind is its own relationship type and also stored as the `kind` property.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub enum EdgeKind {
    /// Entities seen in the same step (accumulated counts).
    Co,
    /// Structural: traceback error -> file, command -> error, symbol -> file.
    Mentions,
    /// Learned: when the agent is working on `src`, it soon needs `dst` again.
    Next,
    /// Learned across sessions: error signature -> file whose edit made it go away.
    FixedBy,
}

impl EdgeKind {
    pub const ALL: [EdgeKind; 4] = [EdgeKind::Co, EdgeKind::Mentions, EdgeKind::Next, EdgeKind::FixedBy];
    pub fn rel_type(&self) -> &'static str {
        match self {
            EdgeKind::Co => "CO",
            EdgeKind::Mentions => "MENTIONS",
            EdgeKind::Next => "NEXT",
            EdgeKind::FixedBy => "FIXED_BY",
        }
    }
    pub fn parse(s: &str) -> Option<EdgeKind> {
        EdgeKind::ALL.into_iter().find(|k| k.rel_type() == s)
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Edge {
    pub src: u64,
    pub dst: u64,
    pub kind: EdgeKind,
    /// Learned weight (NEXT/FIXED_BY) or saturating strength derived from `n` (CO/MENTIONS).
    pub w: f64,
    /// Observation count.
    pub n: u64,
}

/// 64-bit FNV-1a: stable across Rust versions and platforms.
fn fnv1a(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for b in bytes {
        h ^= *b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

/// HydraDB node ids are non-negative integers. The low 56 bits hash the
/// entity; bits 56..62 hold a `space` so several independent graphs (one per
/// experiment arm, or per tenant) live side by side in one HydraDB graph.
pub fn entity_id(space: u8, repo: &str, ent: &Ent) -> u64 {
    let h = fnv1a(format!("{}|{}|{}", repo, ent.kind.as_str(), ent.key).as_bytes());
    ((space as u64 & 0x7f) << 56) | (h & 0x00ff_ffff_ffff_ffff)
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct MemGraph {
    pub ents: HashMap<u64, Ent>,
    pub edges: HashMap<(u64, u64, EdgeKind), Edge>,
    pub out: HashMap<u64, Vec<(u64, EdgeKind)>>,
}

impl MemGraph {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add_entity(&mut self, id: u64, ent: &Ent) {
        self.ents.entry(id).or_insert_with(|| ent.clone());
    }

    pub fn get(&self, src: u64, dst: u64, kind: EdgeKind) -> Option<&Edge> {
        self.edges.get(&(src, dst, kind))
    }

    pub fn upsert(&mut self, e: Edge) {
        let key = (e.src, e.dst, e.kind);
        if !self.edges.contains_key(&key) {
            self.out.entry(e.src).or_default().push((e.dst, e.kind));
        }
        self.edges.insert(key, e);
    }

    pub fn neighbors(&self, src: u64) -> Vec<Edge> {
        self.out
            .get(&src)
            .map(|v| v.iter().filter_map(|(d, k)| self.edges.get(&(src, *d, *k)).cloned()).collect())
            .unwrap_or_default()
    }

    pub fn edge_count(&self) -> usize {
        self.edges.len()
    }

    pub fn kind_counts(&self) -> HashMap<EdgeKind, usize> {
        let mut m = HashMap::new();
        for e in self.edges.values() {
            *m.entry(e.kind).or_insert(0) += 1;
        }
        m
    }

    pub fn entity_counts(&self) -> HashMap<Kind, usize> {
        let mut m = HashMap::new();
        for e in self.ents.values() {
            *m.entry(e.kind).or_insert(0) += 1;
        }
        m
    }
}

/// String key HydraDB's `algo.MSpaths` selects vertices by (`sel` property).
/// Scoped by repo so repos sharing one space never match each other's vertices.
pub fn selector(repo: &str, e: &Ent) -> String {
    format!("{repo}|{}", e.label())
}

/// A path returned by `GraphStore::paths`: vertex ids (with labels when the
/// store knows them) and the relationships between consecutive vertices.
#[derive(Clone, Debug)]
pub struct GPath {
    pub nodes: Vec<(u64, Option<String>)>,
    pub edges: Vec<Edge>,
}

impl GPath {
    /// Canonical order key, so float sums do not depend on store result order.
    pub fn sort_key(&self) -> (Vec<u64>, Vec<&'static str>) {
        (self.nodes.iter().map(|n| n.0).collect(), self.edges.iter().map(|e| e.kind.rel_type()).collect())
    }
}

/// Everything a learner needs to resume from the store after a restart.
#[derive(Default)]
pub struct SpaceState {
    /// (id, entity, repo, fault count)
    pub ents: Vec<(u64, Ent, String, f64)>,
    pub edges: Vec<Edge>,
    /// repo -> learned feature weights
    pub weights: Vec<(String, Vec<f64>)>,
}

/// Persistence + traversal backend for the repo graph.
pub trait GraphStore: Send {
    fn name(&self) -> String;
    /// Delete everything in `space`.
    fn clear_space(&mut self, space: u8, ids: &[u64]) -> anyhow::Result<()>;
    fn write_entities(&mut self, repo: &str, ents: &[(u64, Ent)]) -> anyhow::Result<()>;
    fn write_edges(&mut self, edges: &[Edge]) -> anyhow::Result<()>;
    /// Out-edges of every id in `ids`, all relationship types, with weights.
    fn neighbors(&mut self, ids: &[u64]) -> anyhow::Result<Vec<Edge>>;
    /// All simple paths of 1..=`max_len` edges over `kinds` from any source to
    /// any target (self-pairs excluded), at most `per_pair` per pair. Sources and
    /// targets are (id, selector). HydraDB answers this with one `algo.MSpaths`.
    fn paths(&mut self, sources: &[(u64, String)], targets: &[(u64, String)], kinds: &[EdgeKind], max_len: usize, per_pair: usize) -> anyhow::Result<Vec<GPath>>;
    /// Out-degree over `kind` for every vertex in `space` (HydraDB: `algo.degree`).
    fn degrees(&mut self, space: u8, kind: EdgeKind) -> anyhow::Result<std::sync::Arc<HashMap<u64, usize>>>;
    /// Persist a learner's weights and fault counts (no-op for the in-memory store).
    fn save_learner(&mut self, _space: u8, _repo: &str, _weights: &[f64], _faults: &[(u64, f64)]) -> anyhow::Result<()> {
        Ok(())
    }
    /// Read back everything stored in `space` (None for the in-memory store).
    fn load_space(&mut self, _space: u8) -> anyhow::Result<Option<SpaceState>> {
        Ok(None)
    }
}

/// In-memory store: same contract as HydraDB, used for tests and as a fallback.
#[derive(Default)]
pub struct MemStore {
    pub g: MemGraph,
    degree_cache: HashMap<(u8, EdgeKind), std::sync::Arc<HashMap<u64, usize>>>,
}

impl GraphStore for MemStore {
    fn name(&self) -> String {
        "memory".into()
    }
    fn clear_space(&mut self, space: u8, _ids: &[u64]) -> anyhow::Result<()> {
        let keep = |id: &u64| (id >> 56) as u8 != space;
        self.g.ents.retain(|k, _| keep(k));
        self.g.edges.retain(|(s, _, _), _| keep(s));
        self.g.out.retain(|k, _| keep(k));
        self.degree_cache.clear();
        Ok(())
    }
    fn write_entities(&mut self, _repo: &str, ents: &[(u64, Ent)]) -> anyhow::Result<()> {
        for (id, e) in ents {
            self.g.add_entity(*id, e);
        }
        Ok(())
    }
    fn write_edges(&mut self, edges: &[Edge]) -> anyhow::Result<()> {
        for e in edges {
            self.g.upsert(e.clone());
        }
        if !edges.is_empty() {
            self.degree_cache.clear();
        }
        Ok(())
    }
    fn neighbors(&mut self, ids: &[u64]) -> anyhow::Result<Vec<Edge>> {
        Ok(ids.iter().flat_map(|id| self.g.neighbors(*id)).collect())
    }
    fn paths(&mut self, sources: &[(u64, String)], targets: &[(u64, String)], kinds: &[EdgeKind], max_len: usize, per_pair: usize) -> anyhow::Result<Vec<GPath>> {
        let tset: std::collections::HashSet<u64> = targets.iter().map(|t| t.0).collect();
        let label = |id: u64| self.g.ents.get(&id).map(|e| e.label());
        let mut out = Vec::new();
        let mut per: HashMap<(u64, u64), usize> = HashMap::new();
        for (s, _) in sources {
            let mut firsts: Vec<Edge> = self.g.neighbors(*s).into_iter().filter(|e| kinds.contains(&e.kind)).collect();
            firsts.sort_by_key(|e| (e.dst, e.kind));
            for e1 in &firsts {
                if e1.dst != *s && tset.contains(&e1.dst) {
                    let c = per.entry((*s, e1.dst)).or_insert(0);
                    if *c < per_pair {
                        *c += 1;
                        out.push(GPath { nodes: vec![(*s, label(*s)), (e1.dst, label(e1.dst))], edges: vec![e1.clone()] });
                    }
                }
            }
            if max_len < 2 {
                continue;
            }
            for e1 in &firsts {
                if e1.dst == *s {
                    continue;
                }
                let mut seconds: Vec<Edge> = self.g.neighbors(e1.dst).into_iter().filter(|e| kinds.contains(&e.kind)).collect();
                seconds.sort_by_key(|e| (e.dst, e.kind));
                for e2 in seconds {
                    if e2.dst == *s || e2.dst == e1.dst || !tset.contains(&e2.dst) {
                        continue;
                    }
                    let c = per.entry((*s, e2.dst)).or_insert(0);
                    if *c < per_pair {
                        *c += 1;
                        out.push(GPath {
                            nodes: vec![(*s, label(*s)), (e1.dst, label(e1.dst)), (e2.dst, label(e2.dst))],
                            edges: vec![e1.clone(), e2.clone()],
                        });
                    }
                }
            }
        }
        Ok(out)
    }
    fn degrees(&mut self, space: u8, kind: EdgeKind) -> anyhow::Result<std::sync::Arc<HashMap<u64, usize>>> {
        if let Some(d) = self.degree_cache.get(&(space, kind)) {
            return Ok(d.clone());
        }
        let mut d: HashMap<u64, usize> = HashMap::new();
        for e in self.g.edges.values() {
            if e.kind == kind && (e.src >> 56) as u8 == space {
                *d.entry(e.src).or_insert(0) += 1;
            }
        }
        let d = std::sync::Arc::new(d);
        self.degree_cache.insert((space, kind), d.clone());
        Ok(d)
    }
}
