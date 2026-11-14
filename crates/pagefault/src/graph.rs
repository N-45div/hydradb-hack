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

/// Persistence + traversal backend for the repo graph.
pub trait GraphStore: Send {
    fn name(&self) -> String;
    /// Delete everything in `space`.
    fn clear_space(&mut self, space: u8, ids: &[u64]) -> anyhow::Result<()>;
    fn write_entities(&mut self, ents: &[(u64, Ent)]) -> anyhow::Result<()>;
    fn write_edges(&mut self, edges: &[Edge]) -> anyhow::Result<()>;
    /// Out-edges of every id in `ids`, all relationship types, with weights.
    fn neighbors(&mut self, ids: &[u64]) -> anyhow::Result<Vec<Edge>>;
}

/// In-memory store: same contract as HydraDB, used for tests and as a fallback.
#[derive(Default)]
pub struct MemStore {
    pub g: MemGraph,
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
        Ok(())
    }
    fn write_entities(&mut self, ents: &[(u64, Ent)]) -> anyhow::Result<()> {
        for (id, e) in ents {
            self.g.add_entity(*id, e);
        }
        Ok(())
    }
    fn write_edges(&mut self, edges: &[Edge]) -> anyhow::Result<()> {
        for e in edges {
            self.g.upsert(e.clone());
        }
        Ok(())
    }
    fn neighbors(&mut self, ids: &[u64]) -> anyhow::Result<Vec<Edge>> {
        Ok(ids.iter().flat_map(|id| self.g.neighbors(*id)).collect())
    }
}
