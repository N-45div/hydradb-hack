//! Compaction replay. Walks a recorded session with a small context budget,
//! finds the points where a harness would have to compact, and records which
//! earlier information the agent actually went back for afterwards.
//!
//! Ground truth for a compaction at step `t`:
//!   needed(t) = entities the agent fetches again in steps (t, t+window]
//!               whose content an *evicted* step had already shown,
//!               and which the raw recent window does not show.
//! A policy that keeps a step showing that entity saves the re-fetch (hit);
//! otherwise the agent pays a page fault: it re-opens the file, re-runs the
//! command or repeats the search.

use crate::extract::{Ent, Kind, StepInfo};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeSet, HashSet};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SimCfg {
    /// Context size that triggers compaction (tokens).
    pub ctx_limit: usize,
    /// Most recent tokens every arm keeps verbatim (pi's `keepRecentTokens`).
    pub recent: usize,
    /// Tokens each policy may fill with older material.
    pub budget: usize,
    /// How many steps ahead count as "the next step needs it".
    pub window: usize,
    /// Steps at the end of the recent window that define the current focus.
    pub focus_steps: usize,
}

impl Default for SimCfg {
    fn default() -> Self {
        SimCfg { ctx_limit: 8_000, recent: 2_000, budget: 2_000, window: 20, focus_steps: 3 }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Need {
    pub ent: Ent,
    /// Tokens the agent spent re-fetching it.
    pub refetch_tokens: usize,
    pub at: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CompactionPoint {
    /// First step after the compaction.
    pub t: usize,
    /// Steps `recent_start..t` stay verbatim; `0..recent_start` are evicted.
    pub recent_start: usize,
    pub needed: Vec<Need>,
}

pub fn provided_by(infos: &[StepInfo], range: std::ops::Range<usize>) -> HashSet<&Ent> {
    infos[range].iter().flat_map(|s| s.provides.iter()).collect()
}

pub fn compaction_points(infos: &[StepInfo], cfg: &SimCfg) -> Vec<CompactionPoint> {
    let mut out = Vec::new();
    let mut ctx = 0usize;
    let n = infos.len();
    for i in 0..n {
        ctx += infos[i].tokens;
        if ctx <= cfg.ctx_limit || i + 2 >= n {
            continue;
        }
        let t = i + 1;
        let mut recent_start = t;
        let mut acc = 0;
        while recent_start > 0 {
            let tok = infos[recent_start - 1].tokens;
            if acc + tok > cfg.recent && recent_start < t {
                break;
            }
            acc += tok;
            recent_start -= 1;
        }
        if recent_start == 0 {
            ctx = acc;
            continue;
        }
        let older = provided_by(infos, 0..recent_start);
        let recent = provided_by(infos, recent_start..t);
        let mut seen = BTreeSet::new();
        let mut needed = Vec::new();
        for u in t..(t + cfg.window).min(n) {
            for e in &infos[u].fetches {
                if !older.contains(e) || recent.contains(e) || seen.contains(e) {
                    continue;
                }
                // Re-running a command after an edit is a new experiment, not a
                // re-fetch of something forgotten. Only count unchanged re-runs.
                if e.kind == Kind::Cmd {
                    let last = (0..recent_start).rev().find(|&j| infos[j].provides.contains(e)).unwrap_or(0);
                    if infos[last + 1..u].iter().any(|s| s.edited.is_some()) {
                        continue;
                    }
                }
                seen.insert(e.clone());
                needed.push(Need { ent: e.clone(), refetch_tokens: infos[u].tokens, at: u });
            }
        }
        out.push(CompactionPoint { t, recent_start, needed });
        ctx = acc + cfg.budget;
    }
    out
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Outcome {
    pub compactions: usize,
    pub needed: usize,
    pub hits: usize,
    pub rework_tokens: usize,
    pub kept_tokens: usize,
    pub waste_tokens: usize,
}

impl Outcome {
    pub fn add(&mut self, o: &Outcome) {
        self.compactions += o.compactions;
        self.needed += o.needed;
        self.hits += o.hits;
        self.rework_tokens += o.rework_tokens;
        self.kept_tokens += o.kept_tokens;
        self.waste_tokens += o.waste_tokens;
    }
    pub fn recall(&self) -> f64 {
        if self.needed == 0 {
            0.0
        } else {
            self.hits as f64 / self.needed as f64
        }
    }
    pub fn faults_per_compaction(&self) -> f64 {
        if self.compactions == 0 {
            0.0
        } else {
            (self.needed - self.hits) as f64 / self.compactions as f64
        }
    }
    pub fn rework_per_compaction(&self) -> f64 {
        if self.compactions == 0 {
            0.0
        } else {
            self.rework_tokens as f64 / self.compactions as f64
        }
    }
}

pub fn score_selection(infos: &[StepInfo], cp: &CompactionPoint, selected: &[usize]) -> Outcome {
    let covered: HashSet<&Ent> = selected.iter().flat_map(|&i| infos[i].provides.iter()).collect();
    let needed: HashSet<&Ent> = cp.needed.iter().map(|n| &n.ent).collect();
    let mut o = Outcome { compactions: 1, needed: cp.needed.len(), ..Default::default() };
    for n in &cp.needed {
        if covered.contains(&n.ent) {
            o.hits += 1;
        } else {
            o.rework_tokens += n.refetch_tokens;
        }
    }
    for &i in selected {
        o.kept_tokens += infos[i].tokens;
        if !infos[i].provides.iter().any(|e| needed.contains(e)) {
            o.waste_tokens += infos[i].tokens;
        }
    }
    o
}
