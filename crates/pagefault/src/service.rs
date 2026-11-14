//! The compaction API, independent of transport. `bin/server.rs` exposes it
//! over HTTP; the pi extension and MCP clients call it.

use crate::adapter::{tool_step, ToolEvent};
use crate::experiment::{evaluate, fill_recent, ArmStores, Point, RepoSplit};
use crate::extract::{Ent, Extractor, Kind, StepInfo};
use crate::graph::{entity_id, EdgeKind, GraphStore};
use crate::learn::{LearnStats, Learner};
use crate::model::Step;
use crate::policy::{bm25, graph_score, select_scored, Ctx, FEATURES, PRIOR};
use crate::sim::{CompactionPoint, Outcome, SimCfg};
use serde::Serialize;
use std::collections::{BTreeMap, HashMap};

pub struct LiveSession {
    pub id: String,
    pub repo: String,
    pub harness: String,
    pub model: String,
    pub steps: Vec<Step>,
    pub ex: Extractor,
    pub infos: Vec<StepInfo>,
    pub recalls: Vec<String>,
    pub compactions: usize,
}

#[derive(Serialize)]
pub struct Item {
    pub step: usize,
    pub tokens: usize,
    pub action: String,
    pub why: String,
    pub text: String,
}

#[derive(Serialize)]
pub struct CompactResponse {
    pub session_id: String,
    pub evicted_steps: usize,
    pub evicted_tokens: usize,
    pub kept: Vec<Item>,
    pub kept_tokens: usize,
    pub recent_start: usize,
    /// Entities still in memory but not restored; reachable with `recall`.
    pub in_memory: Vec<String>,
    /// Ready-to-use compaction summary for the harness.
    pub summary_md: String,
}

#[derive(Serialize)]
pub struct MemoryView {
    pub repo: String,
    pub summary: HashMap<String, usize>,
    pub weights: Vec<(String, f64, f64)>,
    pub top_next: Vec<(String, String, f64)>,
    pub stats: LearnStats,
}

pub struct Demo {
    pub split: RepoSplit,
    pub cfg: SimCfg,
    pub stat: Learner,
    pub learned: Learner,
    pub stat_store: Box<dyn GraphStore>,
    pub learned_store: Box<dyn GraphStore>,
    pub ingested: usize,
    pub fixed: BTreeMap<String, Outcome>,
    pub curve: Vec<BTreeMap<String, Point>>,
}

impl Demo {
    pub fn new(split: RepoSplit, cfg: SimCfg, mut stat_store: Box<dyn GraphStore>, mut learned_store: Box<dyn GraphStore>) -> anyhow::Result<Self> {
        stat_store.clear_space(120, &[])?;
        learned_store.clear_space(121, &[])?;
        let repo = split.repo.clone();
        let mut d = Demo {
            stat: Learner::new(120, &repo, false),
            learned: Learner::new(121, &repo, true),
            split,
            cfg,
            stat_store,
            learned_store,
            ingested: 0,
            fixed: BTreeMap::new(),
            curve: Vec::new(),
        };
        let mut stores = ArmStores { stat: (d.stat_store.as_mut(), 120), learned: (d.learned_store.as_mut(), 121), learned_params: d.learned.params.clone() };
        d.fixed = evaluate(&d.split, &d.cfg, &["recency", "bm25", "oracle"], &mut stores)?;
        d.point()?;
        Ok(d)
    }

    fn point(&mut self) -> anyhow::Result<()> {
        let mut stores = ArmStores {
            stat: (self.stat_store.as_mut(), 120),
            learned: (self.learned_store.as_mut(), 121),
            learned_params: self.learned.params.clone(),
        };
        let mut res = evaluate(&self.split, &self.cfg, &["graph_static", "graph_learned"], &mut stores)?;
        res.extend(self.fixed.clone());
        let row: BTreeMap<String, Point> = res
            .iter()
            .map(|(k, o)| {
                let mut p = Point::default();
                p.sessions = self.ingested;
                p.recall = o.recall();
                p.faults_per_compaction = o.faults_per_compaction();
                p.rework_tokens_per_compaction = o.rework_per_compaction();
                p.compactions = o.compactions;
                p.needed = o.needed;
                (k.clone(), p)
            })
            .collect();
        self.curve.push(row);
        Ok(())
    }

    /// Ingest the next `n` training sessions into both graphs, then re-evaluate.
    pub fn learn(&mut self, n: usize) -> anyhow::Result<()> {
        let end = (self.ingested + n).min(self.split.train.len());
        for i in self.ingested..end {
            let p = &self.split.train[i];
            self.stat.ingest(&p.infos, p.session.resolved, &self.cfg, self.stat_store.as_mut())?;
            self.learned.ingest(&p.infos, p.session.resolved, &self.cfg, self.learned_store.as_mut())?;
        }
        self.ingested = end;
        self.point()
    }
}

pub struct Service {
    pub store: Box<dyn GraphStore>,
    pub space: u8,
    pub cfg: SimCfg,
    pub learners: HashMap<String, Learner>,
    pub sessions: HashMap<String, LiveSession>,
    pub demo: Option<Demo>,
    next_id: u64,
}

fn clip(s: &str, max_chars: usize) -> String {
    if s.len() <= max_chars {
        return s.to_string();
    }
    let mut end = max_chars;
    while !s.is_char_boundary(end) {
        end -= 1;
    }
    format!("{}\n…[clipped]", &s[..end])
}

impl Service {
    pub fn new(store: Box<dyn GraphStore>, cfg: SimCfg) -> Self {
        Service { store, space: 0, cfg, learners: HashMap::new(), sessions: HashMap::new(), demo: None, next_id: 1 }
    }

    fn learner(&mut self, repo: &str) -> &mut Learner {
        let space = self.space;
        self.learners.entry(repo.to_string()).or_insert_with(|| Learner::new(space, repo, true))
    }

    pub fn create_session(&mut self, repo: &str, harness: &str, model: &str, root: &str) -> String {
        let id = format!("s{}-{}", self.next_id, std::process::id());
        self.next_id += 1;
        self.sessions.insert(
            id.clone(),
            LiveSession {
                id: id.clone(),
                repo: repo.to_string(),
                harness: harness.to_string(),
                model: model.to_string(),
                steps: Vec::new(),
                ex: Extractor::with_root(root),
                infos: Vec::new(),
                recalls: Vec::new(),
                compactions: 0,
            },
        );
        self.learner(repo);
        id
    }

    fn session(&mut self, id: &str) -> anyhow::Result<&mut LiveSession> {
        self.sessions.get_mut(id).ok_or_else(|| anyhow::anyhow!("unknown session {id}"))
    }

    pub fn add_steps(&mut self, id: &str, steps: Vec<Step>) -> anyhow::Result<usize> {
        let s = self.session(id)?;
        for st in steps {
            let info = s.ex.step(&st);
            s.steps.push(st);
            s.infos.push(info);
        }
        Ok(s.steps.len())
    }

    pub fn add_events(&mut self, id: &str, events: Vec<ToolEvent>) -> anyhow::Result<usize> {
        self.add_steps(id, events.iter().map(tool_step).collect())
    }

    pub fn compact(&mut self, id: &str, keep_recent: Option<usize>, budget: Option<usize>) -> anyhow::Result<CompactResponse> {
        let keep_recent = keep_recent.unwrap_or(self.cfg.recent);
        let budget = budget.unwrap_or(self.cfg.budget);
        let focus_steps = self.cfg.focus_steps;
        let space = self.space;
        let (repo, infos, steps) = {
            let s = self.session(id)?;
            s.compactions += 1;
            (s.repo.clone(), s.infos.clone(), s.steps.clone())
        };
        let t = infos.len();
        let mut recent_start = t;
        let mut acc = 0;
        while recent_start > 0 && (acc + infos[recent_start - 1].tokens <= keep_recent || recent_start == t) {
            acc += infos[recent_start - 1].tokens;
            recent_start -= 1;
        }
        let cp = CompactionPoint { t, recent_start, needed: vec![] };
        let params = self.learner(&repo).params.clone();
        let ctx = Ctx { repo: &repo, infos: &infos, cp: &cp, budget, focus_steps };
        let scored = graph_score(&ctx, self.store.as_mut(), space, &params)?;
        let mut sel = select_scored(&ctx, &scored, space);
        fill_recent(&ctx, &mut sel);
        sel.items.sort();
        let kept_set: std::collections::HashSet<u64> =
            sel.items.iter().flat_map(|&i| infos[i].provides.iter()).map(|e| entity_id(space, &repo, e)).collect();
        let mut rest: Vec<(f64, String)> = scored
            .scores
            .iter()
            .filter(|(k, _)| !kept_set.contains(k))
            .filter_map(|(k, v)| scored.cand.get(k).map(|e| (*v, e.label())))
            .collect();
        rest.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
        let in_memory: Vec<String> = rest.into_iter().take(12).map(|x| x.1).collect();
        let kept: Vec<Item> = sel
            .items
            .iter()
            .map(|&i| Item {
                step: i,
                tokens: infos[i].tokens,
                action: steps[i].action.lines().next().unwrap_or("").to_string(),
                why: sel.why.get(&i).cloned().unwrap_or_else(|| "recent".into()),
                text: clip(&format!("$ {}\n{}", steps[i].action, steps[i].observation), infos[i].tokens * 4),
            })
            .collect();
        let evicted_tokens: usize = infos[..recent_start].iter().map(|i| i.tokens).sum();
        let kept_tokens: usize = kept.iter().map(|k| k.tokens).sum();
        let mut md = format!(
            "## Context paged out by pagefault\n{} earlier steps ({} tokens) moved to the HydraDB memory graph. {} steps the next work most likely needs were restored below. Anything else is one `recall` away.\n",
            recent_start, evicted_tokens, kept.len()
        );
        if !in_memory.is_empty() {
            md.push_str(&format!("\nIn memory, not restored: {}\n", in_memory.join(", ")));
        }
        for k in &kept {
            md.push_str(&format!("\n### step {} · {} (kept: {})\n```\n{}\n```\n", k.step, k.action, k.why, k.text));
        }
        Ok(CompactResponse { session_id: id.to_string(), evicted_steps: recent_start, evicted_tokens, kept, kept_tokens, recent_start, in_memory, summary_md: md })
    }

    /// On-demand recall: a page fault. Returns matching steps and records the
    /// miss so the graph learns to restore this next time.
    pub fn recall(&mut self, id: &str, q: &str, k: usize) -> anyhow::Result<Vec<Item>> {
        let space = self.space;
        let (repo, infos, steps) = {
            let s = self.session(id)?;
            s.recalls.push(q.to_string());
            (s.repo.clone(), s.infos.clone(), s.steps.clone())
        };
        if infos.is_empty() {
            return Ok(vec![]);
        }
        let t = infos.len();
        // Score every step against the query with BM25 over the whole session.
        let qstep = StepInfo { text: q.to_string(), ..Default::default() };
        let mut all = infos.clone();
        all.push(qstep);
        let cp = CompactionPoint { t: t + 1, recent_start: t, needed: vec![] };
        let ctx = Ctx { repo: &repo, infos: &all, cp: &cp, budget: 6000, focus_steps: 1 };
        let sel = bm25(&ctx);
        let items: Vec<Item> = sel
            .items
            .into_iter()
            .take(k)
            .map(|i| Item {
                step: i,
                tokens: infos[i].tokens,
                action: steps[i].action.lines().next().unwrap_or("").to_string(),
                why: format!("recall: {q}"),
                text: clip(&format!("$ {}\n{}", steps[i].action, steps[i].observation), 6000),
            })
            .collect();
        // Online learning from the fault: focus -> recalled entities.
        if let Some(first) = items.first() {
            let focus: Vec<Ent> = infos[t.saturating_sub(3)..].iter().flat_map(|s| s.all_entities().into_iter().cloned()).collect();
            let targets: Vec<Ent> = infos[first.step].provides.clone();
            let learner = self.learner(&repo);
            let mut edges = Vec::new();
            for f in &focus {
                for e in &targets {
                    let (fi, ei) = (entity_id(space, &repo, f), entity_id(space, &repo, e));
                    if fi == ei {
                        continue;
                    }
                    let mut ed = learner.mirror.get(fi, ei, EdgeKind::Next).cloned().unwrap_or(crate::graph::Edge { src: fi, dst: ei, kind: EdgeKind::Next, w: 0.0, n: 0 });
                    ed.n += 1;
                    ed.w = (ed.w + learner.eta).min(3.0);
                    learner.mirror.upsert(ed.clone());
                    edges.push(ed);
                }
            }
            learner.stats.faults_seen += 1;
            let ents: Vec<(u64, Ent)> = focus.iter().chain(&targets).map(|e| (entity_id(space, &repo, e), e.clone())).collect();
            self.store.write_entities(&ents)?;
            self.store.write_edges(&edges)?;
        }
        Ok(items)
    }

    /// Warn before re-running something that already failed.
    pub fn check(&mut self, id: &str, command: &str) -> anyhow::Result<Option<String>> {
        let space = self.space;
        let probe = Extractor::new().step(&Step { action: command.to_string(), ..Default::default() });
        let Some(cmd) = probe.fetches.iter().find(|e| e.kind == Kind::Cmd).cloned() else { return Ok(None) };
        let (repo, infos) = {
            let s = self.session(id)?;
            (s.repo.clone(), s.infos.clone())
        };
        if let Some(j) = (0..infos.len()).rev().find(|&j| infos[j].fetches.contains(&cmd)) {
            if infos[j].failed && !infos[j + 1..].iter().any(|s| s.edited.is_some()) {
                let err = infos[j].mentions.iter().find(|e| e.kind == Kind::Error).map(|e| e.key.clone()).unwrap_or_else(|| "an error".into());
                return Ok(Some(format!("`{}` already failed at step {} with {} and nothing was edited since.", cmd.key, j, err)));
            }
        }
        let cid = entity_id(space, &repo, &cmd);
        let edges = self.store.neighbors(&[cid])?;
        let learner = self.learner(&repo);
        let mut errs: Vec<(u64, String)> = edges
            .iter()
            .filter(|e| e.kind == EdgeKind::Mentions && e.n >= 2)
            .filter_map(|e| learner.mirror.ents.get(&e.dst).map(|x| (e.n, x.key.clone())))
            .collect();
        errs.sort_by(|a, b| b.0.cmp(&a.0));
        Ok(errs.first().map(|(n, k)| format!("In {} past sessions `{}` ended with {}.", n, cmd.key, k)))
    }

    pub fn end(&mut self, id: &str, resolved: bool) -> anyhow::Result<LearnStats> {
        let s = self.sessions.remove(id).ok_or_else(|| anyhow::anyhow!("unknown session {id}"))?;
        let cfg = self.cfg.clone();
        let learner = self.learners.entry(s.repo.clone()).or_insert_with(|| Learner::new(self.space, &s.repo, true));
        learner.ingest(&s.infos, resolved, &cfg, self.store.as_mut())?;
        Ok(learner.stats.clone())
    }

    pub fn memory(&mut self, repo: &str) -> MemoryView {
        let l = self.learner(repo);
        let weights = FEATURES.iter().enumerate().map(|(i, f)| (f.to_string(), PRIOR[i], l.params.weights[i])).collect();
        let mut next: Vec<(String, String, f64)> = l
            .mirror
            .edges
            .values()
            .filter(|e| e.kind == EdgeKind::Next || e.kind == EdgeKind::FixedBy)
            .filter_map(|e| Some((l.mirror.ents.get(&e.src)?.label(), l.mirror.ents.get(&e.dst)?.label(), e.w)))
            .collect();
        next.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap());
        next.truncate(25);
        MemoryView { repo: repo.to_string(), summary: l.summary(), weights, top_next: next, stats: l.stats.clone() }
    }
}
