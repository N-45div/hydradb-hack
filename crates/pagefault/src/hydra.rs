//! HydraDB (open-source engine) store over its HTTP JSON query API.
//!
//! Shapes follow HydraDB's OpenCypher subset:
//! * node ids are non-negative integers -> `graph::entity_id`
//! * vertex upserts are `UNWIND $rows AS row MERGE (n {id: row.id}) SET ...`
//! * traversal at compaction time is one `algo.MSpaths` call: many sources and
//!   targets, every relationship type, resolved server-side over GraphBLAS
//!   topology (selectors are the `sel` string property; lists are inlined)
//! * repo-graph degree is `algo.degree`, cached until the next write
//! * learner state (feature weights, fault counts) lives on vertices, so a
//!   restarted service resumes from HydraDB instead of from zero
//! * batches are capped at 1024 rows

use crate::extract::{Ent, Kind};
use crate::graph::{selector, Edge, EdgeKind, GPath, GraphStore, SpaceState};
use anyhow::{anyhow, bail, Context};
use serde_json::{json, Value};
use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;

const BATCH: usize = 1000;
const UNION_ARMS: usize = 24;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum EdgeWriteMode {
    /// `UNWIND ... MATCH (a),(b) MERGE (a)-[r:T {id: row.eid}]->(b) SET r.w = ...` in one batch.
    MergeSetBatch,
    /// Batch `CREATE` for missing edges, then one `MATCH ... SET` per edge.
    PerEdgeSet,
}

pub struct HydraStore {
    client: reqwest::blocking::Client,
    url: String,
    token: String,
    graph: String,
    namespace: String,
    cell: String,
    pub mode: EdgeWriteMode,
    pub requests: u64,
    degree_cache: HashMap<(u8, EdgeKind), Arc<HashMap<u64, usize>>>,
}

/// Inline a list of strings as a Cypher literal (MSpaths rejects list parameters).
fn cypher_strings(values: &[&str]) -> String {
    let parts: Vec<String> = values
        .iter()
        .map(|v| {
            let clean: String = v.chars().filter(|c| !c.is_control()).collect();
            format!("'{}'", clean.replace('\\', "\\\\").replace('\'', "\\'"))
        })
        .collect();
    format!("[{}]", parts.join(", "))
}

/// Unwrap a path-cell property: `{"Float": 1.5}` / `{"Integer": 2}` / `{"String": "x"}`.
fn tagged(v: &Value) -> Option<&Value> {
    v.as_object().and_then(|o| o.values().next())
}

fn meta_id(space: u8, repo: &str) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for b in format!("meta|{repo}").bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    ((space as u64 & 0x7f) << 56) | (h & 0x00ff_ffff_ffff_ffff)
}

fn cell_value(c: &Value) -> &Value {
    c.get("value").unwrap_or(&Value::Null)
}

pub fn cell_u64(c: &Value) -> Option<u64> {
    let v = cell_value(c);
    v.as_u64().or_else(|| v.as_i64().map(|x| x as u64)).or_else(|| v.as_f64().map(|x| x as u64))
}

pub fn cell_f64(c: &Value) -> Option<f64> {
    let v = cell_value(c);
    v.as_f64().or_else(|| v.as_i64().map(|x| x as f64))
}

pub fn cell_str(c: &Value) -> Option<String> {
    cell_value(c).as_str().map(|s| s.to_string())
}

fn edge_id(e: &Edge) -> u64 {
    // Relationship identity for MERGE: stable hash of (src, dst, kind).
    let mut h: u64 = 0xcbf29ce484222325;
    for b in e.src.to_le_bytes().iter().chain(e.dst.to_le_bytes().iter()).chain(e.kind.rel_type().as_bytes()) {
        h ^= *b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h & 0x7fff_ffff_ffff_ffff
}

impl HydraStore {
    pub fn new(url: &str, token: &str) -> anyhow::Result<Self> {
        let env = |k: &str, d: &str| std::env::var(k).unwrap_or_else(|_| d.to_string());
        let mode = match env("HYDRADB_EDGE_MODE", "merge_set").as_str() {
            "per_edge" => EdgeWriteMode::PerEdgeSet,
            _ => EdgeWriteMode::MergeSetBatch,
        };
        Ok(HydraStore {
            client: reqwest::blocking::Client::builder().timeout(std::time::Duration::from_secs(60)).build()?,
            url: url.trim_end_matches('/').to_string(),
            token: token.to_string(),
            graph: env("HYDRADB_GRAPH", "default"),
            namespace: env("HYDRADB_NAMESPACE", "default"),
            cell: env("HYDRADB_CELL", "cell-0"),
            mode,
            requests: 0,
            degree_cache: HashMap::new(),
        })
    }

    /// Run one statement. Returns rows of typed cells.
    pub fn query(&mut self, q: &str, params: Value) -> anyhow::Result<Vec<Vec<Value>>> {
        self.requests += 1;
        let params = if params.is_null() { json!({}) } else { params };
        let mut body = json!({"cell_id": self.cell, "query": q, "parameters": params, "page_size": 4096});
        let mut out = Vec::new();
        loop {
            let resp = self
                .client
                .post(format!("{}/v1/graphs/{}/query", self.url, self.graph))
                .bearer_auth(&self.token)
                .header("X-Graph-Namespace", &self.namespace)
                .json(&body)
                .send()
                .with_context(|| format!("HydraDB request to {}", self.url))?;
            let status = resp.status();
            let text = resp.text()?;
            let v: Value = serde_json::from_str(&text).map_err(|_| anyhow!("HydraDB {status}: {text}"))?;
            if let Some(err) = v.get("error") {
                bail!("HydraDB error: {err} in query: {}", &q[..q.len().min(300)]);
            }
            let rows = v.get("rows").and_then(|r| r.as_array()).ok_or_else(|| anyhow!("HydraDB {status}: unexpected body {text}"))?;
            for r in rows {
                out.push(r.as_array().cloned().unwrap_or_default());
            }
            match v.get("next_cursor") {
                Some(c) if !c.is_null() => {
                    body["cursor"] = c.clone();
                    body["query_id"] = v.get("query_id").cloned().unwrap_or(Value::Null);
                }
                _ => break,
            }
        }
        Ok(out)
    }

    pub fn ping(&mut self) -> anyhow::Result<()> {
        self.query("MATCH (n:S0) RETURN count(*) AS c", Value::Null).map(|_| ())
    }

    fn write_edges_merge_set(&mut self, kind: EdgeKind, edges: &[&Edge]) -> anyhow::Result<()> {
        // Batched endpoints need exactly one label: the space label every entity carries.
        let space = edges.first().map(|e| (e.src >> 56) as u8).unwrap_or(0);
        let q = format!(
            "UNWIND $rows AS row MATCH (a:S{space} {{id: row.src}}), (b:S{space} {{id: row.dst}}) MERGE (a)-[r:{} {{id: row.eid}}]->(b) SET r.w = row.w, r.n = row.n",
            kind.rel_type()
        );
        for chunk in edges.chunks(BATCH) {
            let rows: Vec<Value> = chunk.iter().map(|e| json!({"src": e.src, "dst": e.dst, "eid": edge_id(e), "w": e.w, "n": e.n})).collect();
            self.query(&q, json!({ "rows": rows }))?;
        }
        Ok(())
    }

    fn write_edges_per_edge(&mut self, kind: EdgeKind, edges: &[&Edge]) -> anyhow::Result<()> {
        // Batch CREATE dedups on (src, dst, type): safe to re-run for existing edges.
        let q = format!("UNWIND $rows AS row CREATE (a {{id: row.src}})-[:{}]->(b {{id: row.dst}})", kind.rel_type());
        for chunk in edges.chunks(BATCH) {
            let rows: Vec<Value> = chunk.iter().map(|e| json!({"src": e.src, "dst": e.dst})).collect();
            self.query(&q, json!({ "rows": rows }))?;
        }
        let q = format!("MATCH (a {{id: $s}})-[r:{}]->(b {{id: $d}}) SET r.w = $w, r.n = $n", kind.rel_type());
        for e in edges {
            self.query(&q, json!({"s": e.src, "d": e.dst, "w": e.w, "n": e.n}))?;
        }
        Ok(())
    }
}

impl GraphStore for HydraStore {
    fn name(&self) -> String {
        format!("hydradb({})", self.url)
    }

    fn clear_space(&mut self, space: u8, _ids: &[u64]) -> anyhow::Result<()> {
        self.degree_cache.clear();
        self.query(&format!("MATCH (n:S{space}) DETACH DELETE n"), Value::Null)?;
        self.query(&format!("MATCH (n:M{space}) DETACH DELETE n"), Value::Null)?;
        Ok(())
    }

    fn write_entities(&mut self, repo: &str, ents: &[(u64, Ent)]) -> anyhow::Result<()> {
        let mut by_space: BTreeMap<u8, Vec<Value>> = BTreeMap::new();
        for (id, e) in ents {
            by_space
                .entry((id >> 56) as u8)
                .or_default()
                .push(json!({"id": id, "kind": e.kind.as_str(), "key": e.key, "sel": selector(repo, e), "repo": repo}));
        }
        for (space, rows) in by_space {
            // Batched vertex upserts take exactly one label: the space.
            let q = format!("UNWIND $rows AS row MERGE (n {{id: row.id}}) SET n:S{space}, n.kind = row.kind, n.key = row.key, n.sel = row.sel, n.repo = row.repo");
            for chunk in rows.chunks(BATCH) {
                self.query(&q, json!({ "rows": chunk }))?;
            }
        }
        Ok(())
    }

    fn write_edges(&mut self, edges: &[Edge]) -> anyhow::Result<()> {
        let mut groups: BTreeMap<(EdgeKind, u8), Vec<&Edge>> = BTreeMap::new();
        for e in edges {
            groups.entry((e.kind, (e.src >> 56) as u8)).or_default().push(e);
        }
        if !groups.is_empty() {
            self.degree_cache.clear();
        }
        for ((kind, _space), es) in groups {
            match self.mode {
                EdgeWriteMode::MergeSetBatch => self.write_edges_merge_set(kind, &es)?,
                EdgeWriteMode::PerEdgeSet => self.write_edges_per_edge(kind, &es)?,
            }
        }
        Ok(())
    }

    fn neighbors(&mut self, ids: &[u64]) -> anyhow::Result<Vec<Edge>> {
        let mut out = Vec::new();
        if ids.is_empty() {
            return Ok(out);
        }
        for kind in EdgeKind::ALL {
            for chunk in ids.chunks(UNION_ARMS) {
                let q = chunk
                    .iter()
                    .map(|id| {
                        format!(
                            "MATCH (a {{id: {id}}})-[r:{}]->(b) RETURN a.id AS src, b.id AS dst, r.w AS w, r.n AS n",
                            kind.rel_type()
                        )
                    })
                    .collect::<Vec<_>>()
                    .join(" UNION ALL ");
                for row in self.query(&q, Value::Null)? {
                    if row.len() < 4 {
                        continue;
                    }
                    let (Some(src), Some(dst)) = (cell_u64(&row[0]), cell_u64(&row[1])) else { continue };
                    out.push(Edge {
                        src,
                        dst,
                        kind,
                        w: cell_f64(&row[2]).unwrap_or(0.0),
                        n: cell_u64(&row[3]).unwrap_or(1),
                    });
                }
            }
        }
        Ok(out)
    }

    fn paths(&mut self, sources: &[(u64, String)], targets: &[(u64, String)], kinds: &[EdgeKind], max_len: usize, per_pair: usize) -> anyhow::Result<Vec<GPath>> {
        if sources.is_empty() || targets.is_empty() || kinds.is_empty() {
            return Ok(Vec::new());
        }
        let space = (sources[0].0 >> 56) as u8;
        let srcs: Vec<&str> = sources.iter().map(|s| s.1.as_str()).collect();
        let tgts: Vec<&str> = targets.iter().map(|t| t.1.as_str()).collect();
        let rels: Vec<&str> = kinds.iter().map(|k| k.rel_type()).collect();
        let q = format!(
            "CALL algo.MSpaths({{sourceLabel: 'S{space}', sourceProperty: 'sel', sourceValues: {}, targetValues: {}, relTypes: {}, relDirection: 'outgoing', maxLen: {max_len}, pathCount: {per_pair}, pairwise: false, resultLimit: 20000}}) YIELD path RETURN path",
            cypher_strings(&srcs),
            cypher_strings(&tgts),
            cypher_strings(&rels)
        );
        let mut out = Vec::new();
        for row in self.query(&q, Value::Null)? {
            let Some(p) = row.first().and_then(|c| c.get("value")) else { continue };
            let nodes: Vec<(u64, Option<String>)> = p
                .get("nodes")
                .and_then(|n| n.as_array())
                .map(|ns| {
                    ns.iter()
                        .filter_map(|n| {
                            let id = n.get("id")?.as_u64()?;
                            let props = n.get("properties");
                            let kind = props.and_then(|pr| pr.get("kind")).and_then(tagged).and_then(|v| v.as_str());
                            let key = props.and_then(|pr| pr.get("key")).and_then(tagged).and_then(|v| v.as_str());
                            let label = match (kind, key) {
                                (Some(k), Some(key)) => Kind::parse(k).map(|k| Ent::new(k, key).label()),
                                _ => None,
                            };
                            Some((id, label))
                        })
                        .collect()
                })
                .unwrap_or_default();
            let edges: Vec<Edge> = p
                .get("relationships")
                .and_then(|r| r.as_array())
                .map(|rs| {
                    rs.iter()
                        .filter_map(|r| {
                            let kind = EdgeKind::parse(r.get("edge_type")?.as_str()?)?;
                            let props = r.get("properties");
                            let w = props.and_then(|pr| pr.get("w")).and_then(tagged).and_then(|v| v.as_f64()).unwrap_or(0.0);
                            let n = props.and_then(|pr| pr.get("n")).and_then(tagged).and_then(|v| v.as_u64()).unwrap_or(1);
                            Some(Edge { src: r.get("src")?.as_u64()?, dst: r.get("dst")?.as_u64()?, kind, w, n })
                        })
                        .collect()
                })
                .unwrap_or_default();
            if !edges.is_empty() && nodes.len() == edges.len() + 1 {
                out.push(GPath { nodes, edges });
            }
        }
        Ok(out)
    }

    fn degrees(&mut self, space: u8, kind: EdgeKind) -> anyhow::Result<Arc<HashMap<u64, usize>>> {
        if let Some(d) = self.degree_cache.get(&(space, kind)) {
            return Ok(d.clone());
        }
        let q = format!(
            "CALL algo.degree({{label: 'S{space}', relTypes: ['{}'], relDirection: 'outgoing', minDegree: 1, limit: 100000}}) YIELD node, degree RETURN node.id AS id, degree",
            kind.rel_type()
        );
        let mut d = HashMap::new();
        for row in self.query(&q, Value::Null)? {
            if let (Some(id), Some(deg)) = (row.first().and_then(cell_u64), row.get(1).and_then(cell_u64)) {
                d.insert(id, deg as usize);
            }
        }
        let d = Arc::new(d);
        self.degree_cache.insert((space, kind), d.clone());
        Ok(d)
    }

    fn save_learner(&mut self, space: u8, repo: &str, weights: &[f64], faults: &[(u64, f64)]) -> anyhow::Result<()> {
        let w = serde_json::to_string(weights)?;
        self.query(
            &format!("UNWIND $rows AS row MERGE (n {{id: row.id}}) SET n:M{space}, n.repo = row.repo, n.weights = row.weights"),
            json!({"rows": [{"id": meta_id(space, repo), "repo": repo, "weights": w}]}),
        )?;
        let rows: Vec<Value> = faults.iter().map(|(id, f)| json!({"id": id, "f": f})).collect();
        let q = format!("UNWIND $rows AS row MERGE (n {{id: row.id}}) SET n:S{space}, n.faults = row.f");
        for chunk in rows.chunks(BATCH) {
            self.query(&q, json!({ "rows": chunk }))?;
        }
        Ok(())
    }

    fn load_space(&mut self, space: u8) -> anyhow::Result<Option<SpaceState>> {
        let mut st = SpaceState::default();
        let q = format!("MATCH (n:S{space}) RETURN n.id AS id, n.kind AS kind, n.key AS key, n.repo AS repo, n.faults AS faults");
        for row in self.query(&q, Value::Null)? {
            let (Some(id), Some(kind), Some(key), Some(repo)) = (
                row.first().and_then(cell_u64),
                row.get(1).and_then(cell_str).and_then(|k| Kind::parse(&k)),
                row.get(2).and_then(cell_str),
                row.get(3).and_then(cell_str),
            ) else {
                continue;
            };
            st.ents.push((id, Ent::new(kind, key), repo, row.get(4).and_then(cell_f64).unwrap_or(0.0)));
        }
        for kind in EdgeKind::ALL {
            let q = format!("MATCH (a:S{space})-[r:{}]->(b) RETURN a.id AS src, b.id AS dst, r.w AS w, r.n AS n", kind.rel_type());
            for row in self.query(&q, Value::Null)? {
                let (Some(src), Some(dst)) = (row.first().and_then(cell_u64), row.get(1).and_then(cell_u64)) else { continue };
                st.edges.push(Edge { src, dst, kind, w: row.get(2).and_then(cell_f64).unwrap_or(0.0), n: row.get(3).and_then(cell_u64).unwrap_or(1) });
            }
        }
        for row in self.query(&format!("MATCH (m:M{space}) RETURN m.repo AS repo, m.weights AS weights"), Value::Null)? {
            if let (Some(repo), Some(w)) = (row.first().and_then(cell_str), row.get(1).and_then(cell_str)) {
                if let Ok(w) = serde_json::from_str::<Vec<f64>>(&w) {
                    st.weights.push((repo, w));
                }
            }
        }
        Ok(Some(st))
    }
}
