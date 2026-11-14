//! HydraDB (open-source engine) store over its HTTP JSON query API.
//!
//! Shapes follow HydraDB's OpenCypher subset:
//! * node ids are non-negative integers -> `graph::entity_id`
//! * vertex upserts are `UNWIND $rows AS row MERGE (n {id: row.id}) SET ...`
//! * a `MATCH` pattern takes one relationship type, so neighbor reads are one
//!   `UNION ALL` request per relationship type, covering every source id
//! * batches are capped at 1024 rows

use crate::extract::Ent;
use crate::graph::{Edge, EdgeKind, GraphStore};
use anyhow::{anyhow, bail, Context};
use serde_json::{json, Value};
use std::collections::BTreeMap;

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
        self.query(&format!("MATCH (n:S{space}) DETACH DELETE n"), Value::Null)?;
        Ok(())
    }

    fn write_entities(&mut self, ents: &[(u64, Ent)]) -> anyhow::Result<()> {
        let mut by_space: BTreeMap<u8, Vec<Value>> = BTreeMap::new();
        for (id, e) in ents {
            by_space.entry((id >> 56) as u8).or_default().push(json!({"id": id, "kind": e.kind.as_str(), "key": e.key}));
        }
        for (space, rows) in by_space {
            // Batched vertex upserts take exactly one label: the space.
            let q = format!("UNWIND $rows AS row MERGE (n {{id: row.id}}) SET n:S{space}, n.kind = row.kind, n.key = row.key");
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
}
