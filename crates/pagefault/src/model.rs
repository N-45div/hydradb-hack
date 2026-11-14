//! Harness-agnostic session model. Every harness adapter (SWE-agent replay,
//! pi extension, HTTP API) produces these.

use serde::{Deserialize, Serialize};
use std::io::BufRead;

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Step {
    #[serde(default)]
    pub thought: String,
    #[serde(default)]
    pub action: String,
    #[serde(default)]
    pub observation: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Session {
    pub id: String,
    #[serde(default)]
    pub instance_id: String,
    pub repo: String,
    #[serde(default)]
    pub model: String,
    #[serde(default)]
    pub resolved: bool,
    #[serde(default)]
    pub issue: String,
    pub steps: Vec<Step>,
}

impl Session {
    /// Numeric part of a SWE-bench style instance id (`owner__repo-1234`), used
    /// to replay sessions in roughly chronological order.
    pub fn instance_number(&self) -> u64 {
        self.instance_id
            .rsplit('-')
            .next()
            .and_then(|s| s.parse().ok())
            .unwrap_or(0)
    }
}

/// Load sessions from a `.jsonl` or `.jsonl.gz` file.
pub fn load_sessions(path: &std::path::Path) -> anyhow::Result<Vec<Session>> {
    let f = std::fs::File::open(path)?;
    let reader: Box<dyn BufRead> = if path.extension().map(|e| e == "gz").unwrap_or(false) {
        Box::new(std::io::BufReader::new(flate2::read::GzDecoder::new(f)))
    } else {
        Box::new(std::io::BufReader::new(f))
    };
    let mut out = Vec::new();
    for line in reader.lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        out.push(serde_json::from_str(&line)?);
    }
    Ok(out)
}
