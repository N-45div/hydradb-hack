//! Deterministic extraction: turn raw agent steps into entities.
//! No LLM involved. Files, commands, searches, errors and symbols come
//! straight from tool calls, file headers and tracebacks.

use crate::model::Step;
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use std::sync::LazyLock;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub enum Kind {
    File,
    Cmd,
    Search,
    Error,
    Symbol,
}

impl Kind {
    pub fn as_str(&self) -> &'static str {
        match self {
            Kind::File => "File",
            Kind::Cmd => "Cmd",
            Kind::Search => "Search",
            Kind::Error => "Error",
            Kind::Symbol => "Symbol",
        }
    }
    pub fn parse(s: &str) -> Option<Kind> {
        Some(match s {
            "File" => Kind::File,
            "Cmd" => Kind::Cmd,
            "Search" => Kind::Search,
            "Error" => Kind::Error,
            "Symbol" => Kind::Symbol,
            _ => return None,
        })
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct Ent {
    pub kind: Kind,
    pub key: String,
}

impl Ent {
    pub fn new(kind: Kind, key: impl Into<String>) -> Self {
        Ent { kind, key: key.into() }
    }
    pub fn label(&self) -> String {
        format!("{}:{}", self.kind.as_str(), self.key)
    }
}

/// What one step means for memory.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct StepInfo {
    pub idx: usize,
    pub tokens: usize,
    /// Entities this step's action goes out and fetches (open a file, run a command, search).
    pub fetches: Vec<Ent>,
    /// Entities whose content this step's observation shows. Keeping this step in
    /// context means the agent does not need to fetch them again.
    pub provides: Vec<Ent>,
    /// Entities referenced without their content (traceback frames, search hits, errors).
    pub mentions: Vec<Ent>,
    /// File edited by this step, if any.
    pub edited: Option<Ent>,
    /// The step's output contained an error or traceback.
    pub failed: bool,
    /// Short text used for keyword retrieval and for rendering memory items.
    pub text: String,
}

impl StepInfo {
    pub fn all_entities(&self) -> BTreeSet<&Ent> {
        self.fetches.iter().chain(&self.provides).chain(&self.mentions).collect()
    }
}

static OPEN_FILE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"\(Open file: ([^)\n]+)\)").unwrap());
static CUR_DIR: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"\(Current directory: ([^)\n]+)\)").unwrap());
static FILE_HDR: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"\[File: ([^\s\]]+) \(\d+ lines total\)\]").unwrap());
static TB: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r#"File "([^"]+)", line (\d+), in ([\w<>]+)"#).unwrap());
static ERR: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?m)^([A-Za-z_][\w.]*(?:Error|Exception|Exit|Warning|Interrupt))(?::\s*(.*))?$").unwrap()
});
static SEARCH_HIT: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?m)^(\S+\.\w+) \(\d+ matches\)\s*$").unwrap());
static GREP_HIT: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?m)^(\.?/?[\w./-]+\.\w{1,6}):\d+:").unwrap());
static FIND_HIT: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?m)^(\.?/[\w./-]+\.\w{1,6})\s*$").unwrap());
static DIGITS: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"\d+").unwrap());
static HEX: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"0x[0-9a-fA-F]+").unwrap());
static WS: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"\s+").unwrap());

pub fn approx_tokens(s: &str) -> usize {
    s.len().div_ceil(4)
}

fn clip(s: &str, n: usize) -> &str {
    if s.len() <= n {
        return s;
    }
    let mut end = n;
    while !s.is_char_boundary(end) {
        end -= 1;
    }
    &s[..end]
}

/// Normalise a path to repo-relative form. Returns None for paths outside the
/// repository (site-packages, system dirs) or placeholders.
pub fn norm_path(p: &str, root: &str) -> Option<String> {
    let p = p.trim().trim_matches(|c| c == '"' || c == '\'' || c == '`');
    if p.is_empty() || p == "n/a" {
        return None;
    }
    for bad in ["site-packages", "/opt/", "/usr/", "/tmp/", "<frozen", "<string>", "/root/"] {
        if p.contains(bad) {
            return None;
        }
    }
    let mut s = p.to_string();
    if !root.is_empty() && root != "/" {
        let r = format!("{}/", root.trim_end_matches('/'));
        if let Some(rest) = s.strip_prefix(&r) {
            s = rest.to_string();
        }
    }
    let s = s.trim_start_matches("./").trim_start_matches('/').to_string();
    if s.is_empty() || s.contains(' ') {
        return None;
    }
    Some(s)
}

fn norm_error(ty: &str, msg: &str) -> String {
    let msg = HEX.replace_all(msg, "0x_");
    let msg = DIGITS.replace_all(&msg, "N");
    let msg = WS.replace_all(msg.trim(), " ");
    let ty = ty.rsplit('.').next().unwrap_or(ty);
    if msg.is_empty() {
        ty.to_string()
    } else {
        format!("{}: {}", ty, clip(&msg, 80))
    }
}

fn norm_cmd(line: &str) -> String {
    clip(&WS.replace_all(line.trim(), " "), 100).to_string()
}

fn push_unique(v: &mut Vec<Ent>, e: Ent) {
    if !v.contains(&e) {
        v.push(e);
    }
}

/// Stateful extractor: tracks the repository root and the file open in the
/// agent's editor across steps.
#[derive(Default)]
pub struct Extractor {
    root: String,
    cur_file: Option<String>,
    idx: usize,
}

impl Extractor {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_root(root: &str) -> Self {
        Extractor { root: root.to_string(), ..Default::default() }
    }

    pub fn step(&mut self, step: &Step) -> StepInfo {
        let mut info = StepInfo {
            idx: self.idx,
            tokens: approx_tokens(&step.thought) + approx_tokens(&step.action) + approx_tokens(&step.observation),
            ..Default::default()
        };
        self.idx += 1;
        if self.root.is_empty() {
            if let Some(c) = CUR_DIR.captures(&step.observation) {
                self.root = c[1].trim().to_string();
            }
        }
        let root = self.root.clone();
        let first = step.action.lines().next().unwrap_or("").trim();
        let mut parts = first.split_whitespace();
        let cmd = parts.next().unwrap_or("");
        let args: Vec<&str> = parts.collect();
        let file_ent = |p: &str| norm_path(p, &root).map(|k| Ent::new(Kind::File, k));

        match cmd {
            "open" => {
                if let Some(e) = args.first().and_then(|p| file_ent(p)) {
                    push_unique(&mut info.fetches, e.clone());
                    push_unique(&mut info.provides, e);
                }
            }
            "cat" | "head" | "tail" | "less" | "nl" => {
                if let Some(e) = args.iter().rev().find(|a| !a.starts_with('-')).and_then(|p| file_ent(p)) {
                    push_unique(&mut info.fetches, e.clone());
                    push_unique(&mut info.provides, e);
                }
            }
            "goto" | "scroll_down" | "scroll_up" => {
                if let Some(e) = self.cur_file.as_deref().and_then(file_ent) {
                    push_unique(&mut info.provides, e);
                }
            }
            "edit" => {
                if let Some(e) = self.cur_file.as_deref().and_then(file_ent) {
                    push_unique(&mut info.provides, e.clone());
                    info.edited = Some(e);
                }
            }
            "create" => {
                if let Some(e) = args.first().and_then(|p| file_ent(p)) {
                    push_unique(&mut info.provides, e.clone());
                    info.edited = Some(e);
                }
            }
            "search_dir" | "search_file" | "find_file" | "grep" | "find" | "find." | "rg" => {
                let e = Ent::new(Kind::Search, norm_cmd(first));
                push_unique(&mut info.fetches, e.clone());
                push_unique(&mut info.provides, e);
            }
            "" | "ls" | "pwd" | "cd" | "cd.." | "submit" | "rm" | "mv" | "echo" | "chmod" | "mkdir" | "cp"
            | "touch" | "exit" => {}
            _ => {
                let e = Ent::new(Kind::Cmd, norm_cmd(first));
                push_unique(&mut info.fetches, e.clone());
                push_unique(&mut info.provides, e);
            }
        }

        let obs = &step.observation;
        for c in FILE_HDR.captures_iter(obs) {
            if let Some(e) = file_ent(&c[1]) {
                push_unique(&mut info.provides, e);
            }
        }
        for c in TB.captures_iter(obs) {
            if let Some(path) = norm_path(&c[1], &root) {
                push_unique(&mut info.mentions, Ent::new(Kind::File, path.clone()));
                let func = &c[3];
                if func != "<module>" {
                    push_unique(&mut info.mentions, Ent::new(Kind::Symbol, format!("{}@{}", func, path)));
                }
            }
        }
        let mut errors = 0;
        for c in ERR.captures_iter(obs) {
            let msg = c.get(2).map(|m| m.as_str()).unwrap_or("");
            push_unique(&mut info.mentions, Ent::new(Kind::Error, norm_error(&c[1], msg)));
            errors += 1;
        }
        info.failed = errors > 0 || obs.contains("Traceback (most recent call last)");
        for re in [&*SEARCH_HIT, &*GREP_HIT, &*FIND_HIT] {
            for c in re.captures_iter(obs).take(15) {
                if let Some(e) = file_ent(&c[1]) {
                    if !info.provides.contains(&e) {
                        push_unique(&mut info.mentions, e);
                    }
                }
            }
        }

        info.text = format!(
            "{}\n{}\n{}",
            clip(&step.thought, 300),
            clip(&step.action, 400),
            clip(obs, 1500)
        );
        if let Some(c) = OPEN_FILE.captures(obs) {
            let f = c[1].trim();
            self.cur_file = if f == "n/a" { None } else { Some(f.to_string()) };
        }
        info
    }
}

pub fn extract_session(steps: &[Step]) -> Vec<StepInfo> {
    let mut ex = Extractor::new();
    steps.iter().map(|s| ex.step(s)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn st(action: &str, obs: &str) -> Step {
        Step { thought: String::new(), action: action.into(), observation: obs.into() }
    }

    #[test]
    fn extracts_files_errors_and_commands() {
        let steps = vec![
            st("ls", "setup.py\n(Open file: n/a)\n(Current directory: /lexicon)\nbash-$"),
            st(
                "python reproduce.py",
                "Traceback (most recent call last):\n  File \"/lexicon/lexicon/providers/memset.py\", line 26, in authenticate\n    x\nrequests.exceptions.HTTPError: 403 Client\n(Open file: n/a)",
            ),
            st(
                "open lexicon/providers/memset.py",
                "[File: /lexicon/lexicon/providers/memset.py (150 lines total)]\n1:import json\n(Open file: /lexicon/lexicon/providers/memset.py)",
            ),
            st("edit 20:22\n    self.domain = 1\nend_of_edit", "[File: /lexicon/lexicon/providers/memset.py (151 lines total)]"),
        ];
        let infos = extract_session(&steps);
        assert!(infos[1].fetches.contains(&Ent::new(Kind::Cmd, "python reproduce.py")));
        assert!(infos[1].mentions.contains(&Ent::new(Kind::File, "lexicon/providers/memset.py")));
        assert!(infos[1].mentions.contains(&Ent::new(Kind::Error, "HTTPError: N Client")));
        assert!(infos[1].failed);
        assert!(infos[2].fetches.contains(&Ent::new(Kind::File, "lexicon/providers/memset.py")));
        assert_eq!(infos[3].edited, Some(Ent::new(Kind::File, "lexicon/providers/memset.py")));
    }
}
