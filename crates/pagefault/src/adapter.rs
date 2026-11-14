//! Harness adapters: map a tool call from any harness (pi, Claude Code, Codex,
//! MCP clients) to a `Step` the extractor understands.

use crate::model::Step;
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// One tool call and its result, as harnesses report them.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct ToolEvent {
    pub tool: String,
    #[serde(default)]
    pub args: Value,
    #[serde(default)]
    pub output: String,
    #[serde(default)]
    pub is_error: bool,
    /// Optional assistant text that preceded the call.
    #[serde(default)]
    pub thought: String,
}

fn arg(args: &Value, keys: &[&str]) -> Option<String> {
    for k in keys {
        if let Some(v) = args.get(*k) {
            if let Some(s) = v.as_str() {
                return Some(s.to_string());
            }
        }
    }
    None
}

pub fn tool_step(ev: &ToolEvent) -> Step {
    let tool = ev.tool.to_lowercase();
    let path = arg(&ev.args, &["path", "file_path", "filePath", "file", "filename"]);
    let lines = ev.output.lines().count();
    let mut observation = ev.output.clone();
    if ev.is_error && !observation.contains("Error") {
        observation = format!("ToolError: {}\n{}", ev.tool, observation);
    }
    let action = match tool.as_str() {
        "read" | "view" | "open" | "cat" | "read_file" => match &path {
            Some(p) => {
                observation = format!("[File: {p} ({lines} lines total)]\n{observation}");
                format!("open {p}")
            }
            None => format!("{} {}", ev.tool, ev.args),
        },
        "edit" | "write" | "create" | "multi_edit" | "multiedit" | "str_replace" | "write_file" | "apply_patch" => match &path {
            Some(p) => {
                observation = format!("[File: {p} ({lines} lines total)]\n{observation}");
                format!("create {p}")
            }
            None => "edit".to_string(),
        },
        "bash" | "shell" | "run" | "exec" | "terminal" => {
            arg(&ev.args, &["command", "cmd", "script"]).unwrap_or_default().lines().next().unwrap_or("").to_string()
        }
        "grep" | "rg" | "find" | "glob" | "ls" | "search" | "search_dir" | "codebase_search" => {
            let pat = arg(&ev.args, &["pattern", "query", "glob", "regex", "path"]).unwrap_or_default();
            format!("search_dir {pat}")
        }
        _ => format!("{} {}", ev.tool, ev.args),
    };
    Step { thought: ev.thought.clone(), action, observation }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::extract::{Ent, Extractor, Kind};
    use serde_json::json;

    #[test]
    fn pi_tools_map_to_entities() {
        let mut ex = Extractor::with_root("/repo");
        let r = ex.step(&tool_step(&ToolEvent { tool: "read".into(), args: json!({"path": "/repo/src/app.py"}), output: "a\nb".into(), ..Default::default() }));
        assert!(r.fetches.contains(&Ent::new(Kind::File, "src/app.py")));
        let b = ex.step(&tool_step(&ToolEvent { tool: "bash".into(), args: json!({"command": "pytest -x tests"}), output: "ValueError: bad".into(), is_error: true, ..Default::default() }));
        assert!(b.fetches.contains(&Ent::new(Kind::Cmd, "pytest -x tests")));
        assert!(b.failed);
        let e = ex.step(&tool_step(&ToolEvent { tool: "edit".into(), args: json!({"path": "src/app.py"}), output: "ok".into(), ..Default::default() }));
        assert_eq!(e.edited, Some(Ent::new(Kind::File, "src/app.py")));
    }
}
