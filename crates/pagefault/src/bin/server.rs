//! pagefault HTTP server: compaction API + benchmark dashboard.
use axum::extract::{Path, Query, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Json, Response};
use axum::routing::{get, post};
use axum::Router;
use clap::Parser;
use pagefault::adapter::ToolEvent;
use pagefault::experiment::{prepare, RepoSplit};
use pagefault::graph::{GraphStore, MemStore};
use pagefault::hydra::HydraStore;
use pagefault::model::{load_sessions, Step};
use pagefault::service::{Demo, Service};
use pagefault::sim::{compaction_points, SimCfg};
use serde::Deserialize;
use serde_json::{json, Value};
use std::sync::{Arc, Mutex};
use tower_http::cors::CorsLayer;
use tower_http::services::ServeDir;

#[derive(Parser, Clone)]
struct Args {
    #[arg(long, env = "PORT", default_value_t = 8080)]
    port: u16,
    #[arg(long, env = "PAGEFAULT_STORE", default_value = "hydra")]
    store: String,
    #[arg(long, env = "HYDRADB_URL", default_value = "http://127.0.0.1:8443")]
    hydra_url: String,
    #[arg(long, env = "HYDRADB_TOKEN", default_value = "local-development-token-32-bytes")]
    hydra_token: String,
    #[arg(long, env = "PAGEFAULT_WEB", default_value = "web")]
    web: String,
    #[arg(long, env = "PAGEFAULT_RESULTS", default_value = "results")]
    results: String,
    #[arg(long, env = "PAGEFAULT_DEMO", default_value = "data/demo/pydantic__pydantic")]
    demo: String,
    /// Optional bearer token required on /v1/sessions endpoints.
    #[arg(long, env = "PAGEFAULT_API_KEY")]
    api_key: Option<String>,
}

type Shared = Arc<(Mutex<Service>, Args)>;

/// Set when the background demo load gives up, so the dashboard can say so.
static DEMO_FAILED: Mutex<Option<String>> = Mutex::new(None);

struct ApiError(anyhow::Error);
impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        (StatusCode::BAD_REQUEST, Json(json!({"error": self.0.to_string()}))).into_response()
    }
}
impl<E: Into<anyhow::Error>> From<E> for ApiError {
    fn from(e: E) -> Self {
        ApiError(e.into())
    }
}
type R = Result<Json<Value>, ApiError>;

fn store(a: &Args) -> anyhow::Result<Box<dyn GraphStore>> {
    Ok(match a.store.as_str() {
        "memory" => Box::new(MemStore::default()),
        _ => Box::new(HydraStore::new(&a.hydra_url, &a.hydra_token)?),
    })
}

async fn blocking<T: Send + 'static>(s: Shared, f: impl FnOnce(&mut Service) -> anyhow::Result<T> + Send + 'static) -> Result<T, ApiError> {
    tokio::task::spawn_blocking(move || {
        let mut svc = s.0.lock().map_err(|_| anyhow::anyhow!("lock poisoned"))?;
        f(&mut svc)
    })
    .await
    .map_err(|e| ApiError(e.into()))?
    .map_err(ApiError)
}

fn authorize(s: &Shared, headers: &axum::http::HeaderMap) -> Result<(), ApiError> {
    if let Some(k) = &s.1.api_key {
        let ok = headers.get("authorization").and_then(|v| v.to_str().ok()).map(|v| v == format!("Bearer {k}")).unwrap_or(false);
        if !ok {
            return Err(ApiError(anyhow::anyhow!("missing or wrong bearer token")));
        }
    }
    Ok(())
}

#[derive(Deserialize)]
struct NewSession {
    repo: String,
    #[serde(default)]
    harness: String,
    #[serde(default)]
    model: String,
    #[serde(default)]
    root: String,
}

async fn create_session(State(s): State<Shared>, h: axum::http::HeaderMap, Json(b): Json<NewSession>) -> R {
    authorize(&s, &h)?;
    let id = blocking(s, move |svc| Ok(svc.create_session(&b.repo, &b.harness, &b.model, &b.root))).await?;
    Ok(Json(json!({"session_id": id})))
}

/// Standard base64 (RFC 4648) decoder; avoids a new dependency.
fn b64_decode(s: &str) -> Option<Vec<u8>> {
    let mut out = Vec::with_capacity(s.len() * 3 / 4);
    let (mut acc, mut bits) = (0u32, 0u32);
    for c in s.bytes() {
        let v = match c {
            b'A'..=b'Z' => c - b'A',
            b'a'..=b'z' => c - b'a' + 26,
            b'0'..=b'9' => c - b'0' + 52,
            b'+' | b'-' => 62,
            b'/' | b'_' => 63,
            b'=' | b'\n' | b'\r' | b' ' => continue,
            _ => return None,
        } as u32;
        acc = (acc << 6) | v;
        bits += 6;
        if bits >= 8 {
            bits -= 8;
            out.push((acc >> bits) as u8);
            acc &= (1 << bits) - 1;
        }
    }
    Some(out)
}

#[derive(Deserialize)]
struct Events {
    #[serde(default)]
    events: Vec<ToolEvent>,
    /// Base64 of a JSON array of events. Tool output is source code and shell
    /// text, which edge firewalls (Render's included) can flag as injection.
    #[serde(default)]
    events_b64: Option<String>,
    #[serde(default)]
    steps: Vec<Step>,
}

async fn add_events(State(s): State<Shared>, h: axum::http::HeaderMap, Path(id): Path<String>, Json(b): Json<Events>) -> R {
    authorize(&s, &h)?;
    let mut events = b.events;
    if let Some(enc) = &b.events_b64 {
        let raw = b64_decode(enc).ok_or_else(|| ApiError(anyhow::anyhow!("events_b64 is not valid base64")))?;
        let more: Vec<ToolEvent> = serde_json::from_slice(&raw).map_err(|e| ApiError(e.into()))?;
        events.extend(more);
    }
    let n = blocking(s, move |svc| {
        if !b.steps.is_empty() {
            svc.add_steps(&id, b.steps)?;
        }
        svc.add_events(&id, events)
    })
    .await?;
    Ok(Json(json!({"steps": n})))
}

#[derive(Deserialize)]
struct CompactReq {
    keep_recent_tokens: Option<usize>,
    budget_tokens: Option<usize>,
}

async fn compact(State(s): State<Shared>, h: axum::http::HeaderMap, Path(id): Path<String>, Json(b): Json<CompactReq>) -> R {
    authorize(&s, &h)?;
    let r = blocking(s, move |svc| svc.compact(&id, b.keep_recent_tokens, b.budget_tokens)).await?;
    Ok(Json(serde_json::to_value(r)?))
}

#[derive(Deserialize)]
struct RecallQ {
    q: String,
    k: Option<usize>,
}

async fn recall(State(s): State<Shared>, h: axum::http::HeaderMap, Path(id): Path<String>, Query(q): Query<RecallQ>) -> R {
    authorize(&s, &h)?;
    let r = blocking(s, move |svc| svc.recall(&id, &q.q, q.k.unwrap_or(3))).await?;
    Ok(Json(json!({"items": r})))
}

#[derive(Deserialize)]
struct CheckReq {
    command: String,
}

async fn check(State(s): State<Shared>, h: axum::http::HeaderMap, Path(id): Path<String>, Json(b): Json<CheckReq>) -> R {
    authorize(&s, &h)?;
    let w = blocking(s, move |svc| svc.check(&id, &b.command)).await?;
    Ok(Json(json!({"warning": w})))
}

#[derive(Deserialize)]
struct EndReq {
    #[serde(default)]
    resolved: bool,
}

async fn end(State(s): State<Shared>, h: axum::http::HeaderMap, Path(id): Path<String>, Json(b): Json<EndReq>) -> R {
    authorize(&s, &h)?;
    let st = blocking(s, move |svc| svc.end(&id, b.resolved)).await?;
    Ok(Json(json!({"learned": st})))
}

async fn memory(State(s): State<Shared>, Path(repo): Path<String>) -> R {
    let m = blocking(s, move |svc| Ok(svc.memory(&repo))).await?;
    Ok(Json(serde_json::to_value(m)?))
}

async fn results(State(s): State<Shared>) -> R {
    let dir = s.1.results.clone();
    let mut out = Vec::new();
    if let Ok(rd) = std::fs::read_dir(&dir) {
        let mut paths: Vec<_> = rd.flatten().map(|e| e.path()).filter(|p| p.extension().map(|e| e == "json").unwrap_or(false)).collect();
        paths.sort();
        for p in paths {
            if let Ok(t) = std::fs::read_to_string(&p) {
                if let Ok(v) = serde_json::from_str::<Value>(&t) {
                    out.push(json!({"file": p.file_name().unwrap().to_string_lossy(), "data": v}));
                }
            }
        }
    }
    Ok(Json(json!({ "results": out })))
}

fn load_demo(a: &Args, cfg: &SimCfg) -> anyhow::Result<Demo> {
    let train = prepare(load_sessions(std::path::Path::new(&format!("{}.train.jsonl.gz", a.demo)))?);
    let test: Vec<_> = prepare(load_sessions(std::path::Path::new(&format!("{}.test.jsonl.gz", a.demo)))?)
        .into_iter()
        .map(|p| {
            let cps = compaction_points(&p.infos, cfg);
            (p, cps)
        })
        .collect();
    // Tag the demo repo with a per-load nonce: entity ids hash the repo name, so
    // each (re)load writes fresh vertex ids instead of re-creating ids that the
    // previous load's graph (persisted on disk) used and that were just deleted.
    let nonce = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0);
    let repo = format!("{}#{nonce}", train.first().map(|p| p.session.repo.clone()).unwrap_or_default());
    Demo::new(RepoSplit { repo, train, test }, cfg.clone(), store(a)?, store(a)?)
}

async fn demo_state(State(s): State<Shared>) -> R {
    let v = blocking(s, |svc| {
        Ok(match &svc.demo {
            Some(d) => json!({"repo": d.split.repo.split('#').next().unwrap_or(""), "ingested": d.ingested, "train": d.split.train.len(), "test": d.split.test.len(), "curve": d.curve, "stats": d.learned.stats, "weights": d.learned.params.weights, "features": pagefault::policy::FEATURES, "prior": pagefault::policy::PRIOR}),
            None => match DEMO_FAILED.lock().ok().and_then(|f| f.clone()) {
                Some(e) => json!({"error": format!("live demo failed to load: {e}")}),
                None => json!({"loading": true, "error": "Loading the live HydraDB demo graph… (about a minute after a cold start)"}),
            },
        })
    })
    .await?;
    Ok(Json(v))
}

#[derive(Deserialize)]
struct LearnReq {
    n: Option<usize>,
}

async fn demo_learn(State(s): State<Shared>, Json(b): Json<LearnReq>) -> R {
    let n = b.n.unwrap_or(25).min(100);
    blocking(s.clone(), move |svc| {
        let d = svc.demo.as_mut().ok_or_else(|| anyhow::anyhow!("the live demo is still loading, try again in a moment"))?;
        d.learn(n)
    })
    .await?;
    demo_state(State(s)).await
}

async fn demo_reset(State(s): State<Shared>) -> R {
    let a = s.1.clone();
    blocking(s.clone(), move |svc| {
        let cfg = svc.cfg.clone();
        svc.demo = Some(load_demo(&a, &cfg)?);
        Ok(())
    })
    .await?;
    demo_state(State(s)).await
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let a = Args::parse();
    let cfg = SimCfg::default();
    let svc = Service::new(store(&a)?, cfg.clone());
    let shared: Shared = Arc::new((Mutex::new(svc), a.clone()));
    // Open the port first and load the live demo in the background. On small
    // hosts the load takes a while, and platforms such as Render fail a deploy
    // whose port stays closed. Retries until HydraDB accepts queries.
    let bg = shared.clone();
    tokio::task::spawn_blocking(move || {
        let mut last = String::new();
        for attempt in 1..=60 {
            match load_demo(&bg.1, &cfg) {
                Ok(d) => {
                    eprintln!("demo loaded: {} train / {} test sessions", d.split.train.len(), d.split.test.len());
                    if let Ok(mut svc) = bg.0.lock() {
                        svc.demo = Some(d);
                    }
                    return;
                }
                Err(e) => {
                    last = e.to_string();
                    eprintln!("demo not loaded yet (attempt {attempt}): {last}");
                    std::thread::sleep(std::time::Duration::from_secs(5));
                }
            }
        }
        if let Ok(mut f) = DEMO_FAILED.lock() {
            *f = Some(last);
        }
    });
    let app = Router::new()
        .route("/healthz", get(|| async { "ok" }))
        .route("/v1/sessions", post(create_session))
        .route("/v1/sessions/:id/events", post(add_events))
        .route("/v1/sessions/:id/compact", post(compact))
        .route("/v1/sessions/:id/recall", get(recall))
        .route("/v1/sessions/:id/check", post(check))
        .route("/v1/sessions/:id/end", post(end))
        .route("/v1/repos/:repo/memory", get(memory))
        .route("/v1/results", get(results))
        .route("/v1/demo", get(demo_state))
        .route("/v1/demo/learn", post(demo_learn))
        .route("/v1/demo/reset", post(demo_reset))
        .fallback_service(ServeDir::new(&a.web))
        .layer(CorsLayer::permissive())
        .with_state(shared);
    let addr = format!("0.0.0.0:{}", a.port);
    eprintln!("pagefault listening on {addr} (store: {})", a.store);
    axum::serve(tokio::net::TcpListener::bind(&addr).await?, app).await?;
    Ok(())
}
