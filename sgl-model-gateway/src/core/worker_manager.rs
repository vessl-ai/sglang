//! Worker Management Module
//!
//! Provides worker lifecycle operations and fan-out request utilities.

use std::{
    collections::HashMap,
    sync::Arc,
    time::{Duration, Instant},
};

use axum::response::{IntoResponse, Response};
use futures::{
    future,
    stream::{self, StreamExt},
};
use http::StatusCode;
use serde_json::Value;
use tokio::{
    sync::{watch, Mutex},
    task::JoinHandle,
};
use tracing::{debug, info, warn};

use crate::{
    core::{metrics_aggregator::MetricPack, ConnectionMode, Worker, WorkerRegistry, WorkerType},
    policies::{LoadReport, PolicyRegistry},
    protocols::worker_spec::{FlushCacheResult, WorkerLoadInfo, WorkerLoadsResult},
};

const REQUEST_TIMEOUT: Duration = Duration::from_secs(5);
const MAX_CONCURRENT: usize = 32;

/// Result of a fan-out request to a single worker
struct WorkerResponse {
    url: String,
    result: Result<reqwest::Response, reqwest::Error>,
}

/// Fan out requests to workers in parallel
async fn fan_out(
    workers: &[Arc<dyn Worker>],
    client: &reqwest::Client,
    endpoint: &str,
    method: reqwest::Method,
) -> Vec<WorkerResponse> {
    let futures: Vec<_> = workers
        .iter()
        .map(|worker| {
            let client = client.clone();
            let url = worker.url().to_string();
            let full_url = format!("{}/{}", url, endpoint);
            let api_key = worker.api_key().clone();
            let method = method.clone();

            async move {
                let mut req = client.request(method, &full_url).timeout(REQUEST_TIMEOUT);
                if let Some(key) = api_key {
                    req = req.bearer_auth(key);
                }
                WorkerResponse {
                    url,
                    result: req.send().await,
                }
            }
        })
        .collect();

    stream::iter(futures)
        .buffer_unordered(MAX_CONCURRENT)
        .collect()
        .await
}

pub enum EngineMetricsResult {
    Ok(String),
    Err(String),
}

impl IntoResponse for EngineMetricsResult {
    fn into_response(self) -> Response {
        match self {
            Self::Ok(text) => (StatusCode::OK, text).into_response(),
            Self::Err(msg) => (StatusCode::INTERNAL_SERVER_ERROR, msg).into_response(),
        }
    }
}

pub struct WorkerManager;

impl WorkerManager {
    pub fn get_worker_urls(registry: &Arc<WorkerRegistry>) -> Vec<String> {
        registry
            .get_all()
            .iter()
            .map(|w| w.url().to_string())
            .collect()
    }

    pub async fn flush_cache_all(
        worker_registry: &WorkerRegistry,
        client: &reqwest::Client,
    ) -> FlushCacheResult {
        let workers = worker_registry.get_all();
        let total_workers = workers.len();

        let http_workers: Vec<_> = workers
            .into_iter()
            .filter(|w| matches!(w.connection_mode(), ConnectionMode::Http))
            .collect();

        if http_workers.is_empty() {
            return FlushCacheResult {
                successful: vec![],
                failed: vec![],
                total_workers,
                http_workers: 0,
                message: "No HTTP workers available for cache flush".to_string(),
            };
        }

        info!(
            "Flushing cache on {} HTTP workers (out of {} total)",
            http_workers.len(),
            total_workers
        );

        let responses = fan_out(&http_workers, client, "flush_cache", reqwest::Method::POST).await;

        let mut successful = Vec::new();
        let mut failed = Vec::new();

        for resp in responses {
            match resp.result {
                Ok(r) if r.status().is_success() => successful.push(resp.url),
                Ok(r) => failed.push((resp.url, format!("HTTP {}", r.status()))),
                Err(e) => failed.push((resp.url, e.to_string())),
            }
        }

        let message = if failed.is_empty() {
            format!(
                "Successfully flushed cache on all {} HTTP workers",
                successful.len()
            )
        } else {
            format!(
                "Cache flush: {} succeeded, {} failed",
                successful.len(),
                failed.len()
            )
        };

        info!("{}", message);

        FlushCacheResult {
            successful,
            failed,
            total_workers,
            http_workers: http_workers.len(),
            message,
        }
    }

    pub async fn get_all_worker_loads(
        worker_registry: &WorkerRegistry,
        client: &reqwest::Client,
    ) -> WorkerLoadsResult {
        let loads: Vec<WorkerLoadInfo> = Self::fetch_worker_loads(worker_registry, client)
            .await
            .into_iter()
            .map(|(load, _, _)| load)
            .collect();
        let total_workers = loads.len();
        let successful = loads.iter().filter(|l| l.load >= 0).count();
        let failed = loads.iter().filter(|l| l.load < 0).count();

        WorkerLoadsResult {
            loads,
            total_workers,
            successful,
            failed,
        }
    }

    /// Each load comes with the time its `/v1/loads` request started.
    async fn fetch_worker_loads(
        worker_registry: &WorkerRegistry,
        client: &reqwest::Client,
    ) -> Vec<(WorkerLoadInfo, Instant, Option<u64>)> {
        let workers = worker_registry.get_all();

        let futures: Vec<_> = workers
            .iter()
            .map(|worker| {
                let url = worker.url().to_string();
                let api_key = worker.api_key().clone();
                let worker_type = match worker.worker_type() {
                    WorkerType::Regular => None,
                    WorkerType::Prefill { .. } => Some("prefill".to_string()),
                    WorkerType::Decode => Some("decode".to_string()),
                };
                let is_http = matches!(worker.connection_mode(), ConnectionMode::Http);
                // A worker registered per DP rank has a rank-suffixed URL that
                // does not serve `/v1/loads`, so it gets no report.
                let dp_size = (!worker.is_dp_aware())
                    .then(|| worker.metadata().labels.get("dp_size")?.parse().ok())
                    .flatten();
                let client = client.clone();

                async move {
                    let queried_at = Instant::now();
                    let (load, requests) = if is_http {
                        Self::parse_load_response(&client, &url, api_key.as_deref(), dp_size).await
                    } else {
                        (-1, None)
                    };
                    let info = WorkerLoadInfo {
                        worker: url,
                        worker_type,
                        load,
                    };
                    (info, queried_at, requests)
                }
            })
            .collect();

        future::join_all(futures).await
    }

    async fn parse_load_response(
        client: &reqwest::Client,
        url: &str,
        api_key: Option<&str>,
        dp_size: Option<usize>,
    ) -> (isize, Option<u64>) {
        let load_url = format!("{}/v1/loads?include=core", url);
        let mut req = client.get(&load_url).timeout(REQUEST_TIMEOUT);
        if let Some(key) = api_key {
            req = req.bearer_auth(key);
        }

        match req.send().await {
            Ok(r) if r.status().is_success() => match r.json::<Value>().await {
                Ok(json) => {
                    let tokens = json
                        .get("aggregate")
                        .and_then(|aggregate| aggregate.get("total_tokens"))
                        .and_then(Value::as_i64)
                        .map(|tokens| tokens as isize)
                        .unwrap_or(-1);
                    let requests = dp_size.and_then(|size| requests_across_ranks(&json, size));
                    (tokens, requests)
                }
                _ => (-1, None),
            },
            _ => (-1, None),
        }
    }

    pub async fn get_engine_metrics(
        worker_registry: &WorkerRegistry,
        client: &reqwest::Client,
    ) -> EngineMetricsResult {
        let workers = worker_registry.get_all();

        if workers.is_empty() {
            return EngineMetricsResult::Err("No available workers".to_string());
        }

        let responses = fan_out(&workers, client, "metrics", reqwest::Method::GET).await;

        let mut metric_packs = Vec::new();
        for resp in responses {
            if let Ok(r) = resp.result {
                if r.status().is_success() {
                    if let Ok(text) = r.text().await {
                        metric_packs.push(MetricPack {
                            labels: vec![("worker_addr".into(), resp.url)],
                            metrics_text: text,
                        });
                    }
                }
            }
        }

        if metric_packs.is_empty() {
            return EngineMetricsResult::Err("All backend requests failed".to_string());
        }

        match crate::core::metrics_aggregator::aggregate_metrics(metric_packs) {
            Ok(text) => EngineMetricsResult::Ok(text),
            Err(e) => EngineMetricsResult::Err(format!("Failed to aggregate metrics: {}", e)),
        }
    }
}

fn requests_across_ranks(json: &Value, dp_size: usize) -> Option<u64> {
    let loads = json.get("loads")?.as_array()?;
    if loads.is_empty() || loads.len() != dp_size {
        return None;
    }
    let mut seen = vec![false; dp_size];
    let mut total: u64 = 0;
    for load in loads {
        let rank = usize::try_from(load.get("dp_rank")?.as_u64()?).ok()?;
        let running = load.get("num_running_reqs")?.as_u64()?;
        let waiting = load.get("num_waiting_reqs")?.as_u64()?;
        if std::mem::replace(seen.get_mut(rank)?, true) {
            return None;
        }
        total = total.checked_add(running)?.checked_add(waiting)?;
    }
    Some(total)
}

/// Load monitoring service that periodically fetches worker loads
pub struct LoadMonitor {
    worker_registry: Arc<WorkerRegistry>,
    policy_registry: Arc<PolicyRegistry>,
    client: reqwest::Client,
    interval: Duration,
    tx: watch::Sender<HashMap<String, LoadReport>>,
    rx: watch::Receiver<HashMap<String, LoadReport>>,
    monitor_handle: Arc<Mutex<Option<JoinHandle<()>>>>,
}

impl LoadMonitor {
    pub fn new(
        worker_registry: Arc<WorkerRegistry>,
        policy_registry: Arc<PolicyRegistry>,
        client: reqwest::Client,
        interval_secs: u64,
    ) -> Self {
        let (tx, rx) = watch::channel(HashMap::new());

        Self {
            worker_registry,
            policy_registry,
            client,
            interval: Duration::from_secs(interval_secs),
            tx,
            rx,
            monitor_handle: Arc::new(Mutex::new(None)),
        }
    }

    pub async fn start(&self) {
        let mut handle_guard = self.monitor_handle.lock().await;
        if handle_guard.is_some() {
            debug!("Load monitoring already running");
            return;
        }

        let interval = if self.policy_registry.get_prefill_policy().name() == "cache_load_weighted"
        {
            Duration::from_secs(1)
        } else {
            self.interval
        };
        info!("Starting load monitoring with interval: {:?}", interval);

        let worker_registry = Arc::clone(&self.worker_registry);
        let policy_registry = Arc::clone(&self.policy_registry);
        let client = self.client.clone();
        let tx = self.tx.clone();

        let handle = tokio::spawn(async move {
            Self::monitor_loop(worker_registry, policy_registry, client, interval, tx).await;
        });

        *handle_guard = Some(handle);
    }

    pub async fn stop(&self) {
        let mut handle_guard = self.monitor_handle.lock().await;
        if let Some(handle) = handle_guard.take() {
            info!("Stopping load monitoring");
            handle.abort();
            let _ = handle.await; // Wait for task to finish
        }
    }

    pub fn subscribe(&self) -> watch::Receiver<HashMap<String, LoadReport>> {
        self.rx.clone()
    }

    async fn monitor_loop(
        worker_registry: Arc<WorkerRegistry>,
        policy_registry: Arc<PolicyRegistry>,
        client: reqwest::Client,
        interval: Duration,
        tx: watch::Sender<HashMap<String, LoadReport>>,
    ) {
        let mut interval_timer = tokio::time::interval(interval);

        loop {
            interval_timer.tick().await;

            let policies = policy_registry.get_load_monitor_policies();

            if policies.is_empty() {
                debug!("No load-aware policies found, skipping load fetch");
                continue;
            }

            let loads: HashMap<String, LoadReport> =
                WorkerManager::fetch_worker_loads(&worker_registry, &client)
                    .await
                    .into_iter()
                    .map(|(info, queried_at, requests)| {
                        (
                            info.worker,
                            LoadReport {
                                tokens: info.load,
                                requests,
                                queried_at,
                            },
                        )
                    })
                    .collect();

            if !loads.is_empty() {
                debug!(
                    "Fetched loads from {} workers, updating {} load-aware policies",
                    loads.len(),
                    policies.len()
                );
                for policy in &policies {
                    policy.update_loads(&loads);
                }
                let _ = tx.send(loads);
            } else {
                warn!("No loads fetched from workers");
            }
        }
    }

    pub async fn is_running(&self) -> bool {
        let handle_guard = self.monitor_handle.lock().await;
        handle_guard.is_some()
    }
}

impl Drop for LoadMonitor {
    fn drop(&mut self) {
        if let Ok(mut handle_guard) = self.monitor_handle.try_lock() {
            if let Some(handle) = handle_guard.take() {
                handle.abort();
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;
    use crate::core::{BasicWorkerBuilder, WorkerType};

    fn loads_response(loads: Value) -> Value {
        json!({
            "timestamp": "2026-10-01T00:00:00+00:00",
            "version": "0.5.0",
            "accelerator": "H100",
            "num_accelerators": 4,
            "aggregate": {"total_tokens": 350},
            "loads": loads,
        })
    }

    fn rank(dp_rank: Value, num_total_tokens: Value) -> Value {
        json!({
            "timestamp": 1.0,
            "dp_rank": dp_rank,
            "num_running_reqs": 3,
            "num_waiting_reqs": 1,
            "num_used_tokens": 10,
            "num_total_tokens": num_total_tokens,
            "max_total_num_tokens": 100000,
        })
    }

    #[test]
    fn requests_sum_running_and_waiting_across_every_rank() {
        let response = loads_response(json!([
            rank(json!(1), json!(250)),
            rank(json!(0), json!(100)),
        ]));
        assert_eq!(requests_across_ranks(&response, 2), Some(8));
    }

    #[test]
    fn requests_reject_incomplete_or_malformed_reports() {
        let cases = [
            ("no loads", json!({"timestamp": "x"}), 1),
            ("empty", loads_response(json!([])), 1),
            (
                "duplicate rank",
                loads_response(json!([rank(json!(0), json!(1)), rank(json!(0), json!(1))])),
                2,
            ),
            (
                "missing rank",
                loads_response(json!([rank(json!(0), json!(1))])),
                2,
            ),
            (
                "rank out of range",
                loads_response(json!([rank(json!(0), json!(1)), rank(json!(2), json!(1))])),
                2,
            ),
            (
                "negative running requests",
                loads_response(json!([{"dp_rank":0,"num_running_reqs":-1,"num_waiting_reqs":1}])),
                1,
            ),
            (
                "fractional running requests",
                loads_response(json!([{"dp_rank":0,"num_running_reqs":1.5,"num_waiting_reqs":1}])),
                1,
            ),
            (
                "string waiting requests",
                loads_response(json!([{"dp_rank":0,"num_running_reqs":1,"num_waiting_reqs":"7"}])),
                1,
            ),
            (
                "negative rank",
                loads_response(json!([rank(json!(-1), json!(1))])),
                1,
            ),
            (
                "missing request counts",
                loads_response(json!([{"dp_rank": 0}])),
                1,
            ),
        ];
        for (name, response, dp_size) in cases {
            assert_eq!(requests_across_ranks(&response, dp_size), None, "{name}");
        }
    }

    #[test]
    fn requests_ignore_tokens_and_reject_overflow() {
        let response = loads_response(json!([
            {"dp_rank":0,"num_running_reqs":2,"num_waiting_reqs":5},
            {"dp_rank":1,"num_running_reqs":4,"num_waiting_reqs":7,"num_total_tokens":-1},
        ]));
        assert_eq!(requests_across_ranks(&response, 2), Some(18));
        let overflow = loads_response(json!([
            {"dp_rank":0,"num_running_reqs":u64::MAX,"num_waiting_reqs":1},
        ]));
        assert_eq!(requests_across_ranks(&overflow, 1), None);
    }

    #[tokio::test]
    async fn weighted_monitor_refreshes_before_reports_expire() {
        use crate::{config::PolicyConfig, policies::PolicyFactory};

        let body = loads_response(json!([rank(json!(0), json!(100))]));
        let app = axum::Router::new().route(
            "/v1/loads",
            axum::routing::get(move || {
                let body = body.clone();
                async move { axum::Json(body) }
            }),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        let server = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        let workers = Arc::new(WorkerRegistry::new());
        workers.register(Arc::new(
            BasicWorkerBuilder::new(url)
                .worker_type(WorkerType::Prefill {
                    bootstrap_port: None,
                })
                .label("dp_size", "1")
                .build(),
        ));
        let policies = Arc::new(PolicyRegistry::new(PolicyConfig::RoundRobin));
        policies.set_prefill_policy(PolicyFactory::create_from_config(
            &PolicyConfig::CacheLoadWeighted {
                cache_weight: 1.0,
                load_weight: 1.0,
                eviction_interval_secs: 0,
                max_tree_size: 10000,
            },
        ));
        let monitor = LoadMonitor::new(workers, policies, reqwest::Client::new(), 30);
        let mut reports = monitor.subscribe();
        monitor.start().await;
        for _ in 0..2 {
            tokio::time::timeout(Duration::from_secs(2), reports.changed())
                .await
                .unwrap()
                .unwrap();
            assert_eq!(
                reports
                    .borrow_and_update()
                    .values()
                    .next()
                    .unwrap()
                    .requests,
                Some(4)
            );
        }
        monitor.stop().await;
        server.abort();
    }

    /// The DP size comes from the worker's `dp_size` label; without it the
    /// response cannot be checked for missing ranks, so the report is unusable.
    #[tokio::test]
    async fn fetch_uses_dp_size_label_and_rejects_unknown_dp_size() {
        let body = loads_response(json!([
            rank(json!(0), json!(100)),
            rank(json!(1), json!(250)),
        ]));
        let app = axum::Router::new().route(
            "/v1/loads",
            axum::routing::get(move || {
                let body = body.clone();
                async move { axum::Json(body) }
            }),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        let server = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });

        let client = reqwest::Client::new();
        for (label, expected) in [(Some("2"), Some(8)), (None, None)] {
            let registry = WorkerRegistry::new();
            let mut builder =
                BasicWorkerBuilder::new(url.clone()).worker_type(WorkerType::Prefill {
                    bootstrap_port: None,
                });
            if let Some(dp_size) = label {
                builder = builder.label("dp_size", dp_size);
            }
            registry.register(Arc::new(builder.build()));
            let loads = WorkerManager::fetch_worker_loads(&registry, &client).await;
            assert_eq!(loads.len(), 1);
            assert_eq!(loads[0].0.load, 350, "legacy aggregate token load");
            assert_eq!(loads[0].2, expected, "dp_size label {label:?}");
        }
        server.abort();
    }
}
