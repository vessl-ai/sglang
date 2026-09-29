use std::{
    collections::HashMap,
    sync::{Arc, RwLock},
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

use async_trait::async_trait;
use futures::{stream, StreamExt};
use rand::seq::IndexedRandom;
use serde::Deserialize;
use tracing::warn;

use super::{get_healthy_worker_indices, LoadBalancingPolicy, SelectWorkerInfo};
use crate::core::{ConnectionMode, Worker, WorkerType};

const MAX_SAMPLE_AGE: Duration = Duration::from_secs(5);
const REQUEST_TIMEOUT: Duration = Duration::from_secs(1);

#[derive(Debug, Deserialize)]
struct EngineLoads {
    loads: Vec<RankLoad>,
}

#[derive(Debug, Deserialize)]
struct RankLoad {
    dp_rank: usize,
    timestamp: f64,
    num_total_tokens: u64,
}

#[derive(Debug)]
struct TokenLoad {
    tokens: u64,
    expires_at: Instant,
}

impl EngineLoads {
    fn token_load(
        self,
        expected_ranks: usize,
        selected_rank: Option<usize>,
        now_secs: f64,
        now: Instant,
    ) -> Result<TokenLoad, &'static str> {
        if expected_ranks == 0 || self.loads.len() != expected_ranks {
            return Err("incomplete rank coverage");
        }
        let mut seen = vec![false; expected_ranks];
        let mut tokens = 0u64;
        let mut max_age = 0.0f64;
        for load in self.loads {
            if load.dp_rank >= expected_ranks || seen[load.dp_rank] {
                return Err("invalid or duplicate rank");
            }
            seen[load.dp_rank] = true;
            let age = now_secs - load.timestamp;
            if !(-1.0..MAX_SAMPLE_AGE.as_secs_f64()).contains(&age) {
                return Err("stale or invalid engine timestamp");
            }
            max_age = max_age.max(age);
            if selected_rank.is_none_or(|rank| rank == load.dp_rank) {
                tokens = tokens
                    .checked_add(load.num_total_tokens)
                    .ok_or("token count overflow")?;
            }
        }
        if selected_rank.is_some_and(|rank| rank >= expected_ranks) {
            return Err("selected rank out of range");
        }
        Ok(TokenLoad {
            tokens,
            expires_at: now + MAX_SAMPLE_AGE - Duration::from_secs_f64(max_age),
        })
    }
}

#[derive(Debug, Default)]
pub struct PrefillTokensPolicy {
    loads: RwLock<HashMap<String, TokenLoad>>,
}

impl PrefillTokensPolicy {
    pub async fn refresh(&self, workers: &[Arc<dyn Worker>], client: &reqwest::Client) {
        let futures: Vec<_> = workers
            .iter()
            .filter(|w| {
                matches!(w.worker_type(), WorkerType::Prefill { .. })
                    && matches!(w.connection_mode(), ConnectionMode::Http)
                    && w.is_healthy()
            })
            .map(|worker| {
                let worker = Arc::clone(worker);
                let client = client.clone();
                async move {
                    match Self::fetch(worker.as_ref(), &client).await {
                        Ok(load) => Some((worker.url().to_owned(), load)),
                        Err(error) => {
                            warn!(worker = worker.url(), %error, "Prefill token load unavailable");
                            None
                        }
                    }
                }
            })
            .collect();
        let loads = stream::iter(futures)
            .buffer_unordered(32)
            .filter_map(|load| async move { load })
            .collect::<HashMap<_, _>>()
            .await;
        *self.loads.write().unwrap() = loads;
    }

    async fn fetch(worker: &dyn Worker, client: &reqwest::Client) -> Result<TokenLoad, String> {
        let expected_ranks = match worker.dp_size() {
            Some(size) => size,
            None => worker
                .metadata()
                .labels
                .get("dp_size")
                .map(|size| size.parse::<usize>())
                .transpose()
                .map_err(|_| "invalid worker dp_size")?
                .unwrap_or(1),
        };
        let mut request = client
            .get(worker.endpoint_url("/v1/loads?include=core"))
            .timeout(REQUEST_TIMEOUT);
        if let Some(key) = worker.api_key() {
            request = request.bearer_auth(key);
        }
        let response = request
            .send()
            .await
            .map_err(|e| e.to_string())?
            .error_for_status()
            .map_err(|e| e.to_string())?
            .json::<EngineLoads>()
            .await
            .map_err(|e| e.to_string())?;
        let now_secs = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(|e| e.to_string())?
            .as_secs_f64();
        response
            .token_load(expected_ranks, worker.dp_rank(), now_secs, Instant::now())
            .map_err(str::to_owned)
    }
}

#[async_trait]
impl LoadBalancingPolicy for PrefillTokensPolicy {
    async fn select_worker(
        &self,
        workers: &[Arc<dyn Worker>],
        _info: &SelectWorkerInfo<'_>,
    ) -> Option<usize> {
        let loads = self.loads.read().unwrap();
        let now = Instant::now();
        let candidates: Vec<_> = get_healthy_worker_indices(workers)
            .into_iter()
            .filter(|&i| matches!(workers[i].worker_type(), WorkerType::Prefill { .. }))
            .filter_map(|i| {
                loads
                    .get(workers[i].url())
                    .filter(|load| now < load.expires_at)
                    .map(|load| (i, load.tokens))
            })
            .collect();
        let min = candidates.iter().map(|(_, tokens)| tokens).min()?;
        let minima: Vec<_> = candidates
            .iter()
            .filter(|(_, tokens)| tokens == min)
            .collect();
        let (selected, _) = **minima.choose(&mut rand::rng())?;
        workers[selected].increment_processed();
        Some(selected)
    }

    fn name(&self) -> &'static str {
        "prefill_tokens"
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        config::{PolicyConfig, RouterConfig, RoutingMode},
        core::{BasicWorkerBuilder, LoadMonitor, WorkerRegistry},
        policies::{PolicyFactory, PolicyRegistry},
    };
    use axum::{extract::State, http::StatusCode, routing::get, Json, Router};
    use serde_json::{json, Value};
    use std::sync::atomic::{AtomicUsize, Ordering};

    fn worker(url: &str) -> Arc<dyn Worker> {
        Arc::new(
            BasicWorkerBuilder::new(url)
                .worker_type(WorkerType::Prefill {
                    bootstrap_port: Some(8998),
                })
                .build(),
        )
    }

    fn snapshot(tokens: u64, timestamp: f64) -> Value {
        json!({"loads": [{"dp_rank": 0, "timestamp": timestamp, "num_total_tokens": tokens}]})
    }

    fn now_secs() -> f64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs_f64()
    }

    fn parse(value: Value, ranks: usize, rank: Option<usize>) -> Result<TokenLoad, String> {
        let response: EngineLoads = serde_json::from_value(value).map_err(|e| e.to_string())?;
        response
            .token_load(ranks, rank, 100.0, Instant::now())
            .map_err(str::to_owned)
    }

    #[test]
    fn parses_real_engine_shape_and_complete_ranks() {
        let value = json!({"timestamp": "unused", "loads": [
            {"dp_rank": 1, "timestamp": 99.0, "num_total_tokens": 200},
            {"dp_rank": 0, "timestamp": 99.0, "num_total_tokens": 100}
        ]});
        assert_eq!(parse(value.clone(), 2, None).unwrap().tokens, 300);
        assert_eq!(parse(value.clone(), 2, Some(1)).unwrap().tokens, 200);
        assert!(parse(value, 3, None).is_err());
        assert_eq!(parse(snapshot(0, 100.0), 1, None).unwrap().tokens, 0);
    }

    #[test]
    fn rejects_bad_samples_instead_of_treating_them_as_zero() {
        for value in [
            json!({"aggregate": {"total_tokens": 100}}),
            json!({"loads": []}),
            json!({"loads": [{"dp_rank": 0, "timestamp": 100.0}]}),
            json!({"loads": [{"dp_rank": 0, "timestamp": 100.0, "num_total_tokens": -1}]}),
            snapshot(0, 95.0),
            snapshot(0, 102.0),
        ] {
            assert!(parse(value, 1, None).is_err());
        }
        let duplicate = json!({"loads": [
            {"dp_rank": 0, "timestamp": 100.0, "num_total_tokens": 10},
            {"dp_rank": 0, "timestamp": 100.0, "num_total_tokens": 20}
        ]});
        assert!(parse(duplicate, 2, None).is_err());
        assert!(parse(snapshot(10, 100.0), 1, Some(1)).is_err());
        let overflow = json!({"loads": [
            {"dp_rank": 0, "timestamp": 100.0, "num_total_tokens": u64::MAX},
            {"dp_rank": 1, "timestamp": 100.0, "num_total_tokens": 1}
        ]});
        assert!(parse(overflow, 2, None).is_err());
    }

    #[tokio::test]
    async fn selects_tokens_not_request_counts_and_excludes_stale_or_unhealthy_workers() {
        let policy = PrefillTokensPolicy::default();
        let workers = vec![worker("http://a"), worker("http://b"), worker("http://c")];
        for _ in 0..30 {
            workers[1].increment_load();
        }
        for (i, tokens) in [8000, 1000, 10].into_iter().enumerate() {
            policy.loads.write().unwrap().insert(
                workers[i].url().into(),
                TokenLoad {
                    tokens,
                    expires_at: Instant::now() + MAX_SAMPLE_AGE,
                },
            );
        }
        workers[2].set_healthy(false);
        assert_eq!(
            policy
                .select_worker(&workers, &SelectWorkerInfo::default())
                .await,
            Some(1)
        );
        policy
            .loads
            .write()
            .unwrap()
            .get_mut("http://b")
            .unwrap()
            .expires_at = Instant::now();
        assert_eq!(
            policy
                .select_worker(&workers, &SelectWorkerInfo::default())
                .await,
            Some(0)
        );
        policy.loads.write().unwrap().clear();
        assert_eq!(
            policy
                .select_worker(&workers, &SelectWorkerInfo::default())
                .await,
            None
        );
        assert_eq!(
            policy
                .select_worker(&workers[..1], &SelectWorkerInfo::default())
                .await,
            None
        );
    }

    #[tokio::test]
    async fn randomizes_ties_across_all_four_prefill_workers() {
        let policy = PrefillTokensPolicy::default();
        let workers: Vec<_> = (0..4).map(|i| worker(&format!("http://w{i}"))).collect();
        for w in &workers {
            policy.loads.write().unwrap().insert(
                w.url().into(),
                TokenLoad {
                    tokens: 100,
                    expires_at: Instant::now() + MAX_SAMPLE_AGE,
                },
            );
        }
        let mut counts = [0; 4];
        for _ in 0..200 {
            counts[policy
                .select_worker(&workers, &SelectWorkerInfo::default())
                .await
                .unwrap()] += 1;
        }
        assert!(counts.iter().all(|&count| count > 0), "{counts:?}");
    }

    type MockState = (Arc<RwLock<Value>>, Arc<AtomicUsize>);

    async fn mock_loads(
        State((value, count)): State<MockState>,
        headers: http::HeaderMap,
    ) -> (StatusCode, Json<Value>) {
        count.fetch_add(1, Ordering::Relaxed);
        assert_eq!(headers.get("authorization").unwrap(), "Bearer test-key");
        let value = value.read().unwrap().clone();
        (
            if value.is_null() {
                StatusCode::INTERNAL_SERVER_ERROR
            } else {
                StatusCode::OK
            },
            Json(value),
        )
    }

    async fn mock_server() -> (Arc<dyn Worker>, MockState, tokio::task::JoinHandle<()>) {
        let state = (
            Arc::new(RwLock::new(snapshot(42, now_secs()))),
            Arc::new(AtomicUsize::new(0)),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        let app = Router::new()
            .route("/v1/loads", get(mock_loads))
            .with_state(state.clone());
        let server = tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });
        let worker = Arc::new(
            BasicWorkerBuilder::new(url)
                .worker_type(WorkerType::Prefill {
                    bootstrap_port: Some(8998),
                })
                .api_key("test-key")
                .build(),
        );
        (worker, state, server)
    }

    #[tokio::test]
    async fn http_refresh_clears_failed_stale_and_removed_samples() {
        let (worker, state, server) = mock_server().await;
        let policy = PrefillTokensPolicy::default();
        let client = reqwest::Client::new();
        let workers = [worker];
        policy.refresh(&workers, &client).await;
        assert_eq!(
            policy
                .select_worker(&workers, &SelectWorkerInfo::default())
                .await,
            Some(0)
        );
        for invalid in [
            Value::Null,
            snapshot(0, now_secs() - 60.0),
            json!({"loads": []}),
        ] {
            *state.0.write().unwrap() = invalid;
            policy.refresh(&workers, &client).await;
            assert!(policy.loads.read().unwrap().is_empty());
        }
        *state.0.write().unwrap() = snapshot(0, now_secs());
        policy.refresh(&workers, &client).await;
        assert_eq!(
            policy
                .select_worker(&workers, &SelectWorkerInfo::default())
                .await,
            Some(0)
        );
        policy.refresh(&[], &client).await;
        assert!(policy.loads.read().unwrap().is_empty());
        server.abort();
    }

    #[tokio::test]
    async fn monitor_updates_prefill_policy_and_stops_with_its_owner() {
        let (worker, state, server) = mock_server().await;
        let workers = Arc::new(WorkerRegistry::new());
        workers.register(worker.clone());
        let policies = Arc::new(PolicyRegistry::new(PolicyConfig::Random));
        let policy = PolicyFactory::create_from_config(&PolicyConfig::PrefillTokens);
        policies.set_prefill_policy(policy.clone());
        let monitor = LoadMonitor::new(workers, policies, reqwest::Client::new(), 30);
        monitor.start().await;
        tokio::time::timeout(Duration::from_secs(3), async {
            loop {
                if policy
                    .select_worker(std::slice::from_ref(&worker), &SelectWorkerInfo::default())
                    .await
                    .is_some()
                {
                    break;
                }
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        })
        .await
        .unwrap();
        monitor.stop().await;
        let count = state.1.load(Ordering::Relaxed);
        tokio::time::sleep(Duration::from_millis(350)).await;
        assert_eq!(state.1.load(Ordering::Relaxed), count);
        server.abort();
    }

    #[test]
    fn configuration_limits_policy_to_http_prefill_and_preserves_decode() {
        let mut config = RouterConfig::new(
            RoutingMode::PrefillDecode {
                prefill_urls: vec![("http://p:8000".into(), None)],
                decode_urls: vec!["http://d:8000".into()],
                prefill_policy: Some(PolicyConfig::PrefillTokens),
                decode_policy: Some(PolicyConfig::LeastLoad),
            },
            PolicyConfig::Random,
        );
        assert!(config.validate().is_ok());
        let encoded = serde_json::to_string(&config).unwrap();
        let decoded: RouterConfig = serde_json::from_str(&encoded).unwrap();
        assert!(decoded.validate().is_ok());
        assert!(matches!(
            decoded.mode,
            RoutingMode::PrefillDecode {
                decode_policy: Some(PolicyConfig::LeastLoad),
                ..
            }
        ));
        config.policy = PolicyConfig::PrefillTokens;
        assert!(config.validate().is_err());
        config.policy = PolicyConfig::Random;
        if let RoutingMode::PrefillDecode { decode_policy, .. } = &mut config.mode {
            *decode_policy = Some(PolicyConfig::PrefillTokens);
        }
        assert!(config.validate().is_err());
    }
}
