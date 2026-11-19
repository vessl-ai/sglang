//! Least-load balancing policy (vessl addition).
//!
//! Routes each request to the decode worker with the fewest live in-flight requests
//! (`Worker::load()`, the RAII `WorkerLoadGuard` counter — incremented on dispatch,
//! decremented when the response/stream finishes). Unlike `power_of_two` this reads the
//! live counter directly (no polling lag), and unlike `round_robin` it accounts for the
//! actual residency of long streaming generations.
//!
//! ## Tie-break is RANDOM among the minimum (the fix for the streaming thundering herd)
//!
//! For a PD streaming reply the load counter is only incremented AFTER the first token
//! (the dispatch path skips the guard for streams; the guard is attached when the
//! streaming response body is built). So during the select→increment window (≈ one TTFT)
//! every arriving request sees the SAME set of still-zero workers. A deterministic argmin
//! (`min_by_key`, which returns the first minimum) therefore sends the entire burst to the
//! single lowest-index worker — that worker spikes to tens of in-flight while the rest sit
//! at zero (observed: max 28 / p50 1 / min 0). Picking RANDOMLY among all workers tied at
//! the minimum load spreads the burst across the idle pool and breaks the herd.

use std::sync::Arc;

use async_trait::async_trait;
use rand::Rng;

use super::{get_healthy_worker_indices, LoadBalancingPolicy, SelectWorkerInfo};
use crate::core::Worker;

/// Least-load policy: pick (one of) the healthy worker(s) with the smallest live load.
#[derive(Debug, Default)]
pub struct LeastLoadPolicy;

impl LeastLoadPolicy {
    pub fn new() -> Self {
        Self
    }
}

#[async_trait]
impl LoadBalancingPolicy for LeastLoadPolicy {
    async fn select_worker(
        &self,
        workers: &[Arc<dyn Worker>],
        _info: &SelectWorkerInfo<'_>,
    ) -> Option<usize> {
        let healthy = get_healthy_worker_indices(workers);
        if healthy.is_empty() {
            return None;
        }
        if healthy.len() == 1 {
            let sel = healthy[0];
            workers[sel].increment_processed();
            return Some(sel);
        }

        // Live in-flight minimum across healthy workers.
        let min_load = healthy.iter().map(|&i| workers[i].load()).min()?;

        // All workers tied at the minimum — random tie-break breaks the streaming herd.
        // (Single-pass reservoir pick so we don't allocate a candidates Vec on the hot path.)
        let mut rng = rand::rng();
        let mut chosen = None;
        let mut seen = 0u32;
        for &i in &healthy {
            if workers[i].load() == min_load {
                seen += 1;
                // Replace with probability 1/seen => uniform over all minima.
                if rng.random_range(0..seen) == 0 {
                    chosen = Some(i);
                }
            }
        }

        let sel = chosen?;
        workers[sel].increment_processed();
        Some(sel)
    }

    fn name(&self) -> &'static str {
        "least_load"
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}
