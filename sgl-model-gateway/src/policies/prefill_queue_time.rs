//! Prefill policy that compares engine-reported prefill load (vessl addition).
//!
//! Each selection samples two healthy workers at random and dispatches to the one
//! with less pending prefill work, so requests arriving close together spread
//! over the pool instead of all landing on the same minimum.
//! Engine reports are compared only when every candidate has a fresh one;
//! otherwise every comparison uses the router-local in-flight count.
//! Reports arrive on each engine's load PUB sockets (see `prefill_load_feed`).

use std::{
    cmp::Ordering,
    collections::HashMap,
    sync::{Arc, Mutex, RwLock},
    time::{Duration, Instant},
};

use async_trait::async_trait;
use rand::Rng;
use tokio::task::JoinHandle;
use tracing::debug;

use super::{
    get_healthy_worker_indices, prefill_load_feed::run_worker_feed, LoadBalancingPolicy,
    SelectWorkerInfo,
};
use crate::core::{ConnectionMode, Worker, WorkerType};

/// A rank with no message for this long counts as missing; the engine re-sends
/// an unchanged load every second, so silence means the rank or the link is down.
/// Same window as `experimental/sgl-router`'s engine-reported load table.
const REPORT_FRESHNESS: Duration = Duration::from_secs(5);

/// The fields of one rank's `LoadStat` message this policy uses.
#[derive(Debug, Clone, PartialEq)]
pub(super) struct RankLoad {
    /// Waiting input tokens with the prefix-cache hits already subtracted.
    pub num_waiting_uncached_tokens: u64,
    pub num_waiting_reqs: u64,
    pub num_running_reqs: u64,
    /// Cumulative input tokens actually computed by prefill.
    pub total_prefill_uncached_tokens: u64,
    /// Cumulative time spent in prefill batches, in microseconds.
    pub total_prefill_busy_us: u64,
}

impl RankLoad {
    fn prefill_counters(&self) -> (u64, u64) {
        (
            self.total_prefill_uncached_tokens,
            self.total_prefill_busy_us,
        )
    }
}

#[derive(Debug)]
struct RankEntry {
    current: RankLoad,
    /// The last message whose prefill counters differ from `current`'s.
    previous: Option<RankLoad>,
    received_at: Instant,
}

impl RankEntry {
    /// Input tokens per second between `previous` and `current`; `None` before
    /// a second distinct sample or when a counter went backwards (engine restart).
    fn prefill_tokens_per_s(&self) -> Option<f64> {
        let previous = self.previous.as_ref()?;
        let tokens = self
            .current
            .total_prefill_uncached_tokens
            .checked_sub(previous.total_prefill_uncached_tokens)
            .filter(|&tokens| tokens > 0)?;
        let busy_us = self
            .current
            .total_prefill_busy_us
            .checked_sub(previous.total_prefill_busy_us)
            .filter(|&busy_us| busy_us > 0)?;
        Some(1_000_000.0 * tokens as f64 / busy_us as f64)
    }
}

/// Engine-reported prefill load of one worker, summed over its DP ranks.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) struct PrefillLoad {
    /// Waiting uncached tokens divided by prefill throughput; `None` unless
    /// every rank has a throughput.
    estimated_queue_ms: Option<f64>,
    pub num_waiting_uncached_tokens: u64,
    num_waiting_reqs: u64,
    num_running_reqs: u64,
}

impl PrefillLoad {
    fn summarize<'a>(ranks: impl Iterator<Item = &'a RankEntry> + Clone) -> Self {
        let num_waiting_uncached_tokens = ranks
            .clone()
            .map(|rank| rank.current.num_waiting_uncached_tokens)
            .sum();
        let tokens_per_s: Option<f64> = ranks.clone().map(RankEntry::prefill_tokens_per_s).sum();
        Self {
            estimated_queue_ms: tokens_per_s
                .map(|rate| 1_000.0 * num_waiting_uncached_tokens as f64 / rate),
            num_waiting_uncached_tokens,
            num_waiting_reqs: ranks
                .clone()
                .map(|rank| rank.current.num_waiting_reqs)
                .sum(),
            num_running_reqs: ranks.map(|rank| rank.current.num_running_reqs).sum(),
        }
    }
}

#[derive(Debug)]
pub(super) struct WorkerReport {
    num_ranks: usize,
    ranks: HashMap<u32, RankEntry>,
    /// The summed load and its least recently heard rank's receive time;
    /// `None` until every rank has sent a message.
    summary: Option<(PrefillLoad, Instant)>,
}

impl WorkerReport {
    fn record(&mut self, rank: u32, load: RankLoad, now: Instant) {
        match self.ranks.get_mut(&rank) {
            // The engine re-sends unchanged counters as a heartbeat; keeping the
            // older `previous` stops the queue-time estimate from vanishing between batches.
            Some(entry) if entry.current.prefill_counters() == load.prefill_counters() => {
                entry.current = load;
                entry.received_at = now;
            }
            Some(entry) => {
                entry.previous = Some(std::mem::replace(&mut entry.current, load));
                entry.received_at = now;
            }
            None => {
                self.ranks.insert(
                    rank,
                    RankEntry {
                        current: load,
                        previous: None,
                        received_at: now,
                    },
                );
            }
        }
        self.summary = (self.ranks.len() == self.num_ranks).then(|| {
            let oldest = self.ranks.values().map(|rank| rank.received_at).min();
            (
                PrefillLoad::summarize(self.ranks.values()),
                oldest.unwrap_or(now),
            )
        });
    }

    pub(super) fn fresh_load(&self, now: Instant) -> Option<PrefillLoad> {
        self.summary
            .filter(|(_, oldest)| now.saturating_duration_since(*oldest) <= REPORT_FRESHNESS)
            .map(|(load, _)| load)
    }
}

/// Worker URL to its engine-reported load, shared with the feed tasks.
pub(super) type ReportTable = Arc<RwLock<HashMap<String, WorkerReport>>>;

/// Starts a fresh report for a worker whose load sockets are about to be read.
pub(super) fn expect_ranks(reports: &ReportTable, worker_url: &str, num_ranks: usize) {
    if let Ok(mut reports) = reports.write() {
        reports.insert(
            worker_url.to_string(),
            WorkerReport {
                num_ranks,
                ranks: HashMap::new(),
                summary: None,
            },
        );
    }
}

/// Applies one rank's message; ignored when the worker is no longer tracked.
pub(super) fn record_rank(
    reports: &ReportTable,
    worker_url: &str,
    rank: u32,
    load: RankLoad,
    now: Instant,
) {
    if let Ok(mut reports) = reports.write() {
        if let Some(report) = reports.get_mut(worker_url) {
            report.record(rank, load, now);
        }
    }
}

/// Queue-time estimate first when both sides have one, then the waiting and
/// running counts.
fn compare_prefill_load(left: &PrefillLoad, right: &PrefillLoad) -> Ordering {
    let queue_time = match (left.estimated_queue_ms, right.estimated_queue_ms) {
        (Some(left_ms), Some(right_ms)) => left_ms.total_cmp(&right_ms),
        _ => Ordering::Equal,
    };
    queue_time.then_with(|| {
        (
            left.num_waiting_uncached_tokens,
            left.num_waiting_reqs,
            left.num_running_reqs,
        )
            .cmp(&(
                right.num_waiting_uncached_tokens,
                right.num_waiting_reqs,
                right.num_running_reqs,
            ))
    })
}

#[derive(Debug, Default)]
pub struct PrefillQueueTimePolicy {
    reports: ReportTable,
    /// Worker URL to the task reading that worker's load sockets.
    feeds: Mutex<HashMap<String, JoinHandle<()>>>,
}

impl PrefillQueueTimePolicy {
    pub fn new() -> Self {
        Self::default()
    }

    /// `policy` as this policy, when it is one.
    pub fn of(policy: &dyn LoadBalancingPolicy) -> Option<&Self> {
        policy.as_any().downcast_ref()
    }

    /// Starts reading the load sockets of the HTTP prefill workers in `workers`,
    /// replacing any feed already running for the same URL.
    pub fn subscribe_workers(&self, workers: &[Arc<dyn Worker>]) {
        let Ok(mut feeds) = self.feeds.lock() else {
            return;
        };
        for worker in workers {
            if !matches!(worker.worker_type(), WorkerType::Prefill { .. })
                || !matches!(worker.connection_mode(), ConnectionMode::Http)
            {
                continue;
            }
            if let Ok(mut reports) = self.reports.write() {
                reports.remove(worker.url());
            }
            let feed = tokio::spawn(run_worker_feed(
                Arc::clone(&self.reports),
                worker.url().to_string(),
                worker.base_url().to_string(),
                worker.api_key().clone(),
                worker.dp_rank(),
            ));
            if let Some(old) = feeds.insert(worker.url().to_string(), feed) {
                old.abort();
            }
        }
    }

    /// Stops the worker's feed and drops its report.
    pub fn remove_worker(&self, worker_url: &str) {
        if let Some(feed) = self
            .feeds
            .lock()
            .ok()
            .and_then(|mut feeds| feeds.remove(worker_url))
        {
            feed.abort();
        }
        if let Ok(mut reports) = self.reports.write() {
            reports.remove(worker_url);
        }
    }

    /// Engine loads of `pair`, or `None` unless every candidate has a fresh report.
    fn pair_loads(
        &self,
        workers: &[Arc<dyn Worker>],
        candidates: &[usize],
        pair: (usize, usize),
        now: Instant,
    ) -> Option<(PrefillLoad, PrefillLoad)> {
        let reports = self.reports.read().ok()?;
        let fresh_load = |idx: usize| {
            reports
                .get(workers[idx].url())
                .and_then(|report| report.fresh_load(now))
        };
        if !candidates.iter().all(|&idx| fresh_load(idx).is_some()) {
            return None;
        }
        Some((fresh_load(pair.0)?, fresh_load(pair.1)?))
    }

    fn select_at(&self, workers: &[Arc<dyn Worker>], now: Instant) -> Option<usize> {
        let healthy = get_healthy_worker_indices(workers);
        if healthy.len() <= 1 {
            return healthy.first().copied();
        }

        let mut rng = rand::rng();
        let first = rng.random_range(0..healthy.len());
        let second = (first + 1 + rng.random_range(0..healthy.len() - 1)) % healthy.len();
        let pair = (healthy[first], healthy[second]);

        let loads = self.pair_loads(workers, &healthy, pair, now);
        let second_vs_first = loads
            .map_or(Ordering::Equal, |(first_load, second_load)| {
                compare_prefill_load(&second_load, &first_load)
            })
            .then_with(|| workers[pair.1].load().cmp(&workers[pair.0].load()));
        // A full tie keeps the first sample, which was itself drawn at random.
        let selected = if second_vs_first.is_lt() {
            pair.1
        } else {
            pair.0
        };

        debug!(
            "prefill_queue_time: {} vs {} (engine loads: {:?}) -> {}",
            workers[pair.0].url(),
            workers[pair.1].url(),
            loads,
            workers[selected].url()
        );
        Some(selected)
    }
}

impl Drop for PrefillQueueTimePolicy {
    fn drop(&mut self) {
        if let Ok(feeds) = self.feeds.get_mut() {
            for feed in feeds.values() {
                feed.abort();
            }
        }
    }
}

#[async_trait]
impl LoadBalancingPolicy for PrefillQueueTimePolicy {
    async fn select_worker(
        &self,
        workers: &[Arc<dyn Worker>],
        _info: &SelectWorkerInfo<'_>,
    ) -> Option<usize> {
        let selected = self.select_at(workers, Instant::now())?;
        workers[selected].increment_processed();
        Some(selected)
    }

    fn name(&self) -> &'static str {
        "prefill_queue_time"
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::BasicWorkerBuilder;

    fn rank_load(waiting_tokens: u64, prefill_tokens_total: u64, busy_us_total: u64) -> RankLoad {
        RankLoad {
            num_waiting_uncached_tokens: waiting_tokens,
            num_waiting_reqs: 0,
            num_running_reqs: 0,
            total_prefill_uncached_tokens: prefill_tokens_total,
            total_prefill_busy_us: busy_us_total,
        }
    }

    fn prefill_workers(urls: &[&str]) -> Vec<Arc<dyn Worker>> {
        urls.iter()
            .map(|url| {
                Arc::new(
                    BasicWorkerBuilder::new(*url)
                        .worker_type(WorkerType::Prefill {
                            bootstrap_port: None,
                        })
                        .build(),
                ) as Arc<dyn Worker>
            })
            .collect()
    }

    /// Tracks each URL as a single-rank worker reporting `waiting_tokens`.
    fn report_waiting(policy: &PrefillQueueTimePolicy, urls_tokens: &[(&str, u64)], now: Instant) {
        for &(url, tokens) in urls_tokens {
            expect_ranks(&policy.reports, url, 1);
            record_rank(&policy.reports, url, 0, rank_load(tokens, 0, 0), now);
        }
    }

    fn summary_of(policy: &PrefillQueueTimePolicy, url: &str) -> PrefillLoad {
        policy.reports.read().unwrap()[url].summary.unwrap().0
    }

    fn load(queue_ms: Option<f64>, tokens: u64, waiting: u64, running: u64) -> PrefillLoad {
        PrefillLoad {
            estimated_queue_ms: queue_ms,
            num_waiting_uncached_tokens: tokens,
            num_waiting_reqs: waiting,
            num_running_reqs: running,
        }
    }

    #[test]
    fn test_queue_time_is_waiting_tokens_over_prefill_throughput() {
        let policy = PrefillQueueTimePolicy::new();
        let t0 = Instant::now();
        expect_ranks(&policy.reports, "http://p", 1);
        record_rank(
            &policy.reports,
            "http://p",
            0,
            rank_load(0, 1_000, 1_000_000),
            t0,
        );
        assert_eq!(summary_of(&policy, "http://p").estimated_queue_ms, None);

        // 4000 tokens in 0.5 s of prefill = 8000 tok/s; 2000 waiting tokens = 250 ms.
        record_rank(
            &policy.reports,
            "http://p",
            0,
            rank_load(2_000, 5_000, 1_500_000),
            t0,
        );
        assert_eq!(
            summary_of(&policy, "http://p").estimated_queue_ms,
            Some(250.0)
        );

        // A heartbeat with unchanged counters keeps the throughput; the new queue counts.
        record_rank(
            &policy.reports,
            "http://p",
            0,
            rank_load(4_000, 5_000, 1_500_000),
            t0,
        );
        assert_eq!(
            summary_of(&policy, "http://p").estimated_queue_ms,
            Some(500.0)
        );
    }

    #[test]
    fn test_counter_reset_gives_no_queue_time() {
        let policy = PrefillQueueTimePolicy::new();
        let t0 = Instant::now();
        expect_ranks(&policy.reports, "http://p", 1);
        record_rank(
            &policy.reports,
            "http://p",
            0,
            rank_load(0, 1_000, 1_000_000),
            t0,
        );
        record_rank(
            &policy.reports,
            "http://p",
            0,
            rank_load(0, 5_000, 1_500_000),
            t0,
        );
        record_rank(&policy.reports, "http://p", 0, rank_load(100, 10, 10), t0);
        let summary = summary_of(&policy, "http://p");
        assert_eq!(summary.estimated_queue_ms, None);
        assert_eq!(summary.num_waiting_uncached_tokens, 100);
    }

    #[test]
    fn test_dp_ranks_are_summed() {
        let policy = PrefillQueueTimePolicy::new();
        let t0 = Instant::now();
        expect_ranks(&policy.reports, "http://p", 2);
        record_rank(&policy.reports, "http://p", 0, rank_load(0, 0, 0), t0);
        assert!(policy.reports.read().unwrap()["http://p"].summary.is_none());
        record_rank(&policy.reports, "http://p", 1, rank_load(0, 0, 0), t0);

        // Rank 0 runs 1000 tok/s, rank 1 runs 3000 tok/s; 2000 waiting tokens over 4000 tok/s.
        record_rank(
            &policy.reports,
            "http://p",
            0,
            rank_load(500, 1_000, 1_000_000),
            t0,
        );
        record_rank(
            &policy.reports,
            "http://p",
            1,
            rank_load(1_500, 3_000, 1_000_000),
            t0,
        );
        let summed = summary_of(&policy, "http://p");
        assert_eq!(summed.num_waiting_uncached_tokens, 2_000);
        assert_eq!(summed.estimated_queue_ms, Some(500.0));

        // A rank without a second distinct sample leaves the worker without an estimate.
        expect_ranks(&policy.reports, "http://q", 2);
        record_rank(&policy.reports, "http://q", 0, rank_load(0, 0, 0), t0);
        record_rank(
            &policy.reports,
            "http://q",
            0,
            rank_load(500, 1_000, 1_000_000),
            t0,
        );
        record_rank(
            &policy.reports,
            "http://q",
            1,
            rank_load(1_500, 3_000, 1_000_000),
            t0,
        );
        assert_eq!(summary_of(&policy, "http://q").estimated_queue_ms, None);
    }

    #[test]
    fn test_freshness_follows_the_least_recently_heard_rank() {
        let policy = PrefillQueueTimePolicy::new();
        let workers = prefill_workers(&["http://p"]);
        let t0 = Instant::now();
        expect_ranks(&policy.reports, "http://p", 2);
        record_rank(&policy.reports, "http://p", 0, rank_load(0, 0, 0), t0);
        record_rank(&policy.reports, "http://p", 1, rank_load(0, 0, 0), t0);
        // Rank 1 goes silent; rank 0 keeps sending.
        record_rank(
            &policy.reports,
            "http://p",
            0,
            rank_load(0, 0, 0),
            t0 + Duration::from_secs(4),
        );

        assert!(policy
            .pair_loads(&workers, &[0], (0, 0), t0 + Duration::from_secs(5))
            .is_some());
        assert!(policy
            .pair_loads(&workers, &[0], (0, 0), t0 + Duration::from_secs(6))
            .is_none());
    }

    #[tokio::test]
    async fn test_removed_worker_loses_feed_and_report() {
        let policy = PrefillQueueTimePolicy::new();
        // Nothing listens on port 1, so the feed stays in its discovery retry.
        let workers = prefill_workers(&["http://127.0.0.1:1", "http://127.0.0.1:2"]);
        policy.subscribe_workers(&workers);
        report_waiting(
            &policy,
            &[("http://127.0.0.1:1", 7), ("http://127.0.0.1:2", 9)],
            Instant::now(),
        );

        policy.remove_worker("http://127.0.0.1:1");
        assert!(!policy
            .feeds
            .lock()
            .unwrap()
            .contains_key("http://127.0.0.1:1"));
        assert!(!policy
            .reports
            .read()
            .unwrap()
            .contains_key("http://127.0.0.1:1"));
        assert!(policy
            .reports
            .read()
            .unwrap()
            .contains_key("http://127.0.0.1:2"));

        // A late message from the removed worker does not bring its report back.
        record_rank(
            &policy.reports,
            "http://127.0.0.1:1",
            0,
            rank_load(1, 0, 0),
            Instant::now(),
        );
        assert!(!policy
            .reports
            .read()
            .unwrap()
            .contains_key("http://127.0.0.1:1"));
    }

    #[tokio::test]
    async fn test_only_http_prefill_workers_are_subscribed() {
        let policy = PrefillQueueTimePolicy::new();
        let decode: Arc<dyn Worker> = Arc::new(
            BasicWorkerBuilder::new("http://127.0.0.1:3")
                .worker_type(WorkerType::Decode)
                .build(),
        );
        let mut workers = prefill_workers(&["http://127.0.0.1:1"]);
        workers.push(decode);
        policy.subscribe_workers(&workers);
        let feeds = policy.feeds.lock().unwrap();
        assert!(feeds.contains_key("http://127.0.0.1:1"));
        assert!(!feeds.contains_key("http://127.0.0.1:3"));
    }

    #[test]
    fn test_compare_order() {
        // Queue time decides when both sides have one, even against fewer waiting tokens.
        assert_eq!(
            compare_prefill_load(&load(Some(10.0), 9_000, 0, 0), &load(Some(20.0), 100, 0, 0)),
            Ordering::Less
        );
        // Without an estimate on one side: waiting tokens, then waiting reqs, then running reqs.
        assert_eq!(
            compare_prefill_load(&load(None, 100, 9, 9), &load(Some(1.0), 200, 0, 0)),
            Ordering::Less
        );
        assert_eq!(
            compare_prefill_load(&load(None, 100, 1, 9), &load(None, 100, 2, 0)),
            Ordering::Less
        );
        assert_eq!(
            compare_prefill_load(&load(None, 100, 1, 1), &load(None, 100, 1, 2)),
            Ordering::Less
        );
        // Equal queue times fall through to the counts.
        assert_eq!(
            compare_prefill_load(&load(Some(5.0), 100, 1, 1), &load(Some(5.0), 50, 1, 1)),
            Ordering::Greater
        );
    }

    #[test]
    fn test_equal_engine_load_breaks_tie_on_local_inflight() {
        let policy = PrefillQueueTimePolicy::new();
        let workers = prefill_workers(&["http://a", "http://b"]);
        workers[0].increment_load();
        let now = Instant::now();
        report_waiting(&policy, &[("http://a", 100), ("http://b", 100)], now);
        for _ in 0..20 {
            assert_eq!(policy.select_at(&workers, now), Some(1));
        }
    }

    #[test]
    fn test_missing_report_on_any_candidate_falls_back_to_local_inflight() {
        let policy = PrefillQueueTimePolicy::new();
        let workers = prefill_workers(&["http://a", "http://b", "http://c"]);
        let now = Instant::now();
        report_waiting(&policy, &[("http://a", 0), ("http://b", 9_000)], now);
        assert!(policy
            .pair_loads(&workers, &[0, 1, 2], (0, 1), now)
            .is_none());
        assert!(policy.pair_loads(&workers, &[0, 1], (0, 1), now).is_some());

        // a has the lighter engine report but more local in-flight; c's missing report
        // makes the whole set compare local counts, so a is never picked over b or c.
        workers[0].increment_load();
        for _ in 0..50 {
            assert_ne!(policy.select_at(&workers, now), Some(0));
        }
    }

    #[test]
    fn test_power_of_two_picks_the_lighter_of_two_samples() {
        let policy = PrefillQueueTimePolicy::new();
        let workers = prefill_workers(&["http://a", "http://b", "http://c"]);
        let now = Instant::now();
        report_waiting(
            &policy,
            &[("http://a", 100), ("http://b", 200), ("http://c", 300)],
            now,
        );
        let mut counts = [0; 3];
        for _ in 0..300 {
            counts[policy.select_at(&workers, now).unwrap()] += 1;
        }
        // The heaviest worker loses every pair; the lightest wins every pair it is drawn into.
        assert_eq!(counts[2], 0);
        assert!(counts[0] > counts[1]);
        assert!(counts[1] > 0);

        let single = prefill_workers(&["http://a"]);
        assert_eq!(policy.select_at(&single, now), Some(0));
        assert_eq!(policy.select_at(&[], now), None);
    }
}
