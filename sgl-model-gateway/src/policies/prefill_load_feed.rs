//! Reads a prefill engine's per-rank load PUB sockets into the
//! `prefill_queue_time` report table.
//!
//! The engine opts in with `--kv-events-config` plus `--load-publish-endpoint`;
//! its `/server_info` then advertises the base port under `kv_events`, and
//! DP rank `r` publishes on `base + r`.

use std::{
    fmt,
    time::{Duration, Instant},
};

use serde::{
    de::{self, IgnoredAny, SeqAccess, Visitor},
    Deserialize, Deserializer,
};
use tokio::time::sleep;
use tracing::{debug, info, warn};
use zeromq::{Socket, SocketRecv, SubSocket, ZmqMessage};

use super::prefill_queue_time::{expect_ranks, record_rank, RankLoad, ReportTable};
use crate::core::steps::worker::local::get_server_info;

/// Arbitrary; bounds how long a worker stays without a report after a
/// transient discovery, connect, or receive failure.
const RETRY_DELAY: Duration = Duration::from_secs(2);

/// Discovers the worker's load ports, then reads every rank until aborted.
/// A worker that advertises no load socket is left without a report.
pub(super) async fn run_worker_feed(
    reports: ReportTable,
    worker_url: String,
    base_url: String,
    api_key: Option<String>,
    dp_rank: Option<usize>,
) {
    let kv_events = loop {
        match get_server_info(&base_url, api_key.as_deref()).await {
            Ok(info) => break info.kv_events,
            Err(e) => {
                debug!("prefill_queue_time: /server_info failed for {worker_url}: {e}");
                sleep(RETRY_DELAY).await;
            }
        }
    };
    let Some((port_base, dp_size)) =
        kv_events.and_then(|block| Some((block.load_endpoint_port_base?, block.dp_size)))
    else {
        info!(
            "prefill_queue_time: {worker_url} advertises no load socket; \
             it is compared on router-local in-flight counts"
        );
        return;
    };
    let Some(host) = url::Url::parse(&base_url)
        .ok()
        .and_then(|url| url.host_str().map(str::to_string))
    else {
        warn!("prefill_queue_time: cannot take a host from {base_url}");
        return;
    };

    // A DP-aware worker object stands for one rank of the engine.
    let ranks: Vec<u32> = match dp_rank {
        Some(rank) => vec![rank as u32],
        None => (0..dp_size).collect(),
    };
    expect_ranks(&reports, &worker_url, ranks.len());
    let feeds = rank_endpoints(&host, port_base, &ranks)
        .into_iter()
        .map(|(rank, endpoint)| read_rank(&reports, &worker_url, rank, endpoint));
    futures::future::join_all(feeds).await;
}

/// `(rank, tcp endpoint)` per rank; a rank whose port would pass 65535 is left out.
fn rank_endpoints(host: &str, port_base: u16, ranks: &[u32]) -> Vec<(u32, String)> {
    ranks
        .iter()
        .filter_map(|&rank| {
            let port = u16::try_from(rank)
                .ok()
                .and_then(|rank| port_base.checked_add(rank))?;
            Some((rank, format!("tcp://{host}:{port}")))
        })
        .collect()
}

async fn read_rank(reports: &ReportTable, worker_url: &str, rank: u32, endpoint: String) {
    loop {
        let mut socket = SubSocket::new();
        if let Err(e) = socket.connect(&endpoint).await {
            debug!("prefill_queue_time: connect {endpoint} for {worker_url} failed: {e}");
            sleep(RETRY_DELAY).await;
            continue;
        }
        // The load socket carries only load messages, so no topic filter.
        if let Err(e) = socket.subscribe("").await {
            debug!("prefill_queue_time: subscribe {endpoint} for {worker_url} failed: {e}");
            sleep(RETRY_DELAY).await;
            continue;
        }
        loop {
            match socket.recv().await {
                Ok(message) => {
                    if let Some(load) = decode_message(&message) {
                        record_rank(reports, worker_url, rank, load, Instant::now());
                    }
                }
                Err(e) => {
                    warn!("prefill_queue_time: receive from {endpoint} failed: {e}");
                    break;
                }
            }
        }
        sleep(RETRY_DELAY).await;
    }
}

/// `[topic, big-endian i64 seq, msgpack LoadStat]`, as `load_publisher.py` frames it.
fn decode_message(message: &ZmqMessage) -> Option<RankLoad> {
    if message.len() != 3 {
        return None;
    }
    rmp_serde::from_slice(message.get(2)?.as_ref()).ok()
}

/// Decodes the engine's `LoadStat`, a msgspec tagged array:
/// `["LoadStat", num_running_reqs, num_waiting_reqs, num_tokens,
/// max_total_num_tokens, attn_dp_rank, num_waiting_uncached_tokens,
/// num_total_tokens, max_running_requests, total_prefill_uncached_tokens,
/// total_prefill_busy_us, ...]`. A shorter message is rejected rather than
/// read as zeros, which would make the worker look idle.
impl<'de> Deserialize<'de> for RankLoad {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct LoadStatVisitor;

        impl<'de> Visitor<'de> for LoadStatVisitor {
            type Value = RankLoad;

            fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
                f.write_str("a msgpack array [\"LoadStat\", ...]")
            }

            fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<RankLoad, A::Error> {
                let tag: String = required(&mut seq, "tag")?;
                if tag != "LoadStat" {
                    return Err(de::Error::custom(format!("unexpected tag {tag:?}")));
                }
                let num_running_reqs = required(&mut seq, "num_running_reqs")?;
                let num_waiting_reqs = required(&mut seq, "num_waiting_reqs")?;
                required::<_, IgnoredAny>(&mut seq, "num_tokens")?;
                required::<_, IgnoredAny>(&mut seq, "max_total_num_tokens")?;
                required::<_, IgnoredAny>(&mut seq, "attn_dp_rank")?;
                let num_waiting_uncached_tokens =
                    required(&mut seq, "num_waiting_uncached_tokens")?;
                required::<_, IgnoredAny>(&mut seq, "num_total_tokens")?;
                required::<_, IgnoredAny>(&mut seq, "max_running_requests")?;
                let total_prefill_uncached_tokens =
                    required(&mut seq, "total_prefill_uncached_tokens")?;
                let total_prefill_busy_us = required(&mut seq, "total_prefill_busy_us")?;
                while seq.next_element::<IgnoredAny>()?.is_some() {}
                Ok(RankLoad {
                    num_waiting_uncached_tokens,
                    num_waiting_reqs,
                    num_running_reqs,
                    total_prefill_uncached_tokens,
                    total_prefill_busy_us,
                })
            }
        }

        fn required<'de, A: SeqAccess<'de>, T: Deserialize<'de>>(
            seq: &mut A,
            field: &'static str,
        ) -> Result<T, A::Error> {
            seq.next_element()?
                .ok_or_else(|| de::Error::missing_field(field))
        }

        deserializer.deserialize_seq(LoadStatVisitor)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use bytes::Bytes;
    use zeromq::{Endpoint, PubSocket, SocketSend};

    use super::*;

    /// msgspec's encoding of the engine's `LoadStat(num_running_reqs=3,
    /// num_waiting_reqs=2, num_tokens=1000, max_total_num_tokens=500000,
    /// attn_dp_rank=1, num_waiting_uncached_tokens=4096, num_total_tokens=5096,
    /// max_running_requests=64, total_prefill_uncached_tokens=123456789,
    /// total_prefill_busy_us=987654321)`.
    const ENGINE_LOAD_STAT: &str =
        "9ba84c6f6164537461740302cd03e8ce0007a12001cd1000cd13e840ce075bcd15ce3ade68b1";

    fn engine_load_stat() -> Vec<u8> {
        (0..ENGINE_LOAD_STAT.len())
            .step_by(2)
            .map(|i| u8::from_str_radix(&ENGINE_LOAD_STAT[i..i + 2], 16).unwrap())
            .collect()
    }

    fn multipart(payload: Vec<u8>) -> ZmqMessage {
        let mut message = ZmqMessage::from(Bytes::from_static(b"load"));
        message.push_back(Bytes::copy_from_slice(&7i64.to_be_bytes()));
        message.push_back(Bytes::from(payload));
        message
    }

    #[test]
    fn test_decodes_the_engine_load_stat_bytes() {
        let load = decode_message(&multipart(engine_load_stat())).unwrap();
        assert_eq!(
            load,
            RankLoad {
                num_waiting_uncached_tokens: 4096,
                num_waiting_reqs: 2,
                num_running_reqs: 3,
                total_prefill_uncached_tokens: 123_456_789,
                total_prefill_busy_us: 987_654_321,
            }
        );
    }

    #[test]
    fn test_decoder_accepts_appended_fields_and_rejects_short_or_foreign_messages() {
        // Array header 0x9b (11 items) -> 0x9c (12 items) plus one trailing value.
        let mut extended = engine_load_stat();
        extended[0] = 0x9c;
        extended.push(0x05);
        assert!(decode_message(&multipart(extended)).is_some());

        let short =
            rmp_serde::to_vec(&("LoadStat", 3u64, 2u64, 1000u64, 500_000u64, 1u64)).unwrap();
        assert!(decode_message(&multipart(short)).is_none());

        let mut foreign = engine_load_stat();
        foreign[2..10].copy_from_slice(b"KvEvents");
        assert!(decode_message(&multipart(foreign)).is_none());

        let mut two_frames = ZmqMessage::from(Bytes::from_static(b"load"));
        two_frames.push_back(Bytes::from(engine_load_stat()));
        assert!(decode_message(&two_frames).is_none());
    }

    #[test]
    fn test_rank_ports_are_base_plus_rank() {
        assert_eq!(
            rank_endpoints("10.0.0.5", 5559, &[0, 1, 2]),
            vec![
                (0, "tcp://10.0.0.5:5559".to_string()),
                (1, "tcp://10.0.0.5:5560".to_string()),
                (2, "tcp://10.0.0.5:5561".to_string()),
            ]
        );
        assert_eq!(
            rank_endpoints("10.0.0.5", 5559, &[3]),
            vec![(3, "tcp://10.0.0.5:5562".to_string())]
        );
        assert_eq!(
            rank_endpoints("h", 65535, &[0, 1]),
            vec![(0, "tcp://h:65535".to_string())]
        );
    }

    #[tokio::test]
    async fn test_receives_load_from_a_pub_socket() {
        let mut publisher = PubSocket::new();
        let port = match publisher.bind("tcp://127.0.0.1:0").await.unwrap() {
            Endpoint::Tcp(_, port) => port,
            other => panic!("unexpected endpoint {other:?}"),
        };
        let reports = ReportTable::default();
        expect_ranks(&reports, "http://p", 1);
        let reader = tokio::spawn({
            let reports = Arc::clone(&reports);
            async move { read_rank(&reports, "http://p", 0, format!("tcp://127.0.0.1:{port}")).await }
        });

        // PUB drops messages sent before the subscription lands, so keep sending.
        let received = tokio::time::timeout(Duration::from_secs(10), async {
            loop {
                publisher.send(multipart(engine_load_stat())).await.unwrap();
                sleep(Duration::from_millis(50)).await;
                if reports.read().unwrap()["http://p"]
                    .fresh_load(Instant::now())
                    .is_some()
                {
                    break;
                }
            }
        })
        .await;
        reader.abort();
        assert!(received.is_ok(), "no load message arrived over ZMQ");
        let load = reports.read().unwrap()["http://p"]
            .fresh_load(Instant::now())
            .unwrap();
        assert_eq!(load.num_waiting_uncached_tokens, 4096);
    }
}
