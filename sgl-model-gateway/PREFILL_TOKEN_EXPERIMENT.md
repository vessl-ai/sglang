# INF-556: prefill token-load experiment (B)

Base: `vessl-ai/sglang` commit `897472fe3baeda9121960657d0fe26b2c851f30d`,
the same base as PR #70. Branch: `feat/pd-prefill-token-load`. PR #70 is
not included. Decode selection and request lifetime accounting are unchanged.

Enable with `--pd-disaggregation --prefill-policy prefill_tokens` and keep
the experiment's existing decode policy and other arguments.

## Decision before implementation

Use `loads[].num_total_tokens` from each engine's `/v1/loads?include=core`.
In this revision, `SchedulerLoadInquirer.get_loads()` computes this as
non-evictable KV tokens plus the input lengths of requests in the waiting
and prefill bootstrap queues. It includes KV retained during transfer;
it is not a count of remaining uncached compute tokens or tokens per minute.
Completed, evictable cached KV is excluded from the occupied-token component.
The waiting component does not deduct prefix-cache hits.

`num_waiting_uncached_tokens` sums the ordinary waiting queue and the remainder
of the chunked request. It omits bootstrap requests and does not account for all
running batches. Waiting cache matches are populated during scheduling, and may
not be available yet. It cannot represent total remaining prefill compute across
all stages without further engine instrumentation, so B uses total tokens.
No engine modification is needed. The endpoint returns a
`loads` array, not `aggregate.total_tokens` (the existing power-of-two poller
expects the latter). B has its own parser and does not change that policy.

The engine publishes snapshots on prefill batch execution and idle iterations;
the default interval of 15 applies to decode iterations, not wall-clock seconds.
The router polls prefill workers concurrently every 250 ms with a 1 s request
timeout. It validates engine timestamps, rank coverage and nonnegative token
counts. Snapshots older than 5 s are unavailable. Invalid or failed responses
remove that worker's sample; no request-count or zero-load fallback is used.
Workers without a fresh sample are excluded; no usable sample means no worker
can be selected (the PD router returns service unavailable).

Among available workers, choose the lowest token count with random tie-breaking.
For a rank-addressed worker use its own rank; otherwise sum all expected ranks.
The expected rank count comes from discovered worker metadata (default 1).
Routing performs no network I/O. The poller shares the existing monitor lifecycle.

## Limits and comparison

Polling can miss arrivals between updates and send a burst to the currently
lowest worker; random tie-breaking only spreads equal minima. Cache locality is
not a selection input in this policy. These are experiment tradeoffs, not claims
of improved performance. Engine/router clocks must be synchronized (up to 1 s
future skew is tolerated). A long prefill that prevents snapshot publication for
more than 5 s makes its sample unavailable.

Compare A (#70) and B on Betelgeuse Solar Mini 4, with 4 prefill and 4 decode
workers, identical request inputs, arrival pattern, concurrency, cache state and
warm-up. Measure all four prefill workers' running/waiting requests and wait times,
TTFT median/p95/p99, completed requests and token throughput. Local tests and
builds do not establish the experiment outcome. Deployment and load generation
are separate operations.

## Local validation

Using Rust 1.90 and the repository's default features:

- `cargo +1.90 test --locked --lib policies::`: 133 passed, including 7 new
  parser, selection, HTTP collection, monitor-lifecycle and configuration tests.
- `cargo +1.90 test --locked --lib config::`: 53 passed.
- `cargo +1.90 build --locked --bin sgl-model-gateway`: passed (native macOS dev build).
- `cargo +1.90 build --locked -p sgl-model-gateway-python`: passed.
- Loaded the built Python extension and verified `policy_from_str("prefill_tokens")`.
- Ran the binary and checked that `prefill_tokens` appears only in prefill choices.
- Targeted rustfmt check and `git diff --check`: passed.

The 7 new tests also passed with `--no-default-features` on Rust 1.90.
No image build, deployment or A/B traffic experiment has been performed.
