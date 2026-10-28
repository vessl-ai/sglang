import torch

from sglang.kernels.jit.benchmark import marker
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kernels.deepseek_v4.v41_prefill_indexer import (
    make_prefill_indexer_case,
    prefill_scores,
)

register_cuda_ci(
    est_time=30, stage="base-b-kernel-benchmark", runner_config="1-gpu-large"
)


@marker.parametrize("heads", [32, 64], [64])
@marker.parametrize("ratio", [1, 2], [1])
@marker.parametrize(
    "context,rows",
    [(8192, 1024), (32768, 1024), (70000, 256), (70000, 1024), (70000, 2048)],
    [(70000, 1024)],
)
@marker.benchmark("impl", ["torch", "fused"])
def benchmark(heads, ratio, context, rows, impl):
    """Scores of one request's newest ``rows`` query rows after ``context - rows``
    cached tokens, without the top-k that follows."""
    if torch.cuda.get_device_capability()[0] != 9:
        marker.skip("Hopper prefill indexer")
    inputs, pool, req_to_token = make_prefill_indexer_case(
        heads=heads, ratio=ratio, requests=[(context - rows, rows)], topk=1024
    )
    return marker.do_bench(
        prefill_scores,
        input_kwargs=dict(
            inputs=inputs,
            token_to_kv_pool=pool,
            req_to_token=req_to_token,
            fused=impl == "fused",
        ),
        # The indexer reads the visible length back to the host per request.
        use_cuda_graph=False,
        disable_log_bandwidth=True,
    )


if __name__ == "__main__":
    benchmark.run()
