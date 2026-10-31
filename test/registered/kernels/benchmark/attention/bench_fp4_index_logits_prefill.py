import torch

from sglang.kernels.jit.benchmark import marker
from sglang.kernels.ops.attention.dsv4.fp4_indexer import fp4_index_logits_prefill
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kernels.deepseek_v4.v41_prefill_indexer import (
    make_prefill_indexer_case,
    prefill_kernel_args,
    prefill_scores,
)

register_cuda_ci(
    est_time=60, stage="base-b-kernel-benchmark", runner_config="1-gpu-large"
)


@marker.parametrize("heads", [32, 64], [64])
@marker.parametrize("ratio", [1, 2], [1])
@marker.parametrize(
    "context,rows",
    [(8192, 1024), (32768, 1024), (70000, 256), (70000, 1024), (70000, 2048)],
    [(70000, 1024)],
)
@marker.benchmark("impl", ["torch", "fused", "triton", "fp8_keys_major"])
def benchmark(heads, ratio, context, rows, impl):
    """Scores of one request's newest ``rows`` query rows after ``context - rows``
    cached tokens, without the top-k that follows. torch and fused run the whole
    prefill indexer scoring path; triton and fp8_keys_major time the kernel alone."""
    if torch.cuda.get_device_capability()[0] != 9:
        marker.skip("Hopper prefill indexer")
    from sglang.kernels.ops.attention.dsv4.fp4_indexer_keys_major import (
        fp4_index_logits_prefill_keys_major,
    )

    inputs, pool, req_to_token = make_prefill_indexer_case(
        heads=heads, ratio=ratio, requests=[(context - rows, rows)], topk=1024
    )
    common = dict(inputs=inputs, token_to_kv_pool=pool, req_to_token=req_to_token)
    if impl in ("torch", "fused"):
        fn, kwargs = prefill_scores, dict(**common, fused=impl == "fused")
    else:
        fn = (
            fp4_index_logits_prefill
            if impl == "triton"
            else fp4_index_logits_prefill_keys_major
        )
        kwargs = prefill_kernel_args(**common, request=0)
    return marker.do_bench(
        fn,
        input_kwargs=kwargs,
        # The indexer reads the visible length back to the host per request.
        use_cuda_graph=False,
        disable_log_bandwidth=True,
    )


if __name__ == "__main__":
    benchmark.run()
