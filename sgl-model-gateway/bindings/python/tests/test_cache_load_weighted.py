import argparse

import pytest

from sglang_router.router import Router as RouterWrapper
from sglang_router.router import policy_from_str
from sglang_router.router_args import RouterArgs
from sglang_router.sglang_router_rs import PolicyType, Router


def test_weighted_prefill_cli_and_binding():
    parser = argparse.ArgumentParser()
    RouterArgs.add_cli_args(parser)
    parsed = parser.parse_args(
        [
            "--pd-disaggregation",
            "--prefill-policy",
            "cache_load_weighted",
            "--prefill-cache-weight",
            "0.4",
        ]
    )
    args = RouterArgs.from_cli_args(parsed)
    assert args.prefill_cache_weight == 0.4
    assert policy_from_str(args.prefill_policy) == PolicyType.CacheLoadWeighted
    router = Router(
        worker_urls=[],
        pd_disaggregation=True,
        prefill_policy=PolicyType.CacheLoadWeighted,
        prefill_cache_weight=args.prefill_cache_weight,
    )
    assert router.prefill_cache_weight == 0.4
    wrapped = RouterWrapper.from_args(args)
    assert wrapped._router.prefill_cache_weight == 0.4


@pytest.mark.parametrize("weight", [None, float("nan"), float("inf"), -0.1, 1.1])
def test_weighted_prefill_requires_a_valid_weight(weight):
    with pytest.raises(ValueError):
        RouterArgs(
            pd_disaggregation=True,
            prefill_policy="cache_load_weighted",
            prefill_cache_weight=weight,
        )._validate_router_args()


def test_weighted_policy_rejected_for_decode_cli():
    parser = argparse.ArgumentParser()
    RouterArgs.add_cli_args(parser)
    with pytest.raises(SystemExit):
        parser.parse_args(["--decode-policy", "cache_load_weighted"])
