import argparse

import pytest
from sglang_router.router import Router as RouterWrapper
from sglang_router.router import policy_from_str
from sglang_router.router_args import RouterArgs
from sglang_router.sglang_router_rs import PolicyType, Router


def _parse(argv):
    parser = argparse.ArgumentParser()
    RouterArgs.add_cli_args(parser)
    return RouterArgs.from_cli_args(parser.parse_args(argv))


def test_weighted_prefill_cli_and_binding():
    args = _parse(
        [
            "--pd-disaggregation",
            "--prefill-policy",
            "cache_load_weighted",
            "--prefill-cache-weight",
            "0.4",
            "--prefill-load-weight",
            "2.5",
        ]
    )
    assert (args.prefill_cache_weight, args.prefill_load_weight) == (0.4, 2.5)
    assert policy_from_str(args.prefill_policy) == PolicyType.CacheLoadWeighted
    router = Router(
        worker_urls=[],
        pd_disaggregation=True,
        prefill_policy=PolicyType.CacheLoadWeighted,
        prefill_cache_weight=args.prefill_cache_weight,
        prefill_load_weight=args.prefill_load_weight,
    )
    assert (router.prefill_cache_weight, router.prefill_load_weight) == (0.4, 2.5)
    wrapped = RouterWrapper.from_args(args)
    assert wrapped._router.prefill_load_weight == 2.5


def test_weighted_prefill_weights_default_to_one():
    args = _parse(["--pd-disaggregation", "--prefill-policy", "cache_load_weighted"])
    assert (args.prefill_cache_weight, args.prefill_load_weight) == (1.0, 1.0)
    router = Router(
        worker_urls=[],
        pd_disaggregation=True,
        prefill_policy=PolicyType.CacheLoadWeighted,
    )
    assert (router.prefill_cache_weight, router.prefill_load_weight) == (1.0, 1.0)


@pytest.mark.parametrize(
    "cache_weight,load_weight",
    [
        (float("nan"), 1.0),
        (1.0, float("inf")),
        (-0.1, 1.0),
        (1.0, -0.1),
        (0.0, 0.0),
    ],
)
def test_weighted_prefill_rejects_invalid_weights(cache_weight, load_weight):
    with pytest.raises(ValueError):
        RouterArgs(
            pd_disaggregation=True,
            prefill_policy="cache_load_weighted",
            prefill_cache_weight=cache_weight,
            prefill_load_weight=load_weight,
        )._validate_router_args()


def test_weighted_policy_rejected_for_decode_cli():
    parser = argparse.ArgumentParser()
    RouterArgs.add_cli_args(parser)
    with pytest.raises(SystemExit):
        parser.parse_args(["--decode-policy", "cache_load_weighted"])
