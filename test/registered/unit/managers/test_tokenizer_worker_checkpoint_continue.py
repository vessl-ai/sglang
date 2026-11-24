import asyncio
import unittest
from types import SimpleNamespace

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.io_struct import ContinueGenerationReqInput
from sglang.srt.managers.multi_tokenizer_mixin import TokenizerWorker

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _worker(refusal=None):
    """A tokenizer worker whose checkpoint check raises `refusal`, recording
    what it sends to the router."""
    sent = []
    worker = SimpleNamespace(_pause_continue_future=None)

    async def refuse():
        if refusal is not None:
            raise refusal

    def dispatch(obj):
        sent.append(obj)
        # The router's broadcast resolves the originator's future.
        worker._pause_continue_future.set_result(True)

    worker._refuse_continue_during_checkpoint = refuse
    worker._dispatch_to_scheduler = dispatch
    return worker, sent


class TestTokenizerWorkerContinueDuringCheckpoint(CustomTestCase):
    def test_continue_is_refused_mid_checkpoint(self):
        """With --tokenizer-worker-num > 1 the worker's own continue_generation
        replaces TokenizerManager's, so it must refuse too, or a continue sent
        to another worker would schedule requests without a KV cache."""
        worker, sent = _worker(RuntimeError("during a process checkpoint"))
        with self.assertRaises(RuntimeError):
            asyncio.run(
                TokenizerWorker.continue_generation(
                    worker, ContinueGenerationReqInput()
                )
            )
        self.assertEqual(sent, [])

    def test_continue_outside_a_checkpoint_goes_to_the_router(self):
        worker, sent = _worker()
        asyncio.run(
            TokenizerWorker.continue_generation(worker, ContinueGenerationReqInput())
        )
        self.assertEqual(len(sent), 1)


if __name__ == "__main__":
    unittest.main()
