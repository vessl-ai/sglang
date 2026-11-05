import unittest

import torch

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-large")


def _encode(x):
    import triton
    import triton.language as tl

    from sglang.kernels.ops.attention.dsa.fp8_e4m3_sw import f32_to_e4m3fn_u8

    @triton.jit
    def kernel(x_ptr, y_ptr, n, BLOCK: tl.constexpr):
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        v = tl.load(x_ptr + offs, mask=mask)
        tl.store(y_ptr + offs, f32_to_e4m3fn_u8(v), mask=mask)

    y = torch.empty(x.numel(), dtype=torch.uint8, device=x.device)
    kernel[(triton.cdiv(x.numel(), 1024),)](x, y, x.numel(), BLOCK=1024)
    return y


@unittest.skipUnless(torch.cuda.is_available(), "needs a CUDA GPU")
class TestSoftwareE4m3Encoder(CustomTestCase):
    def setUp(self):
        try:
            import triton  # noqa: F401
        except ImportError:
            self.skipTest("triton is not installed")

    def _check(self, x):
        expected = x.to(torch.float8_e4m3fn).view(torch.uint8)
        self.assertTrue(torch.equal(_encode(x), expected))

    def test_random_values_match_torch(self):
        gen = torch.Generator(device="cuda").manual_seed(0)
        # Normals up to the saturation point and e4m3 subnormals, both signs.
        self._check(
            torch.cat(
                [
                    (torch.rand(1 << 16, device="cuda", generator=gen) * 2 - 1) * 448,
                    (torch.rand(1 << 16, device="cuda", generator=gen) * 2 - 1) * 2**-5,
                ]
            )
        )

    def test_saturation_and_nan_bytes(self):
        # torch maps out-of-range values to NaN; the encoder saturates to +-448 instead.
        x = torch.tensor([1000.0, -1000.0, 464.0, float("nan")], device="cuda")
        self.assertEqual(_encode(x).tolist(), [0x7E, 0xFE, 0x7E, 0x7F])

    def test_ties_to_even_and_exponent_carry(self):
        # 1.0625 and 1.1875 are exact midpoints (round to the even mantissa); 15.5 is
        # the midpoint below 16, so rounding carries into the exponent.
        ties = [1.0625, 1.1875, 15.5, 2**-9, 2**-10, 448.0, 0.0]
        values = ties + [-v for v in ties]
        self._check(torch.tensor(values, device="cuda"))


if __name__ == "__main__":
    unittest.main()
