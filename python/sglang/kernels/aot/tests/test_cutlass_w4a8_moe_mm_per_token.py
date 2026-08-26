"""INF-405: cutlass_w4a8_moe_mm의 per-token activation-scale 경로 회귀 재현 테스트.

per-token 벡터를 GEMM epilogue(Sm90RowBroadcast, bool N-stride)로 넘기면 벡터가
사실상 스칼라로 붕괴해 행별 스케일이 오적용된다. PR#31의 자체 검증(상수 벡터
bit-identical 대조)은 상수 벡터가 스칼라 붕괴와 구별 불가능해 이 결함을 잡지
못했다. 이 테스트는 행별 스케일 분산이 큰 입력으로 그 경로를 직접 검증한다.

epilogue가 고쳐질 때까지 xfail(strict)로 둔다 — 파이썬 쪽은 이 경로를 더 이상
쓰지 않고(scalar=1 GEMM + triton fold), 이 테스트가 통과하기 시작하면 epilogue가
고쳐졌다는 신호이므로 strict 실패로 알아차린다.
"""
import pytest
import torch
from sgl_kernel import cutlass_w4a8_moe_mm
from utils import is_hopper


def _pack_int4(v):
    t = v.to(torch.int8)
    return ((t[..., 1::2] << 4) | (t[..., 0::2] & 0x0F)).to(torch.int8)


def _pack_interleave(E, ref_w, ref_s):
    n, k = ref_w.shape[1], ref_w.shape[2]
    w_q = _pack_int4(ref_w.cpu()).cuda().view(E, n, k // 2).view(torch.int8).contiguous()
    al = 4 if k % 512 == 0 else 1
    s = (
        ref_s.reshape(E, n, k // 128 // al, al)
        .permute(0, 2, 1, 3)
        .reshape(E, k // 128 // al, n * al)
    )
    return w_q, s.contiguous()


@pytest.mark.skipif(not is_hopper(), reason="sm90 only")
@pytest.mark.xfail(
    strict=True,
    reason="INF-405: per-token epilogue collapses the scale vector to a scalar",
)
@pytest.mark.parametrize("rows_per_e", [[16], [3, 7, 1, 13]])
def test_per_token_scale_with_row_variance(rows_per_e):
    torch.manual_seed(0)
    dev = "cuda"
    E, k, n = len(rows_per_e), 512, 1024
    m = sum(rows_per_e)

    a = torch.randn(m, k, dtype=torch.bfloat16, device=dev)
    a = a * torch.logspace(-2, 2, m, device=dev)[:, None]
    ref_w = torch.randint(-8, 8, (E, n, k), dtype=torch.int8, device=dev)
    ref_s = torch.randn(E, n, k // 128, dtype=torch.bfloat16, device=dev) * 0.005
    w, w_s = _pack_interleave(E, ref_w, ref_s)

    a_scale = (a.float().abs().amax(dim=1) / 448.0).clamp_min(1e-10)
    a_q = (a.float() / a_scale[:, None]).to(torch.float8_e4m3fn)

    offs = [0]
    for r in rows_per_e:
        offs.append(offs[-1] + r)
    expert_offsets = torch.tensor(offs, dtype=torch.int32, device=dev)
    problem_sizes = torch.tensor(
        [[n, r, k] for r in rows_per_e], dtype=torch.int32, device=dev
    )
    a_strides = torch.full((E, 3), k, device=dev, dtype=torch.int64)
    c_strides = torch.full((E, 3), n, device=dev, dtype=torch.int64)

    c = torch.empty((m, n), dtype=torch.bfloat16, device=dev)
    cutlass_w4a8_moe_mm(
        c, a_q, w, a_scale.float().contiguous(), w_s,
        expert_offsets[:-1], problem_sizes,
        a_strides, a_strides, c_strides, c_strides, 128, 8,
    )

    c_ref = torch.empty((m, n), dtype=torch.float32, device=dev)
    for e in range(E):
        lo, hi = offs[e], offs[e + 1]
        wd = ref_w[e].to(torch.float32) * ref_s[e].repeat_interleave(128, dim=1).to(
            torch.float32
        )
        c_ref[lo:hi] = (a_q[lo:hi].to(torch.float32) @ wd.t()) * a_scale[lo:hi, None]

    torch.testing.assert_close(c.float(), c_ref, rtol=2e-2, atol=0.1)
