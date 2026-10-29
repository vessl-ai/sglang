// INF-681 k-qcp: SM90 CUDA version of mhc._hc_mix_stats_bf16x3_kernel, the first half of
// hc_mix_stats_sinkhorn_bf16x3. RESEARCH KERNEL, not wired into the model: bit-identical but slower
// than the Triton kernel (H200, 16,302 rows: 0.29 vs 0.22 ms; it reloads the weight tiles per 64 rows
// where Triton's CTA covers 128). Same partial outputs bit for bit as the Triton kernel's prefill
// configuration (BLOCK_M 128, BLOCK_K 64, 16 slices, 4 warps), at any row count; Triton's own
// BLOCK_M 64 configuration (used below 4096 rows) differs from its BLOCK_M 128 one in ~0.02-0.2% of
// the partials (|diff| <= 4e-6):
//
//   part_mix[s, m, :] = (hi + mid) + lo   with  hi/mid/lo = x[m, slice s] @ w_{hi,mid,lo}[:, slice s]^T
//                                          accumulated per 64-column block by wgmma.m64nNk16 (k16 in order)
//   part_sq[s, m]     = sum over the slice's 64-column blocks, in order, of Triton's per-block
//                       ((fma(e0,e0,e1^2) + fma(e2,e2,e3^2)) + (fma(e4,e4,e5^2) + fma(e6,e6,e7^2)))
//                       per 8-column lane, folded across the 8 lanes as xor 4, 2, 1.
//
// (The Triton kernel feeds A from shared memory and uses n32 with zero padding; this one feeds A from
// registers via ldmatrix and uses n24: the instruction and each element's sum are the same.)
// CTA = one warpgroup, 64 rows x one 1280-column slice; a 4-stage cp.async ring of the x tile
// (64 x 64 bf16) and the three 24 x 64 weight tiles, all 128B-swizzled.
#pragma once
#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>

#include <tvm/ffi/container/tensor.h>

#include <cstdint>

namespace sglang {
namespace mhc_stats {

constexpr int kK = 20480;
constexpr int kSlices = 16;
constexpr int kSlice = kK / kSlices;  // 1280
constexpr int kRows = 64;
constexpr int kBlk = 64;
constexpr int kBlocks = kSlice / kBlk;  // 20
constexpr int kThreads = 128;
constexpr int kStages = 4;
constexpr int kXTile = kRows * 128;     // 8192 B
constexpr int kWTile = 24 * 128;        // 3072 B
constexpr int kStage = kXTile + 3 * kWTile;  // 17408 B = 17 x 1024
constexpr int kSmem = 1024 + kStages * kStage;

struct Params {
  const bf16_t *x, *w_hi, *w_mid, *w_lo;
  float *part_mix, *part_sq;
  int m;
};

SGL_DEVICE uint32_t smem_u32(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}
SGL_DEVICE void cp_async16(uint32_t dst, const void* src, bool valid) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" ::"r"(dst), "l"(src), "r"(valid ? 16 : 0) : "memory");
}
SGL_DEVICE void cp_async_commit() {
  asm volatile("cp.async.commit_group;" ::: "memory");
}
template <int N>
SGL_DEVICE void cp_async_wait() {
  asm volatile("cp.async.wait_group %0;" ::"n"(N) : "memory");
}
SGL_DEVICE void fence_proxy_async() {
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
}
SGL_DEVICE uint64_t make_desc(uint32_t smem_addr) {  // K-major, 128B swizzle, 8-row groups 1024 B apart
  uint64_t d = static_cast<uint64_t>((smem_addr & 0x3FFFF) >> 4);
  d |= static_cast<uint64_t>(1) << 16;
  d |= static_cast<uint64_t>(1024 >> 4) << 32;
  d |= static_cast<uint64_t>(1) << 62;
  return d;
}
SGL_DEVICE int swz(int row, int chunk) {
  return chunk ^ (row & 7);
}
SGL_DEVICE void fence_acc(float (&d)[12]) {
#pragma unroll
  for (int i = 0; i < 12; ++i)
    asm volatile("" : "+f"(d[i])::"memory");
}
SGL_DEVICE void fence_a(uint32_t (&a)[4]) {
#pragma unroll
  for (int i = 0; i < 4; ++i)
    asm volatile("" : "+r"(a[i])::"memory");
}
SGL_DEVICE void wgmma_bf16(float (&d)[12], const uint32_t (&a)[4], uint64_t desc) {
  asm volatile(
      "{\n.reg .pred p;\nsetp.ne.b32 p, %17, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n24k16.f32.bf16.bf16 "
      "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11}, {%12, %13, %14, %15}, %16, p, 1, 1, 0;\n}\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]),
        "+f"(d[9]), "+f"(d[10]), "+f"(d[11])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc), "r"(1));
}

__global__ __launch_bounds__(kThreads) void mhc_stats_bf16x3_kernel(const __grid_constant__ Params p) {
  extern __shared__ __align__(1024) uint8_t smem_raw[];
  const uint32_t raw = smem_u32(smem_raw);
  uint8_t* base = smem_raw + (((raw + 1023) & ~1023u) - raw);

  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int gid = lane >> 2, tig = lane & 3;
  const int row0 = blockIdx.x * kRows;
  const int slice = blockIdx.y;
  const int k0 = slice * kSlice;

  auto load = [&](int blk) {
    uint8_t* st = base + (blk % kStages) * kStage;
    const int kb = k0 + blk * kBlk;
    for (int i = tid; i < kRows * 8; i += kThreads) {  // x: 64 rows x 8 chunks
      const int chunk = i & 7, r = i >> 3, row = row0 + r;
      const bool valid = row < p.m;
      cp_async16(
          smem_u32(st + r * 128 + swz(r, chunk) * 16),
          p.x + static_cast<int64_t>(valid ? row : 0) * kK + kb + chunk * 8,
          valid);
    }
    for (int i = tid; i < 3 * 24 * 8; i += kThreads) {  // w: 3 parts x 24 rows x 8 chunks
      const int chunk = i & 7, n = (i >> 3) % 24, part = i / (24 * 8);
      const bf16_t* w = part == 0 ? p.w_hi : (part == 1 ? p.w_mid : p.w_lo);
      cp_async16(
          smem_u32(st + kXTile + part * kWTile + n * 128 + swz(n, chunk) * 16),
          w + static_cast<int64_t>(n) * kK + kb + chunk * 8,
          true);
    }
  };

  // ldmatrix.x4 address of this lane in a 16 x 16 A tile of warp `warp` (rows 16 warp .. +15)
  const int lm_row = 16 * warp + ((lane >> 3) & 1) * 8 + (lane & 7);
  const int lm_chunk = lane >> 4;
  // sq: Triton's #blocked layout, lane t = tid % 8 holds 8 consecutive columns of rows tid / 8 + 16 j
  const int st_t = tid & 7, st_r = tid >> 3;

  float acc[3][12];
  float sq[4];
#pragma unroll
  for (int t = 0; t < 3; ++t)
#pragma unroll
    for (int e = 0; e < 12; ++e) acc[t][e] = 0.f;
#pragma unroll
  for (int j = 0; j < 4; ++j) sq[j] = 0.f;

#pragma unroll
  for (int b = 0; b < kStages - 1; ++b) {
    load(b);
    cp_async_commit();
  }
  for (int blk = 0; blk < kBlocks; ++blk) {
    cp_async_wait<kStages - 2>();
    __syncthreads();  // block blk landed; block blk - 1's stage is free (every wgmma waited)
    if (blk + kStages - 1 < kBlocks) load(blk + kStages - 1);
    cp_async_commit();
    fence_proxy_async();
    const uint8_t* st = base + (blk % kStages) * kStage;

    uint32_t a[4][4];
#pragma unroll
    for (int c = 0; c < 4; ++c) {
      const uint32_t addr = smem_u32(st + lm_row * 128 + swz(lm_row, 2 * c + lm_chunk) * 16);
      asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
                   : "=r"(a[c][0]), "=r"(a[c][1]), "=r"(a[c][2]), "=r"(a[c][3])
                   : "r"(addr));
    }
#pragma unroll
    for (int c = 0; c < 4; ++c) fence_a(a[c]);
    asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
#pragma unroll
    for (int part = 0; part < 3; ++part) {
      const uint32_t wt = smem_u32(st + kXTile + part * kWTile);
#pragma unroll
      for (int c = 0; c < 4; ++c) wgmma_bf16(acc[part], a[c], make_desc(wt + c * 32));
    }
    asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");

    // sum of squares beside the wgmmas
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const int r = st_r + 16 * j;
      const uint4 raw4 = *reinterpret_cast<const uint4*>(st + r * 128 + swz(r, st_t) * 16);
      const __nv_bfloat162* h2 = reinterpret_cast<const __nv_bfloat162*>(&raw4);
      float v[8];
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        const float2 f = __bfloat1622float2(h2[e]);
        v[2 * e] = f.x;
        v[2 * e + 1] = f.y;
      }
      float s8 = __fadd_rn(
          __fadd_rn(__fmaf_rn(v[0], v[0], __fmul_rn(v[1], v[1])), __fmaf_rn(v[2], v[2], __fmul_rn(v[3], v[3]))),
          __fadd_rn(__fmaf_rn(v[4], v[4], __fmul_rn(v[5], v[5])), __fmaf_rn(v[6], v[6], __fmul_rn(v[7], v[7]))));
      s8 = __fadd_rn(s8, __shfl_xor_sync(0xffffffff, s8, 4));
      s8 = __fadd_rn(s8, __shfl_xor_sync(0xffffffff, s8, 2));
      s8 = __fadd_rn(s8, __shfl_xor_sync(0xffffffff, s8, 1));
      sq[j] = __fadd_rn(sq[j], s8);
    }
    asm volatile("wgmma.wait_group.sync.aligned 0;" ::: "memory");
#pragma unroll
    for (int t = 0; t < 3; ++t) fence_acc(acc[t]);
  }

#pragma unroll
  for (int hlf = 0; hlf < 2; ++hlf) {
    const int row = row0 + 16 * warp + gid + 8 * hlf;
    if (row < p.m) {
      float* dst = p.part_mix + (static_cast<int64_t>(slice) * p.m + row) * 24;
#pragma unroll
      for (int nb = 0; nb < 3; ++nb) {
        const int n = 8 * nb + 2 * tig, e = 4 * nb + 2 * hlf;
        dst[n] = __fadd_rn(__fadd_rn(acc[0][e], acc[1][e]), acc[2][e]);
        dst[n + 1] = __fadd_rn(__fadd_rn(acc[0][e + 1], acc[1][e + 1]), acc[2][e + 1]);
      }
    }
  }
  if (st_t == 0) {
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const int row = row0 + st_r + 16 * j;
      if (row < p.m) p.part_sq[static_cast<int64_t>(slice) * p.m + row] = sq[j];
    }
  }
}

struct MhcStatsBf16x3 {
  static void
  run(tvm::ffi::TensorView x,
      tvm::ffi::TensorView w_hi,
      tvm::ffi::TensorView w_mid,
      tvm::ffi::TensorView w_lo,
      tvm::ffi::TensorView part_mix,
      tvm::ffi::TensorView part_sq) {
    using namespace host;
    auto m = SymbolicSize{"num_tokens"};
    auto dev = SymbolicDevice{};
    dev.set_options<kDLCUDA>();
    TensorMatcher({m, kK}).with_dtype<bf16_t>().with_device(dev).verify(x);
    TensorMatcher({24, kK}).with_dtype<bf16_t>().with_device(dev).verify(w_hi).verify(w_mid).verify(w_lo);
    TensorMatcher({kSlices, m, 24}).with_dtype<float>().with_device(dev).verify(part_mix);
    TensorMatcher({kSlices, m}).with_dtype<float>().with_device(dev).verify(part_sq);
    const int rows = static_cast<int>(m.unwrap());
    CHECK_HOST(rows >= 1) << "rows must be >= 1";
    const Params prm{
        static_cast<const bf16_t*>(x.data_ptr()),
        static_cast<const bf16_t*>(w_hi.data_ptr()),
        static_cast<const bf16_t*>(w_mid.data_ptr()),
        static_cast<const bf16_t*>(w_lo.data_ptr()),
        static_cast<float*>(part_mix.data_ptr()),
        static_cast<float*>(part_sq.data_ptr()),
        rows};
    auto kernel = mhc_stats_bf16x3_kernel;
    RuntimeDeviceCheck(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmem));
    LaunchKernel(dim3((rows + kRows - 1) / kRows, kSlices), dim3(kThreads), dev.unwrap(), kSmem)(kernel, prm);
  }
};

}  // namespace mhc_stats
}  // namespace sglang
