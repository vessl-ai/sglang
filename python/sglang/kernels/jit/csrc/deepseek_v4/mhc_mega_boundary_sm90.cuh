// INF-681 k-qcp: one SM90 kernel for a DeepSeek-V4.1 mHC boundary in prefill. RESEARCH KERNEL, not
// wired into the model: it is bit-identical to today's path but slower (H200, 16,302 rows: 0.75 ms vs
// 0.66 ms for mhc_post_combine_norm_prefill + the Triton stats; floor 0.35 ms). Kept for the exact
// recipe below. From y, R and this hyper-connection's (pre, post, comb) it produces:
//
//   R'      = post * y + comb^T R                     bit-identical to mhc_post_combine_norm_prefill.cuh
//   x       = RMSNorm(sum_i pre[i] R'[i]) * w         bit-identical to the same kernel
//   part_mix[16, M, 24], part_sq[16, M]                bit-identical to mhc._hc_mix_stats_bf16x3_kernel
//                                                      on R' in its BLOCK_M 128 configuration (>= 4096
//                                                      rows; the unchanged sinkhorn kernel finishes them)
//
// Exactness of the stats (also mhc_stats_bf16x3_sm90.cuh):
//  * mix: per 1280-column slice and 64-column block, wgmma m64nNk16 bf16 -> fp32 with k16 chunks in
//    order into separate hi / mid / lo accumulators, then (hi + mid) + lo. Triton issues m64n32k16 with
//    A from shared memory and n 24..31 zero; m64n24k16 with A from registers gives the same elements.
//  * sq: per 64-column block, Triton's #blocked layout gives each lane 8 consecutive columns and folds
//    them as ((fma(e0,e0,e1*e1) + fma(e2,e2,e3*e3)) + (fma(e4,e4,e5*e5) + fma(e6,e6,e7*e7))), then a
//    butterfly over the 8 lanes (xor 4, 2, 1), then sq += block (read from its PTX).
//
// Structure: CTA = 64 rows, 512 threads = 4 warpgroups (warpgroup s runs stream s's wgmmas). Every
// thread also runs the element-wise side for one row and 8 columns of each 64-column block (which is
// exactly Triton's sq lane): R' for the 4 streams, stored to global and to shared memory (rp, read by
// ldmatrix as the wgmma A operand), the collapsed row (stored to `normalized`) and sq. 80 blocks,
// double-buffered with cp.async. At the end each row's collapsed values are re-read and normalized in
// place with the fused kernel's exact reduction. Measured (16,302 rows): loads + element-wise side
// 0.53 ms, wgmma side +0.11, final norm pass +0.11; 512 threads x 128 registers = 1 CTA per SM, and
// the phases of an iteration do not overlap enough to approach the floor.
#pragma once
#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/math.cuh>
#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>

#include <tvm/ffi/container/tensor.h>

#include <cstdint>

namespace sglang {
namespace mhc_mega {

constexpr int kH = 5120;
constexpr int kK = 4 * kH;
constexpr int kRows = 64;          // rows per CTA
constexpr int kBlk = 64;           // columns per block
constexpr int kBlocks = kH / kBlk; // 80
constexpr int kQuarter = 1280;     // one stats slice per stream and quarter
constexpr int kBlocksPerQuarter = kQuarter / kBlk;  // 20
constexpr int kThreads = 512;
constexpr int kPitch = 72;         // padded bf16 row pitch of the y / R tiles (144 B)
constexpr int kTile = kRows * kPitch * 2;           // 9216 B
constexpr int kWTile = 24 * 128;                    // 24 mix rows x 64 k bf16, 128B-swizzled (3 atoms)
constexpr int kWStage = 4 * 3 * kWTile;             // 36864 B (stream x part)
constexpr int kStage = kWStage + 5 * kTile;         // 82944 B = 81 KiB
constexpr int kRp = 4 * kTile;                      // R' of the block, 4 streams (the element-wise side's input)
constexpr int kCoef = kRows * 24 * 4;              // post | comb | pre of the CTA's rows (fp32)
constexpr int kSmem = 1024 + 2 * kStage + kRp + kCoef + 64;  // + warp sums

struct Params {
  const bf16_t *y, *residual, *w_hi, *w_mid, *w_lo, *norm_w;
  bf16_t *residual_out, *normalized;
  const float *post, *comb, *pre;
  float *part_mix, *part_sq;
  int m;
  float eps;
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
SGL_DEVICE void wgmma_fence() {
  asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
}
SGL_DEVICE void wgmma_commit() {
  asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
}
SGL_DEVICE void wgmma_wait0() {
  asm volatile("wgmma.wait_group.sync.aligned 0;" ::: "memory");
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
// D[64x24] += A(regs, 64x16 bf16) x B(desc, 16x24 bf16 k-major). Each output element's arithmetic is
// that of the Triton kernel's m64n32k16 (zero-padded n 24..31): n does not enter an element's sum.
SGL_DEVICE void wgmma_bf16(float (&d)[12], const uint32_t (&a)[4], uint64_t desc) {
  asm volatile(
      "{\n.reg .pred p;\nsetp.ne.b32 p, %17, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n24k16.f32.bf16.bf16 "
      "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11}, {%12, %13, %14, %15}, %16, p, 1, 1, 0;\n}\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]),
        "+f"(d[9]), "+f"(d[10]), "+f"(d[11])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc), "r"(1));
}

SGL_DEVICE float bf(const bf16_t v) {
  return __bfloat162float(v);
}
SGL_DEVICE bf16_t rnd(float v) {
  return __float2bfloat16_rn(v);
}
// One residual element of output stream i: the fused kernel's FMA chain.
SGL_DEVICE float post_elem(float post, float c0, float c1, float c2, float c3, float y, float r0, float r1, float r2, float r3) {
  float a = __fmaf_rn(post, y, __fmul_rn(c0, r0));
  a = __fmaf_rn(c1, r1, a);
  a = __fmaf_rn(c2, r2, a);
  return __fmaf_rn(c3, r3, a);
}
SGL_DEVICE uint32_t pack2(bf16_t lo, bf16_t hi) {
  return static_cast<uint32_t>(__bfloat16_as_ushort(lo)) | (static_cast<uint32_t>(__bfloat16_as_ushort(hi)) << 16);
}

__global__ __launch_bounds__(kThreads, 1) void mhc_mega_boundary_kernel(const __grid_constant__ Params p) {
  extern __shared__ __align__(1024) uint8_t smem_raw[];
  const uint32_t raw = smem_u32(smem_raw);
  uint8_t* base = smem_raw + (((raw + 1023) & ~1023u) - raw);
  bf16_t* rp = reinterpret_cast<bf16_t*>(base + 2 * kStage);               // [4][64][kPitch]
  float* coef = reinterpret_cast<float*>(base + 2 * kStage + kRp);                // [64][24]
  float* warp_sums = reinterpret_cast<float*>(base + 2 * kStage + kRp + kCoef);  // [16] (4 rows at a time)

  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;      // 0..15
  const int wg = warp >> 2;       // warpgroup 0..3 = the stream its wgmmas serve
  const int wq = warp & 3;        // warp within the warpgroup: rows 16 wq .. 16 wq + 15
  const int gid = lane >> 2;      // fragment row
  const int tig = lane & 3;       // fragment column pair
  const int row0 = blockIdx.x * kRows;


  // ---- loads of one 64-column block into a stage ----
  auto load_block = [&](int blk, int stage) {
    uint8_t* st = base + stage * kStage;
    const int h0 = blk * kBlk;
    // W: 4 streams x 3 parts x 24 rows x 8 chunks of 16 B (the 24 x 128 B tile, no padding rows)
    for (int i = tid; i < 4 * 3 * 24 * 8; i += kThreads) {
      const int chunk = i & 7, n = (i >> 3) % 24, tile = i / (24 * 8), s = tile / 3, part = tile % 3;
      const bf16_t* w = part == 0 ? p.w_hi : (part == 1 ? p.w_mid : p.w_lo);
      const bf16_t* src = w + static_cast<int64_t>(n) * kK + s * kH + h0 + chunk * 8;
      const uint32_t dst = smem_u32(st + tile * kWTile + n * 128 + ((chunk ^ (n & 7)) * 16));
      cp_async16(dst, src, true);
    }
    // y and R: 5 tiles x 64 rows x 8 chunks
    for (int i = tid; i < 5 * kRows * 8; i += kThreads) {
      const int chunk = i & 7, r = (i >> 3) & (kRows - 1), t = i / (kRows * 8);
      const int row = row0 + r;
      const bool valid = row < p.m;
      const int srow = valid ? row : 0;
      const bf16_t* src = t == 0 ? p.y + static_cast<int64_t>(srow) * kH + h0 + chunk * 8
                                 : p.residual + static_cast<int64_t>(srow) * kK + (t - 1) * kH + h0 + chunk * 8;
      const uint32_t dst = smem_u32(st + kWStage + t * kTile + r * (kPitch * 2) + chunk * 16);
      cp_async16(dst, src, valid);
    }
  };

  // ---- coefficients of the CTA's rows into shared memory: [row][post 4 | comb 16 | pre 4] ----
  for (int i = tid; i < kRows * 24; i += kThreads) {
    const int r = i / 24, k = i % 24, row = row0 + r;
    float v = 0.f;
    if (row < p.m) v = k < 4 ? p.post[row * 4 + k] : (k < 20 ? p.comb[row * 16 + k - 4] : p.pre[row * 4 + k - 20]);
    coef[i] = v;
  }
  // element-wise side: thread -> row tid / 8, columns 8 (tid % 8) .. +7 of each block: exactly the
  // stats kernel's #blocked lane t = tid % 8 (8 consecutive columns), the 8 lanes of a row consecutive.
  const int er = tid >> 3;
  const int et = tid & 7;
  const int ec = 8 * et;
  const int erow = row0 + er;
  const bool erow_ok = erow < p.m;
  // wgmma side: warp wq of warpgroup wg covers rows 16 wq .. 16 wq + 15; fragment rows rA, rB
  const int rA = 16 * wq + gid, rB = rA + 8;
  // ldmatrix.x4 row address of this lane in a 16 x 16 A tile (matrices: rows 0-7 / 8-15 x k 0-7 / 8-15)
  const int lm_row = 16 * wq + ((lane >> 3) & 1) * 8 + (lane & 7);
  const int lm_col = (lane >> 4) * 8;

  float acc[3][12];  // [part][fragment] for stream wg
  float sq[4];       // element-wise side: [stream] for row er
#pragma unroll
  for (int t = 0; t < 3; ++t)
#pragma unroll
    for (int e = 0; e < 12; ++e)
      acc[t][e] = 0.f;
#pragma unroll
  for (int i = 0; i < 4; ++i) sq[i] = 0.f;

  load_block(0, 0);
  cp_async_commit();
  __syncthreads();  // coef

  for (int blk = 0; blk < kBlocks; ++blk) {
    const int stage = blk & 1;
    if (blk + 1 < kBlocks) load_block(blk + 1, stage ^ 1);
    cp_async_commit();
    cp_async_wait<1>();
    __syncthreads();
    fence_proxy_async();  // cp.async (generic proxy) writes of W -> wgmma (async proxy) reads
    const uint8_t* st = base + stage * kStage;
    auto tile = [&](int t) { return reinterpret_cast<const bf16_t*>(st + kWStage + t * kTile); };
    const int h0 = blk * kBlk;

    // ---- element-wise side: R' (all 4 streams), the collapsed row and the stats' sum of squares ----
    {
      const float* cf = coef + er * 24;
      const int off = er * kPitch + ec;
      float in[5][8];
#pragma unroll
      for (int t = 0; t < 5; ++t) {
        const uint4 raw = *reinterpret_cast<const uint4*>(tile(t) + off);
        const __nv_bfloat162* h2 = reinterpret_cast<const __nv_bfloat162*>(&raw);
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          const float2 f = __bfloat1622float2(h2[e]);
          in[t][2 * e] = f.x;
          in[t][2 * e + 1] = f.y;
        }
      }
      float coll[8];
#pragma unroll
      for (int e = 0; e < 8; ++e) coll[e] = 0.f;
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const float po = cf[i], c0 = cf[4 + i], c1 = cf[8 + i], c2 = cf[12 + i], c3 = cf[16 + i], pr = cf[20 + i];
        bf16_t u[8];
        float v[8];
#pragma unroll
        for (int e = 0; e < 8; ++e) {
          u[e] = rnd(post_elem(po, c0, c1, c2, c3, in[0][e], in[1][e], in[2][e], in[3][e], in[4][e]));
          v[e] = bf(u[e]);
          coll[e] = __fmaf_rn(pr, v[e], coll[e]);
        }
        const uint4 packed = *reinterpret_cast<const uint4*>(u);
        *reinterpret_cast<uint4*>(rp + i * (kRows * kPitch) + off) = packed;
        if (erow_ok)
          *reinterpret_cast<uint4*>(p.residual_out + static_cast<int64_t>(erow) * kK + i * kH + h0 + ec) = packed;
        // Triton's per-lane fold of 8 columns, then its butterfly over the row's 8 lanes (xor 4, 2, 1)
        float s8 = __fadd_rn(
            __fadd_rn(__fmaf_rn(v[0], v[0], __fmul_rn(v[1], v[1])), __fmaf_rn(v[2], v[2], __fmul_rn(v[3], v[3]))),
            __fadd_rn(__fmaf_rn(v[4], v[4], __fmul_rn(v[5], v[5])), __fmaf_rn(v[6], v[6], __fmul_rn(v[7], v[7]))));
        s8 = __fadd_rn(s8, __shfl_xor_sync(0xffffffff, s8, 4));
        s8 = __fadd_rn(s8, __shfl_xor_sync(0xffffffff, s8, 2));
        s8 = __fadd_rn(s8, __shfl_xor_sync(0xffffffff, s8, 1));
        sq[i] = __fadd_rn(sq[i], s8);
      }
      bf16_t cb[8];
#pragma unroll
      for (int e = 0; e < 8; ++e) cb[e] = rnd(coll[e]);
      if (erow_ok)
        *reinterpret_cast<uint4*>(p.normalized + static_cast<int64_t>(erow) * kH + h0 + ec) =
            *reinterpret_cast<const uint4*>(cb);
    }
    __syncthreads();  // rp complete

    // ---- wgmma side: A fragments of R'_wg by ldmatrix, 4 chunks x 3 parts ----
    {
      uint32_t a[4][4];
#pragma unroll
      for (int c = 0; c < 4; ++c) {
        const uint32_t addr = smem_u32(rp + wg * (kRows * kPitch) + lm_row * kPitch + 16 * c + lm_col);
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
                     : "=r"(a[c][0]), "=r"(a[c][1]), "=r"(a[c][2]), "=r"(a[c][3])
                     : "r"(addr));
      }
#pragma unroll
      for (int c = 0; c < 4; ++c) fence_a(a[c]);
      wgmma_fence();
#pragma unroll
      for (int part = 0; part < 3; ++part) {
        const uint32_t wt = smem_u32(st + (wg * 3 + part) * kWTile);
#pragma unroll
        for (int c = 0; c < 4; ++c)
          wgmma_bf16(acc[part], a[c], make_desc(wt + c * 32));
      }
      wgmma_commit();
      wgmma_wait0();
#pragma unroll
      for (int t = 0; t < 3; ++t) fence_acc(acc[t]);
    }

    // ---- end of a quarter: flush the slices ----
    if ((blk % kBlocksPerQuarter) == kBlocksPerQuarter - 1) {
      const int q = blk / kBlocksPerQuarter;
      const int slice = wg * 4 + q;
#pragma unroll
      for (int hlf = 0; hlf < 2; ++hlf) {
        const int row = row0 + (hlf ? rB : rA);
        if (row < p.m) {
          float* dst = p.part_mix + (static_cast<int64_t>(slice) * p.m + row) * 24;
#pragma unroll
          for (int nb = 0; nb < 3; ++nb) {  // the three n8 blocks
            const int n = 8 * nb + 2 * tig;
            const int e = 4 * nb + 2 * hlf;
            dst[n] = __fadd_rn(__fadd_rn(acc[0][e], acc[1][e]), acc[2][e]);
            dst[n + 1] = __fadd_rn(__fadd_rn(acc[0][e + 1], acc[1][e + 1]), acc[2][e + 1]);
          }
        }
      }
#pragma unroll
      for (int t = 0; t < 3; ++t)
#pragma unroll
        for (int e = 0; e < 12; ++e)
          acc[t][e] = 0.f;
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        if (et == 0 && erow_ok) p.part_sq[static_cast<int64_t>(i * 4 + q) * p.m + erow] = sq[i];
        sq[i] = 0.f;
      }
    }
    __syncthreads();  // the stage and rp are refilled by the next iteration
  }
  cp_async_wait<0>();
  __syncthreads();  // the collapsed rows of this CTA are written (global, same CTA)
  __threadfence_block();

  // ---- RMSNorm of the collapsed rows, the fused kernel's exact reduction (128 threads / row) ----
  using V = device::AlignedVector<bf16x2_t, 4>;
  const int lt = tid & 127;           // thread within the row team
  const int team = tid >> 7;          // 4 rows at a time
  const int tl = lt & 31, tw = lt >> 5;
  for (int r = team; r < kRows; r += 4) {
    const int row = row0 + r;
    const bool ok = row < p.m;  // uniform per team
    const bf16_t* coll = p.normalized + static_cast<int64_t>(row) * kH;
    float sum = 0.f;
    if (ok) {
#pragma unroll
      for (int group = 0; group < 5; ++group) {
        V v;
        v.load(coll + group * 1024, lt);
#pragma unroll
        for (int k = 0; k < 4; ++k) {
          const auto value = device::cast<fp32x2_t>(v[k]);
          if (group == 0 && k == 0)
            sum = __fadd_rn(__fmul_rn(value.y, value.y), __fmul_rn(value.x, value.x));
          else {
            sum = __fmaf_rn(value.x, value.x, sum);
            sum = __fadd_rn(__fmul_rn(value.y, value.y), sum);
          }
        }
      }
    }
#pragma unroll
    for (int offset = 16; offset; offset >>= 1)
      sum = __fadd_rn(sum, __shfl_xor_sync(0xffffffff, sum, offset));
    if (tl == 0) warp_sums[team * 4 + tw] = sum;
    asm volatile("bar.sync %0, 128;" ::"r"(1 + team) : "memory");
    float inv_rms = 0.f;
    {
      float s4 = tl < 4 ? warp_sums[team * 4 + tl] : 0.f;
      s4 = __fadd_rn(s4, __shfl_xor_sync(0xffffffff, s4, 2));
      s4 = __fadd_rn(s4, __shfl_xor_sync(0xffffffff, s4, 1));
      float mean, inverse;
      asm("div.full.f32 %0, %1, %2;" : "=f"(mean) : "f"(s4), "f"(5120.f));
      asm("add.f32 %0, %1, %2;" : "=f"(mean) : "f"(mean), "f"(p.eps));
      asm("rsqrt.approx.ftz.f32 %0, %1;" : "=f"(inverse) : "f"(mean));
      inv_rms = __shfl_sync(0xffffffff, inverse, 0);
    }
    asm volatile("bar.sync %0, 128;" ::"r"(1 + team) : "memory");  // warp_sums reused next row
    if (ok) {
      bf16_t* out = p.normalized + static_cast<int64_t>(row) * kH;
#pragma unroll 1
      for (int vid = lt; vid < 640; vid += 128) {
        V v, w, o;
        v.load(coll, vid);
        w.load(p.norm_w, vid);
#pragma unroll
        for (int k = 0; k < 4; ++k) {
          const auto value = device::cast<fp32x2_t>(v[k]);
          const auto weight = device::cast<fp32x2_t>(w[k]);
          o[k] = device::cast<bf16x2_t>(fp32x2_t{
              __fmul_rn(__fmul_rn(value.x, inv_rms), weight.x), __fmul_rn(__fmul_rn(value.y, inv_rms), weight.y)});
        }
        o.store(out, vid);
      }
    }
  }
}

struct MhcMegaBoundary {
  static void
  run(tvm::ffi::TensorView y,
      tvm::ffi::TensorView residual,
      tvm::ffi::TensorView post,
      tvm::ffi::TensorView comb,
      tvm::ffi::TensorView pre,
      tvm::ffi::TensorView w_hi,
      tvm::ffi::TensorView w_mid,
      tvm::ffi::TensorView w_lo,
      tvm::ffi::TensorView norm_w,
      tvm::ffi::TensorView residual_out,
      tvm::ffi::TensorView normalized,
      tvm::ffi::TensorView part_mix,
      tvm::ffi::TensorView part_sq,
      double eps) {
    using namespace host;
    auto m = SymbolicSize{"num_tokens"};
    auto dev = SymbolicDevice{};
    dev.set_options<kDLCUDA>();
    TensorMatcher({m, kH}).with_dtype<bf16_t>().with_device(dev).verify(y).verify(normalized);
    TensorMatcher({m, 4, kH}).with_dtype<bf16_t>().with_device(dev).verify(residual).verify(residual_out);
    TensorMatcher({m, 4}).with_dtype<float>().with_device(dev).verify(post).verify(pre);
    TensorMatcher({m, 4, 4}).with_dtype<float>().with_device(dev).verify(comb);
    TensorMatcher({24, kK}).with_dtype<bf16_t>().with_device(dev).verify(w_hi).verify(w_mid).verify(w_lo);
    TensorMatcher({kH}).with_dtype<bf16_t>().with_device(dev).verify(norm_w);
    TensorMatcher({16, m, 24}).with_dtype<float>().with_device(dev).verify(part_mix);
    TensorMatcher({16, m}).with_dtype<float>().with_device(dev).verify(part_sq);
    const int rows = static_cast<int>(m.unwrap());
    CHECK_HOST(rows >= 1 && rows <= 65536) << "rows must be in [1, 65536]";
    const Params prm{
        static_cast<const bf16_t*>(y.data_ptr()),
        static_cast<const bf16_t*>(residual.data_ptr()),
        static_cast<const bf16_t*>(w_hi.data_ptr()),
        static_cast<const bf16_t*>(w_mid.data_ptr()),
        static_cast<const bf16_t*>(w_lo.data_ptr()),
        static_cast<const bf16_t*>(norm_w.data_ptr()),
        static_cast<bf16_t*>(residual_out.data_ptr()),
        static_cast<bf16_t*>(normalized.data_ptr()),
        static_cast<const float*>(post.data_ptr()),
        static_cast<const float*>(comb.data_ptr()),
        static_cast<const float*>(pre.data_ptr()),
        static_cast<float*>(part_mix.data_ptr()),
        static_cast<float*>(part_sq.data_ptr()),
        rows,
        static_cast<float>(eps)};
    auto kernel = mhc_mega_boundary_kernel;
    RuntimeDeviceCheck(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmem));
    LaunchKernel(dim3((rows + kRows - 1) / kRows), dim3(kThreads), dev.unwrap(), kSmem)(kernel, prm);
  }
};

}  // namespace mhc_mega
}  // namespace sglang
