// SPDX-License-Identifier: Apache-2.0
//
// SM90 decode MoE for DeepSeek-V4 MXFP4 experts (W4A16: e2m1 weights with e8m0 block-32 scales,
// bf16 activations, fp32 accumulation). One CTA owns one (expert, output-channel block) and
// streams that block's weights exactly once for ALL rows routed to the expert (up to 8*kMaxCh
// rows per pass), so the cost per layer follows the number of touched experts, not the number of
// 16-row tiles. The current triton_kernels decode path re-reads and re-dequantizes an expert's
// weights for every 16-row tile.
//
// Weights are repacked at load time (see dsv4_mxfp4_decode_moe.py) so that each lane's
// mma.sync m16n8k16 A fragments for 4 consecutive k16 steps of one 16-channel tile are one
// 16-byte load, with the 8 e2m1 values of a k16 step placed in a 32-bit word such that they
// expand to the 4 bf16x2 A registers with a few shifts and masks:
//   a0 = (q)      & 0x81C081C0      a1 = (q << 3) & 0x81C081C0      a2 = (q << 6) & 0x81C081C0
//   a3 = ((q << 1) & 0x80008000) | ((q >> 4) & 0x00C000C0) | ((q >> 5) & 0x01000100)
// Each a_i is then the exact value * 2^-126 (bf16, subnormal for e=0); one bf16x2 multiply by
// 2^(s-1) (s = e8m0 exponent) gives the exact dequantized weight.
//
// Two kernels:
//   gate/up (kGateUp=true):  h[slot, c] = silu(clamp(g)) * clamp(u)  (bf16 rounding as the
//                            current path: g,u rounded to bf16, silu rounded, product rounded)
//   down    (kGateUp=false): y[orig_slot, n] = bf16(acc * gamma[slot]), summed over top-k later.

#pragma once

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <cuda_bf16.h>

#include <cstdint>
#include <type_traits>

namespace sglang {
namespace dsv4_decode_moe {

struct Params {
  const __nv_bfloat16* __restrict__ x;  // gate/up: [tokens, K] ; down: [slots, K] in sorted order
  const uint4* __restrict__ wq;         // repacked e2m1 [E][NB][W][K/64][T][32]
  const uint4* __restrict__ ws;         // repacked e8m0+126 [E][NB][W][K/64][32]
  const int32_t* __restrict__ counts;   // [E] slots per expert
  const int32_t* __restrict__ gather;   // gate/up: token of each sorted slot
  const int32_t* __restrict__ combine;  // down: original slot of each sorted slot
  const float* __restrict__ gammas;     // down: routing weight of each sorted slot
  __nv_bfloat16* __restrict__ out;      // gate/up: [slots, I] sorted ; down: [slots, N] original
  int num_experts;
  int out_ld;  // row stride of out (elements)
  float limit;
};

// ---------------------------------------------------------------------------------------------
// PTX helpers
SGL_DEVICE uint4 ldg_stream(const uint4* p) {
  uint4 r;
  asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0, %1, %2, %3}, [%4];"
               : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w)
               : "l"(p));
  return r;
}

SGL_DEVICE void cp_async16(uint32_t smem_addr, const void* gmem) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(smem_addr), "l"(gmem));
}
SGL_DEVICE void cp_async_commit() { asm volatile("cp.async.commit_group;"); }
template <int N>
SGL_DEVICE void cp_async_wait() {
  asm volatile("cp.async.wait_group %0;" ::"n"(N));
}

SGL_DEVICE void ldmatrix_x4(uint32_t addr, uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
               : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
               : "r"(addr));
}

SGL_DEVICE void mma_bf16(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
      "{%0, %1, %2, %3};"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

SGL_DEVICE uint32_t bf16x2_mul(uint32_t a, uint32_t b) {
  uint32_t d;
  asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(d) : "r"(a), "r"(b));
  return d;
}

// bf16x2 multiplier 2^(s-1) in both halves from byte `i` of `w` (byte = s + 126).
template <int kByte>
SGL_DEVICE uint32_t scale_mult(uint32_t w) {
  const uint32_t b = __byte_perm(w, 0u, 0x4440u | kByte);
  return b * 0x00800080u;
}

// 8 e2m1 values (repacked word) -> 4 bf16x2 A registers (value * 2^-126), then scaled.
SGL_DEVICE void dequant8(uint32_t q, uint32_t m_lo, uint32_t m_hi, uint32_t (&a)[4]) {
  const uint32_t a0 = q & 0x81C081C0u;
  const uint32_t a1 = (q << 3) & 0x81C081C0u;
  const uint32_t a2 = (q << 6) & 0x81C081C0u;
  const uint32_t a3 = ((q << 1) & 0x80008000u) | ((q >> 4) & 0x00C000C0u) | ((q >> 5) & 0x01000100u);
  a[0] = bf16x2_mul(a0, m_lo);  // channel g,   k 2c..2c+1
  a[1] = bf16x2_mul(a1, m_hi);  // channel g+8, k 2c..2c+1
  a[2] = bf16x2_mul(a2, m_lo);  // channel g,   k 2c+8..2c+9
  a[3] = bf16x2_mul(a3, m_hi);  // channel g+8, k 2c+8..2c+9
}

SGL_DEVICE float bf16_round(float v) { return __bfloat162float(__float2bfloat16_rn(v)); }

// ---------------------------------------------------------------------------------------------
// kWarps warps per CTA, kTiles 16-channel tiles per warp, kK reduction length, kKC elements of K
// staged in shared memory per X stage, kStages X stages, up to 8*kMaxCh rows per weight pass,
// kRing k64 steps in each thread's private weight ring (cp.async, so in-flight bytes cost no
// registers), kMinBlocks CTAs per SM for __launch_bounds__, kKSplit groups of kWarps warps that
// each reduce one contiguous K range of the same channels (partial sums added through shared memory
// at the end; same per-element k order inside a group, so the gate/up split changes rounding only
// in the final fp32 add of the two halves).
template <int kWarps, int kTiles, int kK, int kKC, int kStages, int kMaxCh, int kRing, int kMinBlocks, bool kGateUp, int kKSplit>
struct Config {
  static constexpr int kWarpsTotal = kWarps * kKSplit;
  static constexpr int kThreads = kWarpsTotal * 32;
  static constexpr int kRows = 8 * kMaxCh;
  static constexpr int kUnits = kKC / 8;  // 16-byte units per staged X row
  static constexpr int kSteps = kK / 64 / kKSplit;  // k64 steps per K group
  static constexpr int kStepsPerChunk = kKC / 64;
  static constexpr int kChunks = kK / kKC / kKSplit;  // X chunks per K group
  static constexpr int kStageElems = kRows * kKC;
  // A single-chunk K (down) needs one X buffer whatever kStages is.
  static constexpr int kSmemStages = kStages < kChunks ? kStages : kChunks;
  static constexpr int kFrags = kTiles + 1;  // per k64 step: one uint4 per tile + one of scales
  static constexpr size_t kXBytes = static_cast<size_t>(kKSplit) * kSmemStages * kStageElems * 2;
  static constexpr size_t kRingBytes = static_cast<size_t>(kRing) * kFrags * kThreads * 16;
  static constexpr size_t kSmemBytes = kXBytes + kRingBytes;
  static constexpr size_t kReduceBytes = static_cast<size_t>(kWarps) * 32 * kTiles * kMaxCh * 4 * sizeof(float);
  static_assert(kK % (kKC * kKSplit) == 0 && kKC % 64 == 0, "K chunking");
  static_assert(!kGateUp || kTiles == 4, "gate/up pairs tiles (0,2) and (1,3)");
  static_assert(kStages >= 2 && kRing >= 2, "pipeline depth");
  static_assert(kMaxCh == 4, "the row-chunk dispatch below covers 1..4");
  static_assert(kKSplit == 1 || (kKSplit == 2 && kReduceBytes <= kXBytes), "K split reduces through the X buffers");
  // X chunk c is copied in the group of step (c - kStages + 1) * kStepsPerChunk; step s waits for the
  // groups up to s - kRing + 1, so the chunk is complete when it is first used.
  static_assert(kChunks == 1 || kRing - 1 <= (kStages - 1) * kStepsPerChunk, "X chunk lead");
};

template <int kWarps, int kTiles, int kK, int kKC, int kStages, int kMaxCh, int kRing, int kMinBlocks, bool kGateUp, int kKSplit>
__global__ void __launch_bounds__(kWarps * kKSplit * 32, kMinBlocks) decode_moe_kernel(const __grid_constant__ Params p) {
  using C = Config<kWarps, kTiles, kK, kKC, kStages, kMaxCh, kRing, kMinBlocks, kGateUp, kKSplit>;
  constexpr int kThreads = C::kThreads;
  constexpr int kRows = C::kRows;
  constexpr int kUnits = C::kUnits;
  constexpr int kSteps = C::kSteps;
  constexpr int kStepsPerChunk = C::kStepsPerChunk;
  constexpr int kChunks = C::kChunks;
  constexpr int kStageElems = C::kStageElems;
  constexpr int kSmemStages = C::kSmemStages;
  constexpr int kFrags = C::kFrags;

  extern __shared__ __align__(128) uint8_t smem_raw[];
  __shared__ int32_t row_src[kRows];
  __shared__ int32_t s_offset;

  const int expert = blockIdx.x;
  const int block = blockIdx.y;
  const int tid = threadIdx.x;
  const int warp_all = tid / 32;
  const int warp = warp_all % kWarps;  // channel warp
  const int kgroup = warp_all / kWarps;  // K range of this warp
  const int lane = tid % 32;
  const int g = lane / 4;
  const int c = lane % 4;

  const int count = p.counts[expert];
  if (count == 0) return;

  // Offset of this expert's slots in sorted order = sum of counts of lower experts.
  if (warp_all == 0) {
    int acc = 0;
    for (int e = lane; e < expert; e += 32) acc += p.counts[e];
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, o);
    if (lane == 0) s_offset = acc;
  }
  __syncthreads();
  const int offset = s_offset;

  const size_t wblock = (static_cast<size_t>(expert) * gridDim.y + block) * kWarps + warp;
  const uint4* wq = p.wq + (wblock * kKSplit + kgroup) * (kSteps * kTiles * 32) + lane;
  const uint4* ws = p.ws + (wblock * kKSplit + kgroup) * (kSteps * 32) + lane;
  const uint32_t xs_base = static_cast<uint32_t>(__cvta_generic_to_shared(smem_raw));
  // This thread's private weight ring: slot-major, then fragment, then thread (conflict-free v4).
  const uint32_t ring_base = xs_base + static_cast<uint32_t>(C::kXBytes) + tid * 16;
  const uint4* ring_ptr = reinterpret_cast<const uint4*>(smem_raw + C::kXBytes) + tid;

  for (int r0 = 0; r0 < count; r0 += kRows) {
    const int nrows = min(count - r0, kRows);
    const int nch = (nrows + 7) / 8;
    if (tid < kRows) {
      int src = -1;
      if (tid < nrows) src = kGateUp ? p.gather[offset + r0 + tid] : offset + r0 + tid;
      row_src[tid] = src;
    }
    __syncthreads();

    // One code path per row-chunk count, so no predicated-off mma/ldmatrix take issue slots.
    auto pass = [&](auto nch_c) {
      constexpr int kNch = decltype(nch_c)::value;
      auto issue_chunk = [&](int chunk) {  // chunk `chunk` of every K group
        const int stage = chunk % kSmemStages;
        constexpr int units = kNch * 8 * kUnits;
        for (int idx = tid; idx < kKSplit * units; idx += kThreads) {
          const int grp = idx / units;
          const int r = (idx % units) / kUnits;
          const int u = idx % kUnits;
          const int src = row_src[r];
          if (src >= 0) {
            const __nv_bfloat16* gsrc = p.x + static_cast<size_t>(src) * kK + (grp * kChunks + chunk) * kKC + u * 8;
            cp_async16(xs_base + 2u * ((grp * kSmemStages + stage) * kStageElems + r * kKC + ((u ^ (r & 7)) * 8)), gsrc);
          }
        }
      };
      auto issue_weights = [&](int step) {
        const int slot = step % kRing;
  #pragma unroll
        for (int f = 0; f < kTiles; ++f)
          cp_async16(ring_base + ((slot * kFrags + f) * kThreads) * 16, wq + (step * kTiles + f) * 32);
        cp_async16(ring_base + ((slot * kFrags + kTiles) * kThreads) * 16, ws + step * 32);
      };

      // Prologue: X chunks 0..kStages-2 in the first group, then one group per weight step.
  #pragma unroll
      for (int i = 0; i < kStages - 1; ++i)
        if (i < kChunks) issue_chunk(i);
  #pragma unroll
      for (int i = 0; i < kRing - 1; ++i) {
        if (i < kSteps) issue_weights(i);
        cp_async_commit();
      }

      float acc[kTiles][kNch][4];
  #pragma unroll
      for (int t = 0; t < kTiles; ++t)
  #pragma unroll
        for (int ch = 0; ch < kNch; ++ch)
  #pragma unroll
          for (int i = 0; i < 4; ++i) acc[t][ch][i] = 0.f;

  #pragma unroll 1
      for (int s = 0; s < kSteps; ++s) {
        cp_async_wait<kRing - 2>();  // this thread's step-s weights (and X chunk) have landed
        if (s % kStepsPerChunk == 0) {
          __syncthreads();  // everyone's X chunk is visible; the previous chunk's stage is free
          const int next = s / kStepsPerChunk + kStages - 1;
          if (next < kChunks) issue_chunk(next);
        }
        if (s + kRing - 1 < kSteps) issue_weights(s + kRing - 1);
        cp_async_commit();

        const int stage = kgroup * kSmemStages + (s / kStepsPerChunk) % kSmemStages;
        const int kin = (s % kStepsPerChunk) * 64;  // k offset inside the staged chunk
        const int slot = s % kRing;
        const uint4 scur = ring_ptr[(slot * kFrags + kTiles) * kThreads];
        const uint32_t sw[4] = {scur.x, scur.y, scur.z, scur.w};

  #pragma unroll
        for (int half = 0; half < 2; ++half) {
          // B fragments of k16 sub-steps 2*half and 2*half+1 for every row chunk.
          uint32_t bfrag[kNch][2][2];
  #pragma unroll
          for (int ch = 0; ch < kNch; ++ch) {
            {
              const int r = ch * 8 + (lane % 8);
              const int unit = (kin + half * 32) / 8 + lane / 8;
              ldmatrix_x4(xs_base + 2u * (stage * kStageElems + r * kKC + ((unit ^ (r & 7)) * 8)), bfrag[ch][0][0],
                          bfrag[ch][0][1], bfrag[ch][1][0], bfrag[ch][1][1]);
            }
          }
  #pragma unroll
          for (int t = 0; t < kTiles; ++t) {
            const uint4 wt = ring_ptr[(slot * kFrags + t) * kThreads];
            const uint32_t q2[2] = {half == 0 ? wt.x : wt.z, half == 0 ? wt.y : wt.w};
            // scale bytes of tile t: [blk0 ch g, blk0 ch g+8, blk1 ch g, blk1 ch g+8]; half == k32 block.
            const uint32_t m_lo = half == 0 ? scale_mult<0>(sw[t]) : scale_mult<2>(sw[t]);
            const uint32_t m_hi = half == 0 ? scale_mult<1>(sw[t]) : scale_mult<3>(sw[t]);
  #pragma unroll
            for (int jj = 0; jj < 2; ++jj) {
              uint32_t a[4];
              dequant8(q2[jj], m_lo, m_hi, a);
  #pragma unroll
              for (int ch = 0; ch < kNch; ++ch) {
                mma_bf16(acc[t][ch], a, bfrag[ch][jj][0], bfrag[ch][jj][1]);
              }
            }
          }
        }
      }

      if constexpr (kKSplit == 2) {
        // Add K group 1's partial sums into group 0 through the (now idle) X buffers.
        float* red = reinterpret_cast<float*>(smem_raw);
        __syncthreads();
        if (kgroup == 1) {
#pragma unroll
          for (int t = 0; t < kTiles; ++t)
#pragma unroll
            for (int ch = 0; ch < kNch; ++ch)
#pragma unroll
              for (int i = 0; i < 4; ++i) red[(((warp * kTiles + t) * kNch + ch) * 4 + i) * 32 + lane] = acc[t][ch][i];
        }
        __syncthreads();
        if (kgroup == 0) {
#pragma unroll
          for (int t = 0; t < kTiles; ++t)
#pragma unroll
            for (int ch = 0; ch < kNch; ++ch)
#pragma unroll
              for (int i = 0; i < 4; ++i) acc[t][ch][i] += red[(((warp * kTiles + t) * kNch + ch) * 4 + i) * 32 + lane];
        }
      }
      if (kgroup == 0) {
      // Epilogue. Thread holds D[channel g / g+8][row 2c / 2c+1] of each (tile, chunk).
      if constexpr (kGateUp) {
  #pragma unroll
        for (int tp = 0; tp < 2; ++tp) {
  #pragma unroll
          for (int ch = 0; ch < kNch; ++ch) {
            {
  #pragma unroll
              for (int i = 0; i < 4; ++i) {
                const int r = ch * 8 + 2 * c + (i & 1);
                if (r < nrows) {
                  const int chan = (block * kWarps + warp) * 32 + tp * 16 + g + 8 * (i >> 1);
                  float gate = bf16_round(acc[tp][ch][i]);
                  float up = bf16_round(acc[tp + 2][ch][i]);
                  const float lim = p.limit;
                  gate = gate > lim ? lim : gate;
                  up = up > lim ? lim : (up < -lim ? -lim : up);
                  const float act = bf16_round(gate / (1.0f + expf(-gate)));
                  p.out[static_cast<size_t>(offset + r0 + r) * p.out_ld + chan] = __float2bfloat16_rn(act * up);
                }
              }
            }
          }
        }
      } else {
  #pragma unroll
        for (int t = 0; t < kTiles; ++t) {
  #pragma unroll
          for (int ch = 0; ch < kNch; ++ch) {
            {
  #pragma unroll
              for (int i = 0; i < 4; ++i) {
                const int r = ch * 8 + 2 * c + (i & 1);
                if (r < nrows) {
                  const int sorted = offset + r0 + r;
                  const int chan = ((block * kWarps + warp) * kTiles + t) * 16 + g + 8 * (i >> 1);
                  const float v = acc[t][ch][i] * p.gammas[sorted];
                  p.out[static_cast<size_t>(p.combine[sorted]) * p.out_ld + chan] = __float2bfloat16_rn(v);
                }
              }
            }
          }
        }
      }
      }  // kgroup == 0
      cp_async_wait<0>();
      __syncthreads();  // smem stages and row_src are reused by the next row pass
    };
    switch (nch) {
      case 1:
        pass(std::integral_constant<int, 1>{});
        break;
      case 2:
        pass(std::integral_constant<int, 2>{});
        break;
      case 3:
        pass(std::integral_constant<int, 3>{});
        break;
      case 4:
        pass(std::integral_constant<int, 4>{});
        break;
      default:
        break;
    }
  }
}


// ---------------------------------------------------------------------------------------------
// Host entry points
template <int kWarps, int kTiles, int kK, int kKC, int kStages, int kMaxCh, int kRing, int kMinBlocks, bool kGateUp, int kKSplit>
void launch(const Params& prm, int num_blocks, DLDevice device) {
  using namespace host;
  using C = Config<kWarps, kTiles, kK, kKC, kStages, kMaxCh, kRing, kMinBlocks, kGateUp, kKSplit>;
  auto kernel = decode_moe_kernel<kWarps, kTiles, kK, kKC, kStages, kMaxCh, kRing, kMinBlocks, kGateUp, kKSplit>;
  // Set on every launch (host-only, cheap, not recorded in CUDA graphs): a function-local static flag
  // would be shared by every JIT module that instantiates this template and skip the others.
  RuntimeDeviceCheck(
      cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(C::kSmemBytes)));
  LaunchKernel(dim3(prm.num_experts, num_blocks), dim3(C::kThreads), device, C::kSmemBytes)(kernel, prm);
}

// Gate/up: H = 5120 (K), I = 576 per rank, blocks of 192 gate + 192 up channels.
inline void gate_up(
    tvm::ffi::TensorView x,
    tvm::ffi::TensorView wq,
    tvm::ffi::TensorView ws,
    tvm::ffi::TensorView counts,
    tvm::ffi::TensorView gather,
    tvm::ffi::TensorView out,
    double limit) {
  using namespace host;
  RuntimeCheck(x.size(1) == 5120, "gate_up: hidden must be 5120, got ", x.size(1));
  RuntimeCheck(out.size(1) == 576, "gate_up: intermediate must be 576, got ", out.size(1));
  Params prm{};
  prm.x = static_cast<const __nv_bfloat16*>(x.data_ptr());
  prm.wq = static_cast<const uint4*>(wq.data_ptr());
  prm.ws = static_cast<const uint4*>(ws.data_ptr());
  prm.counts = static_cast<const int32_t*>(counts.data_ptr());
  prm.gather = static_cast<const int32_t*>(gather.data_ptr());
  prm.out = static_cast<__nv_bfloat16*>(out.data_ptr());
  prm.num_experts = static_cast<int>(counts.size(0));
  prm.out_ld = static_cast<int>(out.size(1));
  prm.limit = static_cast<float>(limit);
  launch<6, 4, 5120, 256, 3, 4, 4, 2, true, 1>(prm, 3, x.device());
}

// Down: K = 576, N = 5120, blocks of 256 output channels (4 warps x 4 tiles).
inline void down(
    tvm::ffi::TensorView h,
    tvm::ffi::TensorView wq,
    tvm::ffi::TensorView ws,
    tvm::ffi::TensorView counts,
    tvm::ffi::TensorView combine,
    tvm::ffi::TensorView gammas,
    tvm::ffi::TensorView out) {
  using namespace host;
  RuntimeCheck(h.size(1) == 576, "down: intermediate must be 576, got ", h.size(1));
  RuntimeCheck(out.size(1) == 5120, "down: hidden must be 5120, got ", out.size(1));
  Params prm{};
  prm.x = static_cast<const __nv_bfloat16*>(h.data_ptr());
  prm.wq = static_cast<const uint4*>(wq.data_ptr());
  prm.ws = static_cast<const uint4*>(ws.data_ptr());
  prm.counts = static_cast<const int32_t*>(counts.data_ptr());
  prm.combine = static_cast<const int32_t*>(combine.data_ptr());
  prm.gammas = static_cast<const float*>(gammas.data_ptr());
  prm.out = static_cast<__nv_bfloat16*>(out.data_ptr());
  prm.num_experts = static_cast<int>(counts.size(0));
  prm.out_ld = static_cast<int>(out.size(1));
  launch<4, 4, 576, 576, 2, 4, 3, 3, false, 1>(prm, 20, h.device());
}


}  // namespace dsv4_decode_moe
}  // namespace sglang
