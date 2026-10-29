// SPDX-License-Identifier: Apache-2.0
//
// SM90 persistent, warp-specialized W4A8 decode MoE for the DeepSeek-V4 TP4 MXFP4 experts
// (hidden 5120, 576 intermediate channels per rank).
//
// Numerics (as the W4A8 prefill path): activations e4m3 with one power-of-2 scale per row; weights e2m1
// expanded exactly to e4m3 in registers (value * 2^-6); wgmma.m64nNk32 e4m3 x e4m3 with one MX block (32 k)
// per instruction; each block's fp32 result is scaled by its e8m0 scale (x 2^6) and added to an fp32
// accumulator ("promotion"), so the block scales stay exact.
//
// Structure: one CTA per SM = 2 consumer warpgroups + 2 producer warps. Each (producer warp, consumer
// warpgroup) pair is an independent pipeline with its own ring and its own contiguous, cost-balanced run of
// (expert, channel block) items (see ItemCursor). For every 128-k stage the producer copies the stage's
// weights (one contiguous cp.async.bulk, repacked at load time) and the activation rows' 128-byte k chunk
// (cp.async into a 128B-swizzled atom) into a ring slot; the consumer reads its A fragments from shared
// memory, converts, and issues one wgmma per (MX block, tile) with B = the swizzled atom. A stage's blocks
// are issued as one batch, waited for, and then promoted (see PassState); the SM's other consumer warpgroup
// overlaps the promotion. The slot is released after that.
//
// gate/up item = 64 intermediate channels: m64 tile 0 = gate rows, tile 1 = up rows of the same channels,
// so SwiGLU stays in registers; h is written bf16 in sorted-slot order.
// down item = 128 output channels (2 m64 tiles); y[orig_slot] = acc * row scale * gamma.
// Also here: a one-CTA stable expert sort (sort_slots) and the top-k combine; both leave masked slots (id < 0)
// out, so dp-attention pad rows cost no expert work.
//
// Weight words (per lane, k64 step, 16-channel tile): 4 words [blk0 rows g, blk0 rows g+8, blk1 rows g,
// blk1 rows g+8]; byte i holds A = k 4c+i (a0/a1) and B = k 4c+16+i (a2/a3):
//   bit7 A.s  bit6 B.e1  bit5 B.s  bit4 A.e1  bit3 A.e0  bit2 A.m  bit1 B.e0  bit0 B.m
// Scale bytes (e8m0 + 6) per warp and step: [8 fragment groups][tile][blk][row g / g+8].

#pragma once

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <cstdint>
#include <cuda_bf16.h>
#include <type_traits>

namespace sglang {
namespace dsv4_moe_ws {

struct Params {
  const uint8_t* __restrict__ x8;       // e4m3 rows: gate/up [tokens, 5120], down [slots, 576] (sorted)
  const float* __restrict__ xscale;     // power-of-2 scale per x8 row
  const uint8_t* __restrict__ w;        // stage-ordered weights + scales [E][NB][item bytes]
  const int32_t* __restrict__ counts;   // [E]
  const int32_t* __restrict__ gather;   // gate/up: token of each sorted slot
  const int32_t* __restrict__ combine;  // down: original slot of each sorted slot
  const float* __restrict__ gammas;     // down: routing weight of each sorted slot
  __nv_bfloat16* __restrict__ out;      // gate/up: h [slots, 576] sorted; down: y [slots, 5120] original
  int num_experts;
  float limit;
};

template <bool kGateUp>
struct Shape;
template <>
struct Shape<true> {  // item = 64 intermediate channels; tile 0 = gate rows, tile 1 = up rows
  static constexpr int kK = 5120, kNB = 9, kTiles = 2, kStages = 40, kLastSteps = 2, kRing = 6, kOutLd = 576;
};
template <>
struct Shape<false> {  // item = 128 output channels; tile t = channels 128j + 64t
  static constexpr int kK = 576, kNB = 40, kTiles = 2, kStages = 5, kLastSteps = 1, kRing = 6, kOutLd = 5120;
};

constexpr int kMaxExperts = 512;

// Cost of one item for an expert with c rows, in activation bytes per k: each 32-row pass streams the item's
// weights (128 channels x 17/32 byte per k = 68) and each row its activations (1 byte per k), in both GEMMs.
SGL_DEVICE int item_cost(int c) {
  return 68 * ((c + 31) / 32) + c;
}

// Written once per CTA by warp 0: per expert its first sorted slot, its row count and the start of its items
// in item_cost units; cstart[num_experts] = total cost.
struct Meta {
  int32_t off[kMaxExperts];
  int32_t cnt[kMaxExperts];
  int32_t cstart[kMaxExperts + 1];
};
constexpr size_t kMetaBytes = (sizeof(Meta) + 15) / 16 * 16;

template <bool kGateUp>
struct Layout {
  using S = Shape<kGateUp>;
  static constexpr int kConsumerWarps = 4;  // per pipeline: one consumer warpgroup
  static constexpr int kPipes = 2;
  static constexpr int kThreads = 2 * 128 + 2 * 32;                  // 2 consumer warpgroups + 2 producer warps
  static constexpr int kStepW = kConsumerWarps * S::kTiles * 512;    // weight bytes per k64 step
  static constexpr int kStepS = kConsumerWarps * 8 * 4 * S::kTiles;  // scale bytes per k64 step
  static constexpr int kFullStage = 2 * (kStepW + kStepS);
  static constexpr int kLastStage = S::kLastSteps * (kStepW + kStepS);
  static constexpr size_t kItemBytes = static_cast<size_t>(S::kStages - 1) * kFullStage + kLastStage;
  static constexpr int kXSlot = 32 * 128;  // 32 rows x one 128-byte swizzle atom
  static constexpr size_t kXRing = static_cast<size_t>(S::kRing) * kXSlot;
  static constexpr size_t kWRing = static_cast<size_t>(S::kRing) * kFullStage;
  static constexpr size_t kBars = static_cast<size_t>(S::kRing) * 2 * 8;
  static constexpr size_t kPipeBytes = kXRing + kWRing + kBars;  // multiple of 1024 kept by kXRing first
  static constexpr size_t kPipeStride = (kPipeBytes + 1023) / 1024 * 1024;
  static constexpr size_t kSmem = 1024 + kPipes * kPipeStride + kMetaBytes;
  static_assert(kFullStage % 16 == 0 && kLastStage % 16 == 0 && kItemBytes % 16 == 0, "bulk copy sizes");
};

// ---------------------------------------------------------------------------------------------
SGL_DEVICE uint32_t smem_u32(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

SGL_DEVICE void mbar_init(uint32_t bar, uint32_t count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(bar), "r"(count));
}
SGL_DEVICE void mbar_arrive(uint32_t bar) {
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(bar) : "memory");
}
SGL_DEVICE void mbar_arrive_expect_tx(uint32_t bar, uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(bar), "r"(bytes) : "memory");
}
SGL_DEVICE void mbar_wait(uint32_t bar, uint32_t parity) {
  asm volatile(
      "{\n"
      ".reg .pred p;\n"
      "WAIT_%=:\n"
      "mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n"
      "@!p bra WAIT_%=;\n"
      "}\n" ::"r"(bar),
      "r"(parity)
      : "memory");
}
SGL_DEVICE void bulk_copy(uint32_t dst, const void* src, uint32_t bytes, uint32_t bar) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];" ::"r"(dst),
               "l"(src),
               "r"(bytes),
               "r"(bar)
               : "memory");
}
SGL_DEVICE void cp_async16(uint32_t dst, const void* src) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(dst), "l"(src) : "memory");
}
SGL_DEVICE void cp_async_arrive_noinc(uint32_t bar) {
  asm volatile("cp.async.mbarrier.arrive.noinc.shared::cta.b64 [%0];" ::"r"(bar) : "memory");
}
SGL_DEVICE void fence_proxy_async() {
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
}
SGL_DEVICE void consumer_sync() {
  asm volatile("bar.sync 1, 256;" ::: "memory");
}

SGL_DEVICE void expand_e4m3(uint32_t q, uint32_t& lo, uint32_t& hi) {
  lo = q & 0x9C9C9C9Cu;
  hi = ((q << 2) & 0x8C8C8C8Cu) | ((q >> 2) & 0x10101010u);
}
template <int kByte>
SGL_DEVICE float scale_f32(uint32_t w) {  // fp32 2^(byte - 127); byte = e8m0 + 6
  return __uint_as_float(__byte_perm(w, 0u, 0x4440u | kByte) << 23);
}
SGL_DEVICE float bf16_round(float v) {
  return __bfloat162float(__float2bfloat16_rn(v));
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
template <int N>
SGL_DEVICE void wgmma_wait() {
  asm volatile("wgmma.wait_group.sync.aligned %0;" ::"n"(N) : "memory");
}
template <int N>
SGL_DEVICE void fence_operands(float (&d)[N]) {
#pragma unroll
  for (int i = 0; i < N; ++i)
    asm volatile("" : "+f"(d[i])::"memory");
}

// D[N/2] = A(regs) x B(desc), D overwritten (scale-d = 0).
template <int N>
SGL_DEVICE void wgmma_e4m3(float (&d)[N / 2], const uint32_t (&a)[4], uint64_t desc);
#define DSV4_WS_HEAD(N, ZERO) \
  "{\n.reg .pred p;\nsetp.ne.b32 p, " ZERO ", 0;\nwgmma.mma_async.sync.aligned.m64n" #N "k32.f32.e4m3.e4m3 "
template <>
SGL_DEVICE void wgmma_e4m3<8>(float (&d)[4], const uint32_t (&a)[4], uint64_t desc) {
  asm volatile(DSV4_WS_HEAD(8, "%9") "{%0, %1, %2, %3}, {%4, %5, %6, %7}, %8, p, 1, 1;\n}\n"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc), "r"(0));
}
template <>
SGL_DEVICE void wgmma_e4m3<16>(float (&d)[8], const uint32_t (&a)[4], uint64_t desc) {
  asm volatile(DSV4_WS_HEAD(16, "%13") "{%0, %1, %2, %3, %4, %5, %6, %7}, {%8, %9, %10, %11}, %12, p, 1, 1;\n}\n"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc), "r"(0));
}
template <>
SGL_DEVICE void wgmma_e4m3<24>(float (&d)[12], const uint32_t (&a)[4], uint64_t desc) {
  asm volatile(
      DSV4_WS_HEAD(
          24, "%17") "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11}, {%12, %13, %14, %15}, %16, p, 1, 1;\n}\n"
      : "+f"(d[0]),
        "+f"(d[1]),
        "+f"(d[2]),
        "+f"(d[3]),
        "+f"(d[4]),
        "+f"(d[5]),
        "+f"(d[6]),
        "+f"(d[7]),
        "+f"(d[8]),
        "+f"(d[9]),
        "+f"(d[10]),
        "+f"(d[11])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc), "r"(0));
}
template <>
SGL_DEVICE void wgmma_e4m3<32>(float (&d)[16], const uint32_t (&a)[4], uint64_t desc) {
  asm volatile(DSV4_WS_HEAD(32, "%21")
               "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, {%16, %17, %18, %19}, %20, p, "
               "1, 1;\n}\n"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]),
                 "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc), "r"(0));
}
#undef DSV4_WS_HEAD

// ---------------------------------------------------------------------------------------------
template <bool kGateUp>
struct Smem {       // one pipeline's ring
  uint32_t x_ring;  // shared addresses
  uint32_t w_ring;
  uint32_t full;         // kRing barriers
  uint32_t empty;        // kRing barriers
  const uint8_t* w_ptr;  // generic pointer of the weight ring
  const Meta* meta;
};

// Broadcast from lane 0 so the compiler knows the value is warp-uniform: wgmma in a path it considers
// divergent is serialized (ptxas C7520).
SGL_DEVICE int uniform(int v) {
  return __shfl_sync(0xffffffffu, v, 0);
}

// The (expert, channel block) items of one pipeline. Items are ordered by (expert, block) and each costs
// item_cost(rows of its expert); pipeline q of nq takes the items whose cost start lies in
// [q * total / nq, (q + 1) * total / nq): a contiguous run of about equal streamed bytes. (Round-robin items
// left the busiest pipeline 1.1-1.3x above the mean at bs 88-160, and the kernel ends with it.) Producer and
// consumer of a pipeline walk the same sequence.
template <int kNB>
struct ItemCursor {
  const Meta* m;
  int num_experts, e, j, hi;

  SGL_DEVICE ItemCursor(const Meta* meta, int experts, int q, int nq)
      : m(meta), num_experts(experts), e(experts), j(0) {
    const int total = uniform(m->cstart[experts]);
    const int lo = static_cast<int>(static_cast<long long>(q) * total / nq);
    hi = static_cast<int>(static_cast<long long>(q + 1) * total / nq);
    if (lo >= hi) return;
    int a = 0, b = experts - 1;  // last expert with cstart <= lo; it has rows since cstart[a + 1] > lo
    while (a < b) {
      const int mid = (a + b + 1) / 2;
      if (uniform(m->cstart[mid]) <= lo) {
        a = mid;
      } else {
        b = mid - 1;
      }
    }
    const int cost = item_cost(uniform(m->cnt[a]));
    e = a;
    j = (lo - uniform(m->cstart[a]) + cost - 1) / cost;  // first block starting at or after lo
  }

  SGL_DEVICE bool next(int& item_e, int& item_j, int& cnt, int& off) {
    while (e < num_experts) {
      const int c = uniform(m->cnt[e]);
      if (j >= kNB || c == 0) {
        ++e;
        j = 0;
        continue;
      }
      if (uniform(m->cstart[e]) + j * item_cost(c) >= hi) break;
      item_e = e;
      item_j = j++;
      cnt = c;
      off = uniform(m->off[e]);
      return true;
    }
    e = num_experts;
    return false;
  }
};

template <bool kGateUp>
SGL_DEVICE void producer(const Params& p, const Smem<kGateUp>& sm, int pipe, int lane) {
  using S = Shape<kGateUp>;
  using L = Layout<kGateUp>;
  ItemCursor<S::kNB> items(sm.meta, p.num_experts, blockIdx.x * 2 + pipe, gridDim.x * 2);
  uint32_t g = 0;
  int e, j, cnt, off;
  while (items.next(e, j, cnt, off)) {
    const uint8_t* item = p.w + (static_cast<size_t>(e) * S::kNB + j) * L::kItemBytes;
    for (int r0 = 0; r0 < cnt; r0 += 32) {
      const int nrows = min(cnt - r0, 32);
      int src = -1;
      if (lane < nrows) src = kGateUp ? p.gather[off + r0 + lane] : off + r0 + lane;
      const uint8_t* xrow = src >= 0 ? p.x8 + static_cast<size_t>(src) * S::kK : nullptr;
      for (int st = 0; st < S::kStages; ++st, ++g) {
        const uint32_t slot = g % S::kRing;
        if (g >= static_cast<uint32_t>(S::kRing)) mbar_wait(sm.empty + slot * 8, ((g / S::kRing) - 1) & 1);
        const bool last = st == S::kStages - 1;
        const uint32_t bytes = last ? L::kLastStage : L::kFullStage;
        const uint32_t full = sm.full + slot * 8;
        if (lane == 0) {
          mbar_arrive_expect_tx(full, bytes);
          bulk_copy(sm.w_ring + slot * L::kFullStage, item + static_cast<size_t>(st) * L::kFullStage, bytes, full);
        }
        if (xrow != nullptr) {
          const int units = (last && !kGateUp) ? (S::kK - st * 128) / 16 : 8;  // down: 576 = 4.5 atoms
          const uint32_t xdst = sm.x_ring + slot * L::kXSlot + lane * 128;
          for (int u = 0; u < units; ++u)
            cp_async16(xdst + ((u ^ (lane & 7)) * 16), xrow + st * 128 + u * 16);
        }
        cp_async_arrive_noinc(full);
      }
    }
  }
}

// Per-pass consumer state. MX blocks are issued in batches of kBatch (4 = the two k64 steps of a stage, or 2
// = one step): convert every block of the batch, issue their wgmmas, wait for all of them, then promote each
// block's fp32 result with its scale. ptxas serializes every wgmma of the kernel if accumulator registers are
// read before wgmma.wait_group 0 (C7514), so promotion waits for the whole batch; the other consumer
// warpgroup of the SM overlaps it. kBatch = 4 holds 4 x 2 x N/2 fp32 temporaries, which fit the 168-register
// budget (10 warps per SM) only for N <= 16.
template <int kT, int kN, int kBatch>
struct PassState {
  static_assert(kBatch == 2 || kBatch == 4, "batch = one step or one stage");
  float acc[kT][kN / 2];
  float tmp[kBatch][kT][kN / 2];
  uint32_t a[kBatch][kT][4];
  uint32_t pend_sw[kBatch][kT];

  template <int kB>  // buffer of the batch; its MX block inside the k64 step is kB & 1
  SGL_DEVICE void convert(const uint4 (&wt)[kT], const uint32_t (&swt)[kT]) {
#pragma unroll
    for (int t = 0; t < kT; ++t) {
      expand_e4m3((kB & 1) == 0 ? wt[t].x : wt[t].z, a[kB][t][0], a[kB][t][2]);
      expand_e4m3((kB & 1) == 0 ? wt[t].y : wt[t].w, a[kB][t][1], a[kB][t][3]);
      pend_sw[kB][t] = swt[t];
    }
  }
  template <int kB>
  SGL_DEVICE void mma(uint32_t xslot, int step) {
    const uint64_t desc = make_desc(xslot + step * 64 + (kB & 1) * 32);
#pragma unroll
    for (int t = 0; t < kT; ++t)
      wgmma_e4m3<kN>(tmp[kB][t], a[kB][t], desc);
  }
  template <int kB>
  SGL_DEVICE void promote() {
#pragma unroll
    for (int t = 0; t < kT; ++t) {
      const float f_lo = (kB & 1) == 0 ? scale_f32<0>(pend_sw[kB][t]) : scale_f32<2>(pend_sw[kB][t]);
      const float f_hi = (kB & 1) == 0 ? scale_f32<1>(pend_sw[kB][t]) : scale_f32<3>(pend_sw[kB][t]);
#pragma unroll
      for (int jj = 0; jj < kN / 8; ++jj) {
        acc[t][4 * jj + 0] = fmaf(tmp[kB][t][4 * jj + 0], f_lo, acc[t][4 * jj + 0]);
        acc[t][4 * jj + 1] = fmaf(tmp[kB][t][4 * jj + 1], f_lo, acc[t][4 * jj + 1]);
        acc[t][4 * jj + 2] = fmaf(tmp[kB][t][4 * jj + 2], f_hi, acc[t][4 * jj + 2]);
        acc[t][4 * jj + 3] = fmaf(tmp[kB][t][4 * jj + 3], f_hi, acc[t][4 * jj + 3]);
      }
    }
  }
  // Pin the converted A fragments before wgmma.fence: otherwise ptxas sinks part of the conversion below the
  // fence, sees A registers defined inside the wgmma batch and splits it with injected warpgroup.arrive (C7519).
  SGL_DEVICE void fence_a() {
#pragma unroll
    for (int b = 0; b < kBatch; ++b)
#pragma unroll
      for (int t = 0; t < kT; ++t)
#pragma unroll
        for (int i = 0; i < 4; ++i)
          asm volatile("" : "+r"(a[b][t][i])::"memory");
  }
  SGL_DEVICE void fence_tmp() {
#pragma unroll
    for (int b = 0; b < kBatch; ++b)
#pragma unroll
      for (int t = 0; t < kT; ++t)
        fence_operands(tmp[b][t]);
  }
};

template <bool kGateUp, int kNch>
SGL_DEVICE void consumer_pass(
    const Params& p, const Smem<kGateUp>& sm, uint32_t& g, int e, int j, int off, int r0, int nrows, int cw, int lane) {
  using S = Shape<kGateUp>;
  using L = Layout<kGateUp>;
  constexpr int kN = 8 * kNch;
  constexpr int kT = S::kTiles;
  const int q = cw;  // warp q of the consumer warpgroup owns rows 16q..16q+15 of each m64 tile
  const int grp = lane / 4;
  const int c = lane % 4;

  constexpr int kBatch = kN <= 16 ? 4 : 2;
  PassState<kT, kN, kBatch> ps;
#pragma unroll
  for (int t = 0; t < kT; ++t)
#pragma unroll
    for (int i = 0; i < kN / 2; ++i)
      ps.acc[t][i] = 0.f;

  // The epilogue's per-row inputs (activation row r = 8jj + 2c + e2): scale and destination row. Loaded
  // before the K loop (their global-memory latency then overlaps it) except on the N = 32 path, which has no
  // registers to spare.
  constexpr bool kPrefetch = kNch <= 2;
  auto row_inputs = [&](int r, float& scale, int& dst) {
    const int sorted = off + r0 + r;
    if constexpr (kGateUp) {
      scale = p.xscale[p.gather[sorted]];
      dst = sorted;
    } else {
      scale = p.xscale[sorted] * p.gammas[sorted];
      dst = p.combine[sorted];
    }
  };
  float row_scale[kNch][2];
  int row_dst[kNch][2];
  if constexpr (kPrefetch) {
#pragma unroll
    for (int jj = 0; jj < kNch; ++jj)
#pragma unroll
      for (int e2 = 0; e2 < 2; ++e2) {
        const int r = 8 * jj + 2 * c + e2;
        row_inputs(r < nrows ? r : 0, row_scale[jj][e2], row_dst[jj][e2]);
      }
  }

#pragma unroll 1
  for (int st = 0; st < S::kStages; ++st, ++g) {
    const uint32_t slot = g % S::kRing;
    mbar_wait(sm.full + slot * 8, (g / S::kRing) & 1);
    fence_proxy_async();  // cp.async (generic proxy) writes of the X atom -> wgmma (async proxy) reads
    const bool last_stage = st == S::kStages - 1;
    const int nsteps = last_stage ? S::kLastSteps : 2;
    const uint8_t* wslot = sm.w_ptr + static_cast<size_t>(slot) * L::kFullStage;
    const uint8_t* sslot = wslot + nsteps * L::kStepW;
    const uint32_t xslot = sm.x_ring + slot * L::kXSlot;
    uint4 wt[2][kT];
    uint32_t swt[2][kT];
#pragma unroll
    for (int step = 0; step < 2; ++step) {
      if (step < nsteps) {
#pragma unroll
        for (int t = 0; t < kT; ++t)
          wt[step][t] =
              *reinterpret_cast<const uint4*>(wslot + ((step * L::kConsumerWarps + cw) * kT + t) * 512 + lane * 16);
        const uint8_t* srec = sslot + (step * L::kConsumerWarps + cw) * (32 * kT) + grp * (4 * kT);
        if constexpr (kT == 1) {
          swt[step][0] = *reinterpret_cast<const uint32_t*>(srec);
        } else {
          const uint2 v = *reinterpret_cast<const uint2*>(srec);
          swt[step][0] = v.x;
          swt[step][1] = v.y;
        }
      }
    }
    if constexpr (kBatch == 4) {
      if (nsteps == 2) {
        ps.template convert<0>(wt[0], swt[0]);
        ps.template convert<1>(wt[0], swt[0]);
        ps.template convert<2>(wt[1], swt[1]);
        ps.template convert<3>(wt[1], swt[1]);
        ps.fence_a();
        wgmma_fence();
        ps.template mma<0>(xslot, 0);
        ps.template mma<1>(xslot, 0);
        ps.template mma<2>(xslot, 1);
        ps.template mma<3>(xslot, 1);
        wgmma_commit();
        wgmma_wait<0>();
        ps.fence_tmp();
        ps.template promote<0>();
        ps.template promote<1>();
        ps.template promote<2>();
        ps.template promote<3>();
      } else {
        ps.template convert<0>(wt[0], swt[0]);
        ps.template convert<1>(wt[0], swt[0]);
        ps.fence_a();
        wgmma_fence();
        ps.template mma<0>(xslot, 0);
        ps.template mma<1>(xslot, 0);
        wgmma_commit();
        wgmma_wait<0>();
        ps.fence_tmp();
        ps.template promote<0>();
        ps.template promote<1>();
      }
    } else {
#pragma unroll
      for (int step = 0; step < 2; ++step) {
        if (step < nsteps) {
          ps.template convert<0>(wt[step], swt[step]);
          ps.template convert<1>(wt[step], swt[step]);
          ps.fence_a();
          wgmma_fence();
          ps.template mma<0>(xslot, step);
          ps.template mma<1>(xslot, step);
          wgmma_commit();
          wgmma_wait<0>();
          ps.fence_tmp();
          ps.template promote<0>();
          ps.template promote<1>();
        }
      }
    }
    mbar_arrive(sm.empty + slot * 8);  // every consumer thread: this slot's wgmmas are done
  }
  auto& acc = ps.acc;

  // Epilogue: acc[t][4jj + 2i + e2] = D[row 16q + grp + 8i of tile t][activation row 8jj + 2c + e2].
  if constexpr (kGateUp) {
#pragma unroll
    for (int jj = 0; jj < kNch; ++jj)
#pragma unroll
      for (int e2 = 0; e2 < 2; ++e2) {
        const int r = 8 * jj + 2 * c + e2;
        if (r < nrows) {
          if constexpr (!kPrefetch) row_inputs(r, row_scale[jj][e2], row_dst[jj][e2]);
          const float rs = row_scale[jj][e2];
          __nv_bfloat16* hrow = p.out + static_cast<size_t>(row_dst[jj][e2]) * S::kOutLd;
#pragma unroll
          for (int ih = 0; ih < 2; ++ih) {
            const int idx = 4 * jj + 2 * ih + e2;
            float gate = bf16_round(acc[0][idx] * rs);
            float up = bf16_round(acc[1][idx] * rs);
            const float lim = p.limit;
            gate = gate > lim ? lim : gate;
            up = up > lim ? lim : (up < -lim ? -lim : up);
            const float act = bf16_round(gate / (1.0f + expf(-gate)));
            hrow[64 * j + 16 * q + grp + 8 * ih] = __float2bfloat16_rn(act * up);
          }
        }
      }
  } else {
#pragma unroll
    for (int jj = 0; jj < kNch; ++jj)
#pragma unroll
      for (int e2 = 0; e2 < 2; ++e2) {
        const int r = 8 * jj + 2 * c + e2;
        if (r < nrows) {
          if constexpr (!kPrefetch) row_inputs(r, row_scale[jj][e2], row_dst[jj][e2]);
          const float sc = row_scale[jj][e2];
          __nv_bfloat16* orow = p.out + static_cast<size_t>(row_dst[jj][e2]) * S::kOutLd;
#pragma unroll
          for (int t = 0; t < kT; ++t)
#pragma unroll
            for (int ih = 0; ih < 2; ++ih) {
              const int chan = 128 * j + 64 * t + 16 * q + grp + 8 * ih;
              orow[chan] = __float2bfloat16_rn(acc[t][4 * jj + 2 * ih + e2] * sc);
            }
        }
      }
  }
}

template <bool kGateUp>
SGL_DEVICE void consumer(const Params& p, const Smem<kGateUp>& sm, int pipe, int cw, int lane) {
  using S = Shape<kGateUp>;
  ItemCursor<S::kNB> items(sm.meta, p.num_experts, blockIdx.x * 2 + pipe, gridDim.x * 2);
  uint32_t g = 0;
  int e, j, cnt, off;
  while (items.next(e, j, cnt, off)) {
    for (int r0 = 0; r0 < cnt; r0 += 32) {
      const int nrows = min(cnt - r0, 32);
      // Two code paths (N = 16 in 4-block batches, N = 32 in 2-block batches). An N = 8 path for <= 8 rows
      // measured no faster: the consumer's per-stage cost is latency, not the N-proportional promotion work.
      if (nrows <= 16) {
        consumer_pass<kGateUp, 2>(p, sm, g, e, j, off, r0, nrows, cw, lane);
      } else {
        consumer_pass<kGateUp, 4>(p, sm, g, e, j, off, r0, nrows, cw, lane);
      }
    }
  }
}

template <bool kGateUp>
__global__ void __launch_bounds__(Layout<kGateUp>::kThreads, 1) moe_ws_kernel(const __grid_constant__ Params p) {
  using S = Shape<kGateUp>;
  using L = Layout<kGateUp>;
  extern __shared__ __align__(1024) uint8_t smem_raw[];
  const uint32_t raw = smem_u32(smem_raw);
  const uint32_t base = (raw + 1023u) & ~1023u;
  uint8_t* basep = smem_raw + (base - raw);
  Meta* meta = reinterpret_cast<Meta*>(basep + L::kPipes * L::kPipeStride);

  const int tid = threadIdx.x;
  const int warp = uniform(tid / 32);  // warp-uniform role for the compiler (see uniform())
  const int lane = tid % 32;
  const bool is_producer = warp >= 8;  // warps 0-3, 4-7: consumer warpgroups; 8, 9: their producers
  const int pipe = is_producer ? warp - 8 : warp / 4;
  Smem<kGateUp> sm;
  const uint32_t pbase = base + pipe * L::kPipeStride;
  sm.x_ring = pbase;
  sm.w_ring = pbase + L::kXRing;
  sm.w_ptr = basep + pipe * L::kPipeStride + L::kXRing;
  sm.full = pbase + L::kXRing + L::kWRing;
  sm.empty = sm.full + S::kRing * 8;
  sm.meta = meta;

  if (warp == 0) {  // exclusive prefixes of row counts (sorted offsets) and of item costs
    int rows = 0, cost = 0;
    for (int e0 = 0; e0 < p.num_experts; e0 += 32) {
      const int e = e0 + lane;
      const int c = e < p.num_experts ? p.counts[e] : 0;
      const int k = c > 0 ? S::kNB * item_cost(c) : 0;
      int ic = c, ik = k;
#pragma unroll
      for (int o = 1; o < 32; o <<= 1) {
        const int vc = __shfl_up_sync(0xffffffffu, ic, o);
        const int vk = __shfl_up_sync(0xffffffffu, ik, o);
        if (lane >= o) {
          ic += vc;
          ik += vk;
        }
      }
      if (e < p.num_experts) {
        meta->off[e] = rows + ic - c;
        meta->cnt[e] = c;
        meta->cstart[e] = cost + ik - k;
      }
      rows += __shfl_sync(0xffffffffu, ic, 31);
      cost += __shfl_sync(0xffffffffu, ik, 31);
    }
    if (lane == 0) meta->cstart[p.num_experts] = cost;
  }
  if (tid == 0) {
    for (int k = 0; k < L::kPipes; ++k) {
      const uint32_t fb = base + k * L::kPipeStride + L::kXRing + L::kWRing;
      for (int i = 0; i < S::kRing; ++i) {
        mbar_init(fb + i * 8, 33);                                   // expect_tx + 32 cp.async arrivals
        mbar_init(fb + (S::kRing + i) * 8, 32 * L::kConsumerWarps);  // every consumer thread arrives
      }
    }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  __syncthreads();
  if (is_producer) {
    producer<kGateUp>(p, sm, pipe, lane);
  } else {
    consumer<kGateUp>(p, sm, pipe, warp % 4, lane);
  }
}

template <bool kGateUp>
void launch(const Params& prm, DLDevice device) {
  using namespace host;
  using L = Layout<kGateUp>;
  auto kernel = moe_ws_kernel<kGateUp>;
  RuntimeDeviceCheck(
      cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(L::kSmem)));
  int sms = 0;
  RuntimeDeviceCheck(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device.device_id));
  const int pairs = (prm.num_experts * Shape<kGateUp>::kNB + 1) / 2;  // two item streams per CTA
  LaunchKernel(dim3(sms < pairs ? sms : pairs), dim3(L::kThreads), device, L::kSmem)(kernel, prm);
}

// gate/up: x8 [tokens, 5120] e4m3 + xscale [tokens]; h [slots, 576] bf16 in sorted-slot order.
inline void gate_up(
    tvm::ffi::TensorView x8,
    tvm::ffi::TensorView xscale,
    tvm::ffi::TensorView w,
    tvm::ffi::TensorView counts,
    tvm::ffi::TensorView gather,
    tvm::ffi::TensorView h,
    double limit) {
  using namespace host;
  RuntimeCheck(x8.size(1) == 5120 && h.size(1) == 576, "ws gate_up: TP4 DeepSeek-V4 shapes only");
  Params prm{};
  prm.x8 = static_cast<const uint8_t*>(x8.data_ptr());
  prm.xscale = static_cast<const float*>(xscale.data_ptr());
  prm.w = static_cast<const uint8_t*>(w.data_ptr());
  prm.counts = static_cast<const int32_t*>(counts.data_ptr());
  prm.gather = static_cast<const int32_t*>(gather.data_ptr());
  prm.out = static_cast<__nv_bfloat16*>(h.data_ptr());
  prm.num_experts = static_cast<int>(counts.size(0));
  prm.limit = static_cast<float>(limit);
  RuntimeCheck(prm.num_experts <= 512, "ws: at most 512 local experts");
  launch<true>(prm, x8.device());
}

// down: h8 [slots, 576] e4m3 (sorted order) + hscale [slots]; y [slots, 5120] bf16 at original slots.
inline void down(
    tvm::ffi::TensorView h8,
    tvm::ffi::TensorView hscale,
    tvm::ffi::TensorView w,
    tvm::ffi::TensorView counts,
    tvm::ffi::TensorView combine,
    tvm::ffi::TensorView gammas,
    tvm::ffi::TensorView y) {
  using namespace host;
  RuntimeCheck(h8.size(1) == 576 && y.size(1) == 5120, "ws down: TP4 DeepSeek-V4 shapes only");
  Params prm{};
  prm.x8 = static_cast<const uint8_t*>(h8.data_ptr());
  prm.xscale = static_cast<const float*>(hscale.data_ptr());
  prm.w = static_cast<const uint8_t*>(w.data_ptr());
  prm.counts = static_cast<const int32_t*>(counts.data_ptr());
  prm.combine = static_cast<const int32_t*>(combine.data_ptr());
  prm.gammas = static_cast<const float*>(gammas.data_ptr());
  prm.out = static_cast<__nv_bfloat16*>(y.data_ptr());
  prm.num_experts = static_cast<int>(counts.size(0));
  RuntimeCheck(prm.num_experts <= 512, "ws: at most 512 local experts");
  launch<false>(prm, h8.device());
}

// Stable sort of routed slots by expert in one CTA (decode-sized batches): counts[E], and for the
// sum(counts) valid slots in (expert, slot) order: sorted_slots, sorted_tokens (= slot / top_k), sorted_weights.
// Masked slots (id < 0) belong to no expert and are left out. Warp w owns a contiguous run of slots; pass 1
// counts per (warp, expert) and records each slot's rank in its warp, pass 2 turns the counts into output
// offsets, pass 3 writes every slot at offset + rank. Shared memory: hist [32][E | 1] + rank [slots] (uint16).
constexpr int kSortThreads = 1024;
constexpr int kSortWarps = kSortThreads / 32;
constexpr int kSortMaxSlots = 16384;

__global__ void __launch_bounds__(kSortThreads, 1) sort_kernel(
    const int32_t* __restrict__ ids,
    const float* __restrict__ weights,
    int num_slots,
    int top_k,
    int num_experts,
    int32_t* __restrict__ counts,
    int32_t* __restrict__ sorted_slots,
    int32_t* __restrict__ sorted_tokens,
    float* __restrict__ sorted_weights) {
  constexpr int kChunks = kSortMaxSlots / kSortThreads;  // 32-slot chunks per warp
  extern __shared__ int32_t sort_smem[];
  const int ld = num_experts | 1;  // odd row stride: the per-expert scan over warps is bank-conflict free
  int32_t* hist = sort_smem;       // [kSortWarps][ld]
  uint16_t* rank = reinterpret_cast<uint16_t*>(hist + kSortWarps * ld);
  __shared__ int32_t offset[1024];
  __shared__ int32_t warp_sums[kSortWarps];
  const int tid = threadIdx.x;
  const int warp = tid / 32;
  const int lane = tid % 32;
  const int per_warp = (num_slots + kSortWarps * 32 - 1) / (kSortWarps * 32) * 32;
  const int begin = warp * per_warp;
  const int end = min(num_slots, begin + per_warp);
  // All of this warp's ids at once (independent loads), then the histogram in slot order.
  int e_of[kChunks];
#pragma unroll
  for (int c = 0; c < kChunks; ++c) {
    const int s = begin + 32 * c + lane;
    e_of[c] = s < end ? ids[s] : -1;
  }
  for (int i = tid; i < kSortWarps * ld; i += kSortThreads)
    hist[i] = 0;
  __syncthreads();
  int32_t* my_hist = hist + warp * ld;
  const unsigned lt = (1u << lane) - 1u;
#pragma unroll
  for (int c = 0; c < kChunks; ++c) {
    if (begin + 32 * c >= end) break;  // warp-uniform
    const int e = e_of[c];
    const unsigned peers = __match_any_sync(0xffffffffu, e);
    const int leader = __ffs(peers) - 1;
    int before = 0;
    if (lane == leader && e >= 0) {
      before = my_hist[e];
      my_hist[e] = before + __popc(peers);
    }
    before = __shfl_sync(0xffffffffu, before, leader);
    if (e >= 0) rank[begin + 32 * c + lane] = static_cast<uint16_t>(before + __popc(peers & lt));
  }
  __syncthreads();
  // hist[w][e] <- slots of expert e in warps before w (scan over warps, one expert per warp step).
  for (int e = warp; e < num_experts; e += kSortWarps) {
    const int c = hist[lane * ld + e];
    int incl = c;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
      const int v = __shfl_up_sync(0xffffffffu, incl, o);
      if (lane >= o) incl += v;
    }
    hist[lane * ld + e] = incl - c;
    if (lane == 31) {
      offset[e] = incl;  // count of expert e, for now
      counts[e] = incl;
    }
  }
  __syncthreads();
  const int count = tid < num_experts ? offset[tid] : 0;
  int incl = count;  // block-wide exclusive scan of the counts
#pragma unroll
  for (int o = 1; o < 32; o <<= 1) {
    const int v = __shfl_up_sync(0xffffffffu, incl, o);
    if (lane >= o) incl += v;
  }
  if (lane == 31) warp_sums[warp] = incl;
  __syncthreads();
  if (warp == 0) {
    int ws = warp_sums[lane];
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
      const int v = __shfl_up_sync(0xffffffffu, ws, o);
      if (lane >= o) ws += v;
    }
    warp_sums[lane] = ws;
  }
  __syncthreads();
  if (tid < num_experts) offset[tid] = incl - count + (warp > 0 ? warp_sums[warp - 1] : 0);
  __syncthreads();
#pragma unroll
  for (int c = 0; c < kChunks; ++c) {
    const int s = begin + 32 * c + lane;
    if (begin + 32 * c >= end) break;  // warp-uniform
    const int e = e_of[c];
    if (e < 0) continue;
    const int pos = offset[e] + my_hist[e] + rank[s];
    sorted_slots[pos] = s;
    sorted_tokens[pos] = s / top_k;
    sorted_weights[pos] = weights[s];
  }
}

inline void sort_slots(
    tvm::ffi::TensorView topk_ids,
    tvm::ffi::TensorView topk_weights,
    tvm::ffi::TensorView counts,
    tvm::ffi::TensorView sorted_slots,
    tvm::ffi::TensorView sorted_tokens,
    tvm::ffi::TensorView sorted_weights) {
  using namespace host;
  const int tokens = static_cast<int>(topk_ids.size(0));
  const int top_k = static_cast<int>(topk_ids.size(1));
  const int num_slots = tokens * top_k;
  const int num_experts = static_cast<int>(counts.size(0));
  RuntimeCheck(num_slots <= kSortMaxSlots && num_experts <= 1024, "ws sort: at most 16384 slots, 1024 experts");
  const size_t smem = static_cast<size_t>(kSortWarps) * (num_experts | 1) * 4 + static_cast<size_t>(num_slots) * 2;
  static const bool smem_set = [] {  // once: the largest layout (1024 experts, 16384 slots)
    RuntimeDeviceCheck(cudaFuncSetAttribute(
        sort_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSortWarps * 1025 * 4 + kSortMaxSlots * 2));
    return true;
  }();
  (void)smem_set;
  LaunchKernel(dim3(1), dim3(kSortThreads), topk_ids.device(), smem)(
      sort_kernel,
      static_cast<const int32_t*>(topk_ids.data_ptr()),
      static_cast<const float*>(topk_weights.data_ptr()),
      num_slots,
      top_k,
      num_experts,
      static_cast<int32_t*>(counts.data_ptr()),
      static_cast<int32_t*>(sorted_slots.data_ptr()),
      static_cast<int32_t*>(sorted_tokens.data_ptr()),
      static_cast<float*>(sorted_weights.data_ptr()));
}

// out[t] = sum over k with topk_ids[t, k] >= 0 of y[t * top_k + k], in fp32 with k ascending (as moe_topk_sum).
// down writes no row for a masked slot (it is not sorted to any expert), so those rows are not read.
constexpr int kCombineMaxTopK = 8;
__global__ void combine_kernel(
    const __nv_bfloat16* __restrict__ y, const int32_t* __restrict__ ids, __nv_bfloat16* __restrict__ out, int top_k) {
  constexpr int kVecs = 5120 / 8;
  const int t = blockIdx.y;
  const int v = blockIdx.x * blockDim.x + threadIdx.x;
  if (v >= kVecs) return;
  bool use[kCombineMaxTopK];  // ids first, so that the row loads below can all be in flight together
#pragma unroll
  for (int k = 0; k < kCombineMaxTopK; ++k)
    use[k] = k < top_k && ids[t * top_k + k] >= 0;
  uint4 w[kCombineMaxTopK];
#pragma unroll
  for (int k = 0; k < kCombineMaxTopK; ++k)
    if (use[k]) w[k] = reinterpret_cast<const uint4*>(y + (static_cast<size_t>(t) * top_k + k) * 5120)[v];
  float acc[8];
#pragma unroll
  for (int i = 0; i < 8; ++i)
    acc[i] = 0.f;
#pragma unroll
  for (int k = 0; k < kCombineMaxTopK; ++k) {
    if (!use[k]) continue;
    const __nv_bfloat162* b = reinterpret_cast<const __nv_bfloat162*>(&w[k]);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const float2 f = __bfloat1622float2(b[i]);
      acc[2 * i] += f.x;
      acc[2 * i + 1] += f.y;
    }
  }
  uint4 o;
  __nv_bfloat162* ob = reinterpret_cast<__nv_bfloat162*>(&o);
#pragma unroll
  for (int i = 0; i < 4; ++i)
    ob[i] = __floats2bfloat162_rn(acc[2 * i], acc[2 * i + 1]);
  reinterpret_cast<uint4*>(out + static_cast<size_t>(t) * 5120)[v] = o;
}

// y [tokens * top_k, 5120] bf16 (original slot order), topk_ids [tokens, top_k] -> out [tokens, 5120] bf16.
inline void combine(tvm::ffi::TensorView y, tvm::ffi::TensorView topk_ids, tvm::ffi::TensorView out) {
  using namespace host;
  RuntimeCheck(y.size(1) == 5120 && out.size(1) == 5120, "ws combine: hidden 5120 only");
  const int tokens = static_cast<int>(topk_ids.size(0));
  const int top_k = static_cast<int>(topk_ids.size(1));
  RuntimeCheck(y.size(0) == static_cast<int64_t>(tokens) * top_k && out.size(0) == tokens, "ws combine: shapes");
  RuntimeCheck(top_k <= kCombineMaxTopK, "ws combine: top_k <= 8");
  if (tokens == 0) return;
  LaunchKernel(dim3(5, tokens), dim3(128), y.device())(
      combine_kernel,
      static_cast<const __nv_bfloat16*>(y.data_ptr()),
      static_cast<const int32_t*>(topk_ids.data_ptr()),
      static_cast<__nv_bfloat16*>(out.data_ptr()),
      top_k);
}

}  // namespace dsv4_moe_ws
}  // namespace sglang
