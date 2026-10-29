// SPDX-License-Identifier: Apache-2.0
//
// SM90 W4A8 decode MoE, v2 (DeepSeek-V4 TP4 MXFP4 experts: hidden 5120, 576 intermediate channels per rank).
//
// One persistent kernel per MoE call runs every routed expert's gate/up GEMM, the per-row e4m3 quantization of
// its h and its down GEMM. Work items (one weight pass of up to 32 rows over one channel block) are handed out
// dynamically from one global counter: first every gate/up item, then every down item. A down item starts once
// its expert's h is quantized. Compared with dsv4_mxfp4_decode_moe_ws.cuh (#104: a gate/up kernel, a row-quant
// kernel and a down kernel, each with a static cost-balanced split) this removes two launches, the row-quant
// pass over all slots and the per-kernel tails (with the static split the slowest of the 264 pipelines ended
// 30-35 us after the median one at 1,024 rows), and it hides the per-item metadata loads.
//
// Numerics are #104's, bit for bit: the per-pass math (e2m1 -> e4m3 expansion, one wgmma per MX block, fp32
// promotion with the block scale in the same order, SwiGLU, bf16 h) is the same code; h is quantized as the
// Triton _row_quant kernel does (its scale instruction for instruction: div.full, libdevice log2f, ceil, ex2; the
// division by that power-of-2 scale as an exact multiply; cvt.rn.satfinite); down and the top-k combine are
// unchanged. Neither the order of rows inside an expert nor the order in which items run
// enters any output.
//
// CTA = 2 pipelines + 1 helper warp. A pipeline is a producer warp and a consumer warpgroup with a ring of 6
// stages. For every 128-k stage the producer copies the stage's weights (one cp.async.bulk of the load-time
// repacked bytes) and the activation rows' 128-byte k chunk (cp.async into a 128B-swizzled atom); the consumer
// converts its A fragments in registers and issues one wgmma.m64nNk32 per (MX block, tile) with B = the atom.
// The producer claims the next item while it streams the current one and passes each item's descriptor and
// per-row epilogue table through shared memory. The helper warp counts finished gate/up items per expert,
// quantizes an expert's h when its last item is done and releases the expert's ready flag, and acquires that
// flag (and fills the down epilogue table) for the producers' next down items, so that none of these global
// round trips sits on a producer's or consumer's path.
//
// Also here: the call's prologue in one launch (stable expert sort, token row quantization, workspace zeroing).

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
namespace dsv4_moe_v2 {


struct Params {
  const uint8_t* __restrict__ x8;      // e4m3 token rows [tokens, 5120]
  const float* __restrict__ xscale;    // power-of-2 scale per token row
  const uint8_t* __restrict__ w13;     // stage-ordered gate/up weights + scales [E][9][348160]
  const uint8_t* __restrict__ w2;      // stage-ordered down weights + scales [E][40][39168]
  const int32_t* __restrict__ counts;  // [E]
  const int32_t* __restrict__ gather;  // token of each sorted slot
  const int32_t* __restrict__ combine; // original slot of each sorted slot
  const float* __restrict__ gammas;    // routing weight of each sorted slot
  __nv_bfloat16* __restrict__ h;       // [slots, 576] bf16, sorted-slot order
  uint8_t* __restrict__ h8;            // [slots, 576] e4m3, sorted-slot order
  float* __restrict__ hscale;          // [slots]
  __nv_bfloat16* __restrict__ y;       // [slots, 5120] bf16, original-slot order
  int32_t* __restrict__ ws;            // zeroed per call: [0] queue head, [kDoneBase + e], [kReadyBase(E) + e]
  int num_experts;
  float limit;
};

constexpr int kDoneBase = 32;
SGL_DEVICE int ready_base(int num_experts) {
  return kDoneBase + num_experts;
}
constexpr int kWorkspaceInts(int num_experts) {
  return kDoneBase + 2 * num_experts;
}

template <bool kGateUp>
struct Shape;
template <>
struct Shape<true> {  // item = 64 intermediate channels; tile 0 = gate rows, tile 1 = up rows
  static constexpr int kK = 5120, kNB = 9, kTiles = 2, kStages = 40, kLastSteps = 2, kOutLd = 576;
};
template <>
struct Shape<false> {  // item = 128 output channels; tile t = channels 128j + 64t
  static constexpr int kK = 576, kNB = 40, kTiles = 2, kStages = 5, kLastSteps = 1, kOutLd = 5120;
};

constexpr int kMaxExperts = 512;
constexpr int kRing = 6;
constexpr int kConsumerWarps = 4;  // per pipeline: one consumer warpgroup
constexpr int kPipes = 2;
constexpr int kThreads = kPipes * 128 + kPipes * 32 + 32;  // + one helper warp
constexpr int kTilesAll = 2;
constexpr int kStepW = kConsumerWarps * kTilesAll * 512;    // weight bytes per k64 step
constexpr int kStepS = kConsumerWarps * 8 * 4 * kTilesAll;  // scale bytes per k64 step
constexpr int kFullStage = 2 * (kStepW + kStepS);
template <bool kGateUp>
constexpr int last_stage_bytes() {
  return Shape<kGateUp>::kLastSteps * (kStepW + kStepS);
}
template <bool kGateUp>
constexpr size_t item_bytes() {
  return static_cast<size_t>(Shape<kGateUp>::kStages - 1) * kFullStage + last_stage_bytes<kGateUp>();
}
static_assert(item_bytes<true>() == 348160 && item_bytes<false>() == 39168, "packed item sizes");
constexpr int kXSlot = 32 * 128;  // 32 rows x one 128-byte swizzle atom
constexpr size_t kXRing = static_cast<size_t>(kRing) * kXSlot;
constexpr size_t kWRing = static_cast<size_t>(kRing) * kFullStage;
constexpr size_t kBars = static_cast<size_t>(kRing) * 2 * 8;
constexpr size_t kPipeBytes = kXRing + kWRing + kBars;  // multiple of 1024 kept by kXRing first
constexpr size_t kPipeStride = (kPipeBytes + 1023) / 1024 * 1024;

enum ItemKind : int { kGu = 0, kDn = 1, kEnd = 2 };

// Written once per CTA by warp 0: per expert its first sorted slot, its row count and its first gate/up and down
// item (an expert has 9 resp. 40 items per 32-row pass). Plus the per-slot item headers and a flag per pipeline.
struct Meta {
  int32_t off[kMaxExperts];
  int32_t cnt[kMaxExperts];
  int32_t gstart[kMaxExperts + 1];
  int32_t dstart[kMaxExperts + 1];
  int4 hdr[kPipes][kRing];
  // Per pipeline, per item in flight (item sequence number % kRowSlots): each activation row's epilogue scale
  // and destination row, written by the producer with the item's first stage. At most three items are between
  // the producer and the consumer's epilogue (an item has >= 5 stages, the ring 6), so four slots suffice.
  float2 rowmeta[kPipes][4][32];
  // Consumer -> helper: finished gate/up items (expert id, or -1 at the end), 4 entries per pipeline.
  int32_t done_e[kPipes][4];
  // Producer -> helper: the next down item's expert and rows (e, item sequence, first sorted slot, rows).
  int4 req[kPipes];
  // mbarriers: done_full[pipe][4] (128 consumer threads), done_free[pipe][4] (helper), req_full[pipe]
  // (producer), req_ack[pipe] (helper).
  uint64_t done_full[kPipes][4];
  uint64_t done_free[kPipes][4];
  uint64_t req_full[kPipes];
  uint64_t req_ack[kPipes];
  int32_t qlist[16];  // helper: experts whose h is waiting to be quantized
};
constexpr int kRowSlots = 4;
constexpr int kDoneSlots = 4;
constexpr int kQuantJobs = 16;
constexpr size_t kMetaBytes = (sizeof(Meta) + 15) / 16 * 16;
constexpr size_t kSmem = 1024 + kPipes * kPipeStride + kMetaBytes;

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
SGL_DEVICE bool mbar_test(uint32_t bar, uint32_t parity) {
  uint32_t ok;
  asm volatile(
      "{\n"
      ".reg .pred p;\n"
      "mbarrier.test_wait.parity.shared::cta.b64 p, [%1], %2;\n"
      "selp.u32 %0, 1, 0, p;\n"
      "}\n"
      : "=r"(ok)
      : "r"(bar), "r"(parity)
      : "memory");
  return ok != 0;
}
SGL_DEVICE void fence_acq_rel_gpu() {
  asm volatile("fence.acq_rel.gpu;" ::: "memory");
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
SGL_DEVICE int ld_acquire(const int32_t* p) {
  int v;
  asm volatile("ld.acquire.gpu.global.b32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
  return v;
}
SGL_DEVICE void st_release(int32_t* p, int v) {
  asm volatile("st.release.gpu.global.b32 [%0], %1;" ::"l"(p), "r"(v) : "memory");
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
#define DSV4_V2_HEAD(N, ZERO) \
  "{\n.reg .pred p;\nsetp.ne.b32 p, " ZERO ", 0;\nwgmma.mma_async.sync.aligned.m64n" #N "k32.f32.e4m3.e4m3 "
template <>
SGL_DEVICE void wgmma_e4m3<8>(float (&d)[4], const uint32_t (&a)[4], uint64_t desc) {
  asm volatile(DSV4_V2_HEAD(8, "%9") "{%0, %1, %2, %3}, {%4, %5, %6, %7}, %8, p, 1, 1;\n}\n"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc), "r"(0));
}
template <>
SGL_DEVICE void wgmma_e4m3<16>(float (&d)[8], const uint32_t (&a)[4], uint64_t desc) {
  asm volatile(DSV4_V2_HEAD(16, "%13") "{%0, %1, %2, %3, %4, %5, %6, %7}, {%8, %9, %10, %11}, %12, p, 1, 1;\n}\n"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc), "r"(0));
}
template <>
SGL_DEVICE void wgmma_e4m3<24>(float (&d)[12], const uint32_t (&a)[4], uint64_t desc) {
  asm volatile(
      DSV4_V2_HEAD(
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
  asm volatile(DSV4_V2_HEAD(32, "%21")
               "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, {%16, %17, %18, %19}, %20, p, "
               "1, 1;\n}\n"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]),
                 "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc), "r"(0));
}
#undef DSV4_V2_HEAD

// ---------------------------------------------------------------------------------------------
struct Smem {       // one pipeline's ring
  uint32_t x_ring;  // shared addresses
  uint32_t w_ring;
  uint32_t full;         // kRing barriers
  uint32_t empty;        // kRing barriers
  const uint8_t* w_ptr;  // generic pointer of the weight ring
  Meta* meta;
};

// Broadcast from lane 0 so the compiler knows the value is warp-uniform: wgmma in a path it considers
// divergent is serialized (ptxas C7520).
SGL_DEVICE int uniform(int v) {
  return __shfl_sync(0xffffffffu, v, 0);
}

// Last expert e with start[e] <= i (it has items: start[e + 1] > i).
SGL_DEVICE int find_expert(const int32_t* start, int num_experts, int i) {
  int a = 0, b = num_experts - 1;
  while (a < b) {
    const int mid = (a + b + 1) / 2;
    if (uniform(start[mid]) <= i) {
      a = mid;
    } else {
      b = mid - 1;
    }
  }
  return a;
}

struct Item {
  int kind, e, j, r0, nrows, off;
  int pipe;      // consumer side: pipeline and item sequence number (row table slot)
  uint32_t seq;
};

// Global item index -> item. Gate/up items [0, ngu): (expert, pass, block of 9); down items after them.
SGL_DEVICE Item decode_item(const Meta* m, int num_experts, int i) {
  Item it;
  const int ngu = uniform(m->gstart[num_experts]);
  const bool gu = i < ngu;
  const int32_t* start = gu ? m->gstart : m->dstart;
  const int k = gu ? i : i - ngu;
  const int nb = gu ? Shape<true>::kNB : Shape<false>::kNB;
  it.kind = gu ? kGu : kDn;
  it.e = find_expert(start, num_experts, k);
  // Passes of one block are adjacent in the queue: pipelines that claim them stream the same weights at about
  // the same time, so the later passes mostly hit L2.
  const int local = k - uniform(start[it.e]);
  const int c = uniform(m->cnt[it.e]);
  const int passes = (c + 31) / 32;
  it.r0 = 32 * (local % passes);
  it.j = local / passes;
  it.nrows = min(c - it.r0, 32);
  it.off = uniform(m->off[it.e]);
  return it;
}

// The producer's view of one item: the item, and per lane (activation row r0 + lane) the x row it copies and,
// for gate/up, the epilogue scale and destination row (down's come from the helper).
struct Work {
  Item it;
  int src;
  float rscale;
  int rdst;
  bool scaled;  // gate/up: rscale already loaded
};

// Issue the loads that describe item index i: gate/up its x rows (the scale follows at stage 2 of the current
// item, see produce_item); down asks the helper for the expert's readiness and the rows' epilogue table.
SGL_DEVICE void start_work(
    const Params& p, const Smem& sm, int pipe, int num_experts, int ntot, int i, uint32_t seq, int lane, Work& w) {
  if (i >= ntot) {
    w.it.kind = kEnd;
    return;
  }
  w.it = decode_item(sm.meta, num_experts, i);
  w.src = -1;
  w.rscale = 0.f;
  w.rdst = 0;
  w.scaled = false;
  const int sorted = w.it.off + w.it.r0 + lane;
  if (w.it.kind == kGu) {
    if (lane < w.it.nrows) {
      w.src = p.gather[sorted];
      w.rdst = sorted;
    }
  } else {
    if (lane < w.it.nrows) w.src = sorted;
    if (lane == 0) {  // one request in flight per pipeline: the previous one was acknowledged in finish_work
      sm.meta->req[pipe] = make_int4(w.it.e, static_cast<int>(seq), w.it.off + w.it.r0, w.it.nrows);
      mbar_arrive(smem_u32(&sm.meta->req_full[pipe]));
    }
  }
}

// Before the item's first stage. Gate/up: its scale (if not loaded yet). Down: wait for the helper, which has
// acquired the expert's ready flag and written the item's epilogue table.
SGL_DEVICE void finish_work(const Params& p, const Smem& sm, int pipe, uint32_t& nreq, int lane, Work& w) {
  if (w.it.kind == kGu) {
    if (!w.scaled && lane < w.it.nrows) w.rscale = p.xscale[w.src];
    return;
  }
  mbar_wait(smem_u32(&sm.meta->req_ack[pipe]), nreq & 1);
  ++nreq;
}

// Stream one item's stages into the ring. After stage kPrefetchStage, the next item's description is started
// (its claim was issued before stage 0), so its loads overlap this item's stages.
template <bool kGateUp>
SGL_DEVICE void produce_item(
    const Params& p,
    const Smem& sm,
    int pipe,
    uint32_t& g,
    uint32_t seq,
    const Work& w,
    int lane,
    int& claim,
    int ntot,
    Work& nw) {
  using S = Shape<kGateUp>;
  const Item& it = w.it;
  const uint8_t* item =
      (kGateUp ? p.w13 : p.w2) + (static_cast<size_t>(it.e) * S::kNB + it.j) * item_bytes<kGateUp>();
  const uint8_t* xrow = w.src >= 0 ? (kGateUp ? p.x8 : p.h8) + static_cast<size_t>(w.src) * S::kK : nullptr;
  for (int st = 0; st < S::kStages; ++st, ++g) {
    const uint32_t slot = g % kRing;
    if (g >= static_cast<uint32_t>(kRing)) mbar_wait(sm.empty + slot * 8, ((g / kRing) - 1) & 1);
    const bool last = st == S::kStages - 1;
    const uint32_t bytes = last ? last_stage_bytes<kGateUp>() : kFullStage;
    const uint32_t full = sm.full + slot * 8;
    if (kGateUp && st == 0) {  // published by the arrives below, read by the consumer's epilogue
      sm.meta->rowmeta[pipe][seq % kRowSlots][lane] = make_float2(w.rscale, __int_as_float(w.rdst));
      __syncwarp();
    }
    if (lane == 0) {
      // The item's descriptor in the header of its first slot, after that slot was released (the consumer
      // reads it right after the full barrier, before it releases the slot); the arrive below publishes it.
      if (st == 0) sm.meta->hdr[pipe][slot] = make_int4(it.kind | (it.e << 8), it.j, it.r0, it.nrows);
      mbar_arrive_expect_tx(full, bytes);
      bulk_copy(sm.w_ring + slot * kFullStage, item + static_cast<size_t>(st) * kFullStage, bytes, full);
    }
    if (xrow != nullptr) {
      const int units = (last && !kGateUp) ? (S::kK - st * 128) / 16 : 8;  // down: 576 = 4.5 atoms
      const uint32_t xdst = sm.x_ring + slot * kXSlot + lane * 128;
      for (int u = 0; u < units; ++u)
        cp_async16(xdst + ((u ^ (lane & 7)) * 16), xrow + st * 128 + u * 16);
    }
    cp_async_arrive_noinc(full);
    if (st == 0) start_work(p, sm, pipe, p.num_experts, ntot, uniform(claim), seq + 1, lane, nw);
    if (st == 2 && nw.it.kind == kGu) {  // its x row index has arrived by now
      if (lane < nw.it.nrows) nw.rscale = p.xscale[nw.src];
      nw.scaled = true;
    }
  }
}


SGL_DEVICE void producer(const Params& p, const Smem& sm, int pipe, int lane) {
  const Meta* m = sm.meta;
  const int E = p.num_experts;
  const int ntot = uniform(m->gstart[E]) + uniform(m->dstart[E]);
  int32_t* queue = p.ws;
  uint32_t g = 0, seq = 0, nreq = 0;
  int claim = 0;
  if (lane == 0) claim = atomicAdd(queue, 1);
  Work w, nw;
  start_work(p, sm, pipe, E, ntot, uniform(claim), 0, lane, w);
  for (; w.it.kind != kEnd; ++seq) {
    if (lane == 0) claim = atomicAdd(queue, 1);  // the next item, claimed now; read after this item's stage 0
    finish_work(p, sm, pipe, nreq, lane, w);
    if (w.it.kind == kGu) {
      produce_item<true>(p, sm, pipe, g, seq, w, lane, claim, ntot, nw);
    } else {
      produce_item<false>(p, sm, pipe, g, seq, w, lane, claim, ntot, nw);
    }
    w = nw;
  }
  const uint32_t slot = g % kRing;  // end marker
  if (g >= static_cast<uint32_t>(kRing)) mbar_wait(sm.empty + slot * 8, ((g / kRing) - 1) & 1);
  if (lane == 0) {
    sm.meta->hdr[pipe][slot] = make_int4(kEnd, 0, 0, 0);
    mbar_arrive_expect_tx(sm.full + slot * 8, 0);
  }
  cp_async_arrive_noinc(sm.full + slot * 8);
}

// Per-pass consumer state. MX blocks are issued in batches of kBatch (4 = the two k64 steps of a stage, or 2
// = one step): convert every block of the batch, issue their wgmmas, wait for all of them, then promote each
// block's fp32 result with its scale. ptxas serializes every wgmma of the kernel if accumulator registers are
// read before wgmma.wait_group 0 (C7514), so promotion waits for the whole batch; the other consumer
// warpgroup of the SM overlaps it.
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
SGL_DEVICE void consumer_pass(const Params& p, const Smem& sm, uint32_t& g, const Item& it, int cw, int lane) {
  using S = Shape<kGateUp>;
  constexpr int kN = 8 * kNch;
  constexpr int kT = S::kTiles;
  const int q = cw;  // warp q of the consumer warpgroup owns rows 16q..16q+15 of each m64 tile
  const int grp = lane / 4;
  const int c = lane % 4;
  const int j = it.j, off = it.off, r0 = it.r0, nrows = it.nrows;

  constexpr int kBatch = kN <= 16 ? 4 : 2;
  PassState<kT, kN, kBatch> ps;
#pragma unroll
  for (int t = 0; t < kT; ++t)
#pragma unroll
    for (int i = 0; i < kN / 2; ++i)
      ps.acc[t][i] = 0.f;

  // The epilogue's per-row inputs (activation row r = 8jj + 2c + e2): scale and destination row, from the
  // item's row table (written by the producer; it stays valid until this epilogue is done).
  const float2* rowmeta = sm.meta->rowmeta[it.pipe][it.seq % kRowSlots];

#pragma unroll 1
  for (int st = 0; st < S::kStages; ++st, ++g) {
    const uint32_t slot = g % kRing;
    mbar_wait(sm.full + slot * 8, (g / kRing) & 1);
    fence_proxy_async();  // cp.async (generic proxy) writes of the X atom -> wgmma (async proxy) reads
    const bool last_stage = st == S::kStages - 1;
    const int nsteps = last_stage ? S::kLastSteps : 2;
    const uint8_t* wslot = sm.w_ptr + static_cast<size_t>(slot) * kFullStage;
    const uint8_t* sslot = wslot + nsteps * kStepW;
    const uint32_t xslot = sm.x_ring + slot * kXSlot;
    uint4 wt[2][kT];
    uint32_t swt[2][kT];
#pragma unroll
    for (int step = 0; step < 2; ++step) {
      if (step < nsteps) {
#pragma unroll
        for (int t = 0; t < kT; ++t)
          wt[step][t] = *reinterpret_cast<const uint4*>(wslot + ((step * kConsumerWarps + cw) * kT + t) * 512 + lane * 16);
        const uint8_t* srec = sslot + (step * kConsumerWarps + cw) * (32 * kT) + grp * (4 * kT);
        const uint2 v = *reinterpret_cast<const uint2*>(srec);
        swt[step][0] = v.x;
        swt[step][1] = v.y;
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
          const float2 rm = rowmeta[r];
          const float rs = rm.x;
          __nv_bfloat16* hrow = p.h + static_cast<size_t>(__float_as_int(rm.y)) * S::kOutLd;
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
          const float2 rm = rowmeta[r];
          const float sc = rm.x;
          __nv_bfloat16* orow = p.y + static_cast<size_t>(__float_as_int(rm.y)) * S::kOutLd;
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

// The Triton _row_quant_kernel's scale, instruction for instruction (div.full.f32, libdevice log2f with
// flush-to-zero, cvt.rpi, ex2.approx), so that h8 and hscale are bit-identical to #104's.
SGL_DEVICE float row_scale_of(float amax) {
  float v, s;
  amax = fmaxf(amax, 1.0e-4f);
  asm("div.full.f32 %0, %1, 0f43E00000;" : "=f"(v) : "f"(amax));  // amax / 448
  asm("{\n"
      ".reg .pred p2, p3, p6;\n"
      ".reg .f32 x, k, f, r, t, z;\n"
      ".reg .b32 i, ie, im;\n"
      "setp.lt.f32 p2, %1, 0f00800000;\n"
      "mul.f32 x, %1, 0f4B000000;\n"
      "selp.f32 x, x, %1, p2;\n"
      "selp.f32 k, 0fC1B80000, 0f00000000, p2;\n"
      "mov.b32 i, x;\n"
      "add.s32 ie, i, -1060439283;\n"
      "and.b32 ie, ie, -8388608;\n"
      "sub.s32 im, i, ie;\n"
      "cvt.rn.f32.s32 f, ie;\n"
      "fma.rn.ftz.f32 k, f, 0f34000000, k;\n"
      "mov.b32 r, im;\n"
      "add.f32 r, r, 0fBF800000;\n"
      "fma.rn.ftz.f32 t, 0f3DC6B27F, r, 0fBE2C7F30;\n"
      "fma.rn.ftz.f32 t, t, r, 0f3E2FCF2A;\n"
      "fma.rn.ftz.f32 t, t, r, 0fBE374E43;\n"
      "fma.rn.ftz.f32 t, t, r, 0f3E520BF4;\n"
      "fma.rn.ftz.f32 t, t, r, 0fBE763C8B;\n"
      "fma.rn.ftz.f32 t, t, r, 0f3E93BF99;\n"
      "fma.rn.ftz.f32 t, t, r, 0fBEB8AA49;\n"
      "fma.rn.ftz.f32 t, t, r, 0f3EF6384A;\n"
      "fma.rn.ftz.f32 t, t, r, 0fBF38AA3B;\n"
      "mul.f32 t, r, t;\n"
      "mul.f32 t, r, t;\n"
      "fma.rn.ftz.f32 t, r, 0f3FB8AA3B, t;\n"
      "add.f32 z, k, t;\n"
      "setp.lt.u32 p3, i, 2139095040;\n"
      "@!p3 fma.rn.ftz.f32 z, x, 0f7F800000, 0f7F800000;\n"
      "setp.eq.f32 p6, x, 0f00000000;\n"
      "selp.f32 z, 0fFF800000, z, p6;\n"
      "cvt.rpi.ftz.f32.f32 z, z;\n"
      "ex2.approx.ftz.f32 %0, z;\n"
      "}\n"
      : "=f"(s)
      : "f"(v));
  return s;
}
// (lo / s, hi / s) -> 2 e4m3 bytes, lo first. s is a power of two, so x / s = x * (1 / s) exactly whenever the
// quotient is a normal float (the Triton kernel's div.full gives the same); a subnormal quotient is far below the
// smallest e4m3 value and becomes a signed zero either way. inv_s = 1 / s (exact).
SGL_DEVICE uint32_t e4m3x2(float lo, float hi, float inv_s) {
  const float a = lo * inv_s, b = hi * inv_s;
  uint16_t r;
  asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(r) : "f"(b), "f"(a));
  return r;
}

// Quantize h rows [off + r0, off + min(r0 + 4, cnt)) of one expert to h8 / hscale (as _row_quant), one warp;
// the four rows' loads are in flight together.
constexpr int kQuantRows = 4;
SGL_DEVICE void quant_batch(const Params& p, int off, int cnt, int r0, int lane) {
  constexpr int kVec = 576 / 8;  // 72 uint4 of 8 bf16 per row
  constexpr int kRows = kQuantRows;
  {
    uint4 v[kRows][3];
    float amax[kRows];
#pragma unroll
    for (int i = 0; i < kRows; ++i) {
      amax[i] = 0.f;
      const int r = r0 + i;
      const uint4* src = reinterpret_cast<const uint4*>(p.h + static_cast<size_t>(off + (r < cnt ? r : 0)) * 576);
#pragma unroll
      for (int k = 0; k < 3; ++k) {
        const int idx = lane + 32 * k;
        if (idx < kVec) v[i][k] = __ldcg(src + idx);
      }
    }
#pragma unroll
    for (int i = 0; i < kRows; ++i) {
#pragma unroll
      for (int k = 0; k < 3; ++k) {
        if (lane + 32 * k < kVec) {
          const __nv_bfloat162* b = reinterpret_cast<const __nv_bfloat162*>(&v[i][k]);
#pragma unroll
          for (int t = 0; t < 4; ++t) {
            const float2 f = __bfloat1622float2(b[t]);
            amax[i] = fmaxf(amax[i], fmaxf(fabsf(f.x), fabsf(f.y)));
          }
        }
      }
#pragma unroll
      for (int o = 16; o > 0; o >>= 1)
        amax[i] = fmaxf(amax[i], __shfl_xor_sync(0xffffffffu, amax[i], o));
    }
#pragma unroll
    for (int i = 0; i < kRows; ++i) {
      const int r = r0 + i;
      if (r >= cnt) break;
      const size_t row = static_cast<size_t>(off + r);
      const float s = row_scale_of(amax[i]);
      const float inv = __frcp_rn(s);
      uint2* dst = reinterpret_cast<uint2*>(p.h8 + row * 576);
#pragma unroll
      for (int k = 0; k < 3; ++k) {
        const int idx = lane + 32 * k;
        if (idx < kVec) {
          const __nv_bfloat162* b = reinterpret_cast<const __nv_bfloat162*>(&v[i][k]);
          uint32_t w[2];
#pragma unroll
          for (int t = 0; t < 2; ++t) {
            const float2 f0 = __bfloat1622float2(b[2 * t]);
            const float2 f1 = __bfloat1622float2(b[2 * t + 1]);
            w[t] = e4m3x2(f0.x, f0.y, inv) | (e4m3x2(f1.x, f1.y, inv) << 16);
          }
          dst[idx] = make_uint2(w[0], w[1]);
        }
      }
      if (lane == 0) p.hscale[row] = s;
    }
  }
}

// After a gate/up item's h stores: hand the item to the helper (the mbarrier arrive releases this thread's
// stores at CTA scope; the helper's acq_rel fence makes them visible GPU-wide before it counts the item).
SGL_DEVICE void gate_up_done(const Smem& sm, int pipe, uint32_t& ndone, int e, int cw, int lane) {
  const uint32_t k = ndone % kDoneSlots;
  if (ndone >= kDoneSlots) mbar_wait(smem_u32(&sm.meta->done_free[pipe][k]), ((ndone / kDoneSlots) - 1) & 1);
  if (cw == 0 && lane == 0) sm.meta->done_e[pipe][k] = e;
  mbar_arrive(smem_u32(&sm.meta->done_full[pipe][k]));
  ++ndone;
}

// One warp per CTA, off the producers' and consumers' paths:
//  - counts each finished gate/up item for its expert; an expert whose last item it counts is queued, its h rows
//    are quantized four at a time (between the other duties) and then its ready flag is released;
//  - serves each producer's down request: once the expert's ready flag is set (acquire), writes the item's
//    epilogue table (scale = hscale x routing weight, destination = original slot) and acknowledges.
// It never waits on a flag: the gate/up item that completes an expert may sit in its own done FIFO.
SGL_DEVICE void helper(const Params& p, const Smem& sm, int lane) {
  Meta* m = sm.meta;
  uint32_t ndone[kPipes] = {0, 0}, nreq[kPipes] = {0, 0};
  int open = kPipes;
  int qhead = 0, qtail = 0, qrow = 0;  // quantization jobs m->qlist[qhead..qtail), next row of the head job
  int32_t* ready = p.ws + ready_base(p.num_experts);
  while (open > 0 || qhead != qtail) {
    bool idle = true;
#pragma unroll
    for (int pipe = 0; pipe < kPipes; ++pipe) {
      if (mbar_test(smem_u32(&m->req_full[pipe]), nreq[pipe] & 1) &&
          __shfl_sync(0xffffffffu, lane == 0 ? ld_acquire(ready + m->req[pipe].x) : 0, 0) != 0) {
        idle = false;
        __syncwarp();  // lane 0's acquire orders the other lanes' reads below
        const int4 rq = m->req[pipe];
        if (lane < rq.w) {
          const int sorted = rq.z + lane;
          m->rowmeta[pipe][rq.y % kRowSlots][lane] =
              make_float2(__ldcg(p.hscale + sorted) * p.gammas[sorted], __int_as_float(p.combine[sorted]));
        }
        __syncwarp();
        if (lane == 0) mbar_arrive(smem_u32(&m->req_ack[pipe]));
        ++nreq[pipe];
      }
      const uint32_t k = ndone[pipe] % kDoneSlots;
      if (ndone[pipe] != 0xffffffffu && qtail - qhead < kQuantJobs &&
          mbar_test(smem_u32(&m->done_full[pipe][k]), (ndone[pipe] / kDoneSlots) & 1)) {
        idle = false;
        const int e = m->done_e[pipe][k];
        if (e < 0) {
          ndone[pipe] = 0xffffffffu;  // this pipeline's consumer has finished
          --open;
          continue;
        }
        int last = 0;
        if (lane == 0) {
          fence_acq_rel_gpu();
          const int total = Shape<true>::kNB * ((m->cnt[e] + 31) / 32);
          last = atomicAdd(p.ws + kDoneBase + e, 1) == total - 1;
          if (last) {
            fence_acq_rel_gpu();
            m->qlist[qtail % kQuantJobs] = e;
          }
          mbar_arrive(smem_u32(&m->done_free[pipe][k]));
        }
        ++ndone[pipe];
        if (__shfl_sync(0xffffffffu, last, 0)) ++qtail;
        __syncwarp();
      }
    }
    if (qhead != qtail) {  // one batch of the head job
      idle = false;
      const int e = m->qlist[qhead % kQuantJobs];
      const int cnt = m->cnt[e];
      quant_batch(p, m->off[e], cnt, qrow, lane);
      qrow += kQuantRows;
      if (qrow >= cnt) {
        __syncwarp();
        if (lane == 0) {
          fence_acq_rel_gpu();
          st_release(ready + e, 1);
        }
        ++qhead;
        qrow = 0;
      }
    }
    if (idle) __nanosleep(64);  // fewer issue slots taken from the consumers on this SMSP
  }
}

SGL_DEVICE void consumer(const Params& p, const Smem& sm, int pipe, int cw, int lane) {
  uint32_t g = 0, ndone = 0;
  for (uint32_t seq = 0;; ++seq) {
    const uint32_t slot = g % kRing;
    mbar_wait(sm.full + slot * 8, (g / kRing) & 1);
    const int4 hd = sm.meta->hdr[pipe][slot];  // after the full barrier (memory clobber): not hoisted
    const int kind = uniform(hd.x & 0xff);
    if (kind == kEnd) {
      gate_up_done(sm, pipe, ndone, -1, cw, lane);  // tells the helper this consumer is done
      break;
    }
    Item it;
    it.kind = kind;
    it.e = uniform(hd.x >> 8);
    it.j = uniform(hd.y);
    it.r0 = uniform(hd.z);
    it.nrows = uniform(hd.w);
    it.off = uniform(sm.meta->off[it.e]);
    it.pipe = pipe;
    it.seq = seq;
    // Three code paths (N = 16 in 4-block batches, N = 24 and 32 in 2-block batches).
    if (kind == kGu) {
      if (it.nrows <= 16) {
        consumer_pass<true, 2>(p, sm, g, it, cw, lane);
      } else if (it.nrows <= 24) {
        consumer_pass<true, 3>(p, sm, g, it, cw, lane);
      } else {
        consumer_pass<true, 4>(p, sm, g, it, cw, lane);
      }
      gate_up_done(sm, pipe, ndone, it.e, cw, lane);
    } else {
      if (it.nrows <= 16) {
        consumer_pass<false, 2>(p, sm, g, it, cw, lane);
      } else if (it.nrows <= 24) {
        consumer_pass<false, 3>(p, sm, g, it, cw, lane);
      } else {
        consumer_pass<false, 4>(p, sm, g, it, cw, lane);
      }
    }
  }
}

__global__ void __launch_bounds__(kThreads, 1) moe_v2_kernel(const __grid_constant__ Params p) {
  extern __shared__ __align__(1024) uint8_t smem_raw[];
  const uint32_t raw = smem_u32(smem_raw);
  const uint32_t base = (raw + 1023u) & ~1023u;
  uint8_t* basep = smem_raw + (base - raw);
  Meta* meta = reinterpret_cast<Meta*>(basep + kPipes * kPipeStride);

  const int tid = threadIdx.x;
  const int warp = uniform(tid / 32);  // warp-uniform role for the compiler (see uniform())
  const int lane = tid % 32;
  // warps 0-3, 4-7: consumer warpgroups; 8, 9: their producers; 10: the helper
  const bool is_producer = warp == 8 || warp == 9;
  const bool is_helper = warp == 10;
  const int pipe = is_producer ? warp - 8 : (is_helper ? 0 : warp / 4);
  Smem sm;
  const uint32_t pbase = base + pipe * kPipeStride;
  sm.x_ring = pbase;
  sm.w_ring = pbase + kXRing;
  sm.w_ptr = basep + pipe * kPipeStride + kXRing;
  sm.full = pbase + kXRing + kWRing;
  sm.empty = sm.full + kRing * 8;
  sm.meta = meta;

  if (warp == 0) {  // exclusive prefixes of row counts (sorted offsets) and of item counts
    int cl[kMaxExperts / 32];  // every count loaded up front: one memory latency instead of one per 32 experts
#pragma unroll
    for (int k = 0; k < kMaxExperts / 32; ++k) {
      const int e = 32 * k + lane;
      cl[k] = e < p.num_experts ? p.counts[e] : 0;
    }
    int rows = 0, gi = 0, di = 0;
#pragma unroll
    for (int k = 0; k < kMaxExperts / 32; ++k) {
      const int e0 = 32 * k;
      if (e0 >= p.num_experts) break;
      const int e = e0 + lane;
      const int c = cl[k];
      const int passes = (c + 31) / 32;
      const int kg = Shape<true>::kNB * passes, kd = Shape<false>::kNB * passes;
      int ic = c, ig = kg, id = kd;
#pragma unroll
      for (int o = 1; o < 32; o <<= 1) {
        const int vc = __shfl_up_sync(0xffffffffu, ic, o);
        const int vg = __shfl_up_sync(0xffffffffu, ig, o);
        const int vd = __shfl_up_sync(0xffffffffu, id, o);
        if (lane >= o) {
          ic += vc;
          ig += vg;
          id += vd;
        }
      }
      if (e < p.num_experts) {
        meta->off[e] = rows + ic - c;
        meta->cnt[e] = c;
        meta->gstart[e] = gi + ig - kg;
        meta->dstart[e] = di + id - kd;
      }
      rows += __shfl_sync(0xffffffffu, ic, 31);
      gi += __shfl_sync(0xffffffffu, ig, 31);
      di += __shfl_sync(0xffffffffu, id, 31);
    }
    if (lane == 0) {
      meta->gstart[p.num_experts] = gi;
      meta->dstart[p.num_experts] = di;
    }
  }
  if (tid == 0) {
    for (int k = 0; k < kPipes; ++k) {
      const uint32_t fb = base + k * kPipeStride + kXRing + kWRing;
      for (int i = 0; i < kRing; ++i) {
        mbar_init(fb + i * 8, 33);                                // expect_tx + 32 cp.async arrivals
        mbar_init(fb + (kRing + i) * 8, 32 * kConsumerWarps);    // every consumer thread arrives
      }
      for (int i = 0; i < kDoneSlots; ++i) {
        mbar_init(smem_u32(&meta->done_full[k][i]), 32 * kConsumerWarps);
        mbar_init(smem_u32(&meta->done_free[k][i]), 1);
      }
      mbar_init(smem_u32(&meta->req_full[k]), 1);
      mbar_init(smem_u32(&meta->req_ack[k]), 1);
    }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  __syncthreads();
  if (is_helper) {
    helper(p, sm, lane);
  } else if (is_producer) {
    producer(p, sm, pipe, lane);
  } else {
    consumer(p, sm, pipe, warp % 4, lane);
  }
}

// x8 [tokens, 5120] e4m3 + xscale [tokens]; counts/gather/combine/gammas from the expert sort; h, h8, hscale,
// y scratch; ws int32 [32 + 2 E], zeroed. Writes y [slots, 5120] bf16 at the original slots.
inline void experts(
    tvm::ffi::TensorView x8,
    tvm::ffi::TensorView xscale,
    tvm::ffi::TensorView w13,
    tvm::ffi::TensorView w2,
    tvm::ffi::TensorView counts,
    tvm::ffi::TensorView gather,
    tvm::ffi::TensorView combine,
    tvm::ffi::TensorView gammas,
    tvm::ffi::TensorView h,
    tvm::ffi::TensorView h8,
    tvm::ffi::TensorView hscale,
    tvm::ffi::TensorView y,
    tvm::ffi::TensorView ws,
    double limit) {
  using namespace host;
  RuntimeCheck(x8.size(1) == 5120 && h.size(1) == 576 && h8.size(1) == 576 && y.size(1) == 5120,
               "v2: TP4 DeepSeek-V4 shapes only");
  Params prm{};
  prm.x8 = static_cast<const uint8_t*>(x8.data_ptr());
  prm.xscale = static_cast<const float*>(xscale.data_ptr());
  prm.w13 = static_cast<const uint8_t*>(w13.data_ptr());
  prm.w2 = static_cast<const uint8_t*>(w2.data_ptr());
  prm.counts = static_cast<const int32_t*>(counts.data_ptr());
  prm.gather = static_cast<const int32_t*>(gather.data_ptr());
  prm.combine = static_cast<const int32_t*>(combine.data_ptr());
  prm.gammas = static_cast<const float*>(gammas.data_ptr());
  prm.h = static_cast<__nv_bfloat16*>(h.data_ptr());
  prm.h8 = static_cast<uint8_t*>(h8.data_ptr());
  prm.hscale = static_cast<float*>(hscale.data_ptr());
  prm.y = static_cast<__nv_bfloat16*>(y.data_ptr());
  prm.ws = static_cast<int32_t*>(ws.data_ptr());
  prm.num_experts = static_cast<int>(counts.size(0));
  prm.limit = static_cast<float>(limit);
  RuntimeCheck(prm.num_experts <= kMaxExperts, "v2: at most 512 local experts");
  RuntimeCheck(ws.size(0) >= kWorkspaceInts(prm.num_experts), "v2: workspace too small");
  const DLDevice device = x8.device();
  static const bool attr_set = [] {
    RuntimeDeviceCheck(
        cudaFuncSetAttribute(moe_v2_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(kSmem)));
    return true;
  }();
  (void)attr_set;
  int sms = 0;
  RuntimeDeviceCheck(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device.device_id));
  LaunchKernel(dim3(sms), dim3(kThreads), device, kSmem)(moe_v2_kernel, prm);
}


// ---------------------------------------------------------------------------------------------
// Prologue of a call, one launch: the stable expert sort of the routed slots (as the sorts of #104), the e4m3
// row quantization of the tokens (as the Triton _row_quant) and the zeroed workspace of the experts kernel.
// Sort CTAs own kSortExperts experts each and scan every slot: an expert's first output position is the number
// of slots routed to lower experts, and its slots keep their order. Masked slots (id < 0) belong to no expert.
constexpr int kPrepThreads = 512;
constexpr int kPrepWarps = kPrepThreads / 32;
constexpr int kSortExperts = 4;
constexpr int kPrepMaxSlots = 16384;
constexpr int kPrepRows = 4;  // token rows per quantization CTA

SGL_DEVICE int block_excl_scan(int v, int* tmp, int& total) {  // exclusive scan over the CTA's warps, warp-wise
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  if (lane == 0) tmp[warp] = v;
  __syncthreads();
  int x = lane < kPrepWarps ? tmp[lane] : 0;
  int incl = x;
#pragma unroll
  for (int o = 1; o < 32; o <<= 1) {
    const int y = __shfl_up_sync(0xffffffffu, incl, o);
    if (lane >= o) incl += y;
  }
  const int excl = __shfl_sync(0xffffffffu, incl - x, warp);
  total = __shfl_sync(0xffffffffu, incl, kPrepWarps - 1);
  __syncthreads();
  return excl;
}

__global__ void __launch_bounds__(kPrepThreads) prep_kernel(
    const int32_t* __restrict__ ids,
    const float* __restrict__ weights,
    int num_slots,
    int top_k,
    int num_experts,
    int sort_ctas,
    int32_t* __restrict__ counts,
    int32_t* __restrict__ sorted_slots,
    int32_t* __restrict__ sorted_tokens,
    float* __restrict__ sorted_weights,
    const __nv_bfloat16* __restrict__ x,
    int tokens,
    uint8_t* __restrict__ x8,
    float* __restrict__ xscale,
    int32_t* __restrict__ ws,
    int ws_ints) {
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  if (blockIdx.x == 0) {
    for (int i = threadIdx.x; i < ws_ints; i += kPrepThreads)
      ws[i] = 0;
  }
  if (static_cast<int>(blockIdx.x) < sort_ctas) {
    __shared__ int tmp[kPrepWarps];
    const int e_lo = blockIdx.x * kSortExperts;
    const int chunk = (num_slots + kPrepWarps * 32 - 1) / (kPrepWarps * 32) * 32;  // slots per warp, in order
    const int begin = warp * chunk, end = min(num_slots, begin + chunk);
    const unsigned lt = (1u << lane) - 1u;
    // Every id of this warp's slots up front (one memory latency), then two passes over registers.
    constexpr int kMaxGroups = kPrepMaxSlots / (kPrepWarps * 32);
    int idv[kMaxGroups];
#pragma unroll
    for (int i = 0; i < kMaxGroups; ++i) {
      const int sl = begin + 32 * i + lane;
      idv[i] = sl < end ? ids[sl] : -1;
    }
    // Pass 1: per warp, slots below this CTA's experts and per expert.
    int below = 0, cnt[kSortExperts] = {0, 0, 0, 0};
#pragma unroll
    for (int i = 0; i < kMaxGroups; ++i) {
      if (begin + 32 * i >= end) break;  // warp-uniform
      const int id = idv[i];
      below += __popc(__ballot_sync(0xffffffffu, id >= 0 && id < e_lo));
#pragma unroll
      for (int k = 0; k < kSortExperts; ++k)
        cnt[k] += __popc(__ballot_sync(0xffffffffu, id == e_lo + k));
    }
    int below_total;
    block_excl_scan(below, tmp, below_total);
    int base[kSortExperts], run[kSortExperts];
    int start = below_total;
#pragma unroll
    for (int k = 0; k < kSortExperts; ++k) {
      int total;
      run[k] = block_excl_scan(cnt[k], tmp, total);
      base[k] = start;
      start += total;
      if (threadIdx.x == 0 && e_lo + k < num_experts) counts[e_lo + k] = total;
    }
    // Pass 2: write each of this CTA's slots at its position.
#pragma unroll
    for (int i = 0; i < kMaxGroups; ++i) {
      if (begin + 32 * i >= end) break;  // warp-uniform
      const int sl = begin + 32 * i + lane;
      const int id = idv[i];
#pragma unroll
      for (int k = 0; k < kSortExperts; ++k) {
        const unsigned m = __ballot_sync(0xffffffffu, id == e_lo + k);
        if (id == e_lo + k) {
          const int pos = base[k] + run[k] + __popc(m & lt);
          sorted_slots[pos] = sl;
          sorted_tokens[pos] = sl / top_k;
          sorted_weights[pos] = weights[sl];
        }
        run[k] += __popc(m);
      }
    }
    return;
  }
  // Token rows (as _row_quant: amax over the row, power-of-2 scale, e4m3): kPrepRows rows per CTA, each split
  // over kPrepWarps / kPrepRows warps.
  constexpr int kPart = 5120 / (kPrepWarps / kPrepRows);  // values per warp
  constexpr int kVec = kPart / 8 / 32;                     // uint4 per lane
  __shared__ float amax_s[kPrepWarps];
  const int r = (blockIdx.x - sort_ctas) * kPrepRows + warp / (kPrepWarps / kPrepRows);
  const int part = warp % (kPrepWarps / kPrepRows);
  const bool live = r < tokens;
  uint4 v[kVec];
  float amax = 0.f;
  const uint4* src = reinterpret_cast<const uint4*>(x + static_cast<size_t>(live ? r : 0) * 5120 + part * kPart);
#pragma unroll
  for (int k = 0; k < kVec; ++k) {
    v[k] = live ? src[lane + 32 * k] : make_uint4(0, 0, 0, 0);
    const __nv_bfloat162* b = reinterpret_cast<const __nv_bfloat162*>(&v[k]);
#pragma unroll
    for (int t = 0; t < 4; ++t) {
      const float2 f = __bfloat1622float2(b[t]);
      amax = fmaxf(amax, fmaxf(fabsf(f.x), fabsf(f.y)));
    }
  }
#pragma unroll
  for (int o = 16; o > 0; o >>= 1)
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
  if (lane == 0) amax_s[warp] = amax;
  __syncthreads();
  const int g0 = warp - part;
#pragma unroll
  for (int i = 0; i < kPrepWarps / kPrepRows; ++i)
    amax = fmaxf(amax, amax_s[g0 + i]);
  if (!live) return;
  const float sc = row_scale_of(amax);
  const float inv = __frcp_rn(sc);
  uint2* dst = reinterpret_cast<uint2*>(x8 + static_cast<size_t>(r) * 5120 + part * kPart);
#pragma unroll
  for (int k = 0; k < kVec; ++k) {
    const __nv_bfloat162* b = reinterpret_cast<const __nv_bfloat162*>(&v[k]);
    uint32_t w[2];
#pragma unroll
    for (int t = 0; t < 2; ++t) {
      const float2 f0 = __bfloat1622float2(b[2 * t]);
      const float2 f1 = __bfloat1622float2(b[2 * t + 1]);
      w[t] = e4m3x2(f0.x, f0.y, inv) | (e4m3x2(f1.x, f1.y, inv) << 16);
    }
    dst[lane + 32 * k] = make_uint2(w[0], w[1]);
  }
  if (lane == 0 && part == 0) xscale[r] = sc;
}

// topk_ids [T, top_k] int32, topk_weights [T, top_k] f32, x [T, 5120] bf16 -> counts [E], sorted_slots /
// sorted_tokens / sorted_weights [T * top_k] (the first sum(counts) entries), x8 [T, 5120] e4m3, xscale [T],
// ws zeroed.
inline void prep(
    tvm::ffi::TensorView topk_ids,
    tvm::ffi::TensorView topk_weights,
    tvm::ffi::TensorView x,
    tvm::ffi::TensorView counts,
    tvm::ffi::TensorView sorted_slots,
    tvm::ffi::TensorView sorted_tokens,
    tvm::ffi::TensorView sorted_weights,
    tvm::ffi::TensorView x8,
    tvm::ffi::TensorView xscale,
    tvm::ffi::TensorView ws) {
  using namespace host;
  const int tokens = static_cast<int>(topk_ids.size(0));
  const int top_k = static_cast<int>(topk_ids.size(1));
  const int num_experts = static_cast<int>(counts.size(0));
  RuntimeCheck(x.size(1) == 5120 && x.size(0) == tokens, "v2 prep: x [tokens, 5120]");
  RuntimeCheck(tokens * top_k <= kPrepMaxSlots, "v2 prep: at most 16384 routed slots");
  const int sort_ctas = (num_experts + kSortExperts - 1) / kSortExperts;
  const int quant_ctas = (tokens + kPrepRows - 1) / kPrepRows;
  LaunchKernel(dim3(sort_ctas + (quant_ctas > 0 ? quant_ctas : 1)), dim3(kPrepThreads), topk_ids.device())(
      prep_kernel,
      static_cast<const int32_t*>(topk_ids.data_ptr()),
      static_cast<const float*>(topk_weights.data_ptr()),
      tokens * top_k,
      top_k,
      num_experts,
      sort_ctas,
      static_cast<int32_t*>(counts.data_ptr()),
      static_cast<int32_t*>(sorted_slots.data_ptr()),
      static_cast<int32_t*>(sorted_tokens.data_ptr()),
      static_cast<float*>(sorted_weights.data_ptr()),
      static_cast<const __nv_bfloat16*>(x.data_ptr()),
      tokens,
      static_cast<uint8_t*>(x8.data_ptr()),
      static_cast<float*>(xscale.data_ptr()),
      static_cast<int32_t*>(ws.data_ptr()),
      static_cast<int>(ws.size(0)));
}

// Load both kernels of this module on the current device now (CUDA loads modules lazily, at a kernel's first
// launch otherwise) and set the experts kernel's shared-memory limit, so that the first MoE call inside CUDA graph
// capture loads nothing.
inline void prepare_kernels() {
  using namespace host;
  RuntimeDeviceCheck(
      cudaFuncSetAttribute(moe_v2_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(kSmem)));
  cudaFuncAttributes attr;
  RuntimeDeviceCheck(cudaFuncGetAttributes(&attr, prep_kernel));
}

}  // namespace dsv4_moe_v2
}  // namespace sglang
