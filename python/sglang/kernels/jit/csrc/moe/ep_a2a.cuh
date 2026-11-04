// INF-730: intra-node expert-parallel dispatch / combine over NVLink with peer pointers (symmetric memory).
//
// One kernel for dispatch, one for combine, no host sync, CUDA-graph safe. Every rank owns one symmetric buffer
// (same layout and size on every rank); a kernel reaches a peer's copy through its base pointer.
//
// dispatch (sender rank r, W ranks, T tokens, top-k ids in global expert numbering):
//   every block scans the whole [T, K] routing (dests(t) = ranks owning one of t's experts; slot(t, d) = number of
//   tokens before t that also go to d, so the order inside a region is the token order, deterministic), then the
//   block's warps push their tokens' rows straight into region [r] of every destination's receive buffer:
//   payload row (x_bytes), scale row (s_bytes), the K local expert ids (-1 where the expert lives elsewhere), the K
//   weights and the source token index. Each block then publishes its per-destination counts and raises its flag
//   on every destination (release, system scope), and waits until the same-index block of every source has raised
//   its flag here (acquire). When the kernel ends every source block has delivered, so the next kernel on the stream
//   (the expert GEMM) reads complete data. Rows of a region past its count keep stale data from an earlier call;
//   with mask_stale their expert ids are set to -1 here, so a kernel that walks all W * tmax rows skips them.
//
// combine (pull): the expert kernel wrote one BF16 row per received row (the weighted sum over this rank's experts
//   of that token) into this rank's partial-output region, in the receive layout. Each block raises its "ready"
//   flag on every peer, waits until every peer's same-index block is ready, then every token of this rank sums its
//   rows from the destinations it went to, in rank order (fp32 accumulation, deterministic), into out [T, hidden].
//
// Buffer reuse needs no extra barrier: a peer writes this rank's receive region for layer L+1 only after its
// combine of layer L, which waited for this rank's "ready" of layer L, which this rank raised after its expert
// kernel of layer L finished reading the region.
#pragma once

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <cstdint>
#include <cuda_bf16.h>

namespace sglang {
namespace ep_a2a {

constexpr int kMaxRanks = 8;
constexpr int kMaxTopk = 8;
constexpr int kMaxExperts = 1024;
constexpr int kMaxTokens = 2048;  // per rank per call (shared-memory slot table)
constexpr int kThreads = 512;
constexpr int kWarps = kThreads / 32;
constexpr int kMaxBlocks = 160;  // flag table rows per direction
constexpr int kChunkVecs = 64;   // dispatch work item: 64 x 16 B of one token row

struct Layout {  // byte offsets inside one rank's symmetric buffer
  int64_t x, s, ids, w, src, cnt, dflag, cflag, y;
};

struct DispatchArgs {
  const int64_t* peers;  // [W] base address of every rank's symmetric buffer
  Layout lay;
  // tmax: rows per source region in this call (region s = rows [s * tmax, (s + 1) * tmax) of every receive array).
  // It must be the same on every rank and at most the buffer capacity (checked by the Python wrapper).
  int rank, world, tmax, num_tokens, top_k, num_experts, x_bytes, s_bytes, mask_stale;
  const uint8_t* x;    // [T, x_bytes]
  const uint8_t* s;    // [T, s_bytes]
  const void* ids;     // [T, K] int32 or int64, global expert ids, -1 = no expert
  const float* wts;    // [T, K]
  const int32_t* expert_rank;   // [E]
  const int32_t* expert_local;  // [E]
  int32_t* slot_map;   // [>= T, W] local: slot of token t in destination d's region, -1 if not sent
  uint32_t* ctr;       // [kMaxBlocks] local epoch per block
  int32_t* recv_counts;  // [W] local output: rows received from every source this call
};

struct CombineArgs {
  const int64_t* peers;
  Layout lay;
  int rank, world, tmax, num_tokens, hidden;  // tmax: the region stride of the matching dispatch
  const int32_t* slot_map;  // [>= T, W]
  uint32_t* ctr;            // [kMaxBlocks]
  __nv_bfloat16* out;       // [T, hidden]
};

SGL_DEVICE uint32_t ld_acquire_sys(const uint32_t* p) {
  uint32_t v;
  asm volatile("ld.acquire.sys.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
  return v;
}

SGL_DEVICE void st_release_sys(uint32_t* p, uint32_t v) {
  asm volatile("st.release.sys.global.u32 [%0], %1;" ::"l"(p), "r"(v) : "memory");
}

SGL_DEVICE uint4 ld_relaxed_sys_16B(const void* p) {
  uint4 v;
  asm volatile("ld.relaxed.sys.global.v4.b32 {%0, %1, %2, %3}, [%4];"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
               : "l"(p));
  return v;
}

SGL_DEVICE uint4 ld_nc_16B(const void* p) {
  uint4 v;
  asm volatile("ld.global.nc.v4.b32 {%0, %1, %2, %3}, [%4];"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
               : "l"(p));
  return v;
}

SGL_DEVICE void st_16B(void* p, const uint4& v) {
  asm volatile("st.global.v4.b32 [%4], {%0, %1, %2, %3};" ::"r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w), "l"(p)
               : "memory");
}

// Before a block raises its flag: the release store below already orders this block's earlier writes (bar.sync
// then st.release.sys is cumulative); define EP_A2A_FULL_FENCE to add a fence.sc.sys as well.
#ifdef EP_A2A_FULL_FENCE
#define EP_A2A_PRE_SIGNAL_FENCE() __threadfence_system()
#else
#define EP_A2A_PRE_SIGNAL_FENCE() \
  do {                            \
  } while (0)
#endif

// Wrap-safe "a >= b" for monotonically increasing 32-bit epochs.
SGL_DEVICE bool reached(uint32_t a, uint32_t b) {
  return static_cast<int32_t>(a - b) >= 0;
}

template <typename IdT>
__global__ void __launch_bounds__(kThreads, 1) dispatch_kernel(const DispatchArgs a) {
  __shared__ uint16_t s_slot[kMaxTokens * kMaxRanks];
  __shared__ uint8_t s_mask[kMaxTokens];
  __shared__ int8_t s_rank_of[kMaxExperts];
  __shared__ int16_t s_local_of[kMaxExperts];
  __shared__ unsigned long long s_warp_sum[kWarps][2];
  __shared__ int s_tot[kMaxRanks];
  __shared__ int s_cnt[kMaxRanks];
  __shared__ uint8_t* s_peer[kMaxRanks];

  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int W = a.world, T = a.num_tokens, K = a.top_k, r = a.rank;
  const uint32_t epoch = a.ctr[blockIdx.x] + 1;
  const IdT* ids = static_cast<const IdT*>(a.ids);

  for (int e = tid; e < a.num_experts; e += kThreads) {
    s_rank_of[e] = static_cast<int8_t>(a.expert_rank[e]);
    s_local_of[e] = static_cast<int16_t>(a.expert_local[e]);
  }
  if (tid < W) s_peer[tid] = reinterpret_cast<uint8_t*>(a.peers[tid]);
  __syncthreads();

  // Phase 1: destination masks and per-destination exclusive prefix over tokens (token order).
  // Counts packed 16 bits per destination: lo = ranks 0..3, hi = ranks 4..7.
  const int per_thread = (T + kThreads - 1) / kThreads;
  const int t0 = tid * per_thread;
  unsigned long long lo = 0, hi = 0;
  for (int i = 0; i < per_thread; ++i) {
    const int t = t0 + i;
    if (t >= T) break;
    uint32_t m = 0;
    for (int k = 0; k < K; ++k) {
      const int e = static_cast<int>(ids[static_cast<int64_t>(t) * K + k]);
      if (e >= 0) m |= 1u << s_rank_of[e];
    }
    s_mask[t] = static_cast<uint8_t>(m);
    for (int d = 0; d < W; ++d) {
      if (m >> d & 1) {
        if (d < 4) lo += 1ull << (16 * d);
        else hi += 1ull << (16 * (d - 4));
      }
    }
  }
  // inclusive warp scan of (lo, hi)
  unsigned long long ilo = lo, ihi = hi;
#pragma unroll
  for (int o = 1; o < 32; o <<= 1) {
    const unsigned long long plo = __shfl_up_sync(0xffffffffu, ilo, o);
    const unsigned long long phi = __shfl_up_sync(0xffffffffu, ihi, o);
    if (lane >= o) {
      ilo += plo;
      ihi += phi;
    }
  }
  if (lane == 31) {
    s_warp_sum[warp][0] = ilo;
    s_warp_sum[warp][1] = ihi;
  }
  __syncthreads();
  if (warp == 0) {
    unsigned long long wlo = lane < kWarps ? s_warp_sum[lane][0] : 0, whi = lane < kWarps ? s_warp_sum[lane][1] : 0;
    unsigned long long xlo = wlo, xhi = whi;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
      const unsigned long long plo = __shfl_up_sync(0xffffffffu, xlo, o);
      const unsigned long long phi = __shfl_up_sync(0xffffffffu, xhi, o);
      if (lane >= o) {
        xlo += plo;
        xhi += phi;
      }
    }
    if (lane < kWarps) {  // exclusive per-warp base
      s_warp_sum[lane][0] = xlo - wlo;
      s_warp_sum[lane][1] = xhi - whi;
    }
    if (lane == kWarps - 1) {
      for (int d = 0; d < W; ++d) {
        s_tot[d] = static_cast<int>(((d < 4 ? xlo : xhi) >> (16 * (d & 3))) & 0xffffull);
      }
    }
  }
  __syncthreads();
  {
    unsigned long long blo = s_warp_sum[warp][0] + ilo - lo, bhi = s_warp_sum[warp][1] + ihi - hi;
    for (int i = 0; i < per_thread; ++i) {
      const int t = t0 + i;
      if (t >= T) break;
      const uint32_t m = s_mask[t];
      for (int d = 0; d < W; ++d) {
        const unsigned long long& base = d < 4 ? blo : bhi;
        const int sh = 16 * (d & 3);
        if (m >> d & 1) {
          s_slot[t * kMaxRanks + d] = static_cast<uint16_t>((base >> sh) & 0xffffull);
          if (d < 4) blo += 1ull << sh;
          else bhi += 1ull << sh;
        } else {
          s_slot[t * kMaxRanks + d] = 0xffff;
        }
      }
    }
  }
  __syncthreads();
  if (blockIdx.x == 0) {
    for (int i = tid; i < T * W; i += kThreads) {
      const int t = i / W, d = i - t * W;
      const uint16_t sl = s_slot[t * kMaxRanks + d];
      a.slot_map[i] = sl == 0xffff ? -1 : static_cast<int>(sl);
    }
  }

  // Phase 2: push rows to every destination. Work items are (token, chunk of 64 x 16 B) pairs spread over all warps
  // of the grid, so a call with few tokens still has many independent copies in flight (latency, not one warp's
  // serial loop). The item with chunk 0 also carries the token's scale row, expert ids, weights and source index.
  const int nvec = a.x_bytes >> 4;
  const int nchunk = (nvec + kChunkVecs - 1) / kChunkVecs;
  const int swords = a.s_bytes >> 2;
  const int gw = blockIdx.x * kWarps + warp, nw = gridDim.x * kWarps;
  for (int item = gw; item < T * nchunk; item += nw) {
    const int t = item / nchunk, c = item - t * nchunk;
    const uint32_t m = s_mask[t];
    if (m == 0) continue;
    int64_t row[kMaxRanks];
#pragma unroll
    for (int d = 0; d < kMaxRanks; ++d) {
      row[d] = (d < W && (m >> d & 1)) ? static_cast<int64_t>(r) * a.tmax + s_slot[t * kMaxRanks + d] : 0;
    }
    const uint8_t* xsrc = a.x + static_cast<int64_t>(t) * a.x_bytes;
    uint4 val[kChunkVecs / 32];
#pragma unroll
    for (int u = 0; u < kChunkVecs / 32; ++u) {
      const int v = c * kChunkVecs + u * 32 + lane;
      if (v < nvec) val[u] = ld_nc_16B(xsrc + v * 16);
    }
#pragma unroll
    for (int d = 0; d < kMaxRanks; ++d) {
      if (d < W && (m >> d & 1)) {
        uint8_t* xdst = s_peer[d] + a.lay.x + row[d] * a.x_bytes;
#pragma unroll
        for (int u = 0; u < kChunkVecs / 32; ++u) {
          const int v = c * kChunkVecs + u * 32 + lane;
          if (v < nvec) st_16B(xdst + v * 16, val[u]);
        }
      }
    }
    if (c != 0) continue;
    for (int q = lane; q < swords; q += 32) {
      const uint32_t sv = reinterpret_cast<const uint32_t*>(a.s + static_cast<int64_t>(t) * a.s_bytes)[q];
#pragma unroll
      for (int d = 0; d < kMaxRanks; ++d) {
        if (d < W && (m >> d & 1)) {
          reinterpret_cast<uint32_t*>(s_peer[d] + a.lay.s + row[d] * a.s_bytes)[q] = sv;
        }
      }
    }
    if (lane < K) {
      const int e = static_cast<int>(ids[static_cast<int64_t>(t) * K + lane]);
      const float wv = a.wts[static_cast<int64_t>(t) * K + lane];
      const int er = e >= 0 ? s_rank_of[e] : -1;
      const int el = e >= 0 ? s_local_of[e] : -1;
#pragma unroll
      for (int d = 0; d < kMaxRanks; ++d) {
        if (d < W && (m >> d & 1)) {
          reinterpret_cast<int32_t*>(s_peer[d] + a.lay.ids)[row[d] * K + lane] = er == d ? el : -1;
          reinterpret_cast<float*>(s_peer[d] + a.lay.w)[row[d] * K + lane] = wv;
        }
      }
    } else if (lane == 31) {
#pragma unroll
      for (int d = 0; d < kMaxRanks; ++d) {
        if (d < W && (m >> d & 1)) reinterpret_cast<int32_t*>(s_peer[d] + a.lay.src)[row[d]] = t;
      }
    }
  }

  // Phase 3: publish counts and raise this block's flag on every destination, then wait for every source.
  __syncthreads();
  if (tid < W) {
    const int d = tid;
    EP_A2A_PRE_SIGNAL_FENCE();
    reinterpret_cast<int32_t*>(s_peer[d] + a.lay.cnt)[blockIdx.x * kMaxRanks + r] = s_tot[d];
    st_release_sys(reinterpret_cast<uint32_t*>(s_peer[d] + a.lay.dflag) + blockIdx.x * kMaxRanks + r, epoch);
    const uint32_t* mine = reinterpret_cast<const uint32_t*>(s_peer[r] + a.lay.dflag) + blockIdx.x * kMaxRanks + d;
    while (!reached(ld_acquire_sys(mine), epoch)) {
    }
    s_cnt[d] = reinterpret_cast<const volatile int32_t*>(s_peer[r] + a.lay.cnt)[blockIdx.x * kMaxRanks + d];
  }
  __syncthreads();
  if (blockIdx.x == 0 && tid < W) a.recv_counts[tid] = s_cnt[tid];
  if (a.mask_stale) {
    int32_t* my_ids = reinterpret_cast<int32_t*>(s_peer[r] + a.lay.ids);
    const int rows = W * a.tmax;
    for (int i = blockIdx.x * kThreads + tid; i < rows; i += gridDim.x * kThreads) {
      const int src = i / a.tmax, sl = i - src * a.tmax;
      if (sl >= s_cnt[src]) {
        for (int k = 0; k < K; ++k) my_ids[static_cast<int64_t>(i) * K + k] = -1;
      }
    }
  }
  if (tid == 0) a.ctr[blockIdx.x] = epoch;
}

template <int kUnroll>
__global__ void __launch_bounds__(kThreads, 1) combine_kernel(const CombineArgs a) {
  __shared__ uint8_t* s_peer[kMaxRanks];
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int W = a.world, r = a.rank;
  const uint32_t epoch = a.ctr[blockIdx.x] + 1;
  if (tid < W) s_peer[tid] = reinterpret_cast<uint8_t*>(a.peers[tid]);
  __syncthreads();
  if (tid < W) {
    const int d = tid;
    EP_A2A_PRE_SIGNAL_FENCE();
    st_release_sys(reinterpret_cast<uint32_t*>(s_peer[d] + a.lay.cflag) + blockIdx.x * kMaxRanks + r, epoch);
    const uint32_t* mine = reinterpret_cast<const uint32_t*>(s_peer[r] + a.lay.cflag) + blockIdx.x * kMaxRanks + d;
    while (!reached(ld_acquire_sys(mine), epoch)) {
    }
  }
  __syncthreads();

  // (token, chunk of 32 * kUnroll vectors) items over all warps of the grid: few tokens still give many reads in
  // flight; each lane keeps kUnroll x (#destinations) 16-byte peer loads outstanding before it sums.
  const int nvec = (a.hidden * 2) >> 4;  // 16-byte vectors of 8 bf16
  constexpr int kItemVecs = 32 * kUnroll;
  const int nchunk = (nvec + kItemVecs - 1) / kItemVecs;
  const int64_t row_bytes = static_cast<int64_t>(a.hidden) * 2;
  const int gw = blockIdx.x * kWarps + warp, nw = gridDim.x * kWarps;
  for (int item = gw; item < a.num_tokens * nchunk; item += nw) {
    const int t = item / nchunk, c = item - t * nchunk;
    const int my_slot = lane < W ? a.slot_map[static_cast<int64_t>(t) * W + lane] : -1;
    const uint8_t* src[kMaxRanks];
    uint32_t m = 0;
#pragma unroll
    for (int d = 0; d < kMaxRanks; ++d) {
      const int sl = __shfl_sync(0xffffffffu, my_slot, d);
      src[d] = nullptr;
      if (d < W && sl >= 0) {
        m |= 1u << d;
        src[d] = s_peer[d] + a.lay.y + (static_cast<int64_t>(r) * a.tmax + sl) * row_bytes;
      }
    }
    uint8_t* dst = reinterpret_cast<uint8_t*>(a.out + static_cast<int64_t>(t) * a.hidden);
    uint4 buf[kUnroll][kMaxRanks];
#pragma unroll
    for (int u = 0; u < kUnroll; ++u) {
      const int v = c * kItemVecs + u * 32 + lane;
#pragma unroll
      for (int d = 0; d < kMaxRanks; ++d) {
        if (v < nvec && (m >> d & 1)) buf[u][d] = ld_relaxed_sys_16B(src[d] + v * 16);
      }
    }
#pragma unroll
    for (int u = 0; u < kUnroll; ++u) {
      const int v = c * kItemVecs + u * 32 + lane;
      if (v >= nvec) break;
      float acc[8];
#pragma unroll
      for (int j = 0; j < 8; ++j) acc[j] = 0.f;
#pragma unroll
      for (int d = 0; d < kMaxRanks; ++d) {
        if (m >> d & 1) {
          const __nv_bfloat162* h = reinterpret_cast<const __nv_bfloat162*>(&buf[u][d]);
#pragma unroll
          for (int j = 0; j < 4; ++j) {
            const float2 f = __bfloat1622float2(h[j]);
            acc[2 * j] += f.x;
            acc[2 * j + 1] += f.y;
          }
        }
      }
      uint4 o;
      __nv_bfloat162* oh = reinterpret_cast<__nv_bfloat162*>(&o);
#pragma unroll
      for (int j = 0; j < 4; ++j) oh[j] = __floats2bfloat162_rn(acc[2 * j], acc[2 * j + 1]);
      st_16B(dst + v * 16, o);
    }
  }
  __syncthreads();
  if (tid == 0) a.ctr[blockIdx.x] = epoch;
}

inline Layout make_layout(
    int64_t off_x, int64_t off_s, int64_t off_ids, int64_t off_w, int64_t off_src, int64_t off_cnt,
    int64_t off_dflag, int64_t off_cflag, int64_t off_y) {
  return Layout{off_x, off_s, off_ids, off_w, off_src, off_cnt, off_dflag, off_cflag, off_y};
}

// Host entry points (tvm-ffi). `layout` is an int64 CPU-side description passed as 9 scalars.
inline void dispatch(
    tvm::ffi::TensorView peers,
    tvm::ffi::TensorView x,
    tvm::ffi::TensorView s,
    tvm::ffi::TensorView topk_ids,
    tvm::ffi::TensorView topk_weights,
    tvm::ffi::TensorView expert_rank,
    tvm::ffi::TensorView expert_local,
    tvm::ffi::TensorView slot_map,
    tvm::ffi::TensorView ctr,
    tvm::ffi::TensorView recv_counts,
    int64_t rank,
    int64_t tmax,
    int64_t off_x,
    int64_t off_s,
    int64_t off_ids,
    int64_t off_w,
    int64_t off_src,
    int64_t off_cnt,
    int64_t off_dflag,
    int64_t off_cflag,
    int64_t off_y,
    int64_t num_blocks,
    int64_t mask_stale) {
  using namespace host;
  DispatchArgs a;
  a.peers = static_cast<const int64_t*>(peers.data_ptr());
  a.lay = make_layout(off_x, off_s, off_ids, off_w, off_src, off_cnt, off_dflag, off_cflag, off_y);
  a.rank = static_cast<int>(rank);
  a.world = static_cast<int>(peers.size(0));
  a.tmax = static_cast<int>(tmax);
  a.num_tokens = static_cast<int>(topk_ids.size(0));
  a.top_k = static_cast<int>(topk_ids.size(1));
  a.num_experts = static_cast<int>(expert_rank.size(0));
  a.x_bytes = static_cast<int>(x.size(1)) * static_cast<int>(x.dtype().bits / 8);
  a.s_bytes = static_cast<int>(s.size(1)) * static_cast<int>(s.dtype().bits / 8);
  a.mask_stale = static_cast<int>(mask_stale);
  RuntimeCheck(a.world >= 1 && a.world <= kMaxRanks, "ep_a2a: 1..8 ranks");
  RuntimeCheck(a.num_tokens <= a.tmax && a.tmax <= kMaxTokens, "ep_a2a: tokens <= tmax <= 2048");
  RuntimeCheck(a.top_k <= kMaxTopk, "ep_a2a: top_k <= 8");
  RuntimeCheck(a.num_experts <= kMaxExperts, "ep_a2a: experts <= 1024");
  RuntimeCheck(a.x_bytes % 16 == 0 && a.s_bytes % 4 == 0, "ep_a2a: x row multiple of 16 B, s row of 4 B");
  RuntimeCheck(num_blocks >= 1 && num_blocks <= kMaxBlocks, "ep_a2a: 1..160 blocks");
  a.x = static_cast<const uint8_t*>(x.data_ptr());
  a.s = static_cast<const uint8_t*>(s.data_ptr());
  a.ids = topk_ids.data_ptr();
  a.wts = static_cast<const float*>(topk_weights.data_ptr());
  a.expert_rank = static_cast<const int32_t*>(expert_rank.data_ptr());
  a.expert_local = static_cast<const int32_t*>(expert_local.data_ptr());
  a.slot_map = static_cast<int32_t*>(slot_map.data_ptr());
  a.ctr = static_cast<uint32_t*>(ctr.data_ptr());
  a.recv_counts = static_cast<int32_t*>(recv_counts.data_ptr());
  if (topk_ids.dtype().bits == 64) {
    LaunchKernel(dim3(num_blocks), dim3(kThreads), x.device())(dispatch_kernel<int64_t>, a);
  } else {
    LaunchKernel(dim3(num_blocks), dim3(kThreads), x.device())(dispatch_kernel<int32_t>, a);
  }
}

inline void combine(
    tvm::ffi::TensorView peers,
    tvm::ffi::TensorView slot_map,
    tvm::ffi::TensorView ctr,
    tvm::ffi::TensorView out,
    int64_t rank,
    int64_t tmax,
    int64_t off_x,
    int64_t off_s,
    int64_t off_ids,
    int64_t off_w,
    int64_t off_src,
    int64_t off_cnt,
    int64_t off_dflag,
    int64_t off_cflag,
    int64_t off_y,
    int64_t num_blocks) {
  using namespace host;
  CombineArgs a;
  a.peers = static_cast<const int64_t*>(peers.data_ptr());
  a.lay = make_layout(off_x, off_s, off_ids, off_w, off_src, off_cnt, off_dflag, off_cflag, off_y);
  a.rank = static_cast<int>(rank);
  a.world = static_cast<int>(peers.size(0));
  a.tmax = static_cast<int>(tmax);
  a.num_tokens = static_cast<int>(out.size(0));
  a.hidden = static_cast<int>(out.size(1));
  RuntimeCheck(a.num_tokens <= a.tmax, "ep_a2a combine: tokens <= tmax");
  RuntimeCheck((a.hidden * 2) % 16 == 0, "ep_a2a combine: hidden multiple of 8");
  RuntimeCheck(num_blocks >= 1 && num_blocks <= kMaxBlocks, "ep_a2a: 1..160 blocks");
  a.slot_map = static_cast<const int32_t*>(slot_map.data_ptr());
  a.ctr = static_cast<uint32_t*>(ctr.data_ptr());
  a.out = static_cast<__nv_bfloat16*>(out.data_ptr());
  LaunchKernel(dim3(num_blocks), dim3(kThreads), out.device())(combine_kernel<2>, a);
}

}  // namespace ep_a2a
}  // namespace sglang
