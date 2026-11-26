// SPDX-License-Identifier: Apache-2.0
//
// Counting sort of the W4A8 prefill MoE's top-k slots by expert, with the atomics aggregated per CTA
// (INF-874; replaces the Triton _direct_hist_kernel / _count_scan_kernel / _direct_scatter_kernel of
// mxfp4_w4a8_prefill._count_route behind SGLANG_OPT_DSV4_W4A8_CTA_ROUTE).
//
// The Triton kernels do one global atomic per slot on one of E counters (the histogram, then the scatter
// cursor), so a 65k-row step (390k slots, 384 experts) serializes about a thousand atomics per address.
// Here each CTA owns a contiguous chunk of slots and ranks them in shared memory (warp-aggregated with
// __match_any_sync: one shared atomic per distinct expert per 32 slots); the global atomics are one per
// (CTA, expert present in the chunk):
//   hist_kernel     shared histogram of the chunk; base[cta][e] = atomicAdd(counts[e], hist[e]), i.e. the rows
//                   of expert e that CTAs before this one (in atomic order) took.
//   scatter_kernel  exclusive scan of counts (every CTA, in shared memory); ranks the chunk's slots again in
//                   shared memory; slot s of expert e goes to row scan[e] + base[cta][e] + rank.
// Output layout as _count_route: counts[E]; rows of expert e contiguous and in expert order; gather[row] = token,
// sorted_weights[row] = the slot's top-k weight, pos[slot] = row (-1 for a masked slot, id < 0, which gets no
// row). The order of rows inside an expert is not deterministic (as the Triton kernels'); every row is computed
// on its own and the top-k sum reads rows by position, so the MoE output does not depend on it.

#pragma once

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <cstdint>

namespace sglang {
namespace w4a8_count_route {

constexpr int kThreads = 512;
constexpr int kWarps = kThreads / 32;
constexpr int kMaxExperts = 1024;
constexpr int kScanPer = kMaxExperts / kThreads;  // experts per thread in the block scan

__global__ void __launch_bounds__(kThreads) hist_kernel(
    const int32_t* __restrict__ ids, int n_slots, int chunk, int num_experts, int32_t* __restrict__ counts,
    int32_t* __restrict__ base) {
  __shared__ int32_t hist[kMaxExperts];
  for (int e = threadIdx.x; e < num_experts; e += kThreads)
    hist[e] = 0;
  __syncthreads();
  const int lane = threadIdx.x % 32;
  const int begin = blockIdx.x * chunk;
  const int end = min(n_slots, begin + chunk);
  // Warp-uniform loop: the warp covers 32 consecutive slots per iteration.
  for (int s0 = begin + (threadIdx.x / 32) * 32; s0 < end; s0 += kThreads) {
    const int s = s0 + lane;
    const int e = s < end ? ids[s] : -1;
    const unsigned peers = __match_any_sync(0xffffffffu, e);
    if (e >= 0 && lane == __ffs(peers) - 1) atomicAdd(&hist[e], __popc(peers));
  }
  __syncthreads();
  int32_t* my_base = base + static_cast<int64_t>(blockIdx.x) * num_experts;
  for (int e = threadIdx.x; e < num_experts; e += kThreads) {
    const int c = hist[e];
    if (c > 0) my_base[e] = atomicAdd(&counts[e], c);
  }
}

__global__ void __launch_bounds__(kThreads) scatter_kernel(
    const int32_t* __restrict__ ids,
    const float* __restrict__ weights,
    int n_slots,
    int chunk,
    int num_experts,
    int top_k,
    const int32_t* __restrict__ counts,
    const int32_t* __restrict__ base,
    int32_t* __restrict__ gather,
    float* __restrict__ sorted_weights,
    int32_t* __restrict__ pos) {
  __shared__ int32_t first[kMaxExperts];  // first row of expert e for this CTA
  __shared__ int32_t taken[kMaxExperts];  // rows of expert e handed out in this CTA so far
  __shared__ int32_t warp_sums[kWarps];
  const int tid = threadIdx.x;
  const int lane = tid % 32;
  const int warp = tid / 32;
  const int32_t* my_base = base + static_cast<int64_t>(blockIdx.x) * num_experts;

  // Exclusive scan of counts over the experts (thread t holds experts kScanPer * t ...).
  int c[kScanPer];
  int sum = 0;
#pragma unroll
  for (int i = 0; i < kScanPer; ++i) {
    const int e = tid * kScanPer + i;
    c[i] = e < num_experts ? counts[e] : 0;
    sum += c[i];
  }
  int incl = sum;
#pragma unroll
  for (int o = 1; o < 32; o <<= 1) {
    const int v = __shfl_up_sync(0xffffffffu, incl, o);
    if (lane >= o) incl += v;
  }
  if (lane == 31) warp_sums[warp] = incl;
  __syncthreads();
  if (warp == 0) {
    int ws = lane < kWarps ? warp_sums[lane] : 0;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
      const int v = __shfl_up_sync(0xffffffffu, ws, o);
      if (lane >= o) ws += v;
    }
    if (lane < kWarps) warp_sums[lane] = ws;
  }
  __syncthreads();
  int excl = incl - sum + (warp > 0 ? warp_sums[warp - 1] : 0);
#pragma unroll
  for (int i = 0; i < kScanPer; ++i) {
    const int e = tid * kScanPer + i;
    if (e < num_experts) {
      // my_base[e] was written by hist_kernel only if this chunk has a slot of e; only such slots read first[e].
      first[e] = excl + (c[i] > 0 ? my_base[e] : 0);
      taken[e] = 0;
    }
    excl += c[i];
  }
  __syncthreads();

  const int begin = blockIdx.x * chunk;
  const int end = min(n_slots, begin + chunk);
  const unsigned lt = (1u << lane) - 1u;
  for (int s0 = begin + warp * 32; s0 < end; s0 += kThreads) {
    const int s = s0 + lane;
    const bool in = s < end;
    const int e = in ? ids[s] : -1;
    const unsigned peers = __match_any_sync(0xffffffffu, e);
    const int leader = __ffs(peers) - 1;
    int before = 0;
    if (e >= 0 && lane == leader) before = atomicAdd(&taken[e], __popc(peers));
    before = __shfl_sync(0xffffffffu, before, leader);
    if (!in) continue;
    if (e < 0) {
      pos[s] = -1;
      continue;
    }
    const int row = first[e] + before + __popc(peers & lt);
    gather[row] = s / top_k;
    sorted_weights[row] = weights[s];
    pos[s] = row;
  }
}

// topk_ids / topk_weights: tokens x top_k slots, token-major (any shape). counts must be zero on entry (the caller allocates it with torch.zeros); base is [ceil(slots / chunk), E].
inline void route(
    tvm::ffi::TensorView topk_ids,
    tvm::ffi::TensorView topk_weights,
    tvm::ffi::TensorView counts,
    tvm::ffi::TensorView base,
    tvm::ffi::TensorView gather,
    tvm::ffi::TensorView sorted_weights,
    tvm::ffi::TensorView pos,
    int64_t top_k_arg,
    int64_t chunk) {
  using namespace host;
  const int64_t slots = topk_ids.numel();
  // topk_ids may come flattened, so top_k (slots per token) is an argument.
  const int top_k = static_cast<int>(top_k_arg);
  const int num_experts = static_cast<int>(counts.size(0));
  RuntimeCheck(slots > 0 && slots < (int64_t{1} << 31), "w4a8 count route: 0 < slots < 2^31");
  RuntimeCheck(num_experts > 0 && num_experts <= kMaxExperts, "w4a8 count route: at most 1024 experts");
  RuntimeCheck(top_k > 0 && slots % top_k == 0, "w4a8 count route: slots must be tokens x top_k");
  RuntimeCheck(chunk > 0 && chunk % 32 == 0, "w4a8 count route: chunk must be a positive multiple of 32");
  const int64_t ctas = (slots + chunk - 1) / chunk;
  RuntimeCheck(base.numel() >= ctas * num_experts, "w4a8 count route: base needs ceil(slots / chunk) x E");
  RuntimeCheck(topk_weights.numel() == slots && gather.numel() == slots && sorted_weights.numel() == slots &&
                   pos.numel() == slots,
               "w4a8 count route: per-slot tensors must have one entry per slot");
  const auto n = static_cast<int>(slots);
  const auto ch = static_cast<int>(chunk);
  LaunchKernel(dim3(static_cast<unsigned>(ctas)), dim3(kThreads), topk_ids.device())(
      hist_kernel,
      static_cast<const int32_t*>(topk_ids.data_ptr()),
      n,
      ch,
      num_experts,
      static_cast<int32_t*>(counts.data_ptr()),
      static_cast<int32_t*>(base.data_ptr()));
  LaunchKernel(dim3(static_cast<unsigned>(ctas)), dim3(kThreads), topk_ids.device())(
      scatter_kernel,
      static_cast<const int32_t*>(topk_ids.data_ptr()),
      static_cast<const float*>(topk_weights.data_ptr()),
      n,
      ch,
      num_experts,
      top_k,
      static_cast<const int32_t*>(counts.data_ptr()),
      static_cast<const int32_t*>(base.data_ptr()),
      static_cast<int32_t*>(gather.data_ptr()),
      static_cast<float*>(sorted_weights.data_ptr()),
      static_cast<int32_t*>(pos.data_ptr()));
}

}  // namespace w4a8_count_route
}  // namespace sglang
