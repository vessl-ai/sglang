#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>

#include <tvm/ffi/container/tensor.h>

#include <cstdint>

// FlashMLA's split-KV tile-scheduler metadata for a large batch, computed in
// parallel. Same output as FlashMLA's own get_mla_metadata kernel and as
// flashmla_sched_meta.cuh, for the sparse schedule only (topk >= 0).
//
// FlashMLA walks the batch on one thread of a <<<1, 32>>> kernel: O(batch)
// dependent steps. When the batch is a prefill chunk's query rows (CP4: one
// "request" per query row, 16k rows per rank at chunk 65280), that walk costs
// 7 ms per call in its low-shared-memory variant (>= ~11.3k rows) and
// ~0.07 us per row below that.
//
// Here every row's cost (num_blocks + fixed_overhead) is prefix-summed by the
// whole block, the partition walk on thread 0 jumps over whole rows with a
// binary search in that prefix (O(num_sm_parts * log batch)), and num_splits
// is filled in parallel from the few rows the walk split.
//
// Shared memory: (batch + 1) prefix entries, num_sm_parts * 8 metadata words,
// num_sm_parts split records, so batch <= ~56k rows at 227 KB.

namespace sglang {

namespace flashmla_large {

constexpr int kMetaInts = 8;
constexpr int kThreads = 1024;

struct Params {
  int b;
  int block_size_n;
  int fixed_overhead_num_blocks;
  int topk;        // >= 0: sparse
  int extra_topk;  // 0 when there is no extra cache
  const int* __restrict__ topk_length;
  const int* __restrict__ extra_topk_length;
  int* __restrict__ tile_scheduler_metadata;
  int* __restrict__ num_splits;
  int num_sm_parts;
};

__device__ __forceinline__ int ceil_div_i(int a, int b) {
  return (a + b - 1) / b;
}

__device__ __forceinline__ int ceil_to_i(int a, int b) {
  return (a + b - 1) / b * b;
}

// The row's KV length as FlashMLA's schedule sees it.
__device__ __forceinline__ int seqlen_k_of(const Params& p, int i) {
  int s = p.topk_length ? __ldg(p.topk_length + i) : p.topk;
  if (s == 0) s = 1;  // the main loop must never be empty
  if (p.extra_topk) {
    s = ceil_to_i(s, p.block_size_n);
    s += p.extra_topk_length ? __ldg(p.extra_topk_length + i) : p.extra_topk;
  }
  return s;
}

// num_blocks = last_block_idx + 1 (the first block index is always 0).
__device__ __forceinline__ int num_blocks_of(const Params& p, int s_k) {
  return max(s_k - 1, 0) / p.block_size_n + 1;
}

__global__ void __launch_bounds__(kThreads) flashmla_sched_meta_large_kernel(__grid_constant__ const Params p) {
  extern __shared__ int smem[];
  const int b = p.b;
  const int f = p.fixed_overhead_num_blocks;
  int* cum = smem;                                     // [b + 1], exclusive prefix of num_blocks + f
  int* out_shared = smem + b + 1;                      // [num_sm_parts * kMetaInts]
  int* split_rows = out_shared + p.num_sm_parts * kMetaInts;  // [num_sm_parts], rows split, ascending

  __shared__ int warp_sums[kThreads / 32];
  __shared__ int num_split_rows_shared;
  __shared__ int first_idle_part_shared;

  // 1. Prefix sum of the per-row cost over contiguous per-thread ranges.
  const int per_thread = ceil_div_i(b, kThreads);
  const int lo = min(threadIdx.x * per_thread, b);
  const int hi = min(lo + per_thread, b);
  int local = 0;
  for (int i = lo; i < hi; ++i) {
    local += num_blocks_of(p, seqlen_k_of(p, i)) + f;
  }
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  int incl = local;
#pragma unroll
  for (int o = 1; o < 32; o *= 2) {
    const int v = __shfl_up_sync(uint32_t(-1), incl, o);
    if (lane >= o) incl += v;
  }
  if (lane == 31) warp_sums[warp] = incl;
  __syncthreads();
  if (warp == 0) {
    int w = lane < kThreads / 32 ? warp_sums[lane] : 0;
    int wincl = w;
#pragma unroll
    for (int o = 1; o < 32; o *= 2) {
      const int v = __shfl_up_sync(uint32_t(-1), wincl, o);
      if (lane >= o) wincl += v;
    }
    if (lane < kThreads / 32) warp_sums[lane] = wincl - w;  // exclusive warp offsets
  }
  __syncthreads();
  int run = warp_sums[warp] + incl - local;  // exclusive prefix at `lo`
  for (int i = lo; i < hi; ++i) {
    cum[i] = run;
    run += num_blocks_of(p, seqlen_k_of(p, i)) + f;
  }
  if (hi == b && lo < hi) cum[b] = run;
  if (b == 0 && threadIdx.x == 0) cum[0] = 0;
  __syncthreads();

  // 2. The partition walk, jumping over whole rows.
  if (threadIdx.x == 0) {
    const int total = cum[b];
    const int payload = ceil_div_i(total, p.num_sm_parts) + f;
    int r = 0, blk = 0, split = 0, nsplit_rows = 0;
    int i = 0;
    for (; i < p.num_sm_parts; ++i) {
      int* meta = out_shared + i * kMetaInts;
      const int begin_req_idx = r;
      const int begin_block_idx = blk;
      const int begin_split_idx = split;
      int is_first_req_splitted = (blk != 0);
      int remain = payload;
      const int need = (cum[r + 1] - cum[r] - f) - blk + f;
      if (remain >= need) {
        // Row r ends in this part; then every row that fits whole.
        remain -= need;
        ++r;
        blk = 0;
        split = 0;
        // Largest k in [r, b] with cum[k] - cum[r] <= remain.
        const int target = cum[r] + remain;
        int lo_k = r, hi_k = b;
        while (lo_k < hi_k) {
          const int mid = (lo_k + hi_k + 1) >> 1;
          if (cum[mid] <= target) {
            lo_k = mid;
          } else {
            hi_k = mid - 1;
          }
        }
        remain -= cum[lo_k] - cum[r];
        r = lo_k;
        if (r < b && remain - f > 0) {
          blk = remain - f;
          split = 1;
          split_rows[nsplit_rows++] = r;
          remain = 0;
        }
      } else if (remain - f > 0) {
        blk += remain - f;
        ++split;
        split_rows[nsplit_rows++] = r;
        remain = 0;
      }
      const int split_open = blk > 0;
      const int end_req_idx = split_open ? r : r - 1;
      // A sparse row's KV length is never 0 (0 becomes 1), so FlashMLA's
      // `end_sk == 0` branches never fire and the row's blocks come from `cum`.
      const int end_lb = cum[end_req_idx + 1] - cum[end_req_idx] - f - 1;
      const int end_block_idx = split_open ? blk : end_lb + 1;
      int is_last_req_splitted = (end_block_idx != end_lb + 1);
      if (begin_req_idx == end_req_idx) {
        is_first_req_splitted = is_last_req_splitted = is_first_req_splitted || is_last_req_splitted;
      }
      meta[0] = begin_req_idx;
      meta[1] = end_req_idx;
      meta[2] = begin_block_idx;
      meta[3] = end_block_idx;
      meta[4] = begin_split_idx;
      meta[5] = is_first_req_splitted;
      meta[6] = is_last_req_splitted;
      meta[7] = 0;
      if (r == b) {
        ++i;
        break;
      }
    }
    first_idle_part_shared = i;
    num_split_rows_shared = nsplit_rows;
  }
  __syncthreads();

  // 3. Parts past the walk, exactly as FlashMLA (and flashmla_sched_meta.cuh)
  // writes them: begin_block_idx reads one past FlashMLA's first-block array,
  // which aliases row 0's last block index.
  {
    const int lb_last = cum[b] - cum[b - 1] - f - 1;
    const int end_block_idx = lb_last + 1;
    const int begin_block_idx = cum[1] - cum[0] - f - 1;
    const int is_last_req_splitted = 0;
    for (int i = first_idle_part_shared + threadIdx.x; i < p.num_sm_parts; i += kThreads) {
      int* meta = out_shared + i * kMetaInts;
      meta[0] = b;
      meta[1] = b - 1;
      meta[2] = begin_block_idx;
      meta[3] = end_block_idx;
      meta[4] = 0;
      meta[5] = 0;
      meta[6] = is_last_req_splitted;
      meta[7] = 0;
    }
  }
  __syncthreads();

  const int meta_words = p.num_sm_parts * kMetaInts;
  for (int i = threadIdx.x; i < meta_words; i += kThreads) {
    p.tile_scheduler_metadata[i] = out_shared[i];
  }
  // num_splits[j + 1] = sum over rows <= j of their piece count
  //                   = (j + 1) + (number of split records on rows <= j).
  const int nsr = num_split_rows_shared;
  for (int j = threadIdx.x; j <= b; j += kThreads) {
    if (j == 0) {
      p.num_splits[0] = 0;
      continue;
    }
    // split records with row <= j - 1 (split_rows is non-decreasing)
    int lo_s = 0, hi_s = nsr;
    while (lo_s < hi_s) {
      const int mid = (lo_s + hi_s) >> 1;
      if (split_rows[mid] <= j - 1) {
        lo_s = mid + 1;
      } else {
        hi_s = mid;
      }
    }
    p.num_splits[j] = j + lo_s;
  }
}

}  // namespace flashmla_large

inline std::size_t flashmla_sched_meta_large_smem_bytes(int64_t b, int64_t num_sm_parts) {
  return sizeof(int) * static_cast<std::size_t>(b + 1 + num_sm_parts * flashmla_large::kMetaInts + num_sm_parts);
}

void flashmla_sched_meta_large(
    tvm::ffi::TensorView tile_scheduler_metadata,
    tvm::ffi::TensorView num_splits,
    tvm::ffi::Optional<tvm::ffi::TensorView> topk_length,
    tvm::ffi::Optional<tvm::ffi::TensorView> extra_topk_length,
    int64_t block_size_n,
    int64_t fixed_overhead_num_blocks,
    int64_t topk,
    int64_t extra_topk) {
  using namespace host;
  using namespace flashmla_large;

  auto parts = SymbolicSize{"num_sm_parts"};
  auto meta_ints = SymbolicSize{"meta_ints"};
  auto b_plus_one = SymbolicSize{"batch_size_plus_one"};
  auto device_ = SymbolicDevice{};
  device_.set_options<kDLGPU>();

  TensorMatcher({parts, meta_ints}).with_dtype<int32_t>().with_device(device_).verify(tile_scheduler_metadata);
  TensorMatcher({b_plus_one}).with_strides({1}).with_dtype<int32_t>().with_device(device_).verify(num_splits);

  const int num_sm_parts = static_cast<int>(parts.unwrap());
  const int b = static_cast<int>(b_plus_one.unwrap()) - 1;
  RuntimeCheck(
      static_cast<int>(meta_ints.unwrap()) == kMetaInts,
      "tile_scheduler_metadata must be [num_sm_parts, ",
      kMetaInts,
      "], got last dim ",
      meta_ints.unwrap());
  RuntimeCheck(b >= 1, "batch size must be positive, got ", b);
  RuntimeCheck(num_sm_parts >= 1, "num_sm_parts must be positive, got ", num_sm_parts);
  RuntimeCheck(block_size_n >= 1, "block_size_n must be positive, got ", block_size_n);
  RuntimeCheck(topk >= 0, "the large-batch schedule is sparse only (topk >= 0), got ", topk);

  auto opt_ptr = [&](const tvm::ffi::Optional<tvm::ffi::TensorView>& t, const char* name) -> const int* {
    if (!t.has_value()) return nullptr;
    auto n = SymbolicSize{"batch_size"};
    auto dev = SymbolicDevice{};
    dev.set_options<kDLGPU>();
    TensorMatcher({n}).with_strides({1}).with_dtype<int32_t>().with_device(dev).verify(t.value());
    RuntimeCheck(static_cast<int>(n.unwrap()) == b, name, " must have ", b, " entries, got ", n.unwrap());
    return static_cast<const int*>(t.value().data_ptr());
  };

  const Params p{
      b,
      static_cast<int>(block_size_n),
      static_cast<int>(fixed_overhead_num_blocks),
      static_cast<int>(topk),
      static_cast<int>(extra_topk),
      opt_ptr(topk_length, "topk_length"),
      opt_ptr(extra_topk_length, "extra_topk_length"),
      static_cast<int*>(tile_scheduler_metadata.data_ptr()),
      static_cast<int*>(num_splits.data_ptr()),
      num_sm_parts,
  };

  const std::size_t smem = flashmla_sched_meta_large_smem_bytes(b, num_sm_parts);
  // Opt in to the device's whole per-block shared memory once per process; every
  // later launch then only checks the size.
  // The dynamic part is the opt-in limit minus the kernel's static shared memory.
  static int max_optin = [] {
    int device = 0, bytes = 0;
    RuntimeDeviceCheck(cudaGetDevice(&device));
    RuntimeDeviceCheck(cudaDeviceGetAttribute(&bytes, cudaDevAttrMaxSharedMemoryPerBlockOptin, device));
    cudaFuncAttributes attr{};
    RuntimeDeviceCheck(cudaFuncGetAttributes(&attr, flashmla_sched_meta_large_kernel));
    bytes -= static_cast<int>(attr.sharedSizeBytes);
    RuntimeDeviceCheck(
        cudaFuncSetAttribute(flashmla_sched_meta_large_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes));
    return bytes;
  }();
  RuntimeCheck(smem <= static_cast<std::size_t>(max_optin), "schedule does not fit in shared memory: ", smem, " bytes");
  LaunchKernel(1, kThreads, device_.unwrap(), smem)(flashmla_sched_meta_large_kernel, p);
}

}  // namespace sglang
