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

// 1. Exclusive prefix of the per-row cost (num_blocks + f) over contiguous
// per-thread ranges: cum[0..b]. Ends with __syncthreads().
__device__ __forceinline__ void prefix_costs(const Params& p, int* cum, int* warp_sums) {
  const int b = p.b;
  const int f = p.fixed_overhead_num_blocks;
  const int per_thread = ceil_div_i(b, kThreads);
  const int lo = min(threadIdx.x * per_thread, b);
  const int hi = min(lo + per_thread, b);
  int local = 0;
  for (int i = lo; i < hi; ++i) {
    const int cost = num_blocks_of(p, seqlen_k_of(p, i)) + f;
    cum[i] = cost;  // the row's cost, replaced by its prefix below
    local += cost;
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
    const int cost = cum[i];
    cum[i] = run;
    run += cost;
  }
  if (hi == b && lo < hi) cum[b] = run;
  if (b == 0 && threadIdx.x == 0) cum[0] = 0;
  __syncthreads();
}

// Largest k in [r, b] with cum[k] <= target.
__device__ __forceinline__ int last_row_at_or_below(const int* cum, int r, int b, int target) {
  int lo_k = r, hi_k = b;
  while (lo_k < hi_k) {
    const int mid = (lo_k + hi_k + 1) >> 1;
    if (cum[mid] <= target) {
      lo_k = mid;
    } else {
      hi_k = mid - 1;
    }
  }
  return lo_k;
}

// One part's metadata from its start state (r0, blk0, split0) and end state (r1, blk1).
__device__ __forceinline__ void
write_part_meta(int* meta, const int* cum, int f, int r0, int blk0, int split0, int r1, int blk1) {
  int is_first_req_splitted = (blk0 != 0);
  const int split_open = blk1 > 0;
  const int end_req_idx = split_open ? r1 : r1 - 1;
  // A sparse row's KV length is never 0 (0 becomes 1), so FlashMLA's
  // `end_sk == 0` branches never fire and the row's blocks come from `cum`.
  const int end_lb = cum[end_req_idx + 1] - cum[end_req_idx] - f - 1;
  const int end_block_idx = split_open ? blk1 : end_lb + 1;
  int is_last_req_splitted = (end_block_idx != end_lb + 1);
  if (r0 == end_req_idx) {
    is_first_req_splitted = is_last_req_splitted = is_first_req_splitted || is_last_req_splitted;
  }
  meta[0] = r0;
  meta[1] = end_req_idx;
  meta[2] = blk0;
  meta[3] = end_block_idx;
  meta[4] = split0;
  meta[5] = is_first_req_splitted;
  meta[6] = is_last_req_splitted;
  meta[7] = 0;
}

// 2. The partition walk on thread 0, jumping over whole rows. Writes the parts it
// fills, *first_idle and *num_split_rows.
__device__ __forceinline__ void
serial_walk(const Params& p, const int* cum, int* out_shared, int* split_rows, int* first_idle, int* num_split_rows) {
  const int b = p.b;
  const int f = p.fixed_overhead_num_blocks;
  const int total = cum[b];
  const int payload = ceil_div_i(total, p.num_sm_parts) + f;
  int r = 0, blk = 0, split = 0, nsplit_rows = 0;
  int i = 0;
  for (; i < p.num_sm_parts; ++i) {
    const int r0 = r, blk0 = blk, split0 = split;
    int remain = payload;
    const int need = (cum[r + 1] - cum[r] - f) - blk + f;
    if (remain >= need) {
      // Row r ends in this part; then every row that fits whole.
      remain -= need;
      ++r;
      blk = 0;
      split = 0;
      const int lo_k = last_row_at_or_below(cum, r, b, cum[r] + remain);
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
    write_part_meta(out_shared + i * kMetaInts, cum, f, r0, blk0, split0, r, blk);
    if (r == b) {
      ++i;
      break;
    }
  }
  *first_idle = i;
  *num_split_rows = nsplit_rows;
}

// 3. Parts past the walk, exactly as FlashMLA (and flashmla_sched_meta.cuh)
// writes them: begin_block_idx reads one past FlashMLA's first-block array,
// which aliases row 0's last block index. 4. Copy out, and num_splits.
// Starts and ends with __syncthreads().
__device__ __forceinline__ void
finish(const Params& p, const int* cum, int* out_shared, const int* split_rows, int first_idle, int nsr) {
  const int b = p.b;
  const int f = p.fixed_overhead_num_blocks;
  __syncthreads();
  {
    const int lb_last = cum[b] - cum[b - 1] - f - 1;
    const int end_block_idx = lb_last + 1;
    const int begin_block_idx = cum[1] - cum[0] - f - 1;
    const int is_last_req_splitted = 0;
    for (int i = first_idle + threadIdx.x; i < p.num_sm_parts; i += kThreads) {
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

__global__ void __launch_bounds__(kThreads) flashmla_sched_meta_large_kernel(__grid_constant__ const Params p) {
  extern __shared__ int smem[];
  const int b = p.b;
  int* cum = smem;                                            // [b + 1], exclusive prefix of num_blocks + f
  int* out_shared = smem + b + 1;                             // [num_sm_parts * kMetaInts]
  int* split_rows = out_shared + p.num_sm_parts * kMetaInts;  // [num_sm_parts], rows split, ascending

  __shared__ int warp_sums[kThreads / 32];
  __shared__ int num_split_rows_shared;
  __shared__ int first_idle_part_shared;

  prefix_costs(p, cum, warp_sums);
  if (threadIdx.x == 0) {
    serial_walk(p, cum, out_shared, split_rows, &first_idle_part_shared, &num_split_rows_shared);
  }
  __syncthreads();
  finish(p, cum, out_shared, split_rows, first_idle_part_shared, num_split_rows_shared);
}

// INF-874 decode attention A2: the same metadata for a decode-sized batch, with the
// walk's per-part step taken out of the serial loop. Every walk state (row r, first
// block blk) is numbered cum[r] + blk, and the end state cum[b]. The whole block
// computes, for every state in parallel, the state the walk reaches one part later
// (one row search each), then kJump parts later; thread 0 follows only the kJump
// table (one shared-memory load per kJump parts), the block fills in the parts in
// between, and every part's metadata is written in parallel. The tables need
// cum[b] + 1 entries each; when the real cum[b] exceeds the capacity the host sized
// them for (lengths past the index widths), the kernel falls back to serial_walk.
constexpr int kRowBits = 19;
constexpr int kBlkBits = 12;  // a packed state is (row << kBlkBits) | blk
constexpr int kBlkMask = (1 << kBlkBits) - 1;
constexpr int kJump = 4;

__device__ __forceinline__ int pack_state(int r, int blk) {
  return (r << kBlkBits) | blk;
}

__global__ void __launch_bounds__(kThreads)
    flashmla_sched_meta_chain_kernel(__grid_constant__ const Params p, const int capacity) {
  extern __shared__ int smem[];
  const int b = p.b;
  const int f = p.fixed_overhead_num_blocks;
  const int parts = p.num_sm_parts;
  const int num_hops = ceil_div_i(parts, kJump) + 1;
  int* cum = smem;                                   // [b + 1]
  int* out_shared = cum + b + 1;                     // [parts * kMetaInts]
  int* split_rows = out_shared + parts * kMetaInts;  // [parts]
  int* chain = split_rows + parts;                   // [parts + 1]: state number at each part's start
  int* hops = chain + parts + 1;                     // [num_hops]: chain[kJump * k]
  int* next_num = hops + num_hops;                   // [capacity + 1]: state number one part later
  int* next_state = next_num + capacity + 1;         // [capacity + 1]: that state, packed
  int* jump_num = next_state + capacity + 1;         // [capacity + 1]: state number kJump parts later

  __shared__ int warp_sums[kThreads / 32];
  __shared__ int num_split_rows_shared;
  __shared__ int first_idle_part_shared;

  prefix_costs(p, cum, warp_sums);
  const int total = cum[b];
  if (total > capacity) {
    if (threadIdx.x == 0) {
      serial_walk(p, cum, out_shared, split_rows, &first_idle_part_shared, &num_split_rows_shared);
    }
    __syncthreads();
    finish(p, cum, out_shared, split_rows, first_idle_part_shared, num_split_rows_shared);
    return;
  }
  const int payload = ceil_div_i(total, parts) + f;

  // a. One part's step from every state, exactly as serial_walk's loop body. The
  // numbers that are no state (a row's last f) and the end state lead to the end.
  for (int r = threadIdx.x; r < b; r += kThreads) {
    const int nb = cum[r + 1] - cum[r] - f;
    int lo_k = r + 1;
    for (int blk = 0; blk < nb; ++blk) {
      int r1 = r, blk1 = blk;
      int remain = payload;
      const int need = nb - blk + f;
      if (remain >= need) {
        remain -= need;
        // Largest k in [r + 1, b] with cum[k] <= target. The target grows with blk,
        // so the search gallops forward from the previous answer (a few rows).
        const int target = cum[r + 1] + remain;
        int step = 1;
        while (lo_k + step <= b && cum[lo_k + step] <= target) {
          lo_k += step;
          step <<= 1;
        }
        lo_k = last_row_at_or_below(cum, lo_k, min(lo_k + step - 1, b), target);
        remain -= cum[lo_k] - cum[r + 1];
        r1 = lo_k;
        blk1 = (r1 < b && remain - f > 0) ? remain - f : 0;
      } else if (remain - f > 0) {
        blk1 = blk + remain - f;
      }
      next_num[cum[r] + blk] = cum[r1] + blk1;
      next_state[cum[r] + blk] = pack_state(r1, blk1);
    }
    for (int s = cum[r] + max(nb, 0); s < cum[r + 1]; ++s) {
      next_num[s] = total;
    }
  }
  if (threadIdx.x == 0) {
    next_num[total] = total;
    next_state[total] = pack_state(b, 0);
  }
  __syncthreads();
  for (int s = threadIdx.x; s <= total; s += kThreads) {
    int j = s;
#pragma unroll
    for (int k = 0; k < kJump; ++k)
      j = next_num[j];
    jump_num[s] = j;
  }
  __syncthreads();

  // b. Thread 0 follows the kJump table from state 0; the block fills the parts in
  // between and finds the first part that starts at the end state.
  if (threadIdx.x == 0) {
    int s = 0;
    for (int k = 0; k < num_hops; ++k) {
      hops[k] = s;
      s = jump_num[s];
    }
  }
  __syncthreads();
  // One part per thread (the host keeps parts < kThreads). The walk only moves
  // forward, so the parts that start before the end state are a prefix: their count
  // is the first part that starts at the end state = 1 + the part that reached it
  // (serial_walk's first_idle), or parts + 1 when none did.
  bool walking = false;
  if (threadIdx.x <= parts) {
    const int i = threadIdx.x;
    int s = hops[i / kJump];
    for (int k = 0; k < i % kJump; ++k)
      s = next_num[s];
    chain[i] = s;
    walking = s != total;
  }
  const int first_idle = min(__syncthreads_count(walking), parts);

  // c. Split records (parts that end inside a row), in part order, by warp 0.
  if (threadIdx.x < 32) {
    int base = 0;
    for (int i0 = 0; i0 < first_idle; i0 += 32) {
      const int i = i0 + threadIdx.x;
      const int st = i < first_idle ? next_state[chain[i]] : 0;
      const bool split = i < first_idle && (st & kBlkMask) > 0;
      const unsigned ballot = __ballot_sync(uint32_t(-1), split);
      if (split) split_rows[base + __popc(ballot & ((1u << threadIdx.x) - 1))] = st >> kBlkBits;
      base += __popc(ballot);
    }
    if (threadIdx.x == 0) num_split_rows_shared = base;
  }

  // d. Each walked part's metadata from its start and end states.
  for (int i = threadIdx.x; i < first_idle; i += kThreads) {
    const int st0 = i == 0 ? 0 : next_state[chain[i - 1]];
    const int st1 = next_state[chain[i]];
    const int r0 = st0 >> kBlkBits, blk0 = st0 & kBlkMask;
    // split index = how many earlier parts ended inside row r0 (consecutive ones).
    int split0 = 0;
    for (int k = i; k > 0; --k) {
      const int sk = next_state[chain[k - 1]];
      if ((sk & kBlkMask) == 0 || (sk >> kBlkBits) != r0) break;
      ++split0;
    }
    write_part_meta(out_shared + i * kMetaInts, cum, f, r0, blk0, split0, st1 >> kBlkBits, st1 & kBlkMask);
  }
  __syncthreads();
  finish(p, cum, out_shared, split_rows, first_idle, num_split_rows_shared);
}

}  // namespace flashmla_large

inline std::size_t flashmla_sched_meta_large_smem_bytes(int64_t b, int64_t num_sm_parts) {
  return sizeof(int) * static_cast<std::size_t>(b + 1 + num_sm_parts * flashmla_large::kMetaInts + num_sm_parts);
}

namespace flashmla_large {

inline Params make_params(
    tvm::ffi::TensorView tile_scheduler_metadata,
    tvm::ffi::TensorView num_splits,
    const tvm::ffi::Optional<tvm::ffi::TensorView>& topk_length,
    const tvm::ffi::Optional<tvm::ffi::TensorView>& extra_topk_length,
    int64_t block_size_n,
    int64_t fixed_overhead_num_blocks,
    int64_t topk,
    int64_t extra_topk,
    DLDevice* device) {
  using namespace host;

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

  *device = device_.unwrap();
  return Params{
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
}

// Opt in to the device's whole per-block shared memory once per process and
// kernel; every later launch then only checks the size. The dynamic part is the
// opt-in limit minus the kernel's static shared memory.
template <typename Kernel>
inline int opt_in_smem(Kernel kernel) {
  using namespace host;
  int device = 0, bytes = 0;
  RuntimeDeviceCheck(cudaGetDevice(&device));
  RuntimeDeviceCheck(cudaDeviceGetAttribute(&bytes, cudaDevAttrMaxSharedMemoryPerBlockOptin, device));
  cudaFuncAttributes attr{};
  RuntimeDeviceCheck(cudaFuncGetAttributes(&attr, kernel));
  bytes -= static_cast<int>(attr.sharedSizeBytes);
  RuntimeDeviceCheck(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes));
  return bytes;
}

}  // namespace flashmla_large

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

  DLDevice device;
  const Params p = make_params(
      tile_scheduler_metadata,
      num_splits,
      topk_length,
      extra_topk_length,
      block_size_n,
      fixed_overhead_num_blocks,
      topk,
      extra_topk,
      &device);
  const std::size_t smem = flashmla_sched_meta_large_smem_bytes(p.b, p.num_sm_parts);
  static int max_optin = opt_in_smem(flashmla_sched_meta_large_kernel);
  RuntimeCheck(smem <= static_cast<std::size_t>(max_optin), "schedule does not fit in shared memory: ", smem, " bytes");
  LaunchKernel(1, kThreads, device, smem)(flashmla_sched_meta_large_kernel, p);
}

// The chain kernel's shared memory for tables of `capacity` + 1 states.
inline std::size_t flashmla_sched_meta_chain_smem_bytes(int64_t b, int64_t num_sm_parts, int64_t capacity) {
  const int64_t num_hops = (num_sm_parts + flashmla_large::kJump - 1) / flashmla_large::kJump + 1;
  return sizeof(int) * static_cast<std::size_t>(
                           b + 1 + num_sm_parts * flashmla_large::kMetaInts + num_sm_parts + num_sm_parts + 1 +
                           num_hops + 3 * (capacity + 1));
}

// INF-874 A2: the same metadata as flashmla_sched_meta_large, for decode-sized
// batches (see flashmla_sched_meta_chain_kernel). `capacity` bounds cum[b], the sum of
// every row's num_blocks + fixed overhead (the caller's bound from the index widths).
void flashmla_sched_meta_chain(
    tvm::ffi::TensorView tile_scheduler_metadata,
    tvm::ffi::TensorView num_splits,
    tvm::ffi::Optional<tvm::ffi::TensorView> topk_length,
    tvm::ffi::Optional<tvm::ffi::TensorView> extra_topk_length,
    int64_t block_size_n,
    int64_t fixed_overhead_num_blocks,
    int64_t topk,
    int64_t extra_topk,
    int64_t capacity) {
  using namespace host;
  using namespace flashmla_large;

  DLDevice device;
  const Params p = make_params(
      tile_scheduler_metadata,
      num_splits,
      topk_length,
      extra_topk_length,
      block_size_n,
      fixed_overhead_num_blocks,
      topk,
      extra_topk,
      &device);
  RuntimeCheck(p.b < (1 << kRowBits), "chain schedule: at most 2^19 rows, got ", p.b);
  RuntimeCheck(p.num_sm_parts < kThreads, "chain schedule: at most ", kThreads - 1, " parts");
  RuntimeCheck(capacity >= 1 && capacity < (int64_t{1} << 30), "chain schedule: bad capacity ", capacity);
  const std::size_t smem = flashmla_sched_meta_chain_smem_bytes(p.b, p.num_sm_parts, capacity);
  static int max_optin = opt_in_smem(flashmla_sched_meta_chain_kernel);
  RuntimeCheck(smem <= static_cast<std::size_t>(max_optin), "schedule does not fit in shared memory: ", smem, " bytes");
  LaunchKernel(1, kThreads, device, smem)(flashmla_sched_meta_chain_kernel, p, static_cast<int>(capacity));
}

}  // namespace sglang
