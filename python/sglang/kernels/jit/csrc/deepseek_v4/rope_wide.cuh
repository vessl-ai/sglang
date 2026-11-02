// INF-681 k-attnp: in-place DeepSeek-V4 RoPE over the last 64 dims of every head of a
// [batch, heads, *] bf16 tensor, one CTA per row (token) instead of one warp per (row, head).
// Each thread moves 16 bytes (4 complex pairs) per head pass and reads its 4 frequencies once
// per row, so a CTA keeps 4 x 16 B per thread in flight instead of 4 B per lane. The per-pair
// arithmetic is written exactly as deepseek_rope_kernel (rope.cuh) so the results are the same
// bits.
#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>

#include <tvm/ffi/container/tensor.h>

#include <cstdint>

namespace sglang {

namespace rope_wide {

using DType = bf16_t;
using DType2 = packed_t<DType>;
constexpr int64_t kRopeDim = 64;
constexpr uint32_t kBlockSize = 128;
constexpr uint32_t kPairsPerThread = 4;                               // 16 B of bf16
constexpr uint32_t kThreadsPerHead = kRopeDim / 2 / kPairsPerThread;  // 8
constexpr uint32_t kHeadsPerPass = kBlockSize / kThreadsPerHead;      // 16
constexpr uint32_t kMaxPasses = 8;                                    // up to 128 heads

struct Params {
  void* __restrict__ x;
  const float* __restrict__ freqs_cis;
  const void* __restrict__ positions;
  int64_t stride_batch;  // bytes
  int64_t stride_head;   // bytes
  uint32_t num_heads;
  uint32_t batch_size;
};

template <bool kUsePDL, bool kInverse, typename IndexType, uint32_t kPasses>
__global__ __launch_bounds__(kBlockSize, 8)  //
    void deepseek_rope_wide_kernel(const __grid_constant__ Params param) {
  using namespace device;
  using Vec = AlignedVector<DType2, kPairsPerThread>;
  using FVec = AlignedVector<fp32x2_t, 2>;

  const uint32_t batch_id = blockIdx.x;
  if (batch_id >= param.batch_size) return;
  const uint32_t sub = threadIdx.x % kThreadsPerHead;    // which 4 pairs of the 32
  const uint32_t head0 = threadIdx.x / kThreadsPerHead;  // head within a pass

  const auto position = static_cast<const IndexType*>(param.positions)[batch_id];
  const auto freq_ptr =
      reinterpret_cast<const fp32x2_t*>(param.freqs_cis + position * kRopeDim) + sub * kPairsPerThread;
  FVec f01, f23;
  f01.load(freq_ptr);
  f23.load(freq_ptr + 2);
  const fp32x2_t f[kPairsPerThread] = {f01[0], f01[1], f23[0], f23[1]};

  PDLWaitPrimary<kUsePDL>();

  void* row = pointer::offset(param.x, static_cast<int64_t>(batch_id) * param.stride_batch);
  Vec data[kPasses];
#pragma unroll
  for (uint32_t p = 0; p < kPasses; ++p) {
    const uint32_t head = p * kHeadsPerPass + head0;
    if (head < param.num_heads) {
      data[p].load(pointer::offset(row, static_cast<int64_t>(head) * param.stride_head), sub);
    }
  }
#pragma unroll
  for (uint32_t p = 0; p < kPasses; ++p) {
    const uint32_t head = p * kHeadsPerPass + head0;
    if (head < param.num_heads) {
      Vec out;
#pragma unroll
      for (uint32_t i = 0; i < kPairsPerThread; ++i) {
        const auto [x_real, x_imag] = cast<fp32x2_t>(data[p][i]);
        const auto [f_real, f_imag] = f[i];
        fp32x2_t output;
        if constexpr (kInverse) {
          // (a + bi) * (c - di) = (ac + bd) + (bc - ad)i
          output = {
              x_real * f_real + x_imag * f_imag,
              x_imag * f_real - x_real * f_imag,
          };
        } else {
          // (a + bi) * (c + di) = (ac - bd) + (ad + bc)i
          output = {
              x_real * f_real - x_imag * f_imag,
              x_real * f_imag + x_imag * f_real,
          };
        }
        out[i] = cast<DType2>(output);
      }
      out.store(pointer::offset(row, static_cast<int64_t>(head) * param.stride_head), sub);
    }
  }

  PDLTriggerSecondary<kUsePDL>();
}

}  // namespace rope_wide

template <bool kUsePDL>
struct FusedRopeWideKernel {
  template <bool kInverse, typename IndexType, uint32_t kPasses>
  static constexpr auto kernel = rope_wide::deepseek_rope_wide_kernel<kUsePDL, kInverse, IndexType, kPasses>;

  template <bool kInverse, typename IndexType>
  static auto pick(uint32_t passes) {
    using K = decltype(kernel<kInverse, IndexType, 1>);
    K table[] = {
        kernel<kInverse, IndexType, 1>,
        kernel<kInverse, IndexType, 2>,
        kernel<kInverse, IndexType, 4>,
        kernel<kInverse, IndexType, 4>,
        kernel<kInverse, IndexType, 8>,
        kernel<kInverse, IndexType, 8>,
        kernel<kInverse, IndexType, 8>,
        kernel<kInverse, IndexType, 8>,
    };
    return table[passes - 1];
  }

  static void forward(
      const tvm::ffi::TensorView x,
      const tvm::ffi::TensorView freqs_cis,
      const tvm::ffi::TensorView positions,
      bool inverse) {
    using namespace host;
    using namespace rope_wide;

    auto B = SymbolicSize{"batch_size"};
    auto H = SymbolicSize{"num_heads"};
    constexpr auto D = kRopeDim;
    auto device_ = SymbolicDevice{};
    device_.set_options<kDLCUDA>();

    TensorMatcher({B, H, D})  //
        .with_strides({-1, -1, 1})
        .with_dtype<DType>()
        .with_device(device_)
        .verify(x);
    TensorMatcher({-1, D})  //
        .with_dtype<float>()
        .with_device(device_)
        .verify(freqs_cis);
    auto pos_dtype = SymbolicDType{};
    TensorMatcher({B})  //
        .with_dtype<int32_t, int64_t>(pos_dtype)
        .with_device(device_)
        .verify(positions);
    const bool pos_i32 = pos_dtype.is_type<int32_t>();

    const auto batch_size = static_cast<uint32_t>(B.unwrap());
    if (batch_size == 0) return;
    const auto num_heads = static_cast<uint32_t>(H.unwrap());
    const uint32_t passes = div_ceil(num_heads, kHeadsPerPass);
    RuntimeCheck(passes >= 1 && passes <= kMaxPasses, "rope_wide: at most 128 heads");

    const auto elem_size = static_cast<int64_t>(sizeof(DType));
    const auto stride_batch = x.stride(0) * elem_size;
    const auto stride_head = x.stride(1) * elem_size;
    RuntimeCheck(
        reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0 && stride_batch % 16 == 0 && stride_head % 16 == 0,
        "rope_wide: rows and heads must be 16-byte aligned");
    const auto params = Params{
        .x = x.data_ptr(),
        .freqs_cis = static_cast<const float*>(freqs_cis.data_ptr()),
        .positions = positions.data_ptr(),
        .stride_batch = stride_batch,
        .stride_head = stride_head,
        .num_heads = num_heads,
        .batch_size = batch_size,
    };
    const auto kernel = inverse ? (pos_i32 ? pick<true, int32_t>(passes) : pick<true, int64_t>(passes))
                                : (pos_i32 ? pick<false, int32_t>(passes) : pick<false, int64_t>(passes));
    LaunchKernel(batch_size, kBlockSize, device_.unwrap())  //
        .enable_pdl(kUsePDL)(kernel, params);
  }
};

}  // namespace sglang
