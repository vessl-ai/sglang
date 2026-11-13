#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include "w4a8_grouped_mm_c3x.cuh"

using namespace cute;

namespace {

enum class Sched { PP, CO };

template <int M, int N, int K, int A, Sched S>
void run(
    torch::Tensor& d,
    torch::Tensor const& a,
    torch::Tensor const& b,
    torch::Tensor const& a_scales,
    torch::Tensor const& b_scales,
    torch::Tensor const& expert_offsets,
    torch::Tensor const& problem_sizes,
    torch::Tensor const& a_strides,
    torch::Tensor const& b_strides,
    torch::Tensor const& d_strides,
    torch::Tensor const& s_strides,
    int64_t chunk_size) {
  using KernelSchedule = std::conditional_t<
      S == Sched::PP,
      cutlass::gemm::KernelPtrArrayTmaWarpSpecializedPingpong,
      cutlass::gemm::KernelPtrArrayTmaWarpSpecializedCooperative>;
  using EpilogueSchedule = std::conditional_t<
      S == Sched::PP,
      cutlass::epilogue::PtrArrayTmaWarpSpecializedPingpong,
      cutlass::epilogue::PtrArrayTmaWarpSpecializedCooperative>;
  using Gemm = cutlass_3x_w4a8_group_gemm<Shape<Int<M>, Int<N>, Int<K>>, Shape<Int<A>, _1, _1>, KernelSchedule, EpilogueSchedule>;
  cutlass_w4a8_group_gemm_caller<Gemm>(
      d, a, b, a_scales, b_scales, expert_offsets, problem_sizes, a_strides, b_strides, d_strides, s_strides, chunk_size);
}

}  // namespace

void w4a8_moe_mm_k128(
    torch::Tensor& d,
    torch::Tensor const& a,
    torch::Tensor const& b,
    torch::Tensor const& a_scales,
    torch::Tensor const& b_scales,
    torch::Tensor const& expert_offsets,
    torch::Tensor const& problem_sizes,
    torch::Tensor const& a_strides,
    torch::Tensor const& b_strides,
    torch::Tensor const& d_strides,
    torch::Tensor const& s_strides,
    int64_t chunk_size,
    int64_t topk) {
  const c10::cuda::CUDAGuard device_guard(a.device());
  int64_t const m = a.size(0) / topk;
  int64_t const n = d.size(1);
  int64_t const k = a.size(1);
#define RUN(M, N, K, A, S) \
  run<M, N, K, A, Sched::S>(d, a, b, a_scales, b_scales, expert_offsets, problem_sizes, a_strides, b_strides, d_strides, s_strides, chunk_size)
  // Tiles and thresholds measured on H200; re-tune when the w4a8 kernel changes.
  if (n == 1024 && k == 4096) {
    if (m <= 256) {
      RUN(256, 16, 128, 1, CO);
    } else if (m <= 768) {
      RUN(256, 32, 128, 1, CO);
    } else if (m <= 2048) {
      RUN(256, 64, 128, 1, CO);
    } else {
      RUN(128, 128, 128, 1, CO);
    }
  } else if (n == 4096 && k == 512) {
    if (m <= 256) {
      RUN(128, 16, 128, 1, PP);
    } else if (m <= 768) {
      RUN(128, 32, 128, 2, PP);
    } else if (m <= 2048) {
      RUN(128, 64, 128, 1, PP);
    } else {
      RUN(128, 128, 128, 1, CO);
    }
  } else {
    TORCH_CHECK(false, "w4a8_moe_mm_k128 has no tiles for n=", n, " k=", k);
  }
#undef RUN
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("w4a8_moe_mm_k128", &w4a8_moe_mm_k128);
}
