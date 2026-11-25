// SPDX-License-Identifier: Apache-2.0
//
// K1 (INF-874): SM90 W4A8 MoE expert kernels for DeepSeek-V4 MXFP4 experts in expert parallel (hidden 5120,
// intermediate 2304); the engine's EP decode MoE path with SGLANG_OPT_DSV4_EP_DECODE_MOE_KERNEL=k1
// (dsv4_mxfp4_decode_moe_ep -> dsv4_mxfp4_decode_moe_k1.decode_ep). Compared with v2w (dsv4_mxfp4_decode_moe_v2w.cuh):
//
// - MX block scales folded per channel at load time (as the prefill W4A8 path, mxfp4_w4a8_prefill.py): within one
//   weight row the e2m1 value of block b becomes e4m3(e2m1 x 2^(6 - k_b)), k_b = max_b s_b - s_b <= 11 (exact; a
//   layer with a larger span keeps v2w), and one channel scale 2^(max_b s_b - 133) is applied in the epilogue. The
//   conversion is a per-(row, block) 8-entry magnitude table (2 IMAD) and one prmt + sign merge per 4 values; the
//   nibbles are repacked at load time so that one 32-bit word is one thread's 8 A values of one row and block.
// - The tensor core accumulates all of K (no fp32 promotion per MX block): its error against fp32 math is ~1.05x
//   v2w's (h of gate/up: 0.36% vs 0.35%).
// - One CTA = one tile (expert, 128 weight rows, up to 128 activation rows): two consumer warpgroups share the
//   activation tile (wgmma B, shared memory) and each holds 64 weight rows as A (registers). For gate/up, warp q's
//   rows 0-7 are gate channels and rows 8-15 the same up channels, so SwiGLU ends inside a thread.
// - Activation rows are laid out once per call stage-major and pre-swizzled ([stages][rows][128 B]), so a tile's
//   stage is one bulk copy; a producer warp streams weights + activations through an 8-stage ring.
// - Tiles come from a device-built list (expert, channel block, first sorted row, rows <= 128) through one counter.

#pragma once

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <algorithm>
#include <cstdint>
#include <cuda_bf16.h>

namespace sglang {
namespace dsv4_moe_k1 {

constexpr int kH = 5120;
constexpr int kI = 2304;
constexpr int kWStage = 8192 + 512;  // per stage: 128 rows x 128 k e2m1 (repacked words) + 128 x 4 block offsets

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
SGL_DEVICE int uniform(int v) {
  return __shfl_sync(0xffffffffu, v, 0);
}

// D[N/2] (+)= A(regs) x B(desc); acc = 0: D overwritten.
template <int N>
SGL_DEVICE void wgmma_acc(float (&d)[N / 2], const uint32_t (&a)[4], uint64_t desc, int acc);
template <>
SGL_DEVICE void wgmma_acc<16>(float (&d)[8], const uint32_t (&a)[4], uint64_t desc, int acc) {
  asm volatile(
      "{\n.reg .pred p;\nsetp.ne.b32 p, %13, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n16k32.f32.e4m3.e4m3 "
      "{%0, %1, %2, %3, %4, %5, %6, %7}, "
      "{%8, %9, %10, %11}, %12, p, 1, 1;\n}\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc), "r"(acc));
}
template <>
SGL_DEVICE void wgmma_acc<32>(float (&d)[16], const uint32_t (&a)[4], uint64_t desc, int acc) {
  asm volatile(
      "{\n.reg .pred p;\nsetp.ne.b32 p, %21, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n32k32.f32.e4m3.e4m3 "
      "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, "
      "{%16, %17, %18, %19}, %20, p, 1, 1;\n}\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc), "r"(acc));
}
template <>
SGL_DEVICE void wgmma_acc<48>(float (&d)[24], const uint32_t (&a)[4], uint64_t desc, int acc) {
  asm volatile(
      "{\n.reg .pred p;\nsetp.ne.b32 p, %29, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n48k32.f32.e4m3.e4m3 "
      "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23}, "
      "{%24, %25, %26, %27}, %28, p, 1, 1;\n}\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc), "r"(acc));
}
template <>
SGL_DEVICE void wgmma_acc<64>(float (&d)[32], const uint32_t (&a)[4], uint64_t desc, int acc) {
  asm volatile(
      "{\n.reg .pred p;\nsetp.ne.b32 p, %37, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n64k32.f32.e4m3.e4m3 "
      "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, "
      "{%32, %33, %34, %35}, %36, p, 1, 1;\n}\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc), "r"(acc));
}
template <>
SGL_DEVICE void wgmma_acc<80>(float (&d)[40], const uint32_t (&a)[4], uint64_t desc, int acc) {
  asm volatile(
      "{\n.reg .pred p;\nsetp.ne.b32 p, %45, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n80k32.f32.e4m3.e4m3 "
      "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39}, "
      "{%40, %41, %42, %43}, %44, p, 1, 1;\n}\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]), "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35]), "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc), "r"(acc));
}
template <>
SGL_DEVICE void wgmma_acc<96>(float (&d)[48], const uint32_t (&a)[4], uint64_t desc, int acc) {
  asm volatile(
      "{\n.reg .pred p;\nsetp.ne.b32 p, %53, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n96k32.f32.e4m3.e4m3 "
      "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47}, "
      "{%48, %49, %50, %51}, %52, p, 1, 1;\n}\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]), "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35]), "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]), "+f"(d[40]), "+f"(d[41]), "+f"(d[42]), "+f"(d[43]), "+f"(d[44]), "+f"(d[45]), "+f"(d[46]), "+f"(d[47])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc), "r"(acc));
}
template <>
SGL_DEVICE void wgmma_acc<112>(float (&d)[56], const uint32_t (&a)[4], uint64_t desc, int acc) {
  asm volatile(
      "{\n.reg .pred p;\nsetp.ne.b32 p, %61, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n112k32.f32.e4m3.e4m3 "
      "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55}, "
      "{%56, %57, %58, %59}, %60, p, 1, 1;\n}\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]), "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35]), "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]), "+f"(d[40]), "+f"(d[41]), "+f"(d[42]), "+f"(d[43]), "+f"(d[44]), "+f"(d[45]), "+f"(d[46]), "+f"(d[47]), "+f"(d[48]), "+f"(d[49]), "+f"(d[50]), "+f"(d[51]), "+f"(d[52]), "+f"(d[53]), "+f"(d[54]), "+f"(d[55])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc), "r"(acc));
}
template <>
SGL_DEVICE void wgmma_acc<128>(float (&d)[64], const uint32_t (&a)[4], uint64_t desc, int acc) {
  asm volatile(
      "{\n.reg .pred p;\nsetp.ne.b32 p, %69, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n128k32.f32.e4m3.e4m3 "
      "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, "
      "{%64, %65, %66, %67}, %68, p, 1, 1;\n}\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]), "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35]), "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]), "+f"(d[40]), "+f"(d[41]), "+f"(d[42]), "+f"(d[43]), "+f"(d[44]), "+f"(d[45]), "+f"(d[46]), "+f"(d[47]), "+f"(d[48]), "+f"(d[49]), "+f"(d[50]), "+f"(d[51]), "+f"(d[52]), "+f"(d[53]), "+f"(d[54]), "+f"(d[55]), "+f"(d[56]), "+f"(d[57]), "+f"(d[58]), "+f"(d[59]), "+f"(d[60]), "+f"(d[61]), "+f"(d[62]), "+f"(d[63])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc), "r"(acc));
}

// ---------------------------------------------------------------------------------------------
// One repacked word = one thread's 8 A values of one weight row in one k32 block: values 0-3 (k 4c..4c+3) have
// their magnitude codes in nibbles 0-3 (bits 4i..4i+2) and signs at bits 8i+7; values 4-7 (k 16+4c..16+4c+3) have
// magnitudes in nibbles 4-7 and signs at bits 8i+3. (tlo, thi) = e4m3 bytes of magnitudes 0-7 for this block.
SGL_DEVICE void convert_word(uint32_t q, uint32_t tlo, uint32_t thi, uint32_t& lo, uint32_t& hi) {
  lo = __byte_perm(tlo, thi, q & 0x7777u) | (q & 0x80808080u);
  hi = __byte_perm(tlo, thi, (q >> 16) & 0x7777u) | ((q << 4) & 0x80808080u);
}
// e4m3 of e2m1 magnitudes {0, .5, 1, 1.5, 2, 3, 4, 6} x 2^(6 - k): exponent field minus k (zero stays zero).
template <int kB>
SGL_DEVICE void table(uint32_t sw, uint32_t& tlo, uint32_t& thi) {
  const uint32_t k = (sw >> (8 * kB)) & 0xFFu;
  tlo = 0x6C686000u - k * 0x08080800u;
  thi = 0x7C787470u - k * 0x08080808u;
}
template <int kB>
SGL_DEVICE void convert_block(uint32_t w_h0, uint32_t w_h1, uint2 sc, uint32_t (&a)[4]) {
  uint32_t t0l, t0h, t1l, t1h;
  table<kB>(sc.x, t0l, t0h);
  table<kB>(sc.y, t1l, t1h);
  convert_word(w_h0, t0l, t0h, a[0], a[2]);  // row grp (gate channel)
  convert_word(w_h1, t1l, t1h, a[1], a[3]);  // row grp + 8 (the same up channel)
}

// =============================================================================================
// K1X: the whole expert side of one MoE call with the configuration M1 chose (one pipeline per CTA, producer
// warp + two consumer warpgroups, tiles of up to 128 activation rows, the tensor core accumulates all of K, waits
// drained every few stages, k32-block accumulator chains for tiles of <= 32 rows, staged epilogue). Launches:
//   count, place     : expert sort of the valid slots (rows of each expert start at a multiple of 16), single-slot
//                      tokens, the reduce list, the valid-slot list, gate/up and down tile lists, counters
//   gather           : e4m3 token rows -> [40][rpad][128] stage-major pre-swizzled rows + row scales
//   gate_up (kernel<false>) : h = SwiGLU(x W13^T), bf16 [rpad][2304]
//   quant            : h rows -> e4m3 [18][rpad][128] pre-swizzled + power-of-2 row scale
//   down    (kernel<true>)  : y = gamma * h W2^T, bf16; a token's only slot is written to its output row, others to
//                      y[slot]
//   reduce           : tokens with several slots: out[t] = sum of y over its slots in slot order (fp32, one round)
// Slots: either the engine's top-k layout (slot s = t * K + k, ids [T * K], -1 = not local) or a pair list
// (explicit token of each slot, slots of a token adjacent, num_valid on the device).
namespace k1x {
constexpr int kNMax = 128;
constexpr int kX = kNMax * 128;
constexpr int kSlot = (kX + kWStage + 1023) / 1024 * 1024;
constexpr int kRing = 8;
constexpr int kEpi = kNMax * 64;  // per consumer warpgroup: rows x 32 channels bf16
constexpr int kBarOff = kRing * kSlot;
constexpr size_t kSmem = 1024 + kBarOff + kRing * 32 + 2 * kEpi;
static_assert(kSmem <= 232448, "k1x shared memory");
constexpr int kThreads = 288;
constexpr int kMaxE = 512;

struct XP {
  const uint8_t* __restrict__ xa;  // activations [ST][rpad][128]
  const float* __restrict__ xrs;   // [rpad]
  const uint8_t* __restrict__ w;   // packed weights [E][JB][ST][kWStage]
  const float* __restrict__ cs;    // channel scales: gate/up [E][2][2304], down [E][5120]
  const int4* __restrict__ tiles;
  const int32_t* __restrict__ ntiles;  // device count
  int32_t* __restrict__ ctr;
  int rpad;
  float limit;
  __nv_bfloat16* __restrict__ h;       // gate/up output [rpad][2304]
  const float* __restrict__ gam;       // down: routing weight of each sorted row
  const int32_t* __restrict__ dst;     // down: >= 0 slot of y, < 0 ~token row of out
  __nv_bfloat16* __restrict__ y;       // [slots][5120]
  __nv_bfloat16* __restrict__ out;     // [T][5120]
};

#ifndef K1X_SPLIT_N
#define K1X_SPLIT_N 0
#endif
#ifndef K1X_PGU
#define K1X_PGU 2  // stages between wgmma drains (measured: 8 -> 2 is -12 us, a smaller unrolled body)
#endif
#ifndef K1X_PDN
#define K1X_PDN 2
#endif
template <bool kDown>
struct Geo {
  static constexpr int ST = kDown ? 18 : 40;               // 128-k stages
  static constexpr int JB = kDown ? 40 : 36;               // channel blocks (down: 128 output channels)
  static constexpr int P = kDown ? K1X_PDN : K1X_PGU;      // stages between wgmma drains
};

template <bool kDown>
SGL_DEVICE void produce(const XP& p, uint32_t sbase, int4* hdr, uint32_t full, uint32_t empty) {
  using G = Geo<kDown>;
  const int nt = *p.ntiles;
  uint32_t g = 0;
  int t = atomicAdd(p.ctr, 1);
  int4 td = t < nt ? p.tiles[t] : make_int4(-1, 0, 0, 0);
  while (true) {
    uint32_t slot = g % kRing;
    if (g >= static_cast<uint32_t>(kRing)) mbar_wait(empty + slot * 8, ((g / kRing) - 1) & 1);
    hdr[slot] = td;
    if (td.x < 0) {
      mbar_arrive(full + slot * 8);
      break;
    }
    const uint32_t nb = static_cast<uint32_t>((td.w + 15) & ~15) * 128u;
    const uint8_t* ws = p.w + static_cast<size_t>(td.x * G::JB + td.y) * G::ST * kWStage;
    const uint8_t* xs = p.xa + static_cast<size_t>(td.z) * 128;
    int4 nxt = make_int4(-1, 0, 0, 0);
    for (int st = 0; st < G::ST; ++st, ++g) {
      slot = g % kRing;
      if (st > 0 && g >= static_cast<uint32_t>(kRing)) mbar_wait(empty + slot * 8, ((g / kRing) - 1) & 1);
      mbar_arrive_expect_tx(full + slot * 8, kWStage + nb);
      bulk_copy(sbase + slot * kSlot + kX, ws + static_cast<size_t>(st) * kWStage, kWStage, full + slot * 8);
      bulk_copy(sbase + slot * kSlot, xs + static_cast<size_t>(st) * p.rpad * 128, nb, full + slot * 8);
      if (st == 1) {
        const int tn = atomicAdd(p.ctr, 1);
        nxt = tn < nt ? p.tiles[tn] : make_int4(-1, 0, 0, 0);
      }
    }
    td = nxt;
  }
}

template <bool kDown, int N>
SGL_DEVICE void tile(
    const XP& p, uint32_t sbase, const uint8_t* sbasep, uint32_t full, uint32_t empty, uint32_t& g, int4 td, int cw,
    int q, int lane, __nv_bfloat16* ebuf) {
  using G = Geo<kDown>;
  constexpr int kD = N / 2;
  // Independent accumulator chains for narrow tiles (K1X_SPLIT_N > 0: tiles of <= that many rows keep one chain per
  // k32 block, ~3 us faster per call) change the summation order with the tile width, i.e. a row's output would depend
  // on how many rows its expert got in the call. Off by default: every row's output is independent of the batch.
  constexpr int kS = N <= K1X_SPLIT_N ? 4 : 1;
  static_assert(G::ST % G::P == 0 && G::P % 2 == 0, "drain blocks");
  const int grp = lane >> 2, c = lane & 3;
  float t[kS][kD];
#pragma unroll
  for (int u = 0; u < kS; ++u)
#pragma unroll
    for (int i = 0; i < kD; ++i)
      t[u][i] = 0.f;
  const int woff = ((cw * 4 + q) * 2) * 512 + lane * 16;
  const int soff = 8192 + ((cw * 4 + q) * 8 + grp) * 8;
  auto lc = [&](uint32_t gs, uint32_t (&a)[4][4]) {
    const uint32_t slot = gs % kRing;
    const uint8_t* wb = sbasep + slot * kSlot + kX;
    mbar_wait(full + slot * 8, (gs / kRing) & 1);
    const uint4 w0 = *reinterpret_cast<const uint4*>(wb + woff);
    const uint4 w1 = *reinterpret_cast<const uint4*>(wb + woff + 512);
    const uint2 sc = *reinterpret_cast<const uint2*>(wb + soff);
    convert_block<0>(w0.x, w0.y, sc, a[0]);
    convert_block<1>(w0.z, w0.w, sc, a[1]);
    convert_block<2>(w1.x, w1.y, sc, a[2]);
    convert_block<3>(w1.z, w1.w, sc, a[3]);
#pragma unroll
    for (int i = 0; i < 4; ++i)
#pragma unroll
      for (int k = 0; k < 4; ++k)
        asm volatile("" : "+r"(a[i][k])::"memory");
  };
  auto issue = [&](uint32_t gs, const uint32_t (&a)[4][4], int first) {
    const uint32_t xs = sbase + (gs % kRing) * kSlot;
#pragma unroll
    for (int u = 0; u < kS; ++u)
      fence_operands(t[u]);
    wgmma_fence();
#pragma unroll
    for (int k = 0; k < 4; ++k)
      wgmma_acc<N>(t[k % kS], a[k], make_desc(xs + 32 * k), (first && (kS > 1 || k == 0)) ? 0 : 1);
    wgmma_commit();
#pragma unroll
    for (int u = 0; u < kS; ++u)
      fence_operands(t[u]);
  };
  auto release = [&](uint32_t gs) {
    __syncwarp();
    if (lane == 0) mbar_arrive(empty + (gs % kRing) * 8);
  };
  uint32_t aA[4][4], aB[4][4];
  lc(g, aA);
#pragma unroll 1
  for (int blk = 0; blk < G::ST / G::P; ++blk) {
#pragma unroll
    for (int i = 0; i < G::P; i += 2) {
      issue(g, aA, (blk == 0 && i == 0) ? 1 : 0);
      if (i > 0) {
        wgmma_wait<1>();
        release(g - 1);
      }
      lc(g + 1, aB);
      ++g;
      issue(g, aB, 0);
      wgmma_wait<1>();
      release(g - 1);
      if (i + 2 < G::P || blk + 1 < G::ST / G::P) lc(g + 1, aA);
      ++g;
    }
    wgmma_wait<0>();
    release(g - 1);
  }
  float acc[kD];
#pragma unroll
  for (int i = 0; i < kD; ++i)
    acc[i] = 0.f;
#pragma unroll
  for (int u = 0; u < kS; ++u) {
    fence_operands(t[u]);
#pragma unroll
    for (int i = 0; i < kD; ++i)
      acc[i] += t[u][i];
  }
  // Epilogue: acc[4jj + 2ih + e2] = D[row 16q + grp + 8ih][activation row 8jj + 2c + e2].
  const int e = td.x, j = td.y, r0 = td.z, n = td.w;
  const int t_ = q * 32 + lane;
  if constexpr (!kDown) {
    const int ch = 64 * j + 32 * cw + 8 * q + grp;  // ih 0: gate, ih 1: the same up channel
    const float csg = p.cs[(2 * e) * kI + ch];
    const float csu = p.cs[(2 * e + 1) * kI + ch];
    const float lim = p.limit;
#pragma unroll
    for (int jj = 0; jj < N / 8; ++jj)
#pragma unroll
      for (int e2 = 0; e2 < 2; ++e2) {
        const int col = 8 * jj + 2 * c + e2;
        const float xsr = __ldg(p.xrs + r0 + col);
        float gate = bf16_round(acc[4 * jj + e2] * (xsr * csg));
        float up = bf16_round(acc[4 * jj + 2 + e2] * (xsr * csu));
        gate = gate > lim ? lim : gate;
        up = up > lim ? lim : (up < -lim ? -lim : up);
        const float act = bf16_round(__fdividef(gate, 1.0f + __expf(-gate)));
        ebuf[col * 32 + 8 * q + grp] = __float2bfloat16_rn(act * up);
      }
    asm volatile("bar.sync %0, 128;" ::"r"(1 + cw) : "memory");
    const int cbase = 64 * j + 32 * cw;
    for (int u = t_; u < n * 4; u += 128) {
      const int row = u >> 2, part = u & 3;
      *reinterpret_cast<uint4*>(p.h + static_cast<size_t>(r0 + row) * kI + cbase + part * 8) =
          *reinterpret_cast<const uint4*>(ebuf + row * 32 + part * 8);
    }
    asm volatile("bar.sync %0, 128;" ::"r"(1 + cw) : "memory");
  } else {
    // two passes (ih): 32 channels per warpgroup each, 16-byte pieces of 8 channels per (row, warp). The
    // destination rows of this thread's pieces are loaded once, up front (one L2 latency instead of one per piece).
    constexpr int kIt = (N * 4 + 127) / 128;
    __nv_bfloat16* orows[kIt];
#pragma unroll
    for (int it = 0; it < kIt; ++it) {
      const int u = t_ + 128 * it;
      const int d = u < n * 4 ? __ldg(p.dst + r0 + (u >> 2)) : 0;
      orows[it] = d >= 0 ? p.y + static_cast<size_t>(d) * kH : p.out + static_cast<size_t>(~d) * kH;
    }
#pragma unroll
    for (int ih = 0; ih < 2; ++ih) {
      const int ch = 128 * j + 64 * cw + 16 * q + 8 * ih + grp;
      const float csc = p.cs[e * kH + ch];
#pragma unroll
      for (int jj = 0; jj < N / 8; ++jj)
#pragma unroll
        for (int e2 = 0; e2 < 2; ++e2) {
          const int col = 8 * jj + 2 * c + e2;
          const float v = (acc[4 * jj + 2 * ih + e2] * (csc * __ldg(p.xrs + r0 + col))) * __ldg(p.gam + r0 + col);
          ebuf[col * 32 + 8 * q + grp] = __float2bfloat16_rn(v);
        }
      asm volatile("bar.sync %0, 128;" ::"r"(1 + cw) : "memory");
#pragma unroll
      for (int it = 0; it < kIt; ++it) {
        const int u = t_ + 128 * it;
        if (u < n * 4) {
          const int row = u >> 2, qq = u & 3;
          *reinterpret_cast<uint4*>(orows[it] + 128 * j + 64 * cw + 16 * qq + 8 * ih) =
              *reinterpret_cast<const uint4*>(ebuf + row * 32 + qq * 8);
        }
      }
      asm volatile("bar.sync %0, 128;" ::"r"(1 + cw) : "memory");
    }
  }
}

template <bool kDown>
__global__ void __launch_bounds__(kThreads, 1) kernel(const __grid_constant__ XP p) {
  extern __shared__ __align__(1024) uint8_t smem_raw[];
  const uint32_t raw = smem_u32(smem_raw);
  const uint32_t base = (raw + 1023u) & ~1023u;
  uint8_t* basep = smem_raw + (base - raw);
  const uint32_t full = base + kBarOff, empty = full + kRing * 8;
  int4* hdr = reinterpret_cast<int4*>(basep + kBarOff + kRing * 16);
  const int tid = threadIdx.x;
  const int warp = uniform(tid / 32);
  const int lane = tid % 32;
  if (tid == 0) {
    for (int i = 0; i < kRing; ++i) {
      mbar_init(full + i * 8, 1);
      mbar_init(empty + i * 8, 8);
    }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  __syncthreads();
  if (warp == 8) {
    if (lane == 0) produce<kDown>(p, base, hdr, full, empty);
    return;
  }
  const int cw = warp / 4, q = warp % 4;
  __nv_bfloat16* ebuf = reinterpret_cast<__nv_bfloat16*>(basep + kBarOff + kRing * 32) + cw * (kEpi / 2);
  uint32_t g = 0;
  while (true) {
    const uint32_t slot = g % kRing;
    mbar_wait(full + slot * 8, (g / kRing) & 1);
    int4 td = hdr[slot];
    td.x = uniform(td.x);
    td.y = uniform(td.y);
    td.z = uniform(td.z);
    td.w = uniform(td.w);
    if (td.x < 0) break;
    switch ((td.w + 15) >> 4) {
      case 1: tile<kDown, 16>(p, base, basep, full, empty, g, td, cw, q, lane, ebuf); break;
      case 2: tile<kDown, 32>(p, base, basep, full, empty, g, td, cw, q, lane, ebuf); break;
      case 3: tile<kDown, 48>(p, base, basep, full, empty, g, td, cw, q, lane, ebuf); break;
      case 4: tile<kDown, 64>(p, base, basep, full, empty, g, td, cw, q, lane, ebuf); break;
      case 5: tile<kDown, 80>(p, base, basep, full, empty, g, td, cw, q, lane, ebuf); break;
      case 6: tile<kDown, 96>(p, base, basep, full, empty, g, td, cw, q, lane, ebuf); break;
      case 7: tile<kDown, 112>(p, base, basep, full, empty, g, td, cw, q, lane, ebuf); break;
      default: tile<kDown, 128>(p, base, basep, full, empty, g, td, cw, q, lane, ebuf); break;
    }
  }
}

// ---- prologue (several CTAs): memset of `counters` (host) -> count -> place.
// counters (int32): [0] gate/up tiles, [1] down tiles, [2] / [3] their claim counters, [4] reduce-list length,
// [5] zero-list length, [6] valid slots, [8, 8 + E) expert counts, [8 + kMaxE, 8 + kMaxE + E) expert cursors.
constexpr int kCntBase = 8;
constexpr int kCounterInts = kCntBase + 2 * kMaxE;
constexpr int kPThreads = 1024;

__global__ void __launch_bounds__(kPThreads) count_kernel(
    const int32_t* __restrict__ ids, int S, const int32_t* __restrict__ nvalid, int32_t* __restrict__ counters) {
  __shared__ int hist[kMaxE];
  if (nvalid != nullptr) S = min(S, max(0, *nvalid));
  for (int e = threadIdx.x; e < kMaxE; e += kPThreads) hist[e] = 0;
  __syncthreads();
  const int lane = threadIdx.x & 31;
  for (int s0 = blockIdx.x * kPThreads; s0 < S; s0 += gridDim.x * kPThreads) {
    const int s = s0 + threadIdx.x;
    const int e = s < S ? ids[s] : -1;
    const unsigned peers = __match_any_sync(0xffffffffu, e);
    if (e >= 0 && lane == __ffs(peers) - 1) atomicAdd(&hist[e], __popc(peers));
  }
  __syncthreads();
  for (int e = threadIdx.x; e < kMaxE; e += kPThreads)
    if (hist[e]) atomicAdd(&counters[kCntBase + e], hist[e]);
}

__global__ void __launch_bounds__(kPThreads) place_kernel(
    const int32_t* __restrict__ ids, const float* __restrict__ wts, const int32_t* __restrict__ tok, int K, int S,
    const int32_t* __restrict__ nvalid, int E, int T, int32_t* __restrict__ rowpos, float* __restrict__ gam,
    int32_t* __restrict__ dst, int32_t* __restrict__ tlist, int32_t* __restrict__ trange,
    int32_t* __restrict__ vlist, int4* __restrict__ tiles_gu, int4* __restrict__ tiles_dn,
    int32_t* __restrict__ counters) {
  __shared__ int cnt[kMaxE], off[kMaxE], nch[kMaxE], base[kMaxE], gto[kMaxE], dto[kMaxE];
  const int tid = threadIdx.x, lane = tid & 31;
  if (nvalid != nullptr) S = min(S, max(0, *nvalid));
  for (int e = tid; e < E; e += kPThreads) cnt[e] = counters[kCntBase + e];
  __syncthreads();
  if (tid == 0) {  // every CTA derives the same layout from the counts
    int pos = 0, gt = 0, dt = 0;
    for (int e = 0; e < E; ++e) {
      const int c = cnt[e];
      off[e] = pos;
      pos += (c + 15) / 16 * 16;
      const int k = (c + kNMax - 1) / kNMax;
      nch[e] = k;
      base[e] = k > 0 ? min(kNMax, ((c + k - 1) / k + 15) / 16 * 16) : 0;
      gto[e] = gt;
      dto[e] = dt;
      gt += k * Geo<false>::JB;
      dt += k * Geo<true>::JB;
    }
    if (blockIdx.x == 0) {
      counters[0] = gt;
      counters[1] = dt;
    }
  }
  __syncthreads();
  for (int s0 = blockIdx.x * kPThreads; s0 < S; s0 += gridDim.x * kPThreads) {
    const int s = s0 + tid;
    const int e = s < S ? ids[s] : -1;
    const unsigned peers = __match_any_sync(0xffffffffu, e);
    const int leader = __ffs(peers) - 1;
    int b0 = 0;
    if (e >= 0 && lane == leader) b0 = atomicAdd(&counters[kCntBase + kMaxE + e], __popc(peers));
    b0 = __shfl_sync(0xffffffffu, b0, leader);
    const unsigned valid = __ballot_sync(0xffffffffu, e >= 0);
    int vb = 0;
    if (lane == 0 && valid) vb = atomicAdd(&counters[6], __popc(valid));
    vb = __shfl_sync(0xffffffffu, vb, 0);
    if (s >= S) continue;
    if (e < 0) {
      rowpos[s] = -1;
      continue;
    }
    vlist[vb + __popc(valid & ((1u << lane) - 1u))] = s;
    const int r = off[e] + b0 + __popc(peers & ((1u << lane) - 1u));
    rowpos[s] = r;
    gam[r] = wts[s];
    bool single = true, first = true;
    int t, end;
    if (tok != nullptr) {
      t = tok[s];
      if (s > 0 && tok[s - 1] == t && ids[s - 1] >= 0) single = first = false;
      end = s + 1;
      while (end < S && tok[end] == t) {
        if (ids[end] >= 0) single = false;
        ++end;
      }
    } else {
      t = s / K;
      end = t * K + K;
      for (int k = t * K; k < end; ++k)
        if (k != s && ids[k] >= 0) {
          single = false;
          if (k < s) first = false;
        }
    }
    dst[r] = single ? ~t : s;
    if (!single && first) {
      const int m = atomicAdd(&counters[4], 1);
      tlist[m] = t;
      trange[2 * m] = s;
      trange[2 * m + 1] = end;
    }
  }
  // top-k layout: tokens without a valid slot get a zero row (listed from the end of tlist)
  if (tok == nullptr) {
    for (int t = blockIdx.x * kPThreads + tid; t < T; t += gridDim.x * kPThreads) {
      bool any = false;
      for (int k = 0; k < K; ++k) any |= ids[t * K + k] >= 0;
      if (!any) tlist[T - 1 - atomicAdd(&counters[5], 1)] = t;
    }
  }
  // tiles (grid-stride): chunk c of expert e covers rows [c * base, min(cnt, (c + 1) * base))
  const int ngu = gto[E - 1] + nch[E - 1] * Geo<false>::JB, ndn = dto[E - 1] + nch[E - 1] * Geo<true>::JB;
  for (int i = blockIdx.x * kPThreads + tid; i < ngu + ndn; i += gridDim.x * kPThreads) {
    const bool dn = i >= ngu;
    const int k = dn ? i - ngu : i;
    const int* to = dn ? dto : gto;
    int lo = 0, hi = E - 1;
    while (lo < hi) {
      const int mid = (lo + hi + 1) / 2;
      if (to[mid] <= k) lo = mid; else hi = mid - 1;
    }
    const int ee = lo, jb = dn ? Geo<true>::JB : Geo<false>::JB;
    const int local = k - to[ee], c = local / jb, j = local % jb;
    const int r = c * base[ee];
    const int4 tv = make_int4(ee, j, off[ee] + r, min(base[ee], cnt[ee] - r));
    if (dn) tiles_dn[k] = tv; else tiles_gu[k] = tv;
  }
}

// gather / quant over the compact valid-slot list (counters[6])
__global__ void gather_kernel(
    const uint8_t* __restrict__ x8, const float* __restrict__ xscale, const int32_t* __restrict__ rowpos,
    const int32_t* __restrict__ tok, int K, const int32_t* __restrict__ vlist, const int32_t* __restrict__ counters,
    uint8_t* __restrict__ xa, float* __restrict__ xrs, int rpad) {
  const long total = static_cast<long>(counters[6]) * (kH / 16);
  for (long idx = blockIdx.x * static_cast<long>(blockDim.x) + threadIdx.x; idx < total;
       idx += static_cast<long>(gridDim.x) * blockDim.x) {
    const int s = vlist[idx / (kH / 16)], unit = static_cast<int>(idx % (kH / 16));
    const int r = rowpos[s];
    const int t = tok != nullptr ? tok[s] : s / K;
    const int st = unit >> 3, u = unit & 7;
    const uint4 v = *reinterpret_cast<const uint4*>(x8 + static_cast<size_t>(t) * kH + unit * 16);
    *reinterpret_cast<uint4*>(xa + (static_cast<size_t>(st) * rpad + r) * 128 + ((u ^ (r & 7)) * 16)) = v;
    if (unit == 0) xrs[r] = xscale[t];
  }
}

// h rows -> e4m3 [18][rpad][128] (pre-swizzled) and a power-of-2 scale (amax / 448 rounded up), one warp per slot
__global__ void quant_kernel(
    const __nv_bfloat16* __restrict__ h, const int32_t* __restrict__ rowpos, int S, const int32_t* __restrict__ nvalid,
    uint8_t* __restrict__ ha, float* __restrict__ hs, int rpad, const int32_t* __restrict__ vlist,
    const int32_t* __restrict__ counters) {
  if (vlist != nullptr) S = counters[6];  // compact list of valid slots
  else if (nvalid != nullptr) S = min(S, max(0, *nvalid));
  const int lane = threadIdx.x % 32;
  for (int m = (blockIdx.x * blockDim.x + threadIdx.x) / 32; m < S; m += gridDim.x * blockDim.x / 32) {
    const int r = rowpos[vlist != nullptr ? vlist[m] : m];
    if (r < 0) continue;
    const uint4* src = reinterpret_cast<const uint4*>(h + static_cast<size_t>(r) * kI);  // 288 uint4 (8 bf16 each)
    uint4 v[9];
    float amax = 0.f;
#pragma unroll
    for (int i = 0; i < 9; ++i) {
      v[i] = src[lane + 32 * i];
      const __nv_bfloat162* b = reinterpret_cast<const __nv_bfloat162*>(&v[i]);
#pragma unroll
      for (int k = 0; k < 4; ++k) {
        const float2 f = __bfloat1622float2(b[k]);
        amax = fmaxf(amax, fmaxf(fabsf(f.x), fabsf(f.y)));
      }
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1)
      amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
    const float sc = exp2f(ceilf(log2f(fmaxf(amax, 1.0e-4f) / 448.0f)));
    const float inv = 1.0f / sc;  // exact (power of 2)
#pragma unroll
    for (int i = 0; i < 9; ++i) {
      const __nv_bfloat162* b = reinterpret_cast<const __nv_bfloat162*>(&v[i]);
      uint32_t wd[2];
#pragma unroll
      for (int k = 0; k < 2; ++k) {
        const float2 f0 = __bfloat1622float2(b[2 * k]), f1 = __bfloat1622float2(b[2 * k + 1]);
        uint16_t lo, hi;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(lo) : "f"(f0.y * inv), "f"(f0.x * inv));
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(hi) : "f"(f1.y * inv), "f"(f1.x * inv));
        wd[k] = static_cast<uint32_t>(lo) | (static_cast<uint32_t>(hi) << 16);
      }
      // 8 bf16 = 8 e4m3 bytes = half of a 16-byte unit; element index of v[i] = 8 * (lane + 32 i)
      const int el = 8 * (lane + 32 * i);
      const int st = el / 128, kb = el % 128, unit = kb / 16, half = (kb % 16) / 8;
      uint8_t* dp = ha + (static_cast<size_t>(st) * rpad + r) * 128 + ((unit ^ (r & 7)) * 16) + half * 8;
      *reinterpret_cast<uint2*>(dp) = make_uint2(wd[0], wd[1]);
    }
    if (lane == 0) hs[r] = sc;
  }
}

// tokens with several valid slots: out[t] = sum over its valid slots (slot order) of y[slot], fp32, one bf16 round
__global__ void reduce_kernel(
    const __nv_bfloat16* __restrict__ y, const int32_t* __restrict__ ids, const int32_t* __restrict__ tlist,
    const int32_t* __restrict__ trange, const int32_t* __restrict__ nlist, int T, __nv_bfloat16* __restrict__ out) {
  constexpr int kVecs = kH / 8;
  const int v = blockIdx.x * blockDim.x + threadIdx.x;
  if (v >= kVecs) return;
  const int nm = nlist[0], nz = nlist[1];
  for (int m = blockIdx.y; m < nm + nz; m += gridDim.y) {
    if (m >= nm) {  // a token without a valid slot: zero row
      const int t = tlist[T - 1 - (m - nm)];
      reinterpret_cast<uint4*>(out + static_cast<size_t>(t) * kH)[v] = make_uint4(0, 0, 0, 0);
      continue;
    }
    const int t = tlist[m], s0 = trange[2 * m], s1 = trange[2 * m + 1];
    float acc[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    for (int s = s0; s < s1; ++s) {
      if (ids[s] < 0) continue;
      const uint4 w = reinterpret_cast<const uint4*>(y + static_cast<size_t>(s) * kH)[v];
      const __nv_bfloat162* b = reinterpret_cast<const __nv_bfloat162*>(&w);
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
    reinterpret_cast<uint4*>(out + static_cast<size_t>(t) * kH)[v] = o;
  }
}
}  // namespace k1x

// ---- host entry points
namespace {
bool g_k1x_attr[2] = {false, false};
}  // namespace

inline void x_quant(
    tvm::ffi::TensorView h, tvm::ffi::TensorView rowpos, tvm::ffi::TensorView nvalid, int64_t use_nvalid,
    tvm::ffi::TensorView ha, tvm::ffi::TensorView hs, tvm::ffi::TensorView vlist, int64_t use_vlist,
    tvm::ffi::TensorView counters) {
  using namespace host;
  LaunchKernel(dim3(132 * 4), dim3(256), h.device(), 0)(
      k1x::quant_kernel, static_cast<const __nv_bfloat16*>(h.data_ptr()), static_cast<const int32_t*>(rowpos.data_ptr()),
      static_cast<int>(rowpos.size(0)), use_nvalid ? static_cast<const int32_t*>(nvalid.data_ptr()) : nullptr,
      static_cast<uint8_t*>(ha.data_ptr()), static_cast<float*>(hs.data_ptr()), static_cast<int>(ha.size(1)),
      use_vlist ? static_cast<const int32_t*>(vlist.data_ptr()) : nullptr,
      static_cast<const int32_t*>(counters.data_ptr()));
}

// prologue: counters zeroed by the caller (memset), then count and place over ceil(S / 1024) CTAs
inline void x_prep(
    tvm::ffi::TensorView ids, tvm::ffi::TensorView wts, tvm::ffi::TensorView tok, int64_t use_tok, int64_t K,
    tvm::ffi::TensorView nvalid, int64_t use_nvalid, int64_t E, int64_t T, tvm::ffi::TensorView rowpos,
    tvm::ffi::TensorView gam, tvm::ffi::TensorView dst, tvm::ffi::TensorView tlist, tvm::ffi::TensorView trange,
    tvm::ffi::TensorView vlist, tvm::ffi::TensorView tiles_gu, tvm::ffi::TensorView tiles_dn,
    tvm::ffi::TensorView counters) {
  using namespace host;
  RuntimeCheck(E <= k1x::kMaxE && counters.size(0) >= k1x::kCounterInts, "k1x prep: experts / counters");
  const int S = static_cast<int>(ids.size(0));
  const int nv = use_nvalid ? 1 : 0;
  const int ctas = std::max(1, std::min(132, (S + k1x::kPThreads - 1) / k1x::kPThreads));
  const int32_t* idp = static_cast<const int32_t*>(ids.data_ptr());
  const int32_t* nvp = nv ? static_cast<const int32_t*>(nvalid.data_ptr()) : nullptr;
  int32_t* cnt = static_cast<int32_t*>(counters.data_ptr());
  LaunchKernel(dim3(ctas), dim3(k1x::kPThreads), ids.device(), 0)(k1x::count_kernel, idp, S, nvp, cnt);
  LaunchKernel(dim3(ctas), dim3(k1x::kPThreads), ids.device(), 0)(
      k1x::place_kernel, idp, static_cast<const float*>(wts.data_ptr()),
      use_tok ? static_cast<const int32_t*>(tok.data_ptr()) : nullptr, static_cast<int>(K), S, nvp,
      static_cast<int>(E), static_cast<int>(T), static_cast<int32_t*>(rowpos.data_ptr()),
      static_cast<float*>(gam.data_ptr()), static_cast<int32_t*>(dst.data_ptr()),
      static_cast<int32_t*>(tlist.data_ptr()), static_cast<int32_t*>(trange.data_ptr()),
      static_cast<int32_t*>(vlist.data_ptr()), static_cast<int4*>(tiles_gu.data_ptr()),
      static_cast<int4*>(tiles_dn.data_ptr()), cnt);
}

inline void x_gather(
    tvm::ffi::TensorView x8, tvm::ffi::TensorView xscale, tvm::ffi::TensorView rowpos, tvm::ffi::TensorView tok,
    int64_t use_tok, int64_t K, tvm::ffi::TensorView vlist, tvm::ffi::TensorView counters, tvm::ffi::TensorView xa,
    tvm::ffi::TensorView xrs) {
  using namespace host;
  LaunchKernel(dim3(132 * 8), dim3(256), x8.device(), 0)(
      k1x::gather_kernel, static_cast<const uint8_t*>(x8.data_ptr()), static_cast<const float*>(xscale.data_ptr()),
      static_cast<const int32_t*>(rowpos.data_ptr()), use_tok ? static_cast<const int32_t*>(tok.data_ptr()) : nullptr,
      static_cast<int>(K), static_cast<const int32_t*>(vlist.data_ptr()),
      static_cast<const int32_t*>(counters.data_ptr()), static_cast<uint8_t*>(xa.data_ptr()),
      static_cast<float*>(xrs.data_ptr()), static_cast<int>(xa.size(1)));
}

// down = 0: gate/up (xa [40][rpad][128], w13, cs13, out h); 1: down (xa = h8 [18][rpad][128], w2, cs2, y / out)
inline void x_run(
    int64_t down, tvm::ffi::TensorView xa, tvm::ffi::TensorView xrs, tvm::ffi::TensorView w, tvm::ffi::TensorView cs,
    tvm::ffi::TensorView tiles, tvm::ffi::TensorView counters, tvm::ffi::TensorView h, tvm::ffi::TensorView gam,
    tvm::ffi::TensorView dst, tvm::ffi::TensorView y, tvm::ffi::TensorView out, double limit) {
  using namespace host;
  RuntimeCheck(w.size(3) == kWStage && w.size(2) == (down ? 18 : 40), "k1x: packed weight layout");
  k1x::XP prm{};
  prm.xa = static_cast<const uint8_t*>(xa.data_ptr());
  prm.xrs = static_cast<const float*>(xrs.data_ptr());
  prm.w = static_cast<const uint8_t*>(w.data_ptr());
  prm.cs = static_cast<const float*>(cs.data_ptr());
  prm.tiles = static_cast<const int4*>(tiles.data_ptr());
  int32_t* cnt = static_cast<int32_t*>(counters.data_ptr());
  prm.ntiles = cnt + (down ? 1 : 0);
  prm.ctr = cnt + (down ? 3 : 2);
  prm.rpad = static_cast<int>(xa.size(1));
  prm.limit = static_cast<float>(limit);
  prm.h = static_cast<__nv_bfloat16*>(h.data_ptr());
  prm.gam = static_cast<const float*>(gam.data_ptr());
  prm.dst = static_cast<const int32_t*>(dst.data_ptr());
  prm.y = static_cast<__nv_bfloat16*>(y.data_ptr());
  prm.out = static_cast<__nv_bfloat16*>(out.data_ptr());
  const DLDevice device = xa.device();
  auto kern = down ? k1x::kernel<true> : k1x::kernel<false>;
  if (!g_k1x_attr[down ? 1 : 0]) {
    RuntimeDeviceCheck(cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(k1x::kSmem)));
    g_k1x_attr[down ? 1 : 0] = true;
  }
  int sms = 0;
  RuntimeDeviceCheck(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device.device_id));
  LaunchKernel(dim3(sms), dim3(k1x::kThreads), device, k1x::kSmem)(kern, prm);
}

inline void x_reduce(
    tvm::ffi::TensorView y, tvm::ffi::TensorView ids, tvm::ffi::TensorView tlist, tvm::ffi::TensorView trange,
    tvm::ffi::TensorView counters, tvm::ffi::TensorView out) {
  using namespace host;
  const int T = static_cast<int>(out.size(0));
  LaunchKernel(dim3((kH / 8 + 127) / 128, T < 1024 ? T : 1024), dim3(128), y.device(), 0)(
      k1x::reduce_kernel, static_cast<const __nv_bfloat16*>(y.data_ptr()), static_cast<const int32_t*>(ids.data_ptr()),
      static_cast<const int32_t*>(tlist.data_ptr()), static_cast<const int32_t*>(trange.data_ptr()),
      static_cast<const int32_t*>(counters.data_ptr()) + 4, T, static_cast<__nv_bfloat16*>(out.data_ptr()));
}

}  // namespace dsv4_moe_k1
}  // namespace sglang
