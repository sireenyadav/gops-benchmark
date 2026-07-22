#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <cstdlib>
#include <cstring>
#include <arm_neon.h>

constexpr int M = 1024, K = 1024, N = 1024;

void init_matrix(std::vector<float>& mat, int size) {
    for (int i = 0; i < size; ++i)
        mat[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX) * 2.0f - 1.0f;
}

void gemm_float32(const std::vector<float>& A, const std::vector<float>& B,
                  std::vector<float>& C) {
    for (int i = 0; i < M; ++i)
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k)
                sum += A[i * K + k] * B[k * N + j];
            C[i * N + j] = sum;
        }
}

void quantize_A_per_row(const std::vector<float>& A, int8_t* A_int8,
                        std::vector<float>& scales_A) {
    for (int i = 0; i < M; ++i) {
        const float* row = A.data() + i * K;
        float max_abs = 0.0f;
        for (int k = 0; k < K; ++k)
            max_abs = std::max(max_abs, std::abs(row[k]));
        float scale = max_abs / 127.0f;
        scales_A[i] = scale;
        float inv_scale = 1.0f / scale;
        for (int k = 0; k < K; ++k) {
            float q = std::round(row[k] * inv_scale);
            q = std::max(-127.0f, std::min(127.0f, q));
            A_int8[i * K + k] = static_cast<int8_t>(q);
        }
    }
}

void quantize_B_per_column(const std::vector<float>& B, int8_t* B_int8,
                           std::vector<float>& scales_B) {
    for (int j = 0; j < N; ++j) {
        float max_abs = 0.0f;
        for (int k = 0; k < K; ++k)
            max_abs = std::max(max_abs, std::abs(B[k * N + j]));
        float scale = max_abs / 127.0f;
        scales_B[j] = scale;
        float inv_scale = 1.0f / scale;
        for (int k = 0; k < K; ++k) {
            float q = std::round(B[k * N + j] * inv_scale);
            q = std::max(-127.0f, std::min(127.0f, q));
            B_int8[k * N + j] = static_cast<int8_t>(q);
        }
    }
}

// Pack B from row-major (K×N) to SDOT-friendly layout.
// Output: B_packed[jg * K * 4 + k0 * 16 + col * 4 + m]
//   = B_int8[(k0*4 + m) * N + (jg*4 + col)]
// Each 16-byte load gives: [col0_k0:k0+3 | col1_k0:k0+3 | col2_k0:k0+3 | col3_k0:k0+3]
void pack_B_for_sdot(const int8_t* __restrict__ B_int8, int8_t* __restrict__ B_packed) {
    constexpr int JG = N / 4;
    for (int jg = 0; jg < JG; ++jg) {
        int8_t* dst = B_packed + (size_t)jg * K * 4;
        for (int k0 = 0; k0 < K / 4; ++k0) {
            const int8_t* s0 = B_int8 + (size_t)(k0 * 4 + 0) * N + jg * 4;
            const int8_t* s1 = B_int8 + (size_t)(k0 * 4 + 1) * N + jg * 4;
            const int8_t* s2 = B_int8 + (size_t)(k0 * 4 + 2) * N + jg * 4;
            const int8_t* s3 = B_int8 + (size_t)(k0 * 4 + 3) * N + jg * 4;
            // col 0
            dst[0]  = s0[0]; dst[1]  = s1[0]; dst[2]  = s2[0]; dst[3]  = s3[0];
            // col 1
            dst[4]  = s0[1]; dst[5]  = s1[1]; dst[6]  = s2[1]; dst[7]  = s3[1];
            // col 2
            dst[8]  = s0[2]; dst[9]  = s1[2]; dst[10] = s2[2]; dst[11] = s3[2];
            // col 3
            dst[12] = s0[3]; dst[13] = s1[3]; dst[14] = s2[3]; dst[15] = s3[3];
            dst += 16;
        }
    }
}

int main() {
    srand(42);

    std::vector<float> A(M * K), B(K * N);
    std::vector<float> C_float(M * N, 0.0f), C_quant(M * N, 0.0f);
    init_matrix(A, M * K);
    init_matrix(B, K * N);

    auto t0 = std::chrono::steady_clock::now();
    gemm_float32(A, B, C_float);
    auto t1 = std::chrono::steady_clock::now();
    std::chrono::duration<double> time_f32 = t1 - t0;

    int8_t *A_int8 = nullptr, *B_int8 = nullptr;
    posix_memalign(reinterpret_cast<void**>(&A_int8), 64, (size_t)M * K);
    posix_memalign(reinterpret_cast<void**>(&B_int8), 64, (size_t)K * N);

    std::vector<float> row_scales_A(M), col_scales_B(N);
    quantize_A_per_row(A, A_int8, row_scales_A);
    quantize_B_per_column(B, B_int8, col_scales_B);

    // Pack B into SDOT-friendly layout (no A packing needed — A is read row-major
    // and replicated at load time via vld1q_dup_s32 / LD1R instruction)
    int8_t *B_packed = nullptr;
    posix_memalign(reinterpret_cast<void**>(&B_packed), 64, (size_t)N * K);
    pack_B_for_sdot(B_int8, B_packed);

    auto t2 = std::chrono::steady_clock::now();

    const int8_t* __restrict__ A_ptr = A_int8;
    const int8_t* __restrict__ B_ptr = B_packed;
    float* __restrict__ C_ptr = C_quant.data();
    const float* __restrict__ rsA = row_scales_A.data();
    const float* __restrict__ csB = col_scales_B.data();

    constexpr int K4 = K / 4;  // 256

    // ====================================================================
    // 8×8 output tile GEMM with DOTPROD
    //
    // Per k-step (4 K-elements):
    //   - 8 LD1R loads (4 bytes each, A replicated to 16 bytes)
    //   - 2 LD1 loads (16 bytes each, B packed 4 cols × 4 k)
    //   - 16 SDOT instructions (8 rows × 2 jg-groups)
    //
    // Memory per 16 SDOTs: 8×4 + 2×16 = 64 bytes → 4 bytes/SDOT
    // (original: 4×16 + 1×16 = 80 bytes → 20 bytes/SDOT — 5× worse)
    //
    // Register usage: 16 accumulators + 4 B temps + 1-2 A temps ≈ 22/32
    // ====================================================================

    for (int i = 0; i < M; i += 8) {
        // Pre-load row scales as broadcast vectors (hoisted out of jg loop)
        float32x4_t rs0v = vdupq_n_f32(rsA[i + 0]);
        float32x4_t rs1v = vdupq_n_f32(rsA[i + 1]);
        float32x4_t rs2v = vdupq_n_f32(rsA[i + 2]);
        float32x4_t rs3v = vdupq_n_f32(rsA[i + 3]);
        float32x4_t rs4v = vdupq_n_f32(rsA[i + 4]);
        float32x4_t rs5v = vdupq_n_f32(rsA[i + 5]);
        float32x4_t rs6v = vdupq_n_f32(rsA[i + 6]);
        float32x4_t rs7v = vdupq_n_f32(rsA[i + 7]);

        const int8_t* a0 = A_ptr + (size_t)(i + 0) * K;
        const int8_t* a1 = A_ptr + (size_t)(i + 1) * K;
        const int8_t* a2 = A_ptr + (size_t)(i + 2) * K;
        const int8_t* a3 = A_ptr + (size_t)(i + 3) * K;
        const int8_t* a4 = A_ptr + (size_t)(i + 4) * K;
        const int8_t* a5 = A_ptr + (size_t)(i + 5) * K;
        const int8_t* a6 = A_ptr + (size_t)(i + 6) * K;
        const int8_t* a7 = A_ptr + (size_t)(i + 7) * K;

        // Process 2 jg-groups (8 columns) at a time for better B reuse
        for (int jg = 0; jg < N / 4; jg += 2) {
            // 16 accumulators: 8 rows × 2 jg-groups, each int32x4 = 4 cols
            int32x4_t acc00 = vdupq_n_s32(0);
            int32x4_t acc01 = vdupq_n_s32(0);
            int32x4_t acc10 = vdupq_n_s32(0);
            int32x4_t acc11 = vdupq_n_s32(0);
            int32x4_t acc20 = vdupq_n_s32(0);
            int32x4_t acc21 = vdupq_n_s32(0);
            int32x4_t acc30 = vdupq_n_s32(0);
            int32x4_t acc31 = vdupq_n_s32(0);
            int32x4_t acc40 = vdupq_n_s32(0);
            int32x4_t acc41 = vdupq_n_s32(0);
            int32x4_t acc50 = vdupq_n_s32(0);
            int32x4_t acc51 = vdupq_n_s32(0);
            int32x4_t acc60 = vdupq_n_s32(0);
            int32x4_t acc61 = vdupq_n_s32(0);
            int32x4_t acc70 = vdupq_n_s32(0);
            int32x4_t acc71 = vdupq_n_s32(0);

            const int8_t* b0 = B_ptr + (size_t)jg * K * 4;
            const int8_t* b1 = B_ptr + (size_t)(jg + 1) * K * 4;

            // K-loop unrolled by 2 for better ILP
            // Each iteration: 16 LD1R + 4 LD1 + 32 SDOT
            for (int k0 = 0; k0 < K4; k0 += 2) {
                // Load 4 B vectors (2 jg-groups × 2 unroll steps)
                int8x16_t bv0_0 = vld1q_s8(b0 + (k0 + 0) * 16);
                int8x16_t bv0_1 = vld1q_s8(b0 + (k0 + 1) * 16);
                int8x16_t bv1_0 = vld1q_s8(b1 + (k0 + 0) * 16);
                int8x16_t bv1_1 = vld1q_s8(b1 + (k0 + 1) * 16);

                // Row 0: load A[k0:k0+3], replicate, SDOT with both B groups
                int8x16_t av0 = vreinterpretq_s8_s32(
                    vld1q_dup_s32((const int32_t*)(a0 + (k0 + 0) * 4)));
                acc00 = vdotq_s32(acc00, av0, bv0_0);
                acc01 = vdotq_s32(acc01, av0, bv1_0);
                int8x16_t av0b = vreinterpretq_s8_s32(
                    vld1q_dup_s32((const int32_t*)(a0 + (k0 + 1) * 4)));
                acc00 = vdotq_s32(acc00, av0b, bv0_1);
                acc01 = vdotq_s32(acc01, av0b, bv1_1);

                // Row 1
                int8x16_t av1 = vreinterpretq_s8_s32(
                    vld1q_dup_s32((const int32_t*)(a1 + (k0 + 0) * 4)));
                acc10 = vdotq_s32(acc10, av1, bv0_0);
                acc11 = vdotq_s32(acc11, av1, bv1_0);
                int8x16_t av1b = vreinterpretq_s8_s32(
                    vld1q_dup_s32((const int32_t*)(a1 + (k0 + 1) * 4)));
                acc10 = vdotq_s32(acc10, av1b, bv0_1);
                acc11 = vdotq_s32(acc11, av1b, bv1_1);

                // Row 2
                int8x16_t av2 = vreinterpretq_s8_s32(
                    vld1q_dup_s32((const int32_t*)(a2 + (k0 + 0) * 4)));
                acc20 = vdotq_s32(acc20, av2, bv0_0);
                acc21 = vdotq_s32(acc21, av2, bv1_0);
                int8x16_t av2b = vreinterpretq_s8_s32(
                    vld1q_dup_s32((const int32_t*)(a2 + (k0 + 1) * 4)));
                acc20 = vdotq_s32(acc20, av2b, bv0_1);
                acc21 = vdotq_s32(acc21, av2b, bv1_1);

                // Row 3
                int8x16_t av3 = vreinterpretq_s8_s32(
                    vld1q_dup_s32((const int32_t*)(a3 + (k0 + 0) * 4)));
                acc30 = vdotq_s32(acc30, av3, bv0_0);
                acc31 = vdotq_s32(acc31, av3, bv1_0);
                int8x16_t av3b = vreinterpretq_s8_s32(
                    vld1q_dup_s32((const int32_t*)(a3 + (k0 + 1) * 4)));
                acc30 = vdotq_s32(acc30, av3b, bv0_1);
                acc31 = vdotq_s32(acc31, av3b, bv1_1);

                // Row 4
                int8x16_t av4 = vreinterpretq_s8_s32(
                    vld1q_dup_s32((const int32_t*)(a4 + (k0 + 0) * 4)));
                acc40 = vdotq_s32(acc40, av4, bv0_0);
                acc41 = vdotq_s32(acc41, av4, bv1_0);
                int8x16_t av4b = vreinterpretq_s8_s32(
                    vld1q_dup_s32((const int32_t*)(a4 + (k0 + 1) * 4)));
                acc40 = vdotq_s32(acc40, av4b, bv0_1);
                acc41 = vdotq_s32(acc41, av4b, bv1_1);

                // Row 5
                int8x16_t av5 = vreinterpretq_s8_s32(
                    vld1q_dup_s32((const int32_t*)(a5 + (k0 + 0) * 4)));
                acc50 = vdotq_s32(acc50, av5, bv0_0);
                acc51 = vdotq_s32(acc51, av5, bv1_0);
                int8x16_t av5b = vreinterpretq_s8_s32(
                    vld1q_dup_s32((const int32_t*)(a5 + (k0 + 1) * 4)));
                acc50 = vdotq_s32(acc50, av5b, bv0_1);
                acc51 = vdotq_s32(acc51, av5b, bv1_1);

                // Row 6
                int8x16_t av6 = vreinterpretq_s8_s32(
                    vld1q_dup_s32((const int32_t*)(a6 + (k0 + 0) * 4)));
                acc60 = vdotq_s32(acc60, av6, bv0_0);
                acc61 = vdotq_s32(acc61, av6, bv1_0);
                int8x16_t av6b = vreinterpretq_s8_s32(
                    vld1q_dup_s32((const int32_t*)(a6 + (k0 + 1) * 4)));
                acc60 = vdotq_s32(acc60, av6b, bv0_1);
                acc61 = vdotq_s32(acc61, av6b, bv1_1);

                // Row 7
                int8x16_t av7 = vreinterpretq_s8_s32(
                    vld1q_dup_s32((const int32_t*)(a7 + (k0 + 0) * 4)));
                acc70 = vdotq_s32(acc70, av7, bv0_0);
                acc71 = vdotq_s32(acc71, av7, bv1_0);
                int8x16_t av7b = vreinterpretq_s8_s32(
                    vld1q_dup_s32((const int32_t*)(a7 + (k0 + 1) * 4)));
                acc70 = vdotq_s32(acc70, av7b, bv0_1);
                acc71 = vdotq_s32(acc71, av7b, bv1_1);
            }

            // Dequantize and store: C = int32(acc) * row_scale * col_scale
            float32x4_t csB0 = vld1q_f32(csB + jg * 4);
            float32x4_t csB1 = vld1q_f32(csB + (jg + 1) * 4);

            float* c0 = C_ptr + (size_t)(i + 0) * N + jg * 4;
            float* c1 = C_ptr + (size_t)(i + 1) * N + jg * 4;
            float* c2 = C_ptr + (size_t)(i + 2) * N + jg * 4;
            float* c3 = C_ptr + (size_t)(i + 3) * N + jg * 4;
            float* c4 = C_ptr + (size_t)(i + 4) * N + jg * 4;
            float* c5 = C_ptr + (size_t)(i + 5) * N + jg * 4;
            float* c6 = C_ptr + (size_t)(i + 6) * N + jg * 4;
            float* c7 = C_ptr + (size_t)(i + 7) * N + jg * 4;

            // Row 0
            vst1q_f32(c0,     vmulq_f32(vcvtq_f32_s32(acc00), vmulq_f32(rs0v, csB0)));
            vst1q_f32(c0 + 4, vmulq_f32(vcvtq_f32_s32(acc01), vmulq_f32(rs0v, csB1)));
            // Row 1
            vst1q_f32(c1,     vmulq_f32(vcvtq_f32_s32(acc10), vmulq_f32(rs1v, csB0)));
            vst1q_f32(c1 + 4, vmulq_f32(vcvtq_f32_s32(acc11), vmulq_f32(rs1v, csB1)));
            // Row 2
            vst1q_f32(c2,     vmulq_f32(vcvtq_f32_s32(acc20), vmulq_f32(rs2v, csB0)));
            vst1q_f32(c2 + 4, vmulq_f32(vcvtq_f32_s32(acc21), vmulq_f32(rs2v, csB1)));
            // Row 3
            vst1q_f32(c3,     vmulq_f32(vcvtq_f32_s32(acc30), vmulq_f32(rs3v, csB0)));
            vst1q_f32(c3 + 4, vmulq_f32(vcvtq_f32_s32(acc31), vmulq_f32(rs3v, csB1)));
            // Row 4
            vst1q_f32(c4,     vmulq_f32(vcvtq_f32_s32(acc40), vmulq_f32(rs4v, csB0)));
            vst1q_f32(c4 + 4, vmulq_f32(vcvtq_f32_s32(acc41), vmulq_f32(rs4v, csB1)));
            // Row 5
            vst1q_f32(c5,     vmulq_f32(vcvtq_f32_s32(acc50), vmulq_f32(rs5v, csB0)));
            vst1q_f32(c5 + 4, vmulq_f32(vcvtq_f32_s32(acc51), vmulq_f32(rs5v, csB1)));
            // Row 6
            vst1q_f32(c6,     vmulq_f32(vcvtq_f32_s32(acc60), vmulq_f32(rs6v, csB0)));
            vst1q_f32(c6 + 4, vmulq_f32(vcvtq_f32_s32(acc61), vmulq_f32(rs6v, csB1)));
            // Row 7
            vst1q_f32(c7,     vmulq_f32(vcvtq_f32_s32(acc70), vmulq_f32(rs7v, csB0)));
            vst1q_f32(c7 + 4, vmulq_f32(vcvtq_f32_s32(acc71), vmulq_f32(rs7v, csB1)));
        }
    }

    auto t3 = std::chrono::steady_clock::now();
    std::chrono::duration<double> time_int8 = t3 - t2;

    float max_abs_err = 0.0f, max_c_val = 0.0f;
    for (int i = 0; i < M * N; ++i) {
        float err = std::abs(C_float[i] - C_quant[i]);
        max_abs_err = std::max(max_abs_err, err);
        max_c_val = std::max(max_c_val, std::abs(C_float[i]));
    }
    float rel_err_pct = (max_abs_err / max_c_val) * 100.0f;

    double total_ops = 2.0 * M * N * K;
    double gops_f32  = (total_ops / time_f32.count()) / 1e9;
    double gops_int8 = (total_ops / time_int8.count()) / 1e9;
    double speedup   = gops_int8 / gops_f32;

    std::cout << "======================================================\n"
              << "       ARM64 NEON QUANTIZED GEMM  Z BENCHMARK\n"
              << "======================================================\n"
              << "Matrix Dimensions    : " << M << " x " << K << " x " << N << "\n"
              << "Total Arithmetic Ops : " << static_cast<long long>(total_ops) << "\n\n"
              << "[1] NAIVE SCALAR FLOAT32 ENGINE\n"
              << std::fixed << std::setprecision(3)
              << "Execution Time       : " << time_f32.count() << " seconds\n"
              << std::setprecision(2)
              << "Performance          : " << gops_f32 << " GOPS\n\n"
              << "[2] OPTIMIZED INT8 NEON SIMD ENGINE\n"
              << std::setprecision(3)
              << "Execution Time       : " << time_int8.count() << " seconds\n"
              << std::setprecision(2)
              << "Performance          : " << gops_int8 << " GOPS\n"
              << "Speedup Factor       : " << speedup << "x\n\n"
              << "[3] CORRECTNESS VERIFICATION\n"
              << std::setprecision(4)
              << "Max Absolute Error   : " << max_abs_err << "\n"
              << "Status               : " << (rel_err_pct < 5.0f ? "PASSED" : "FAILED")
              << " (Precision Loss < 5%)\n"
              << "======================================================\n";

    free(A_int8);
    free(B_int8);
    free(B_packed);
    return 0;
}
