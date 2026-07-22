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

// 4x replication of A for direct LD1 compatibility
void pack_A_vdot(const int8_t* A, int8_t* A_packed) {
    for (int i = 0; i < M; ++i) {
        const int8_t* src = A + i * K;
        int8_t* dst = A_packed + i * (K * 4);
        for (int k0 = 0; k0 < K; k0 += 4) {
            for (int rep = 0; rep < 4; ++rep) {
                std::memcpy(dst, src + k0, 4);
                dst += 4;
            }
        }
    }
}

void pack_B_vdot(const int8_t* B_T, int8_t* B_packed) {
    constexpr int JG = N / 4;
    for (int jg = 0; jg < JG; ++jg) {
        int8_t* dst = B_packed + jg * (K * 4);
        for (int k0 = 0; k0 < K; k0 += 4) {
            for (int jj = 0; jj < 4; ++jj) {
                const int8_t* col = B_T + (jg * 4 + jj) * K + k0;
                std::memcpy(dst, col, 4);
                dst += 4;
            }
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

    int8_t *A_int8 = nullptr, *B_int8 = nullptr, *B_int8_T = nullptr;
    posix_memalign(reinterpret_cast<void**>(&A_int8), 64, M * K);
    posix_memalign(reinterpret_cast<void**>(&B_int8), 64, K * N);
    posix_memalign(reinterpret_cast<void**>(&B_int8_T), 64, K * N);

    std::vector<float> row_scales_A(M), col_scales_B(N);
    quantize_A_per_row(A, A_int8, row_scales_A);
    quantize_B_per_column(B, B_int8, col_scales_B);

    for (int i = 0; i < K; ++i)
        for (int j = 0; j < N; ++j)
            B_int8_T[j * K + i] = B_int8[i * N + j];

    int8_t *A_packed = nullptr, *B_packed = nullptr;
    posix_memalign(reinterpret_cast<void**>(&A_packed), 64, M * K * 4);
    posix_memalign(reinterpret_cast<void**>(&B_packed), 64, N * K);
    pack_A_vdot(A_int8, A_packed);
    pack_B_vdot(B_int8_T, B_packed);

    auto t2 = std::chrono::steady_clock::now();

    const int8_t* __restrict__ A_ptr = A_packed;
    const int8_t* __restrict__ B_ptr = B_packed;
    float* __restrict__ C_ptr = C_quant.data();
    const float* __restrict__ rsA = row_scales_A.data();
    const float* __restrict__ csB = col_scales_B.data();

    constexpr int IBLOCK = 8;

    for (int i = 0; i < M; i += IBLOCK) {
        float rs[8];
        for (int r = 0; r < 8; ++r) rs[r] = rsA[i + r];

        // Process 8 columns (2 jg groups) at a time to maximize A reuse
        for (int jg = 0; jg < N / 4; jg += 2) {
            int32x4_t c00 = vdupq_n_s32(0), c01 = vdupq_n_s32(0);
            int32x4_t c10 = vdupq_n_s32(0), c11 = vdupq_n_s32(0);
            int32x4_t c20 = vdupq_n_s32(0), c21 = vdupq_n_s32(0);
            int32x4_t c30 = vdupq_n_s32(0), c31 = vdupq_n_s32(0);
            int32x4_t c40 = vdupq_n_s32(0), c41 = vdupq_n_s32(0);
            int32x4_t c50 = vdupq_n_s32(0), c51 = vdupq_n_s32(0);
            int32x4_t c60 = vdupq_n_s32(0), c61 = vdupq_n_s32(0);
            int32x4_t c70 = vdupq_n_s32(0), c71 = vdupq_n_s32(0);

            const int8_t* b0_base = B_ptr + jg * (K * 4);
            const int8_t* b1_base = B_ptr + (jg + 1) * (K * 4);

            const int8_t* a0_base = A_ptr + (i + 0) * (K * 4);
            const int8_t* a1_base = A_ptr + (i + 1) * (K * 4);
            const int8_t* a2_base = A_ptr + (i + 2) * (K * 4);
            const int8_t* a3_base = A_ptr + (i + 3) * (K * 4);
            const int8_t* a4_base = A_ptr + (i + 4) * (K * 4);
            const int8_t* a5_base = A_ptr + (i + 5) * (K * 4);
            const int8_t* a6_base = A_ptr + (i + 6) * (K * 4);
            const int8_t* a7_base = A_ptr + (i + 7) * (K * 4);

            // K-loop unrolled by 2 (k0 += 8)
            for (int k0 = 0; k0 < K; k0 += 8) {
                int8x16_t b0_0 = vld1q_s8(b0_base + k0 * 4);
                int8x16_t b0_1 = vld1q_s8(b0_base + k0 * 4 + 16);
                int8x16_t b1_0 = vld1q_s8(b1_base + k0 * 4);
                int8x16_t b1_1 = vld1q_s8(b1_base + k0 * 4 + 16);

                // Row 0
                int8x16_t a0_0 = vld1q_s8(a0_base + k0 * 4);
                c00 = vdotq_s32(c00, a0_0, b0_0);
                c01 = vdotq_s32(c01, a0_0, b1_0);
                int8x16_t a0_1 = vld1q_s8(a0_base + k0 * 4 + 16);
                c00 = vdotq_s32(c00, a0_1, b0_1);
                c01 = vdotq_s32(c01, a0_1, b1_1);

                // Row 1
                int8x16_t a1_0 = vld1q_s8(a1_base + k0 * 4);
                c10 = vdotq_s32(c10, a1_0, b0_0);
                c11 = vdotq_s32(c11, a1_0, b1_0);
                int8x16_t a1_1 = vld1q_s8(a1_base + k0 * 4 + 16);
                c10 = vdotq_s32(c10, a1_1, b0_1);
                c11 = vdotq_s32(c11, a1_1, b1_1);

                // Row 2
                int8x16_t a2_0 = vld1q_s8(a2_base + k0 * 4);
                c20 = vdotq_s32(c20, a2_0, b0_0);
                c21 = vdotq_s32(c21, a2_0, b1_0);
                int8x16_t a2_1 = vld1q_s8(a2_base + k0 * 4 + 16);
                c20 = vdotq_s32(c20, a2_1, b0_1);
                c21 = vdotq_s32(c21, a2_1, b1_1);

                // Row 3
                int8x16_t a3_0 = vld1q_s8(a3_base + k0 * 4);
                c30 = vdotq_s32(c30, a3_0, b0_0);
                c31 = vdotq_s32(c31, a3_0, b1_0);
                int8x16_t a3_1 = vld1q_s8(a3_base + k0 * 4 + 16);
                c30 = vdotq_s32(c30, a3_1, b0_1);
                c31 = vdotq_s32(c31, a3_1, b1_1);

                // Row 4
                int8x16_t a4_0 = vld1q_s8(a4_base + k0 * 4);
                c40 = vdotq_s32(c40, a4_0, b0_0);
                c41 = vdotq_s32(c41, a4_0, b1_0);
                int8x16_t a4_1 = vld1q_s8(a4_base + k0 * 4 + 16);
                c40 = vdotq_s32(c40, a4_1, b0_1);
                c41 = vdotq_s32(c41, a4_1, b1_1);

                // Row 5
                int8x16_t a5_0 = vld1q_s8(a5_base + k0 * 4);
                c50 = vdotq_s32(c50, a5_0, b0_0);
                c51 = vdotq_s32(c51, a5_0, b1_0);
                int8x16_t a5_1 = vld1q_s8(a5_base + k0 * 4 + 16);
                c50 = vdotq_s32(c50, a5_1, b0_1);
                c51 = vdotq_s32(c51, a5_1, b1_1);

                // Row 6
                int8x16_t a6_0 = vld1q_s8(a6_base + k0 * 4);
                c60 = vdotq_s32(c60, a6_0, b0_0);
                c61 = vdotq_s32(c61, a6_0, b1_0);
                int8x16_t a6_1 = vld1q_s8(a6_base + k0 * 4 + 16);
                c60 = vdotq_s32(c60, a6_1, b0_1);
                c61 = vdotq_s32(c61, a6_1, b1_1);

                // Row 7
                int8x16_t a7_0 = vld1q_s8(a7_base + k0 * 4);
                c70 = vdotq_s32(c70, a7_0, b0_0);
                c71 = vdotq_s32(c71, a7_0, b1_0);
                int8x16_t a7_1 = vld1q_s8(a7_base + k0 * 4 + 16);
                c70 = vdotq_s32(c70, a7_1, b0_1);
                c71 = vdotq_s32(c71, a7_1, b1_1);
            }

            // Dequantize and store
            float32x4_t csB0 = vld1q_f32(csB + jg * 4);
            float32x4_t csB1 = vld1q_f32(csB + (jg + 1) * 4);

            float* c0 = C_ptr + (i + 0) * N + jg * 4;
            float* c1 = C_ptr + (i + 1) * N + jg * 4;
            float* c2 = C_ptr + (i + 2) * N + jg * 4;
            float* c3 = C_ptr + (i + 3) * N + jg * 4;
            float* c4 = C_ptr + (i + 4) * N + jg * 4;
            float* c5 = C_ptr + (i + 5) * N + jg * 4;
            float* c6 = C_ptr + (i + 6) * N + jg * 4;
            float* c7 = C_ptr + (i + 7) * N + jg * 4;

            float32x4_t rs0 = vdupq_n_f32(rs[0]);
            float32x4_t rs1 = vdupq_n_f32(rs[1]);
            float32x4_t rs2 = vdupq_n_f32(rs[2]);
            float32x4_t rs3 = vdupq_n_f32(rs[3]);
            float32x4_t rs4 = vdupq_n_f32(rs[4]);
            float32x4_t rs5 = vdupq_n_f32(rs[5]);
            float32x4_t rs6 = vdupq_n_f32(rs[6]);
            float32x4_t rs7 = vdupq_n_f32(rs[7]);

            vst1q_f32(c0,     vmulq_f32(vcvtq_f32_s32(c00), vmulq_f32(rs0, csB0)));
            vst1q_f32(c0 + 4, vmulq_f32(vcvtq_f32_s32(c01), vmulq_f32(rs0, csB1)));

            vst1q_f32(c1,     vmulq_f32(vcvtq_f32_s32(c10), vmulq_f32(rs1, csB0)));
            vst1q_f32(c1 + 4, vmulq_f32(vcvtq_f32_s32(c11), vmulq_f32(rs1, csB1)));

            vst1q_f32(c2,     vmulq_f32(vcvtq_f32_s32(c20), vmulq_f32(rs2, csB0)));
            vst1q_f32(c2 + 4, vmulq_f32(vcvtq_f32_s32(c21), vmulq_f32(rs2, csB1)));

            vst1q_f32(c3,     vmulq_f32(vcvtq_f32_s32(c30), vmulq_f32(rs3, csB0)));
            vst1q_f32(c3 + 4, vmulq_f32(vcvtq_f32_s32(c31), vmulq_f32(rs3, csB1)));

            vst1q_f32(c4,     vmulq_f32(vcvtq_f32_s32(c40), vmulq_f32(rs4, csB0)));
            vst1q_f32(c4 + 4, vmulq_f32(vcvtq_f32_s32(c41), vmulq_f32(rs4, csB1)));

            vst1q_f32(c5,     vmulq_f32(vcvtq_f32_s32(c50), vmulq_f32(rs5, csB0)));
            vst1q_f32(c5 + 4, vmulq_f32(vcvtq_f32_s32(c51), vmulq_f32(rs5, csB1)));

            vst1q_f32(c6,     vmulq_f32(vcvtq_f32_s32(c60), vmulq_f32(rs6, csB0)));
            vst1q_f32(c6 + 4, vmulq_f32(vcvtq_f32_s32(c61), vmulq_f32(rs6, csB1)));

            vst1q_f32(c7,     vmulq_f32(vcvtq_f32_s32(c70), vmulq_f32(rs7, csB0)));
            vst1q_f32(c7 + 4, vmulq_f32(vcvtq_f32_s32(c71), vmulq_f32(rs7, csB1)));
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
              << "       ARM64 NEON QUANTIZED GEMM BENCHMARK\n"
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

    free(A_int8); free(B_int8); free(B_int8_T);
    free(A_packed); free(B_packed);
    return 0;
}
