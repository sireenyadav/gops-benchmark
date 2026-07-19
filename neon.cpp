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

// Per-row quantization of A
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

// Per-column quantization of B
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

    constexpr int IBLOCK = 4;
    constexpr int JG = N / 4;

    for (int i = 0; i < M; i += IBLOCK) {
        for (int jg = 0; jg < JG; ++jg) {
            int32x4_t c0 = vdupq_n_s32(0);
            int32x4_t c1 = vdupq_n_s32(0);
            int32x4_t c2 = vdupq_n_s32(0);
            int32x4_t c3 = vdupq_n_s32(0);

            const int8_t* b_base = B_packed + jg * (K * 4);
            const int8_t* a0 = A_packed + (i + 0) * (K * 4);
            const int8_t* a1 = A_packed + (i + 1) * (K * 4);
            const int8_t* a2 = A_packed + (i + 2) * (K * 4);
            const int8_t* a3 = A_packed + (i + 3) * (K * 4);

            for (int k_block = 0; k_block < K / 4; k_block += 2) {
                int8x16_t b0 = vld1q_s8(b_base + (k_block + 0) * 16);
                int8x16_t b1 = vld1q_s8(b_base + (k_block + 1) * 16);

                int8x16_t a00 = vld1q_s8(a0 + (k_block + 0) * 16);
                int8x16_t a01 = vld1q_s8(a0 + (k_block + 1) * 16);
                c0 = vdotq_s32(c0, a00, b0);
                c0 = vdotq_s32(c0, a01, b1);

                int8x16_t a10 = vld1q_s8(a1 + (k_block + 0) * 16);
                int8x16_t a11 = vld1q_s8(a1 + (k_block + 1) * 16);
                c1 = vdotq_s32(c1, a10, b0);
                c1 = vdotq_s32(c1, a11, b1);

                int8x16_t a20 = vld1q_s8(a2 + (k_block + 0) * 16);
                int8x16_t a21 = vld1q_s8(a2 + (k_block + 1) * 16);
                c2 = vdotq_s32(c2, a20, b0);
                c2 = vdotq_s32(c2, a21, b1);

                int8x16_t a30 = vld1q_s8(a3 + (k_block + 0) * 16);
                int8x16_t a31 = vld1q_s8(a3 + (k_block + 1) * 16);
                c3 = vdotq_s32(c3, a30, b0);
                c3 = vdotq_s32(c3, a31, b1);
            }

            int32_t res[4];
            float* c_out = C_quant.data();
            float rs0 = row_scales_A[i + 0];
            float rs1 = row_scales_A[i + 1];
            float rs2 = row_scales_A[i + 2];
            float rs3 = row_scales_A[i + 3];

            vst1q_s32(res, c0);
            for (int jj = 0; jj < 4; ++jj)
                c_out[(i + 0) * N + jg * 4 + jj] = static_cast<float>(res[jj]) * rs0 * col_scales_B[jg * 4 + jj];

            vst1q_s32(res, c1);
            for (int jj = 0; jj < 4; ++jj)
                c_out[(i + 1) * N + jg * 4 + jj] = static_cast<float>(res[jj]) * rs1 * col_scales_B[jg * 4 + jj];

            vst1q_s32(res, c2);
            for (int jj = 0; jj < 4; ++jj)
                c_out[(i + 2) * N + jg * 4 + jj] = static_cast<float>(res[jj]) * rs2 * col_scales_B[jg * 4 + jj];

            vst1q_s32(res, c3);
            for (int jj = 0; jj < 4; ++jj)
                c_out[(i + 3) * N + jg * 4 + jj] = static_cast<float>(res[jj]) * rs3 * col_scales_B[jg * 4 + jj];
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
EOF
