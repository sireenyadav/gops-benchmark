#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <cstdlib>
#include <cstring>
#include <arm_neon.h>

constexpr int M = 1024;
constexpr int K = 1024;
constexpr int N = 1024;

// -----------------------------------------------------------------------------
// Utility: fill matrix with random floats in [-1, 1]
// -----------------------------------------------------------------------------
void init_matrix(std::vector<float>& mat, int size) {
    for (int i = 0; i < size; ++i) {
        mat[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX) * 2.0f - 1.0f;
    }
}

// -----------------------------------------------------------------------------
// Path A: naive scalar float GEMM (unchanged)
// -----------------------------------------------------------------------------
void gemm_float32(const std::vector<float>& A, const std::vector<float>& B, std::vector<float>& C) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

// -----------------------------------------------------------------------------
// Symmetric int8 quantization (unchanged)
// -----------------------------------------------------------------------------
float quantize_to_int8(const std::vector<float>& mat, int8_t* mat_int8, int size) {
    float max_val = 0.0f;
    for (int i = 0; i < size; ++i) {
        max_val = std::max(max_val, std::abs(mat[i]));
    }
    float scale = max_val / 127.0f;
    float inv_scale = 1.0f / scale;

    for (int i = 0; i < size; ++i) {
        float q = std::round(mat[i] * inv_scale);
        q = std::max(-127.0f, std::min(127.0f, q));
        mat_int8[i] = static_cast<int8_t>(q);
    }
    return scale;
}

// -----------------------------------------------------------------------------
// Pack A for vdotq: duplicate every 4-byte block 4 times
// A_packed size = M * K * 4 bytes
// -----------------------------------------------------------------------------
void pack_A_vdot(const int8_t* A, int8_t* A_packed) {
    for (int i = 0; i < M; ++i) {
        const int8_t* row = A + i * K;
        int8_t* dst = A_packed + i * (K * 4);
        for (int k0 = 0; k0 < K; k0 += 4) {
            // copy the 4 bytes four times to form a 16-byte vector
            for (int rep = 0; rep < 4; ++rep) {
                std::memcpy(dst, row + k0, 4);
                dst += 4;
            }
        }
    }
}

// -----------------------------------------------------------------------------
// Pack B (already transposed) for vdotq:
// interleave 4 columns per k‑block of 4 bytes
// B_packed size = N * K bytes
// -----------------------------------------------------------------------------
void pack_B_vdot(const int8_t* B_T, int8_t* B_packed) {
    for (int jg = 0; jg < N / 4; ++jg) {
        int8_t* dst = B_packed + jg * (K * 4);   // each group occupies 4*K bytes
        for (int k0 = 0; k0 < K; k0 += 4) {
            for (int jj = 0; jj < 4; ++jj) {
                const int8_t* col = B_T + (jg * 4 + jj) * K + k0;
                std::memcpy(dst, col, 4);
                dst += 4;
            }
        }
    }
}

// -----------------------------------------------------------------------------
// main
// -----------------------------------------------------------------------------
int main() {
    srand(42);

    std::vector<float> A(M * K);
    std::vector<float> B(K * N);
    std::vector<float> C_float(M * N, 0.0f);
    std::vector<float> C_quant(M * N, 0.0f);

    init_matrix(A, M * K);
    init_matrix(B, K * N);

    // ------------------------------- Path A ---------------------------------
    auto start_f32 = std::chrono::steady_clock::now();
    gemm_float32(A, B, C_float);
    auto end_f32 = std::chrono::steady_clock::now();
    std::chrono::duration<double> diff_f32 = end_f32 - start_f32;

    // ---------------------------- Quantization ------------------------------
    int8_t *A_int8 = nullptr, *B_int8 = nullptr, *B_int8_T = nullptr;
    posix_memalign(reinterpret_cast<void**>(&A_int8), 64, M * K);
    posix_memalign(reinterpret_cast<void**>(&B_int8), 64, K * N);
    posix_memalign(reinterpret_cast<void**>(&B_int8_T), 64, K * N);

    float scale_A = quantize_to_int8(A, A_int8, M * K);
    float scale_B = quantize_to_int8(B, B_int8, K * N);

    // Transpose B (still needed for the packing step)
    for (int i = 0; i < K; ++i)
        for (int j = 0; j < N; ++j)
            B_int8_T[j * K + i] = B_int8[i * N + j];

    // Pack matrices for efficient vdotq usage
    int8_t *A_packed = nullptr, *B_packed = nullptr;
    posix_memalign(reinterpret_cast<void**>(&A_packed), 64, M * K * 4);  // 4x expansion
    posix_memalign(reinterpret_cast<void**>(&B_packed), 64, N * K);

    pack_A_vdot(A_int8, A_packed);
    pack_B_vdot(B_int8_T, B_packed);

    float dequant_scale = scale_A * scale_B;

    // ------------------------------- Path B --------------------------------
    auto start_int8 = std::chrono::steady_clock::now();

    constexpr int IBLOCK = 4;              // rows of A processed together
    constexpr int JG = N / 4;              // number of column groups of 4

    for (int i = 0; i < M; i += IBLOCK) {
        for (int jg = 0; jg < JG; ++jg) {
            // Accumulators for the 4 rows of this block
            int32x4_t c0 = vdupq_n_s32(0);
            int32x4_t c1 = vdupq_n_s32(0);
            int32x4_t c2 = vdupq_n_s32(0);
            int32x4_t c3 = vdupq_n_s32(0);

            const int8_t* b_base = B_packed + jg * (K * 4);
            const int8_t* a_base0 = A_packed + (i + 0) * (K * 4);
            const int8_t* a_base1 = A_packed + (i + 1) * (K * 4);
            const int8_t* a_base2 = A_packed + (i + 2) * (K * 4);
            const int8_t* a_base3 = A_packed + (i + 3) * (K * 4);

            // Unroll the k‑loop by 2 to hide load and vdotq latency
            for (int k_block = 0; k_block < K / 4; k_block += 2) {
                int8x16_t b0 = vld1q_s8(b_base + (k_block + 0) * 16);
                int8x16_t b1 = vld1q_s8(b_base + (k_block + 1) * 16);

                int8x16_t a00 = vld1q_s8(a_base0 + (k_block + 0) * 16);
                int8x16_t a01 = vld1q_s8(a_base0 + (k_block + 1) * 16);
                c0 = vdotq_s32(c0, a00, b0);
                c0 = vdotq_s32(c0, a01, b1);

                int8x16_t a10 = vld1q_s8(a_base1 + (k_block + 0) * 16);
                int8x16_t a11 = vld1q_s8(a_base1 + (k_block + 1) * 16);
                c1 = vdotq_s32(c1, a10, b0);
                c1 = vdotq_s32(c1, a11, b1);

                int8x16_t a20 = vld1q_s8(a_base2 + (k_block + 0) * 16);
                int8x16_t a21 = vld1q_s8(a_base2 + (k_block + 1) * 16);
                c2 = vdotq_s32(c2, a20, b0);
                c2 = vdotq_s32(c2, a21, b1);

                int8x16_t a30 = vld1q_s8(a_base3 + (k_block + 0) * 16);
                int8x16_t a31 = vld1q_s8(a_base3 + (k_block + 1) * 16);
                c3 = vdotq_s32(c3, a30, b0);
                c3 = vdotq_s32(c3, a31, b1);
            }

            // Extract and store 4 rows x 4 columns
            int32_t res[4];
            vst1q_s32(res, c0);
            for (int jj = 0; jj < 4; ++jj)
                C_quant[(i + 0) * N + jg * 4 + jj] = static_cast<float>(res[jj]) * dequant_scale;

            vst1q_s32(res, c1);
            for (int jj = 0; jj < 4; ++jj)
                C_quant[(i + 1) * N + jg * 4 + jj] = static_cast<float>(res[jj]) * dequant_scale;

            vst1q_s32(res, c2);
            for (int jj = 0; jj < 4; ++jj)
                C_quant[(i + 2) * N + jg * 4 + jj] = static_cast<float>(res[jj]) * dequant_scale;

            vst1q_s32(res, c3);
            for (int jj = 0; jj < 4; ++jj)
                C_quant[(i + 3) * N + jg * 4 + jj] = static_cast<float>(res[jj]) * dequant_scale;
        }
    }

    auto end_int8 = std::chrono::steady_clock::now();
    std::chrono::duration<double> diff_int8 = end_int8 - start_int8;

    // ------------------------------- Verification --------------------------
    float max_abs_error = 0.0f;
    float max_c_val = 0.0f;
    for (int i = 0; i < M * N; ++i) {
        float err = std::abs(C_float[i] - C_quant[i]);
        max_abs_error = std::max(max_abs_error, err);
        max_c_val = std::max(max_c_val, std::abs(C_float[i]));
    }
    float relative_error_pct = (max_abs_error / max_c_val) * 100.0f;

    // ------------------------------- Reporting -----------------------------
    double total_ops = 2.0 * static_cast<double>(M) * static_cast<double>(N) * static_cast<double>(K);
    double gops_f32 = (total_ops / diff_f32.count()) / 1e9;
    double gops_int8 = (total_ops / diff_int8.count()) / 1e9;
    double speedup = gops_int8 / gops_f32;

    std::cout << "======================================================\n";
    std::cout << "       ARM64 NEON QUANTIZED GEMM BENCHMARK\n";
    std::cout << "======================================================\n";
    std::cout << "Matrix Dimensions    : " << M << " x " << K << " x " << N << "\n";
    std::cout << "Total Arithmetic Ops : " << static_cast<long long>(total_ops) << "\n\n";

    std::cout << "[1] NAIVE SCALAR FLOAT32 ENGINE\n";
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "Execution Time       : " << diff_f32.count() << " seconds\n";
    std::cout << std::setprecision(2);
    std::cout << "Performance          : " << gops_f32 << " GOPS\n\n";

    std::cout << "[2] OPTIMIZED INT8 NEON SIMD ENGINE\n";
    std::cout << std::setprecision(3);
    std::cout << "Execution Time       : " << diff_int8.count() << " seconds\n";
    std::cout << std::setprecision(2);
    std::cout << "Performance          : " << gops_int8 << " GOPS\n";
    std::cout << "Speedup Factor       : " << speedup << "x\n\n";

    std::cout << "[3] CORRECTNESS VERIFICATION\n";
    std::cout << std::setprecision(4);
    std::cout << "Max Absolute Error   : " << max_abs_error << "\n";
    if (relative_error_pct < 5.0f) {
        std::cout << "Status               : PASSED (Precision Loss < 5%)\n";
    } else {
        std::cout << "Status               : FAILED (Precision Loss >= 5%)\n";
    }
    std::cout << "======================================================\n";

    free(A_int8);
    free(B_int8);
    free(B_int8_T);
    free(A_packed);
    free(B_packed);

    return 0;
}
