```cpp
#include <arm_neon.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <random>
#include <thread>
#include <vector>

using steady_clock = std::chrono::steady_clock;

constexpr int M = 1024;
constexpr int K = 1024;
constexpr int N = 1024;

// Tune this:
// 32 is a good balance for accuracy/speed.
// 16 gives better accuracy, 64 may be faster on some cores.
constexpr int K_BLOCK = 32;
constexpr int K_BLOCKS = K / K_BLOCK;

static_assert(K % K_BLOCK == 0, "K must be divisible by K_BLOCK");
static_assert(N % 4 == 0, "N must be divisible by 4");
static_assert(K % 16 == 0, "K must be divisible by 16 for vdotq loops");

static inline int32_t hsum_s32(int32x4_t v) {
#if defined(__aarch64__)
    return vaddvq_s32(v);
#else
    int32x2_t s = vadd_s32(vget_low_s32(v), vget_high_s32(v));
    s = vpadd_s32(s, s);
    return vget_lane_s32(s, 0);
#endif
}

static inline float rand_uniform_float(std::mt19937 &rng) {
    static thread_local std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    return dist(rng);
}

void init_matrix(std::vector<float> &mat) {
    std::mt19937 rng(42);
    for (auto &x : mat) x = rand_uniform_float(rng);
}

void gemm_float32(const std::vector<float> &A,
                  const std::vector<float> &B,
                  std::vector<float> &C) {
    for (int i = 0; i < M; ++i) {
        const float *a = A.data() + i * K;
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                sum += a[k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

void quantize_A_blockwise(const std::vector<float> &A,
                          std::vector<int8_t> &Aq,
                          std::vector<float> &A_scales) {
    Aq.resize(M * K);
    A_scales.resize(M * K_BLOCKS);

    for (int i = 0; i < M; ++i) {
        const float *row = A.data() + i * K;
        for (int blk = 0; blk < K_BLOCKS; ++blk) {
            const int k0 = blk * K_BLOCK;
            float max_abs = 0.0f;
            for (int k = 0; k < K_BLOCK; ++k) {
                max_abs = std::max(max_abs, std::fabs(row[k0 + k]));
            }

            float scale = (max_abs > 0.0f) ? (max_abs / 127.0f) : 1.0f;
            A_scales[i * K_BLOCKS + blk] = scale;
            const float inv = 1.0f / scale;

            for (int k = 0; k < K_BLOCK; ++k) {
                float q = std::round(row[k0 + k] * inv);
                q = std::max(-127.0f, std::min(127.0f, q));
                Aq[i * K + k0 + k] = static_cast<int8_t>(q);
            }
        }
    }
}

// Store B transposed in column-major form:
// B_T[j * K + k] = B[k * N + j]
void quantize_B_blockwise_transposed(const std::vector<float> &B,
                                     std::vector<int8_t> &BqT,
                                     std::vector<float> &B_scales) {
    BqT.resize(N * K);
    B_scales.resize(N * K_BLOCKS);

    for (int j = 0; j < N; ++j) {
        for (int blk = 0; blk < K_BLOCKS; ++blk) {
            const int k0 = blk * K_BLOCK;
            float max_abs = 0.0f;
            for (int k = 0; k < K_BLOCK; ++k) {
                max_abs = std::max(max_abs, std::fabs(B[(k0 + k) * N + j]));
            }

            float scale = (max_abs > 0.0f) ? (max_abs / 127.0f) : 1.0f;
            B_scales[j * K_BLOCKS + blk] = scale;
            const float inv = 1.0f / scale;

            for (int k = 0; k < K_BLOCK; ++k) {
                float x = B[(k0 + k) * N + j];
                float q = std::round(x * inv);
                q = std::max(-127.0f, std::min(127.0f, q));
                BqT[j * K + (k0 + k)] = static_cast<int8_t>(q);
            }
        }
    }
}

void gemm_int8_neon_blockwise_mt(const std::vector<int8_t> &Aq,
                                 const std::vector<float> &A_scales,
                                 const std::vector<int8_t> &BqT,
                                 const std::vector<float> &B_scales,
                                 std::vector<float> &C,
                                 int num_threads) {
#if !defined(__aarch64__) || !defined(__ARM_FEATURE_DOTPROD)
    std::cerr << "This build needs AArch64 with dotprod support (vdotq_s32).\n";
    std::exit(1);
#endif

    auto worker = [&](int row_begin, int row_end) {
        for (int i = row_begin; i < row_end; ++i) {
            const int8_t *a_row = Aq.data() + i * K;
            const float  *a_scl = A_scales.data() + i * K_BLOCKS;
            float *c_row = C.data() + i * N;

            for (int jg = 0; jg < N / 4; ++jg) {
                const int j0 = jg * 4;

                const int8_t *b0 = BqT.data() + (j0 + 0) * K;
                const int8_t *b1 = BqT.data() + (j0 + 1) * K;
                const int8_t *b2 = BqT.data() + (j0 + 2) * K;
                const int8_t *b3 = BqT.data() + (j0 + 3) * K;

                const float *bs0 = B_scales.data() + (j0 + 0) * K_BLOCKS;
                const float *bs1 = B_scales.data() + (j0 + 1) * K_BLOCKS;
                const float *bs2 = B_scales.data() + (j0 + 2) * K_BLOCKS;
                const float *bs3 = B_scales.data() + (j0 + 3) * K_BLOCKS;

                float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;

                for (int blk = 0; blk < K_BLOCKS; ++blk) {
                    const int k0 = blk * K_BLOCK;

                    int32x4_t sum0 = vdupq_n_s32(0);
                    int32x4_t sum1 = vdupq_n_s32(0);
                    int32x4_t sum2 = vdupq_n_s32(0);
                    int32x4_t sum3 = vdupq_n_s32(0);

                    // 32-byte block = 2 x 16-byte dotprod steps
                    for (int t = 0; t < K_BLOCK; t += 16) {
                        const int8x16_t a_vec = vld1q_s8(a_row + k0 + t);
                        const int8x16_t bv0   = vld1q_s8(b0 + k0 + t);
                        const int8x16_t bv1   = vld1q_s8(b1 + k0 + t);
                        const int8x16_t bv2   = vld1q_s8(b2 + k0 + t);
                        const int8x16_t bv3   = vld1q_s8(b3 + k0 + t);

                        sum0 = vdotq_s32(sum0, a_vec, bv0);
                        sum1 = vdotq_s32(sum1, a_vec, bv1);
                        sum2 = vdotq_s32(sum2, a_vec, bv2);
                        sum3 = vdotq_s32(sum3, a_vec, bv3);
                    }

                    const int32_t s0 = hsum_s32(sum0);
                    const int32_t s1 = hsum_s32(sum1);
                    const int32_t s2 = hsum_s32(sum2);
                    const int32_t s3 = hsum_s32(sum3);

                    const float sa = a_scl[blk];
                    acc0 += static_cast<float>(s0) * sa * bs0[blk];
                    acc1 += static_cast<float>(s1) * sa * bs1[blk];
                    acc2 += static_cast<float>(s2) * sa * bs2[blk];
                    acc3 += static_cast<float>(s3) * sa * bs3[blk];
                }

                c_row[j0 + 0] = acc0;
                c_row[j0 + 1] = acc1;
                c_row[j0 + 2] = acc2;
                c_row[j0 + 3] = acc3;
            }
        }
    };

    if (num_threads < 1) num_threads = 1;
    if (num_threads > M) num_threads = M;

    std::vector<std::thread> threads;
    threads.reserve(num_threads);

    const int rows_per_thread = (M + num_threads - 1) / num_threads;
    for (int t = 0; t < num_threads; ++t) {
        int row_begin = t * rows_per_thread;
        int row_end = std::min(M, row_begin + rows_per_thread);
        if (row_begin >= row_end) break;
        threads.emplace_back(worker, row_begin, row_end);
    }

    for (auto &th : threads) th.join();
}

int main() {
    std::ios::sync_with_stdio(false);
    std::cin.tie(nullptr);

    std::vector<float> A(M * K), B(K * N);
    std::vector<float> C_ref(M * N, 0.0f), C_q(M * N, 0.0f);

    init_matrix(A);
    init_matrix(B);

    const auto t0 = steady_clock::now();
    gemm_float32(A, B, C_ref);
    const auto t1 = steady_clock::now();

    std::vector<int8_t> Aq, BqT;
    std::vector<float> A_scales, B_scales;

    quantize_A_blockwise(A, Aq, A_scales);
    quantize_B_blockwise_transposed(B, BqT, B_scales);

    int num_threads = std::max(1u, std::thread::hardware_concurrency());
    const auto t2 = steady_clock::now();
    gemm_int8_neon_blockwise_mt(Aq, A_scales, BqT, B_scales, C_q, num_threads);
    const auto t3 = steady_clock::now();

    double time_f32 = std::chrono::duration<double>(t1 - t0).count();
    double time_q   = std::chrono::duration<double>(t3 - t2).count();

    float max_abs_err = 0.0f;
    float max_ref_abs = 0.0f;
    float mean_abs_err = 0.0f;

    for (int i = 0; i < M * N; ++i) {
        float diff = std::fabs(C_ref[i] - C_q[i]);
        max_abs_err = std::max(max_abs_err, diff);
        max_ref_abs = std::max(max_ref_abs, std::fabs(C_ref[i]));
        mean_abs_err += diff;
    }
    mean_abs_err /= static_cast<float>(M * N);

    const double total_ops = 2.0 * M * N * K;
    const double gops_f32 = (total_ops / time_f32) / 1e9;
    const double gops_q    = (total_ops / time_q) / 1e9;
    const double speedup   = gops_q / gops_f32;
    const float rel_max_pct = (max_ref_abs > 0.0f) ? (max_abs_err / max_ref_abs * 100.0f) : 0.0f;

    std::cout << "======================================================\n";
    std::cout << "     ARM64 NEON BLOCKED INT8 GEMM BENCHMARK\n";
    std::cout << "======================================================\n";
    std::cout << "Matrix Dimensions    : " << M << " x " << K << " x " << N << "\n";
    std::cout << "K Block Size         : " << K_BLOCK << "\n";
    std::cout << "Threads Used         : " << num_threads << "\n";
    std::cout << "Total Arithmetic Ops  : " << static_cast<long long>(total_ops) << "\n\n";

    std::cout << "[1] NAIVE SCALAR FLOAT32 ENGINE\n";
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "Execution Time       : " << time_f32 << " seconds\n";
    std::cout << std::setprecision(2);
    std::cout << "Performance          : " << gops_f32 << " GOPS\n\n";

    std::cout << "[2] BLOCKED INT8 NEON SIMD ENGINE\n";
    std::cout << std::setprecision(3);
    std::cout << "Execution Time       : " << time_q << " seconds\n";
    std::cout << std::setprecision(2);
    std::cout << "Performance          : " << gops_q << " GOPS\n";
    std::cout << "Speedup Factor       : " << speedup << "x\n\n";

    std::cout << "[3] CORRECTNESS VERIFICATION\n";
    std::cout << std::setprecision(6);
    std::cout << "Max Absolute Error   : " << max_abs_err << "\n";
    std::cout << "Mean Absolute Error  : " << mean_abs_err << "\n";
    std::cout << "Max Ref Abs Value    : " << max_ref_abs << "\n";
    std::cout << "Status               : " << ((max_abs_err <= 0.05f) ? "PASSED" : "FAILED")
              << " (Target Max Abs Error <= 0.05)\n";
    std::cout << "======================================================\n";

    return 0;
}
```
