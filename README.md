# Systems Engineering Assessment: ARM64 NEON Quantized GEMM

**Target Architecture:** ARMv8.2-A (or higher)
**Allowed Languages:** C++ (C++17 or higher)
**Allowed Headers:** `<iostream>`, `<vector>`, `<chrono>`, `<cmath>`, `<arm_neon.h>`
**Forbidden:** Eigen, OpenBLAS, parallelization frameworks (OpenMP, std::thread)

---

## 1. The Objective

Write a standalone, highly optimized C++ engine that performs General Matrix Multiplication (GEMM) of two dense matrices (C = A * B). 

Standard float32 matrix multiplication is heavily memory-bandwidth bound on mobile System-on-Chips (SoCs). Your objective is to bypass this bottleneck by implementing symmetric integer quantization (float32 -> int8) and accelerating the arithmetic using ARM NEON SIMD intrinsics.

## 2. Mathematical Specifications

**Symmetric Quantization**
Map a continuous floating-point array to an 8-bit signed integer array. The zero-point is assumed to be 0. Calculate the scale 'S' for a given matrix using its maximum absolute value:

S = max(|q_min|, |q_max|) / 127

Quantize each floating-point element 'q' to its 8-bit integer representation 'x':

x = round(q / S)

**Quantized Dot Product Accumulation**
For the core matrix multiplication, the floating-point result C_f is approximated by extracting the scales and summing the integer products:

C_f ~ (S_A * S_B) * sum(A_int * B_int)

## 3. Implementation Requirements

You must implement two complete compute paths for an M x K matrix A and a K x N matrix B:

**Path A: The Baseline (Naive Scalar)**
* Implement a standard triple-nested loop matrix multiplication using float32.

**Path B: The Accelerated Engine (NEON Int8)**
* **Memory Alignment:** Allocate your int8 arrays on 64-byte boundaries using `posix_memalign` to prevent cache line splitting.
* **Cache Locality:** Transpose Matrix B in memory *before* multiplication. If you attempt to read Matrix B column-by-column during the inner loop, you will thrash the CPU cache and fail the performance metric.
* **SIMD Execution:** Write the inner loop utilizing ARM NEON intrinsics. You must load 16-byte chunks using `vld1q_s8` and accumulate the dot products into 32-bit integer vectors using the `vdotq_s32` instruction.
* **De-quantization:** Convert the final 32-bit integer accumulations back to float32 and multiply by (S_A * S_B).

## 4. Benchmarking & Evaluation Metrics

Your program must initialize matrices of size M=1024, K=1024, N=1024 with random floats between -1.0 and 1.0.

Measure the execution time of the matrix multiplication phase exclusively using `std::chrono::steady_clock`. Do not include matrix allocation, cache transposition, or the initial quantization phase in your timer.

Calculate and print the throughput in Giga Operations Per Second (GOPS):

Total Operations = 2 * M * N * K
GOPS = Total Operations / (Time_in_Seconds * 1,000,000,000)

## 5. Acceptance Criteria

| Metric | Target Requirement |
|---|---|
| **Compilation** | Must compile with `clang++ -O3 -march=armv8.2-a+dotprod` |
| **Correctness** | Max absolute error between Path A and Path B < 0.05 |
| **Speedup** | Path B must execute at least 6x faster than Path A |
| **Output** | Console must print execution times and GOPS for both paths |
