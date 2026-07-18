#!/bin/bash
clang++ -O3 -march=armv8.2-a+dotprod -std=c++17 gemm_neon.cpp -o gemm_neon
./gemm_neon
