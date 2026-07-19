#!/usr/bin/env bash
set -Eeuo pipefail

SOURCE="${1:-neon.cpp}"
OUTPUT="${2:-benchmark}"

if [[ ! -f "$SOURCE" ]]; then
    echo "Error: '$SOURCE' not found."
    exit 1
fi

echo "======================================================"
echo " ARM64 NEON GEMM BENCHMARK"
echo "======================================================"

echo "[System]"
echo "Date      : $(date)"
echo "Machine   : $(uname -m)"
echo "Kernel    : $(uname -r)"
echo "CPU Cores : $(getconf _NPROCESSORS_ONLN)"

if command -v lscpu >/dev/null 2>&1; then
    CPU_MODEL=$(lscpu | awk -F: '/Model name/ {gsub(/^[ \t]+/,"",$2); print $2; exit}')
    [[ -n "$CPU_MODEL" ]] && echo "CPU       : $CPU_MODEL"
fi

FLAGS=(
    -O3
    -ffast-math
    -funroll-loops
    -fomit-frame-pointer
    -DNDEBUG
    -std=c++20
    -pthread
)

ARCH=$(uname -m)

if [[ "$ARCH" == "aarch64" ]]; then
    FLAGS+=(
        -march=armv8.2-a+dotprod
    )
fi

echo
echo "[Compile]"
echo "Compiler  : $(g++ --version | head -1)"
echo "Flags     : ${FLAGS[*]}"
echo

START=$(date +%s.%N)

g++ "$SOURCE" "${FLAGS[@]}" -o "$OUTPUT"

END=$(date +%s.%N)

echo "Compile Time : $(awk "BEGIN{print $END-$START}") sec"

echo
echo "[Run]"
echo "------------------------------------------------------"

"./$OUTPUT"

RET=$?

echo "------------------------------------------------------"

if [[ $RET -eq 0 ]]; then
    echo "Status : SUCCESS"
else
    echo "Status : FAILED ($RET)"
fi
