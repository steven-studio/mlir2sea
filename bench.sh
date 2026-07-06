#!/bin/bash
# benchmark: scalar vs RVV matmul 8x8

echo "=== Building scalar ==="
./build/mlir2sea test/matmul_8x8_vec4.mlir 2>/dev/null
# wait, scalar 要用 non-vec mlir
./build/mlir2sea test/matmul.mlir 2>/dev/null
cp /tmp/mlir2sea_out.c /tmp/bench_scalar.c

echo "=== Building RVV ==="
./build/mlir2sea test/matmul_8x8_vec4.mlir --emit-rvv 2>/dev/null
cp /tmp/mlir2sea_rvv.c /tmp/bench_rvv.c

echo "=== Done ==="
