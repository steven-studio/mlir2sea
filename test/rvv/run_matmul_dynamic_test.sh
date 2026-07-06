#!/bin/bash
set -e
cd "$(dirname "$0")/../.."

mlir-opt-18 \
    --convert-linalg-to-affine-loops \
    --affine-super-vectorize="virtual-vector-size=4" \
    test/rvv/matmul_dyn.mlir > /tmp/matmul_dyn_vec.mlir

./build/mlir2sea /tmp/matmul_dyn_vec.mlir --emit-rvv

riscv64-linux-gnu-gcc -march=rv64gcv -static \
    test/rvv/matmul_dynamic_multi.c /tmp/mlir2sea_rvv.c \
    -o /tmp/test_rvv_dyn_multi

qemu-riscv64 -cpu rv64,v=true /tmp/test_rvv_dyn_multi
