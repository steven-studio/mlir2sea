#!/bin/bash
# Usage: ./run_rvv_test.sh <kernel_name>
# Expects: test/<kernel_name>.mlir  and  test/harness/<kernel_name>.c

set -e

if [ -z "$1" ]; then
    echo "Usage: $0 <kernel_name>"
    echo "Example: $0 matmul_dynamic_vec4"
    exit 1
fi

KERNEL="$1"
MLIR_FILE="test/${KERNEL}.mlir"
HARNESS_FILE="test/harness/${KERNEL}.c"

if [ ! -f "$MLIR_FILE" ]; then
    echo "ERROR: $MLIR_FILE not found"
    exit 1
fi
if [ ! -f "$HARNESS_FILE" ]; then
    echo "ERROR: $HARNESS_FILE not found (need a scalar-reference harness for this kernel)"
    exit 1
fi

echo "=== [$KERNEL] Translating MLIR -> RVV C ==="
./build/mlir2sea "$MLIR_FILE" --emit-rvv
if [ $? -ne 0 ]; then
    echo "FAIL: mlir2sea translation failed"
    exit 1
fi

KERNEL_C="/tmp/rvv_${KERNEL}.c"
cp /tmp/mlir2sea_rvv.c "$KERNEL_C"

echo "=== [$KERNEL] Compiling kernel + harness for RISC-V ==="
riscv64-linux-gnu-gcc -static -march=rv64gcv -O2 -c "$KERNEL_C" -o "/tmp/${KERNEL}_kernel.o"
riscv64-linux-gnu-gcc -static -O2 -c "$HARNESS_FILE" -o "/tmp/${KERNEL}_harness.o"
riscv64-linux-gnu-gcc -static "/tmp/${KERNEL}_kernel.o" "/tmp/${KERNEL}_harness.o" -o "/tmp/${KERNEL}_test"

echo "=== [$KERNEL] Running under QEMU ==="
qemu-riscv64 "/tmp/${KERNEL}_test"
RESULT=$?

if [ $RESULT -eq 0 ]; then
    echo "=== [$KERNEL] OVERALL: PASS ==="
else
    echo "=== [$KERNEL] OVERALL: FAIL ==="
fi
exit $RESULT
