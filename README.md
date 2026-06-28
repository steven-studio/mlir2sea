
## Vector Dialect Support (feature/rvv-codegen)

### Vectorizing linalg.matmul to vector dialect

```bash
mlir-opt-18 \
    --convert-linalg-to-affine-loops \
    --affine-super-vectorize="virtual-vector-size=4" \
    test/matmul_linalg.mlir > test/matmul_vec4.mlir
```

This produces `vector.transfer_read`, `arith.mulf vector<4xf32>`, `vector.transfer_write` ops.

Next step: mlir2sea vector op handlers → RVV intrinsics.
