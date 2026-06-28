#map = affine_map<(d0, d1) -> (0)>
module {
  func.func @matmul(%arg0: memref<4x4xf32>, %arg1: memref<4x4xf32>, %arg2: memref<4x4xf32>) {
    affine.for %arg3 = 0 to 4 {
      affine.for %arg4 = 0 to 4 step 4 {
        affine.for %arg5 = 0 to 4 {
          %cst = arith.constant 0.000000e+00 : f32
          %0 = vector.transfer_read %arg0[%arg3, %arg5], %cst {permutation_map = #map} : memref<4x4xf32>, vector<4xf32>
          %cst_0 = arith.constant 0.000000e+00 : f32
          %1 = vector.transfer_read %arg1[%arg5, %arg4], %cst_0 : memref<4x4xf32>, vector<4xf32>
          %cst_1 = arith.constant 0.000000e+00 : f32
          %2 = vector.transfer_read %arg2[%arg3, %arg4], %cst_1 : memref<4x4xf32>, vector<4xf32>
          %3 = arith.mulf %0, %1 : vector<4xf32>
          %4 = arith.addf %2, %3 : vector<4xf32>
          vector.transfer_write %4, %arg2[%arg3, %arg4] : vector<4xf32>, memref<4x4xf32>
        }
      }
    }
    return
  }
}

