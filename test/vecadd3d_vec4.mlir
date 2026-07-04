module {
  func.func @vecadd3d(%arg0: memref<2x3x8xf32>, %arg1: memref<2x3x8xf32>, %arg2: memref<2x3x8xf32>) {
    %cst = arith.constant 0.000000e+00 : f32
    affine.for %i = 0 to 2 {
      affine.for %j = 0 to 3 {
        affine.for %k = 0 to 8 step 4 {
          %a = vector.transfer_read %arg0[%i, %j, %k], %cst : memref<2x3x8xf32>, vector<4xf32>
          %b = vector.transfer_read %arg1[%i, %j, %k], %cst : memref<2x3x8xf32>, vector<4xf32>
          %c = arith.addf %a, %b : vector<4xf32>
          vector.transfer_write %c, %arg2[%i, %j, %k] : vector<4xf32>, memref<2x3x8xf32>
        }
      }
    }
    return
  }
}
