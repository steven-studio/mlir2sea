module {
  func.func @mm2(%A: memref<4x4xf32>, %B: memref<4x4xf32>, %C: memref<4x4xf32>,
                 %tmp: memref<4x4xf32>, %D: memref<4x4xf32>) {
    %cst = arith.constant 0.000000e+00 : f32
    // tmp = A * B
    affine.for %i = 0 to 4 {
      affine.for %j = 0 to 4 step 4 {
        affine.for %k = 0 to 4 {
          %a0 = vector.transfer_read %A[%i, %k], %cst {permutation_map = affine_map<(d0,d1)->(0)>} : memref<4x4xf32>, vector<4xf32>
          %b0 = vector.transfer_read %B[%k, %j], %cst : memref<4x4xf32>, vector<4xf32>
          %c0 = vector.transfer_read %tmp[%i, %j], %cst : memref<4x4xf32>, vector<4xf32>
          %m0 = arith.mulf %a0, %b0 : vector<4xf32>
          %s0 = arith.addf %c0, %m0 : vector<4xf32>
          vector.transfer_write %s0, %tmp[%i, %j] : vector<4xf32>, memref<4x4xf32>
        }
      }
    }
    // D = tmp * C
    affine.for %i = 0 to 4 {
      affine.for %j = 0 to 4 step 4 {
        affine.for %k = 0 to 4 {
          %a1 = vector.transfer_read %tmp[%i, %k], %cst {permutation_map = affine_map<(d0,d1)->(0)>} : memref<4x4xf32>, vector<4xf32>
          %b1 = vector.transfer_read %C[%k, %j], %cst : memref<4x4xf32>, vector<4xf32>
          %c1 = vector.transfer_read %D[%i, %j], %cst : memref<4x4xf32>, vector<4xf32>
          %m1 = arith.mulf %a1, %b1 : vector<4xf32>
          %s1 = arith.addf %c1, %m1 : vector<4xf32>
          vector.transfer_write %s1, %D[%i, %j] : vector<4xf32>, memref<4x4xf32>
        }
      }
    }
    return
  }
}
