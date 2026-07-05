module {
  func.func @heat3d(%A: memref<4x4x8xf32>, %B: memref<4x4x8xf32>) {
    %cst = arith.constant 0.000000e+00 : f32
    %cst1 = arith.constant 1.000000e-01 : f32
    %coef = vector.broadcast %cst1 : f32 to vector<4xf32>
    affine.for %i = 1 to 3 {
      affine.for %j = 1 to 3 {
        affine.for %k = 1 to 5 step 4 {
          %center = vector.transfer_read %A[%i, %j, %k], %cst : memref<4x4x8xf32>, vector<4xf32>
          %im1 = affine.apply affine_map<(d0) -> (d0 - 1)>(%i)
          %xm = vector.transfer_read %A[%im1, %j, %k], %cst : memref<4x4x8xf32>, vector<4xf32>
          %ip1 = affine.apply affine_map<(d0) -> (d0 + 1)>(%i)
          %xp = vector.transfer_read %A[%ip1, %j, %k], %cst : memref<4x4x8xf32>, vector<4xf32>
          %jm1 = affine.apply affine_map<(d0) -> (d0 - 1)>(%j)
          %ym = vector.transfer_read %A[%i, %jm1, %k], %cst : memref<4x4x8xf32>, vector<4xf32>
          %jp1 = affine.apply affine_map<(d0) -> (d0 + 1)>(%j)
          %yp = vector.transfer_read %A[%i, %jp1, %k], %cst : memref<4x4x8xf32>, vector<4xf32>
          %km1 = affine.apply affine_map<(d0) -> (d0 - 1)>(%k)
          %zm = vector.transfer_read %A[%i, %j, %km1], %cst : memref<4x4x8xf32>, vector<4xf32>
          %kp1 = affine.apply affine_map<(d0) -> (d0 + 1)>(%k)
          %zp = vector.transfer_read %A[%i, %j, %kp1], %cst : memref<4x4x8xf32>, vector<4xf32>
          %s1 = arith.addf %center, %xm : vector<4xf32>
          %s2 = arith.addf %s1, %xp : vector<4xf32>
          %s3 = arith.addf %s2, %ym : vector<4xf32>
          %s4 = arith.addf %s3, %yp : vector<4xf32>
          %s5 = arith.addf %s4, %zm : vector<4xf32>
          %s6 = arith.addf %s5, %zp : vector<4xf32>
          %res = arith.mulf %s6, %coef : vector<4xf32>
          vector.transfer_write %res, %B[%i, %j, %k] : vector<4xf32>, memref<4x4x8xf32>
        }
      }
    }
    return
  }
}
