module {
  func.func @jacobi2d(%A: memref<8x8xf32>, %B: memref<8x8xf32>) {
    %cst = arith.constant 0.000000e+00 : f32
    %cst5 = arith.constant 2.000000e-01 : f32
    %c1 = arith.constant 1 : index
    %five = vector.broadcast %cst5 : f32 to vector<4xf32>
    affine.for %i = 1 to 7 {
      affine.for %j = 1 to 5 step 4 {
        %center = vector.transfer_read %A[%i, %j], %cst : memref<8x8xf32>, vector<4xf32>
        %jm1 = affine.apply affine_map<(d0) -> (d0 - 1)>(%j)
        %left = vector.transfer_read %A[%i, %jm1], %cst : memref<8x8xf32>, vector<4xf32>
        %jp1 = affine.apply affine_map<(d0) -> (d0 + 1)>(%j)
        %right = vector.transfer_read %A[%i, %jp1], %cst : memref<8x8xf32>, vector<4xf32>
        %im1 = affine.apply affine_map<(d0) -> (d0 - 1)>(%i)
        %up = vector.transfer_read %A[%im1, %j], %cst : memref<8x8xf32>, vector<4xf32>
        %ip1 = affine.apply affine_map<(d0) -> (d0 + 1)>(%i)
        %down = vector.transfer_read %A[%ip1, %j], %cst : memref<8x8xf32>, vector<4xf32>
        %s1 = arith.addf %center, %left : vector<4xf32>
        %s2 = arith.addf %s1, %right : vector<4xf32>
        %s3 = arith.addf %s2, %up : vector<4xf32>
        %s4 = arith.addf %s3, %down : vector<4xf32>
        %res = arith.mulf %s4, %five : vector<4xf32>
        vector.transfer_write %res, %B[%i, %j] : vector<4xf32>, memref<8x8xf32>
      }
    }
    return
  }
}
