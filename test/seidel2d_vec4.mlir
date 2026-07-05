module {
  func.func @seidel2d(%A: memref<8x8xf32>) {
    %cst = arith.constant 0.000000e+00 : f32
    %cst9 = arith.constant 1.111111e-01 : f32
    %ninth = vector.broadcast %cst9 : f32 to vector<4xf32>
    affine.for %i = 1 to 7 {
      affine.for %j = 1 to 5 step 4 {
        %c = vector.transfer_read %A[%i, %j], %cst : memref<8x8xf32>, vector<4xf32>
        %jm1 = affine.apply affine_map<(d0) -> (d0 - 1)>(%j)
        %l = vector.transfer_read %A[%i, %jm1], %cst : memref<8x8xf32>, vector<4xf32>
        %jp1 = affine.apply affine_map<(d0) -> (d0 + 1)>(%j)
        %r = vector.transfer_read %A[%i, %jp1], %cst : memref<8x8xf32>, vector<4xf32>
        %im1 = affine.apply affine_map<(d0) -> (d0 - 1)>(%i)
        %u = vector.transfer_read %A[%im1, %j], %cst : memref<8x8xf32>, vector<4xf32>
        %ip1 = affine.apply affine_map<(d0) -> (d0 + 1)>(%i)
        %d = vector.transfer_read %A[%ip1, %j], %cst : memref<8x8xf32>, vector<4xf32>
        %s1 = arith.addf %c, %l : vector<4xf32>
        %s2 = arith.addf %s1, %r : vector<4xf32>
        %s3 = arith.addf %s2, %u : vector<4xf32>
        %s4 = arith.addf %s3, %d : vector<4xf32>
        %res = arith.mulf %s4, %ninth : vector<4xf32>
        vector.transfer_write %res, %A[%i, %j] : vector<4xf32>, memref<8x8xf32>
      }
    }
    return
  }
}
