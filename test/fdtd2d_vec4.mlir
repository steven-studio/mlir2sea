module {
  func.func @fdtd2d(%ey: memref<8x8xf32>, %ex: memref<8x8xf32>, %hz: memref<8x8xf32>) {
    %cst = arith.constant 0.000000e+00 : f32
    %cst05 = arith.constant 5.000000e-01 : f32
    %cst07 = arith.constant 7.000000e-01 : f32
    %half = vector.broadcast %cst05 : f32 to vector<4xf32>
    %seven = vector.broadcast %cst07 : f32 to vector<4xf32>

    // ey[i][j] -= 0.5*(hz[i][j] - hz[i-1][j]),  i from 1
    affine.for %i = 1 to 7 {
      affine.for %j = 0 to 4 step 4 {
        %eyv = vector.transfer_read %ey[%i, %j], %cst : memref<8x8xf32>, vector<4xf32>
        %hzc = vector.transfer_read %hz[%i, %j], %cst : memref<8x8xf32>, vector<4xf32>
        %im1 = affine.apply affine_map<(d0) -> (d0 - 1)>(%i)
        %hzu = vector.transfer_read %hz[%im1, %j], %cst : memref<8x8xf32>, vector<4xf32>
        %diff = arith.subf %hzc, %hzu : vector<4xf32>
        %scaled = arith.mulf %diff, %half : vector<4xf32>
        %newey = arith.subf %eyv, %scaled : vector<4xf32>
        vector.transfer_write %newey, %ey[%i, %j] : vector<4xf32>, memref<8x8xf32>
      }
    }

    // ex[i][j] -= 0.5*(hz[i][j] - hz[i][j-1]),  j from 1
    affine.for %i = 0 to 8 {
      affine.for %j = 1 to 5 step 4 {
        %exv = vector.transfer_read %ex[%i, %j], %cst : memref<8x8xf32>, vector<4xf32>
        %hzc2 = vector.transfer_read %hz[%i, %j], %cst : memref<8x8xf32>, vector<4xf32>
        %jm1 = affine.apply affine_map<(d0) -> (d0 - 1)>(%j)
        %hzl = vector.transfer_read %hz[%i, %jm1], %cst : memref<8x8xf32>, vector<4xf32>
        %diff2 = arith.subf %hzc2, %hzl : vector<4xf32>
        %scaled2 = arith.mulf %diff2, %half : vector<4xf32>
        %newex = arith.subf %exv, %scaled2 : vector<4xf32>
        vector.transfer_write %newex, %ex[%i, %j] : vector<4xf32>, memref<8x8xf32>
      }
    }

    // hz[i][j] -= 0.7*(ex[i][j+1]-ex[i][j] + ey[i+1][j]-ey[i][j]),  i,j to 6
    affine.for %i = 0 to 6 {
      affine.for %j = 0 to 4 step 4 {
        %hzv = vector.transfer_read %hz[%i, %j], %cst : memref<8x8xf32>, vector<4xf32>
        %exc = vector.transfer_read %ex[%i, %j], %cst : memref<8x8xf32>, vector<4xf32>
        %jp1 = affine.apply affine_map<(d0) -> (d0 + 1)>(%j)
        %exr = vector.transfer_read %ex[%i, %jp1], %cst : memref<8x8xf32>, vector<4xf32>
        %eyc = vector.transfer_read %ey[%i, %j], %cst : memref<8x8xf32>, vector<4xf32>
        %ip1 = affine.apply affine_map<(d0) -> (d0 + 1)>(%i)
        %eyd = vector.transfer_read %ey[%ip1, %j], %cst : memref<8x8xf32>, vector<4xf32>
        %dex = arith.subf %exr, %exc : vector<4xf32>
        %dey = arith.subf %eyd, %eyc : vector<4xf32>
        %sum = arith.addf %dex, %dey : vector<4xf32>
        %scaled3 = arith.mulf %sum, %seven : vector<4xf32>
        %newhz = arith.subf %hzv, %scaled3 : vector<4xf32>
        vector.transfer_write %newhz, %hz[%i, %j] : vector<4xf32>, memref<8x8xf32>
      }
    }
    return
  }
}
