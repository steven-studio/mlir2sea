#bcast2d = affine_map<(d0,d1) -> (0)>
module {
  func.func @covariance(%data: memref<4x4xf32>, %mean: memref<4xf32>, %cov: memref<4x4xf32>) {
    %cst = arith.constant 0.000000e+00 : f32
    %c0 = arith.constant 0 : index
    %invM = arith.constant 2.500000e-01 : f32
    %invM1 = arith.constant 3.333333e-01 : f32
    %invMv = vector.broadcast %invM : f32 to vector<4xf32>
    %invM1v = vector.broadcast %invM1 : f32 to vector<4xf32>
    %zero0 = vector.broadcast %cst : f32 to vector<4xf32>

    // Step 1: mean[j] = sum_i data[i][j] / M
    vector.transfer_write %zero0, %mean[%c0] : vector<4xf32>, memref<4xf32>
    affine.for %i = 0 to 4 {
      %row = vector.transfer_read %data[%i, %c0], %cst : memref<4x4xf32>, vector<4xf32>
      %old = vector.transfer_read %mean[%c0], %cst : memref<4xf32>, vector<4xf32>
      %new = arith.addf %old, %row : vector<4xf32>
      vector.transfer_write %new, %mean[%c0] : vector<4xf32>, memref<4xf32>
    }
    %meanRaw = vector.transfer_read %mean[%c0], %cst : memref<4xf32>, vector<4xf32>
    %meanScaled = arith.mulf %meanRaw, %invMv : vector<4xf32>
    vector.transfer_write %meanScaled, %mean[%c0] : vector<4xf32>, memref<4xf32>

    // Step 2: center data: data[i][j] -= mean[j]
    affine.for %i = 0 to 4 {
      %row2 = vector.transfer_read %data[%i, %c0], %cst : memref<4x4xf32>, vector<4xf32>
      %meanv = vector.transfer_read %mean[%c0], %cst : memref<4xf32>, vector<4xf32>
      %centered = arith.subf %row2, %meanv : vector<4xf32>
      vector.transfer_write %centered, %data[%i, %c0] : vector<4xf32>, memref<4x4xf32>
    }

    // Step 3: cov[i][j] = sum_k data[k][i]*data[k][j] / (M-1)
    affine.for %i = 0 to 4 {
      vector.transfer_write %zero0, %cov[%i, %c0] : vector<4xf32>, memref<4x4xf32>
      affine.for %k = 0 to 4 {
        %dki = vector.transfer_read %data[%k, %i], %cst {permutation_map = #bcast2d} : memref<4x4xf32>, vector<4xf32>
        %dkj = vector.transfer_read %data[%k, %c0], %cst : memref<4x4xf32>, vector<4xf32>
        %prod = arith.mulf %dki, %dkj : vector<4xf32>
        %oldc = vector.transfer_read %cov[%i, %c0], %cst : memref<4x4xf32>, vector<4xf32>
        %newc = arith.addf %oldc, %prod : vector<4xf32>
        vector.transfer_write %newc, %cov[%i, %c0] : vector<4xf32>, memref<4x4xf32>
      }
      %covRow = vector.transfer_read %cov[%i, %c0], %cst : memref<4x4xf32>, vector<4xf32>
      %covScaled = arith.mulf %covRow, %invM1v : vector<4xf32>
      vector.transfer_write %covScaled, %cov[%i, %c0] : vector<4xf32>, memref<4x4xf32>
    }
    return
  }
}
