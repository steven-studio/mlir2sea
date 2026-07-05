#bcast2d = affine_map<(d0,d1) -> (0)>
module {
  func.func @correlation(%data: memref<4x4xf32>, %mean: memref<4xf32>, %stddev: memref<4xf32>, %corr: memref<4x4xf32>) {
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

    // Step 2: stddev[j] = sqrt( sum_i (data[i][j]-mean[j])^2 / M )
    vector.transfer_write %zero0, %stddev[%c0] : vector<4xf32>, memref<4xf32>
    affine.for %i = 0 to 4 {
      %rowd = vector.transfer_read %data[%i, %c0], %cst : memref<4x4xf32>, vector<4xf32>
      %meanv2 = vector.transfer_read %mean[%c0], %cst : memref<4xf32>, vector<4xf32>
      %diff = arith.subf %rowd, %meanv2 : vector<4xf32>
      %sq = arith.mulf %diff, %diff : vector<4xf32>
      %oldsd = vector.transfer_read %stddev[%c0], %cst : memref<4xf32>, vector<4xf32>
      %newsd = arith.addf %oldsd, %sq : vector<4xf32>
      vector.transfer_write %newsd, %stddev[%c0] : vector<4xf32>, memref<4xf32>
    }
    %sdRaw = vector.transfer_read %stddev[%c0], %cst : memref<4xf32>, vector<4xf32>
    %sdMean = arith.mulf %sdRaw, %invMv : vector<4xf32>
    %sdFinal = math.sqrt %sdMean : vector<4xf32>
    vector.transfer_write %sdFinal, %stddev[%c0] : vector<4xf32>, memref<4xf32>

    // Step 3: data[i][j] = (data[i][j] - mean[j]) / stddev[j]
    affine.for %i = 0 to 4 {
      %row3 = vector.transfer_read %data[%i, %c0], %cst : memref<4x4xf32>, vector<4xf32>
      %meanv3 = vector.transfer_read %mean[%c0], %cst : memref<4xf32>, vector<4xf32>
      %sdv3 = vector.transfer_read %stddev[%c0], %cst : memref<4xf32>, vector<4xf32>
      %centered = arith.subf %row3, %meanv3 : vector<4xf32>
      %normed = arith.divf %centered, %sdv3 : vector<4xf32>
      vector.transfer_write %normed, %data[%i, %c0] : vector<4xf32>, memref<4x4xf32>
    }

    // Step 4: corr[i][j] = sum_k data[k][i]*data[k][j] / (M-1)
    affine.for %i = 0 to 4 {
      vector.transfer_write %zero0, %corr[%i, %c0] : vector<4xf32>, memref<4x4xf32>
      affine.for %k = 0 to 4 {
        %dki = vector.transfer_read %data[%k, %i], %cst {permutation_map = #bcast2d} : memref<4x4xf32>, vector<4xf32>
        %dkj = vector.transfer_read %data[%k, %c0], %cst : memref<4x4xf32>, vector<4xf32>
        %prod = arith.mulf %dki, %dkj : vector<4xf32>
        %oldc = vector.transfer_read %corr[%i, %c0], %cst : memref<4x4xf32>, vector<4xf32>
        %newc = arith.addf %oldc, %prod : vector<4xf32>
        vector.transfer_write %newc, %corr[%i, %c0] : vector<4xf32>, memref<4x4xf32>
      }
      %corrRow = vector.transfer_read %corr[%i, %c0], %cst : memref<4x4xf32>, vector<4xf32>
      %corrScaled = arith.mulf %corrRow, %invM1v : vector<4xf32>
      vector.transfer_write %corrScaled, %corr[%i, %c0] : vector<4xf32>, memref<4x4xf32>
    }
    return
  }
}
