#bcast1 = affine_map<(d0) -> (0)>
module {
  func.func @bicg(%A: memref<4x4xf32>, %p: memref<4xf32>, %r: memref<4xf32>,
                  %s: memref<4xf32>, %q: memref<4xf32>) {
    %cst = arith.constant 0.000000e+00 : f32
    %c0 = arith.constant 0 : index

    // s = A^T * p  (row broadcast accumulation)
    affine.for %i = 0 to 4 {
      %ps = vector.transfer_read %p[%i], %cst {permutation_map = #bcast1} : memref<4xf32>, vector<4xf32>
      %row = vector.transfer_read %A[%i, %c0], %cst : memref<4x4xf32>, vector<4xf32>
      %prod = arith.mulf %row, %ps : vector<4xf32>
      %sold = vector.transfer_read %s[%c0], %cst : memref<4xf32>, vector<4xf32>
      %snew = arith.addf %sold, %prod : vector<4xf32>
      vector.transfer_write %snew, %s[%c0] : vector<4xf32>, memref<4xf32>
    }

    // q = A * r  (row reduction)
    affine.for %i = 0 to 4 {
      %row2 = vector.transfer_read %A[%i, %c0], %cst : memref<4x4xf32>, vector<4xf32>
      %rv = vector.transfer_read %r[%c0], %cst : memref<4xf32>, vector<4xf32>
      %prod2 = arith.mulf %row2, %rv : vector<4xf32>
      %sum = vector.reduction <add>, %prod2 : vector<4xf32> into f32
      %sumv = vector.broadcast %sum : f32 to vector<1xf32>
      vector.transfer_write %sumv, %q[%i] : vector<1xf32>, memref<4xf32>
    }
    return
  }
}
