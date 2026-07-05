#bcast1 = affine_map<(d0) -> (0)>
module {
  func.func @atax(%A: memref<4x4xf32>, %x: memref<4xf32>, %tmp: memref<4xf32>, %y: memref<4xf32>) {
    %cst = arith.constant 0.000000e+00 : f32
    %c0 = arith.constant 0 : index

    // Part 1: tmp = A * x  (row reduction)
    affine.for %i = 0 to 4 {
      %row = vector.transfer_read %A[%i, %c0], %cst : memref<4x4xf32>, vector<4xf32>
      %xv = vector.transfer_read %x[%c0], %cst : memref<4xf32>, vector<4xf32>
      %prod = arith.mulf %row, %xv : vector<4xf32>
      %sum = vector.reduction <add>, %prod : vector<4xf32> into f32
      %sumv = vector.broadcast %sum : f32 to vector<1xf32>
      vector.transfer_write %sumv, %tmp[%i] : vector<1xf32>, memref<4xf32>
    }

    // Part 2: y = A^T * tmp  (row broadcast accumulation)
    affine.for %i = 0 to 4 {
      %tmps = vector.transfer_read %tmp[%i], %cst {permutation_map = #bcast1} : memref<4xf32>, vector<4xf32>
      %row2 = vector.transfer_read %A[%i, %c0], %cst : memref<4x4xf32>, vector<4xf32>
      %prod2 = arith.mulf %row2, %tmps : vector<4xf32>
      %yold = vector.transfer_read %y[%c0], %cst : memref<4xf32>, vector<4xf32>
      %ynew = arith.addf %yold, %prod2 : vector<4xf32>
      vector.transfer_write %ynew, %y[%c0] : vector<4xf32>, memref<4xf32>
    }
    return
  }
}
