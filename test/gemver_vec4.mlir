#bcast1 = affine_map<(d0) -> (0)>
module {
  func.func @gemver(%A: memref<4x4xf32>, %u1: memref<4xf32>, %v1: memref<4xf32>,
                    %u2: memref<4xf32>, %v2: memref<4xf32>, %y: memref<4xf32>,
                    %z: memref<4xf32>, %x: memref<4xf32>, %w: memref<4xf32>) {
    %cst = arith.constant 0.000000e+00 : f32
    %c0 = arith.constant 0 : index

    // A = A + u1*v1^T + u2*v2^T  (rank-1 updates)
    affine.for %i = 0 to 4 {
      %u1s = vector.transfer_read %u1[%i], %cst {permutation_map = #bcast1} : memref<4xf32>, vector<4xf32>
      %u2s = vector.transfer_read %u2[%i], %cst {permutation_map = #bcast1} : memref<4xf32>, vector<4xf32>
      %v1v = vector.transfer_read %v1[%c0], %cst : memref<4xf32>, vector<4xf32>
      %v2v = vector.transfer_read %v2[%c0], %cst : memref<4xf32>, vector<4xf32>
      %Arow = vector.transfer_read %A[%i, %c0], %cst : memref<4x4xf32>, vector<4xf32>
      %t1 = arith.mulf %u1s, %v1v : vector<4xf32>
      %t2 = arith.mulf %u2s, %v2v : vector<4xf32>
      %s1 = arith.addf %Arow, %t1 : vector<4xf32>
      %s2 = arith.addf %s1, %t2 : vector<4xf32>
      vector.transfer_write %s2, %A[%i, %c0] : vector<4xf32>, memref<4x4xf32>
    }

    // x = A^T*y + z  (row broadcast accumulation)
    affine.for %i = 0 to 4 {
      %ys = vector.transfer_read %y[%i], %cst {permutation_map = #bcast1} : memref<4xf32>, vector<4xf32>
      %row2 = vector.transfer_read %A[%i, %c0], %cst : memref<4x4xf32>, vector<4xf32>
      %prod2 = arith.mulf %row2, %ys : vector<4xf32>
      %xold = vector.transfer_read %x[%c0], %cst : memref<4xf32>, vector<4xf32>
      %xnew = arith.addf %xold, %prod2 : vector<4xf32>
      vector.transfer_write %xnew, %x[%c0] : vector<4xf32>, memref<4xf32>
    }
    affine.for %i = 0 to 4 {
      %xold2 = vector.transfer_read %x[%i], %cst : memref<4xf32>, vector<1xf32>
      %zv = vector.transfer_read %z[%i], %cst : memref<4xf32>, vector<1xf32>
      %xnew2 = arith.addf %xold2, %zv : vector<1xf32>
      vector.transfer_write %xnew2, %x[%i] : vector<1xf32>, memref<4xf32>
    }

    // w = A*x  (row reduction)
    affine.for %i = 0 to 4 {
      %row3 = vector.transfer_read %A[%i, %c0], %cst : memref<4x4xf32>, vector<4xf32>
      %xv = vector.transfer_read %x[%c0], %cst : memref<4xf32>, vector<4xf32>
      %prod3 = arith.mulf %row3, %xv : vector<4xf32>
      %sum = vector.reduction <add>, %prod3 : vector<4xf32> into f32
      %sumv = vector.broadcast %sum : f32 to vector<1xf32>
      vector.transfer_write %sumv, %w[%i] : vector<1xf32>, memref<4xf32>
    }
    return
  }
}
