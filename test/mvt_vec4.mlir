#bcast4 = affine_map<(d0,d1) -> (0)>
#bcast1 = affine_map<(d0) -> (0)>
module {
  func.func @mvt(%A: memref<4x4xf32>, %y1: memref<4xf32>, %y2: memref<4xf32>,
                 %x1: memref<4xf32>, %x2: memref<4xf32>) {
    %cst = arith.constant 0.000000e+00 : f32
    %c0 = arith.constant 0 : index

    // Part 1: x1 = x1 + A*y1  (row reduction)
    affine.for %i = 0 to 4 {
      %row = vector.transfer_read %A[%i, %c0], %cst : memref<4x4xf32>, vector<4xf32>
      %y1v = vector.transfer_read %y1[%c0], %cst : memref<4xf32>, vector<4xf32>
      %prod = arith.mulf %row, %y1v : vector<4xf32>
      %sum = vector.reduction <add>, %prod : vector<4xf32> into f32
      %sumv = vector.broadcast %sum : f32 to vector<1xf32>
      %x1old = vector.transfer_read %x1[%i], %cst : memref<4xf32>, vector<1xf32>
      %x1new = arith.addf %x1old, %sumv : vector<1xf32>
      vector.transfer_write %x1new, %x1[%i] : vector<1xf32>, memref<4xf32>
    }

    // Part 2: x2 = x2 + A^T*y2  (outer accumulation)
    affine.for %i = 0 to 4 {
      %y2s = vector.transfer_read %y2[%i], %cst {permutation_map = #bcast1} : memref<4xf32>, vector<4xf32>
      %row2 = vector.transfer_read %A[%i, %c0], %cst : memref<4x4xf32>, vector<4xf32>
      %prod2 = arith.mulf %row2, %y2s : vector<4xf32>
      %x2old = vector.transfer_read %x2[%c0], %cst : memref<4xf32>, vector<4xf32>
      %x2new = arith.addf %x2old, %prod2 : vector<4xf32>
      vector.transfer_write %x2new, %x2[%c0] : vector<4xf32>, memref<4xf32>
    }
    return
  }
}
