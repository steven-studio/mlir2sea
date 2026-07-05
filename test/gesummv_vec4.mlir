module {
  func.func @gesummv(%A: memref<4x4xf32>, %B: memref<4x4xf32>, %x: memref<4xf32>, %y: memref<4xf32>) {
    %cst = arith.constant 0.000000e+00 : f32
    %c0 = arith.constant 0 : index
    affine.for %i = 0 to 4 {
      %rowA = vector.transfer_read %A[%i, %c0], %cst : memref<4x4xf32>, vector<4xf32>
      %rowB = vector.transfer_read %B[%i, %c0], %cst : memref<4x4xf32>, vector<4xf32>
      %xv = vector.transfer_read %x[%c0], %cst : memref<4xf32>, vector<4xf32>
      %prodA = arith.mulf %rowA, %xv : vector<4xf32>
      %prodB = arith.mulf %rowB, %xv : vector<4xf32>
      %sumA = vector.reduction <add>, %prodA : vector<4xf32> into f32
      %sumB = vector.reduction <add>, %prodB : vector<4xf32> into f32
      %total = arith.addf %sumA, %sumB : f32
      %totalv = vector.broadcast %total : f32 to vector<1xf32>
      vector.transfer_write %totalv, %y[%i] : vector<1xf32>, memref<4xf32>
    }
    return
  }
}
