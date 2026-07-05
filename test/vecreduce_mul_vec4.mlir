module {
  func.func @vecreduce_mul(%arg0: memref<4xf32>, %arg1: memref<4xf32>) {
    %cst = arith.constant 0.000000e+00 : f32
    %c0 = arith.constant 0 : index
    %v = vector.transfer_read %arg0[%c0], %cst : memref<4xf32>, vector<4xf32>
    %prod = vector.reduction <mul>, %v : vector<4xf32> into f32
    %prodv = vector.broadcast %prod : f32 to vector<4xf32>
    vector.transfer_write %prodv, %arg1[%c0] : vector<4xf32>, memref<4xf32>
    return
  }
}
