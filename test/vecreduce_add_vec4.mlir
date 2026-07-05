module {
  func.func @vecreduce_add(%arg0: memref<4xf32>, %arg1: memref<4xf32>) {
    %cst = arith.constant 0.000000e+00 : f32
    %c0 = arith.constant 0 : index
    %v = vector.transfer_read %arg0[%c0], %cst : memref<4xf32>, vector<4xf32>
    %sum = vector.reduction <add>, %v : vector<4xf32> into f32
    %sumv = vector.broadcast %sum : f32 to vector<4xf32>
    vector.transfer_write %sumv, %arg1[%c0] : vector<4xf32>, memref<4xf32>
    return
  }
}
