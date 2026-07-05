module {
  func.func @relu_cmpsel(%arg0: memref<8xf32>, %arg1: memref<8xf32>) {
    %cst = arith.constant 0.000000e+00 : f32
    %zero = vector.broadcast %cst : f32 to vector<4xf32>
    affine.for %i = 0 to 8 step 4 {
      %x = vector.transfer_read %arg0[%i], %cst : memref<8xf32>, vector<4xf32>
      %mask = arith.cmpf ogt, %x, %zero : vector<4xf32>
      %result = arith.select %mask, %x, %zero : vector<4xi1>, vector<4xf32>
      vector.transfer_write %result, %arg1[%i] : vector<4xf32>, memref<8xf32>
    }
    return
  }
}
