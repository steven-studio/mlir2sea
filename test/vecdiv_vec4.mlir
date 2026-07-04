module {
  func.func @vecdiv(%arg0: memref<8xf32>, %arg1: memref<8xf32>, %arg2: memref<8xf32>) {
    %cst = arith.constant 0.000000e+00 : f32
    affine.for %i = 0 to 8 step 4 {
      %a = vector.transfer_read %arg0[%i], %cst : memref<8xf32>, vector<4xf32>
      %b = vector.transfer_read %arg1[%i], %cst : memref<8xf32>, vector<4xf32>
      %c = arith.divf %a, %b : vector<4xf32>
      vector.transfer_write %c, %arg2[%i] : vector<4xf32>, memref<8xf32>
    }
    return
  }
}
