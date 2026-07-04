module {
  func.func @vecadd_f64(%arg0: memref<8xf64>, %arg1: memref<8xf64>, %arg2: memref<8xf64>) {
    %cst = arith.constant 0.000000e+00 : f64
    affine.for %i = 0 to 8 step 4 {
      %a = vector.transfer_read %arg0[%i], %cst : memref<8xf64>, vector<4xf64>
      %b = vector.transfer_read %arg1[%i], %cst : memref<8xf64>, vector<4xf64>
      %c = arith.addf %a, %b : vector<4xf64>
      vector.transfer_write %c, %arg2[%i] : vector<4xf64>, memref<8xf64>
    }
    return
  }
}
