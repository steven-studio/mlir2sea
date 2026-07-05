module {
  func.func @vecsqrt(%A: memref<8xf32>, %B: memref<8xf32>) {
    %cst = arith.constant 0.000000e+00 : f32
    affine.for %i = 0 to 8 step 4 {
      %x = vector.transfer_read %A[%i], %cst : memref<8xf32>, vector<4xf32>
      %s = math.sqrt %x : vector<4xf32>
      vector.transfer_write %s, %B[%i] : vector<4xf32>, memref<8xf32>
    }
    return
  }
}
