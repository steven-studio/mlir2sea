module {
  func.func @adi_forward(%X: memref<8xf32>, %a: memref<8xf32>, %c: memref<8xf32>) {
    %cst = arith.constant 0.000000e+00 : f32
    affine.for %i = 1 to 5 step 4 {
      %b = vector.transfer_read %X[%i], %cst : memref<8xf32>, vector<4xf32>
      %av = vector.transfer_read %a[%i], %cst : memref<8xf32>, vector<4xf32>
      %cv = vector.transfer_read %c[%i], %cst : memref<8xf32>, vector<4xf32>
      %im1 = affine.apply affine_map<(d0) -> (d0 - 1)>(%i)
      %xprev = vector.transfer_read %X[%im1], %cst : memref<8xf32>, vector<4xf32>
      %prod = arith.mulf %av, %xprev : vector<4xf32>
      %num = arith.subf %b, %prod : vector<4xf32>
      %newx = arith.divf %num, %cv : vector<4xf32>
      vector.transfer_write %newx, %X[%i] : vector<4xf32>, memref<8xf32>
    }
    return
  }
}
