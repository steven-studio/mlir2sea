module {
  func.func @jacobi_1d(%arg0: memref<16xf32>, %arg1: memref<16xf32>) {
    %c1 = arith.constant 1 : index
    %cst = arith.constant 3.333333e-01 : f32
    %cstv = vector.broadcast %cst : f32 to vector<4xf32>
    affine.for %i = 1 to 13 step 4 {
      %im1 = arith.subi %i, %c1 : index
      %ip1 = arith.addi %i, %c1 : index
      %a0 = vector.transfer_read %arg0[%im1], %cst : memref<16xf32>, vector<4xf32>
      %a1 = vector.transfer_read %arg0[%i],   %cst : memref<16xf32>, vector<4xf32>
      %a2 = vector.transfer_read %arg0[%ip1], %cst : memref<16xf32>, vector<4xf32>
      %sum = arith.addf %a0, %a1 : vector<4xf32>
      %sum2 = arith.addf %sum, %a2 : vector<4xf32>
      %res = arith.mulf %sum2, %cstv : vector<4xf32>
      vector.transfer_write %res, %arg1[%i] : vector<4xf32>, memref<16xf32>
    }
    return
  }
}
