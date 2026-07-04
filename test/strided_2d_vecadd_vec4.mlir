module {
  func.func @strided_2d_vecadd(%arg0: memref<2x4xf32, strided<[16, 2], offset: 0>>,
                                %arg1: memref<2x4xf32>,
                                %arg2: memref<2x4xf32>) {
    %cst = arith.constant 0.000000e+00 : f32
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    affine.for %i = 0 to 2 {
      %a = vector.transfer_read %arg0[%i, %c0], %cst : memref<2x4xf32, strided<[16, 2], offset: 0>>, vector<4xf32>
      %b = vector.transfer_read %arg1[%i, %c0], %cst : memref<2x4xf32>, vector<4xf32>
      %c = arith.addf %a, %b : vector<4xf32>
      vector.transfer_write %c, %arg2[%i, %c0] : vector<4xf32>, memref<2x4xf32>
    }
    return
  }
}
