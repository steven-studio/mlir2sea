module {
  func.func @strided_vecadd(%arg0: memref<4xf32, strided<[2], offset: 0>>,
                             %arg1: memref<4xf32>,
                             %arg2: memref<4xf32>) {
    %cst = arith.constant 0.000000e+00 : f32
    %c0 = arith.constant 0 : index
    %a = vector.transfer_read %arg0[%c0], %cst : memref<4xf32, strided<[2], offset: 0>>, vector<4xf32>
    %b = vector.transfer_read %arg1[%c0], %cst : memref<4xf32>, vector<4xf32>
    %c = arith.addf %a, %b : vector<4xf32>
    vector.transfer_write %c, %arg2[%c0] : vector<4xf32>, memref<4xf32>
    return
  }
}
