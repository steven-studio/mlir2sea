module {
  func.func @relu6_vec4(%arg0: memref<8xf32>, %arg1: memref<8xf32>) {
    %cst0 = arith.constant 0.000000e+00 : f32
    %cst6 = arith.constant 6.000000e+00 : f32
    %zero = vector.broadcast %cst0 : f32 to vector<4xf32>
    %six = vector.broadcast %cst6 : f32 to vector<4xf32>
    affine.for %i = 0 to 8 step 4 {
      %x = vector.transfer_read %arg0[%i], %cst0 : memref<8xf32>, vector<4xf32>
      %clamped_low = arith.maximumf %x, %zero : vector<4xf32>
      %clamped = arith.minimumf %clamped_low, %six : vector<4xf32>
      vector.transfer_write %clamped, %arg1[%i] : vector<4xf32>, memref<8xf32>
    }
    return
  }
}
