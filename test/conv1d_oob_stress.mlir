#bcast = affine_map<(d0) -> (0)>
module {
  func.func @conv1d_oob_stress(%arg0: memref<8xf32>, %arg1: memref<3xf32>, %arg2: memref<7xf32>) {
    %cst = arith.constant 0.000000e+00 : f32
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    affine.for %i = 0 to 7 step 4 {
      %in0 = vector.transfer_read %arg0[%i], %cst : memref<8xf32>, vector<4xf32>
      %w0 = vector.transfer_read %arg1[%c0], %cst {permutation_map = #bcast} : memref<3xf32>, vector<4xf32>
      %acc0 = arith.mulf %in0, %w0 : vector<4xf32>

      %idx1 = affine.apply affine_map<(d0) -> (d0 + 1)>(%i)
      %in1 = vector.transfer_read %arg0[%idx1], %cst : memref<8xf32>, vector<4xf32>
      %w1 = vector.transfer_read %arg1[%c1], %cst {permutation_map = #bcast} : memref<3xf32>, vector<4xf32>
      %mul1 = arith.mulf %in1, %w1 : vector<4xf32>
      %acc1 = arith.addf %acc0, %mul1 : vector<4xf32>

      %idx2 = affine.apply affine_map<(d0) -> (d0 + 2)>(%i)
      %in2 = vector.transfer_read %arg0[%idx2], %cst : memref<8xf32>, vector<4xf32>
      %w2 = vector.transfer_read %arg1[%c2], %cst {permutation_map = #bcast} : memref<3xf32>, vector<4xf32>
      %mul2 = arith.mulf %in2, %w2 : vector<4xf32>
      %acc2 = arith.addf %acc1, %mul2 : vector<4xf32>

      vector.transfer_write %acc2, %arg2[%i] : vector<4xf32>, memref<7xf32>
    }
    return
  }
}
