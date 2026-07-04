#bcast = affine_map<(d0, d1) -> (0)>
module {
  func.func @conv2d(%arg0: memref<8x8xf32>, %arg1: memref<3x3xf32>, %arg2: memref<6x6xf32>) {
    %cst = arith.constant 0.000000e+00 : f32
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    affine.for %oh = 0 to 6 {
      affine.for %ow = 0 to 6 step 4 {
        // kh=0
        %r0 = affine.apply affine_map<(d0) -> (d0 + 0)>(%oh)
        %c00 = affine.apply affine_map<(d0) -> (d0 + 0)>(%ow)
        %in00 = vector.transfer_read %arg0[%r0, %c00], %cst : memref<8x8xf32>, vector<4xf32>
        %w00 = vector.transfer_read %arg1[%c0, %c0], %cst {permutation_map = #bcast} : memref<3x3xf32>, vector<4xf32>
        %acc00 = arith.mulf %in00, %w00 : vector<4xf32>

        %c01 = affine.apply affine_map<(d0) -> (d0 + 1)>(%ow)
        %in01 = vector.transfer_read %arg0[%r0, %c01], %cst : memref<8x8xf32>, vector<4xf32>
        %w01 = vector.transfer_read %arg1[%c0, %c1], %cst {permutation_map = #bcast} : memref<3x3xf32>, vector<4xf32>
        %mul01 = arith.mulf %in01, %w01 : vector<4xf32>
        %acc01 = arith.addf %acc00, %mul01 : vector<4xf32>

        %c02 = affine.apply affine_map<(d0) -> (d0 + 2)>(%ow)
        %in02 = vector.transfer_read %arg0[%r0, %c02], %cst : memref<8x8xf32>, vector<4xf32>
        %w02 = vector.transfer_read %arg1[%c0, %c2], %cst {permutation_map = #bcast} : memref<3x3xf32>, vector<4xf32>
        %mul02 = arith.mulf %in02, %w02 : vector<4xf32>
        %acc02 = arith.addf %acc01, %mul02 : vector<4xf32>

        // kh=1
        %r1 = affine.apply affine_map<(d0) -> (d0 + 1)>(%oh)
        %in10 = vector.transfer_read %arg0[%r1, %c00], %cst : memref<8x8xf32>, vector<4xf32>
        %w10 = vector.transfer_read %arg1[%c1, %c0], %cst {permutation_map = #bcast} : memref<3x3xf32>, vector<4xf32>
        %mul10 = arith.mulf %in10, %w10 : vector<4xf32>
        %acc10 = arith.addf %acc02, %mul10 : vector<4xf32>

        %in11 = vector.transfer_read %arg0[%r1, %c01], %cst : memref<8x8xf32>, vector<4xf32>
        %w11 = vector.transfer_read %arg1[%c1, %c1], %cst {permutation_map = #bcast} : memref<3x3xf32>, vector<4xf32>
        %mul11 = arith.mulf %in11, %w11 : vector<4xf32>
        %acc11 = arith.addf %acc10, %mul11 : vector<4xf32>

        %in12 = vector.transfer_read %arg0[%r1, %c02], %cst : memref<8x8xf32>, vector<4xf32>
        %w12 = vector.transfer_read %arg1[%c1, %c2], %cst {permutation_map = #bcast} : memref<3x3xf32>, vector<4xf32>
        %mul12 = arith.mulf %in12, %w12 : vector<4xf32>
        %acc12 = arith.addf %acc11, %mul12 : vector<4xf32>

        // kh=2
        %r2 = affine.apply affine_map<(d0) -> (d0 + 2)>(%oh)
        %in20 = vector.transfer_read %arg0[%r2, %c00], %cst : memref<8x8xf32>, vector<4xf32>
        %w20 = vector.transfer_read %arg1[%c2, %c0], %cst {permutation_map = #bcast} : memref<3x3xf32>, vector<4xf32>
        %mul20 = arith.mulf %in20, %w20 : vector<4xf32>
        %acc20 = arith.addf %acc12, %mul20 : vector<4xf32>

        %in21 = vector.transfer_read %arg0[%r2, %c01], %cst : memref<8x8xf32>, vector<4xf32>
        %w21 = vector.transfer_read %arg1[%c2, %c1], %cst {permutation_map = #bcast} : memref<3x3xf32>, vector<4xf32>
        %mul21 = arith.mulf %in21, %w21 : vector<4xf32>
        %acc21 = arith.addf %acc20, %mul21 : vector<4xf32>

        %in22 = vector.transfer_read %arg0[%r2, %c02], %cst : memref<8x8xf32>, vector<4xf32>
        %w22 = vector.transfer_read %arg1[%c2, %c2], %cst {permutation_map = #bcast} : memref<3x3xf32>, vector<4xf32>
        %mul22 = arith.mulf %in22, %w22 : vector<4xf32>
        %acc22 = arith.addf %acc21, %mul22 : vector<4xf32>

        vector.transfer_write %acc22, %arg2[%oh, %ow] : vector<4xf32>, memref<6x6xf32>
      }
    }
    return
  }
}
