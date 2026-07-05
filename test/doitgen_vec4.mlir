module {
  func.func @doitgen(%A: memref<2x2x4xf32>, %C4: memref<4x4xf32>, %tmp: memref<2x2x4xf32>) {
    %cst = arith.constant 0.000000e+00 : f32
    %c0 = arith.constant 0 : index
    affine.for %r = 0 to 2 {
      affine.for %q = 0 to 2 {
        affine.for %p = 0 to 4 step 4 {
          affine.for %s = 0 to 4 {
            %a0 = vector.transfer_read %A[%r, %q, %s], %cst {permutation_map = affine_map<(d0,d1,d2)->(0)>} : memref<2x2x4xf32>, vector<4xf32>
            %c0v = vector.transfer_read %C4[%s, %p], %cst : memref<4x4xf32>, vector<4xf32>
            %old = vector.transfer_read %tmp[%r, %q, %p], %cst : memref<2x2x4xf32>, vector<4xf32>
            %m0 = arith.mulf %a0, %c0v : vector<4xf32>
            %s0 = arith.addf %old, %m0 : vector<4xf32>
            vector.transfer_write %s0, %tmp[%r, %q, %p] : vector<4xf32>, memref<2x2x4xf32>
          }
        }
      }
    }
    // copy tmp back into A
    affine.for %r = 0 to 2 {
      affine.for %q = 0 to 2 {
        affine.for %p = 0 to 4 step 4 {
          %v = vector.transfer_read %tmp[%r, %q, %p], %cst : memref<2x2x4xf32>, vector<4xf32>
          vector.transfer_write %v, %A[%r, %q, %p] : vector<4xf32>, memref<2x2x4xf32>
        }
      }
    }
    return
  }
}
