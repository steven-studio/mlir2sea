func.func @matmul(%A: memref<4x4xf32>, %B: memref<4x4xf32>, %C: memref<4x4xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index

  scf.for %i = %c0 to %c4 step %c1 {
    scf.for %j = %c0 to %c4 step %c1 {
      scf.for %k = %c0 to %c4 step %c1 {
        %a = memref.load %A[%i, %k] : memref<4x4xf32>
        %b = memref.load %B[%k, %j] : memref<4x4xf32>
        %c = memref.load %C[%i, %j] : memref<4x4xf32>
        %mul = arith.mulf %a, %b : f32
        %acc = arith.addf %c, %mul : f32
        memref.store %acc, %C[%i, %j] : memref<4x4xf32>
      }
    }
  }
  func.return
}
