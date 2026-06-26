// test/matmul_simple.mlir
func.func @dot(%n: index) -> f32 {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %init = arith.constant 0.0 : f32

  %r = scf.for %i = %c0 to %n step %c1 iter_args(%acc = %init) -> (f32) {
    %i32 = arith.index_cast %i : index to i32
    %fi = arith.sitofp %i32 : i32 to f32
    %mul = arith.mulf %fi, %fi : f32
    %next = arith.addf %acc, %mul : f32
    scf.yield %next : f32
  }
  func.return %r : f32
}
