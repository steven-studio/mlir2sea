// test/loop.mlir
func.func @sum(%n: index) -> i32 {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %init = arith.constant 0 : i32
  %r = scf.for %i = %c0 to %n step %c1 iter_args(%acc = %init) -> (i32) {
    %i32 = arith.index_cast %i : index to i32
    %next = arith.addi %acc, %i32 : i32
    scf.yield %next : i32
  }
  func.return %r : i32
}
