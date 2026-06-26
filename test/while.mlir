// test/while.mlir
func.func @countdown(%n: i32) -> i32 {
  %c0 = arith.constant 0 : i32
  %result = scf.while (%i = %n) : (i32) -> i32 {
    %cond = arith.cmpi sgt, %i, %c0 : i32
    scf.condition(%cond) %i : i32
  } do {
  ^bb0(%i: i32):
    %c1 = arith.constant 1 : i32
    %next = arith.subi %i, %c1 : i32
    scf.yield %next : i32
  }
  func.return %result : i32
}
