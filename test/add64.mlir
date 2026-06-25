// test/add64.mlir
func.func @add64(%a: i64, %b: i64) -> i64 {
  %r = arith.addi %a, %b : i64
  func.return %r : i64
}
