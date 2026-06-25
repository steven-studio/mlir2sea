// test/add.mlir
func.func @add(%a: i32, %b: i32) -> i32 {
  %r = arith.addi %a, %b : i32
  func.return %r : i32
}
