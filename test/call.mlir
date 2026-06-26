func.func @add(%a: i32, %b: i32) -> i32 {
  %r = arith.addi %a, %b : i32
  func.return %r : i32
}

func.func @main(%x: i32, %y: i32) -> i32 {
  %r = func.call @add(%x, %y) : (i32, i32) -> i32
  func.return %r : i32
}
