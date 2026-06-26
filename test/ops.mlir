func.func @test(%a: i32, %b: i32) -> i32 {
  %r1 = arith.andi %a, %b : i32
  %r2 = arith.ori %r1, %b : i32
  %r3 = arith.shli %r2, %b : i32
  func.return %r3 : i32
}
