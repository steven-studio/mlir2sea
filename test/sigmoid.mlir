// test/sigmoid.mlir
func.func @sigmoid(%x: f32) -> f32 {
  %one  = arith.constant 1.0 : f32
  %neg  = arith.negf %x : f32
  %e    = math.exp %neg : f32
  %denom = arith.addf %one, %e : f32
  %r    = arith.divf %one, %denom : f32
  func.return %r : f32
}
