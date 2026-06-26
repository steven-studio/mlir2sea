// test/abs.mlir
func.func @myabs(%x: f32) -> f32 {
  %r = math.absf %x : f32
  func.return %r : f32
}
