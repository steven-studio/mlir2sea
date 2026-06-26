// test/sqrt.mlir
func.func @mysqrt(%a: f64) -> f64 {
  %r = math.sqrt %a : f64
  func.return %r : f64
}
