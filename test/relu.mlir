// test/relu.mlir
func.func @relu(%x: f32) -> f32 {
  %zero = arith.constant 0.0 : f32
  %cond = arith.cmpf ogt, %x, %zero : f32
  %r = scf.if %cond -> (f32) {
    scf.yield %x : f32
  } else {
    scf.yield %zero : f32
  }
  func.return %r : f32
}
