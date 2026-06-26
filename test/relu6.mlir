// test/relu6.mlir
func.func @relu6(%x: f32) -> f32 {
  %zero = arith.constant 0.0 : f32
  %six  = arith.constant 6.0 : f32

  // max(x, 0)
  %cond0 = arith.cmpf ogt, %x, %zero : f32
  %r0 = scf.if %cond0 -> (f32) {
    scf.yield %x : f32
  } else {
    scf.yield %zero : f32
  }

  // min(r0, 6)
  %cond6 = arith.cmpf olt, %r0, %six : f32
  %r1 = scf.if %cond6 -> (f32) {
    scf.yield %r0 : f32
  } else {
    scf.yield %six : f32
  }

  func.return %r1 : f32
}
