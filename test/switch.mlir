// test/switch.mlir
func.func @classify(%x: index) -> i32 {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %c99 = arith.constant 99 : i32
  %r = scf.index_switch %x -> i32
  case 0 {
    scf.yield %c0 : i32
  }
  case 1 {
    scf.yield %c1 : i32
  }
  default {
    scf.yield %c99 : i32
  }
  func.return %r : i32
}
