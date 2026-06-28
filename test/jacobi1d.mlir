func.func @jacobi_1d(%A: memref<8xf64>, %B: memref<8xf64>) {
  %c1 = arith.constant 1 : index
  %c7 = arith.constant 7 : index
  %c1i = arith.constant 1 : index
  %cst = arith.constant 3.333333e-01 : f64
  scf.for %i = %c1 to %c7 step %c1i {
    %im1 = arith.subi %i, %c1 : index
    %ip1 = arith.addi %i, %c1 : index
    %a0 = memref.load %A[%im1] : memref<8xf64>
    %a1 = memref.load %A[%i]   : memref<8xf64>
    %a2 = memref.load %A[%ip1] : memref<8xf64>
    %sum = arith.addf %a0, %a1 : f64
    %sum2 = arith.addf %sum, %a2 : f64
    %res = arith.mulf %sum2, %cst : f64
    memref.store %res, %B[%i] : memref<8xf64>
  }
  func.return
}
