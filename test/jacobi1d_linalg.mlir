func.func @jacobi_1d(%A: memref<8xf64>, %B: memref<8xf64>) {
  %subA0 = memref.subview %A[0][6][1] : memref<8xf64> to memref<6xf64, strided<[1], offset: 0>>
  %subA1 = memref.subview %A[1][6][1] : memref<8xf64> to memref<6xf64, strided<[1], offset: 1>>
  %subA2 = memref.subview %A[2][6][1] : memref<8xf64> to memref<6xf64, strided<[1], offset: 2>>
  %subB  = memref.subview %B[1][6][1] : memref<8xf64> to memref<6xf64, strided<[1], offset: 1>>
  %cst = arith.constant 3.333333e-01 : f64
  linalg.generic {
    indexing_maps = [
      affine_map<(i) -> (i)>,
      affine_map<(i) -> (i)>,
      affine_map<(i) -> (i)>,
      affine_map<(i) -> (i)>
    ],
    iterator_types = ["parallel"]
  } ins(%subA0, %subA1, %subA2 : memref<6xf64, strided<[1], offset: 0>>, memref<6xf64, strided<[1], offset: 1>>, memref<6xf64, strided<[1], offset: 2>>)
    outs(%subB : memref<6xf64, strided<[1], offset: 1>>) {
  ^bb0(%a0: f64, %a1: f64, %a2: f64, %b: f64):
    %sum = arith.addf %a0, %a1 : f64
    %sum2 = arith.addf %sum, %a2 : f64
    %res = arith.mulf %sum2, %cst : f64
    linalg.yield %res : f64
  }
  func.return
}
