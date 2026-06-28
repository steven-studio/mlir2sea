func.func @matmul(%A: memref<7x7xf32>, %B: memref<7x7xf32>, %C: memref<7x7xf32>) {
  linalg.matmul ins(%A, %B : memref<7x7xf32>, memref<7x7xf32>)
               outs(%C : memref<7x7xf32>)
  func.return
}
