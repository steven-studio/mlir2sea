func.func @matmul(%A: memref<8x8xf32>, %B: memref<8x8xf32>, %C: memref<8x8xf32>) {
  linalg.matmul ins(%A, %B : memref<8x8xf32>, memref<8x8xf32>)
               outs(%C : memref<8x8xf32>)
  func.return
}
