func.func @gemm(%A: memref<8x8xf32>, %B: memref<8x8xf32>, %C: memref<8x8xf32>) {
  linalg.generic {
    indexing_maps = [
      affine_map<(i,j,k) -> (i,k)>,
      affine_map<(i,j,k) -> (k,j)>,
      affine_map<(i,j,k) -> (i,j)>
    ],
    iterator_types = ["parallel", "parallel", "reduction"]
  } ins(%A, %B : memref<8x8xf32>, memref<8x8xf32>)
    outs(%C : memref<8x8xf32>) {
  ^bb0(%a: f32, %b: f32, %c: f32):
    %mul = arith.mulf %a, %b : f32
    %acc = arith.addf %c, %mul : f32
    linalg.yield %acc : f32
  }
  func.return
}
