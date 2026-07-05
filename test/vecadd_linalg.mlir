func.func @vecadd_linalg(%A: memref<8xf32>, %B: memref<8xf32>, %C: memref<8xf32>) {
  linalg.generic {
    indexing_maps = [
      affine_map<(i) -> (i)>,
      affine_map<(i) -> (i)>,
      affine_map<(i) -> (i)>
    ],
    iterator_types = ["parallel"]
  } ins(%A, %B : memref<8xf32>, memref<8xf32>)
    outs(%C : memref<8xf32>) {
  ^bb0(%a: f32, %b: f32, %c: f32):
    %sum = arith.addf %a, %b : f32
    linalg.yield %sum : f32
  }
  func.return
}
