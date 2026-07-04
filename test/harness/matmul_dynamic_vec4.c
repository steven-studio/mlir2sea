#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// 這是 mlir2sea 產生的 RVV kernel，簽名要跟 /tmp/mlir2sea_rvv.c 一致
extern void matmul(float* p0, float* p1, float* p2,
                    size_t dim_p0_0, size_t dim_p0_1,
                    size_t dim_p1_0, size_t dim_p1_1,
                    size_t dim_p2_0, size_t dim_p2_1);

// 手寫 scalar reference，loop 順序刻意跟 RVV kernel 對齊 (i, j, k)
// 這樣理論上連 floating-point 的運算順序都一致，方便抓出差異
void matmul_ref(float* A, float* B, float* C, int M, int K, int N) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            for (int k = 0; k < K; k++) {
                C[i*N + j] += A[i*K + k] * B[k*N + j];
            }
        }
    }
}

int main() {
    // 8x8x8，N 要是 4 的倍數才符合目前 kernel 的 vlen=4 假設
    int M = 8, K = 8, N = 8;

    float A[64], B[64], C_ref[64], C_rvv[64];

    srand(42);
    for (int i = 0; i < M*K; i++) A[i] = (float)(rand() % 100) / 10.0f;
    for (int i = 0; i < K*N; i++) B[i] = (float)(rand() % 100) / 10.0f;
    for (int i = 0; i < M*N; i++) { C_ref[i] = 0.0f; C_rvv[i] = 0.0f; }

    matmul_ref(A, B, C_ref, M, K, N);
    matmul(A, B, C_rvv, M, K, K, N, M, N);

    int mismatches = 0;
    for (int i = 0; i < M*N; i++) {
        float diff = fabsf(C_ref[i] - C_rvv[i]);
        if (diff > 1e-4f) {
            printf("MISMATCH at %d: ref=%f rvv=%f diff=%f\n", i, C_ref[i], C_rvv[i], diff);
            mismatches++;
        }
    }

    if (mismatches == 0) {
        printf("PASS: all %d elements match\n", M*N);
    } else {
        printf("FAIL: %d/%d mismatches\n", mismatches, M*N);
    }
    return mismatches != 0;
}
