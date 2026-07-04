#include <stdio.h>
#include <stdlib.h>
#include <math.h>

extern void gemm(float* p0, float* p1, float* p2);

// scalar reference，loop 順序跟 RVV kernel 對齊 (i, j, k)
void gemm_ref(float* A, float* B, float* C, int M, int N, int K) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            for (int k = 0; k < K; k++) {
                C[i*N + j] += A[i*K + k] * B[k*N + j];
            }
        }
    }
}

int main() {
    int M = 8, N = 8, K = 8;
    float A[64], B[64], C_ref[64], C_rvv[64];

    srand(123);
    for (int i = 0; i < M*K; i++) A[i] = (float)(rand() % 100) / 10.0f;
    for (int i = 0; i < K*N; i++) B[i] = (float)(rand() % 100) / 10.0f;
    for (int i = 0; i < M*N; i++) { C_ref[i] = 0.0f; C_rvv[i] = 0.0f; }

    gemm_ref(A, B, C_ref, M, N, K);
    gemm(A, B, C_rvv);

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
