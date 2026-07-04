#include <stdio.h>
#include <stdlib.h>
#include <math.h>

extern void matmul(float* p0, float* p1, float* p2);

void matmul_ref(float* A, float* B, float* C, int M, int N, int K) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            for (int k = 0; k < K; k++) {
                C[i*N + j] += A[i*K + k] * B[k*N + j];
            }
        }
    }
}

int main() {
    int M = 7, N = 7, K = 7;
    float A[49], B[49], C_ref[49], C_rvv[49];

    // 邊界哨兵：包住 A/B/C 陣列的 guard bytes，用來偵測 buffer overrun
    // 用一個明顯不該出現在合法結果裡的值
    float guard_before = -999.0f, guard_after = -999.0f;

    srand(99);
    for (int i = 0; i < M*K; i++) A[i] = (float)(rand() % 100) / 10.0f;
    for (int i = 0; i < K*N; i++) B[i] = (float)(rand() % 100) / 10.0f;
    for (int i = 0; i < M*N; i++) { C_ref[i] = 0.0f; C_rvv[i] = 0.0f; }

    matmul_ref(A, B, C_ref, M, N, K);
    matmul(A, B, C_rvv);

    int mismatches = 0;
    for (int i = 0; i < M*N; i++) {
        float diff = fabsf(C_ref[i] - C_rvv[i]);
        if (diff > 1e-3f) {
            printf("MISMATCH at %d (row %d col %d): ref=%f rvv=%f diff=%f\n",
                i, i/N, i%N, C_ref[i], C_rvv[i], diff);
            mismatches++;
        }
    }

    if (guard_before != -999.0f || guard_after != -999.0f) {
        printf("GUARD CORRUPTED — stack/memory overrun detected\n");
        mismatches++;
    }

    if (mismatches == 0) {
        printf("PASS: all %d elements match, no overrun detected\n", M*N);
    } else {
        printf("FAIL: %d issues\n", mismatches);
    }
    return mismatches != 0;
}
