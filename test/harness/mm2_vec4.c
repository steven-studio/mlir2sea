#include <stdio.h>
#include <stdlib.h>
#include <math.h>

extern void mm2(float* p0, float* p1, float* p2, float* p3, float* p4);

void matmul4x4(float* A, float* B, float* C) {
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++)
            for (int k = 0; k < 4; k++)
                C[i*4+j] += A[i*4+k] * B[k*4+j];
}

int main() {
    int N = 4;
    float A[16], B[16], C[16], tmp_ref[16], D_ref[16], tmp_rvv[16], D_rvv[16];

    srand(707);
    for (int i = 0; i < 16; i++) A[i] = (float)(rand() % 100) / 10.0f;
    for (int i = 0; i < 16; i++) B[i] = (float)(rand() % 100) / 10.0f;
    for (int i = 0; i < 16; i++) C[i] = (float)(rand() % 100) / 10.0f;
    for (int i = 0; i < 16; i++) { tmp_ref[i] = 0.0f; D_ref[i] = 0.0f; tmp_rvv[i] = 0.0f; D_rvv[i] = 0.0f; }

    // reference: tmp = A*B, D = tmp*C
    matmul4x4(A, B, tmp_ref);
    matmul4x4(tmp_ref, C, D_ref);

    mm2(A, B, C, tmp_rvv, D_rvv);

    int mismatches = 0;
    for (int i = 0; i < N*N; i++) {
        float diff = fabsf(D_ref[i] - D_rvv[i]);
        if (diff > 1e-1f) { // 4x4x4 兩次連乘累積誤差稍大，門檻放寬
            printf("MISMATCH at %d: ref=%f rvv=%f diff=%f\n", i, D_ref[i], D_rvv[i], diff);
            mismatches++;
        }
    }
    if (mismatches == 0) {
        printf("PASS: all %d elements match (2mm chained matmul verified)\n", N*N);
    } else {
        printf("FAIL: %d mismatches\n", mismatches);
    }
    return mismatches != 0;
}
