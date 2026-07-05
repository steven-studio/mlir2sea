#include <stdio.h>
#include <stdlib.h>
#include <math.h>

extern void mm3(float* p0, float* p1, float* p2, float* p3, float* p4, float* p5, float* p6);

void matmul4x4(float* A, float* B, float* C) {
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++)
            for (int k = 0; k < 4; k++)
                C[i*4+j] += A[i*4+k] * B[k*4+j];
}

int main() {
    int N = 4;
    float A[16], B[16], C[16], D[16];
    float E_ref[16], F_ref[16], G_ref[16];
    float E_rvv[16], F_rvv[16], G_rvv[16];

    srand(1212);
    for (int i = 0; i < 16; i++) A[i] = (float)(rand() % 100) / 10.0f;
    for (int i = 0; i < 16; i++) B[i] = (float)(rand() % 100) / 10.0f;
    for (int i = 0; i < 16; i++) C[i] = (float)(rand() % 100) / 10.0f;
    for (int i = 0; i < 16; i++) D[i] = (float)(rand() % 100) / 10.0f;
    for (int i = 0; i < 16; i++) {
        E_ref[i] = 0.0f; F_ref[i] = 0.0f; G_ref[i] = 0.0f;
        E_rvv[i] = 0.0f; F_rvv[i] = 0.0f; G_rvv[i] = 0.0f;
    }

    matmul4x4(A, B, E_ref);
    matmul4x4(C, D, F_ref);
    matmul4x4(E_ref, F_ref, G_ref);

    mm3(A, B, C, D, E_rvv, F_rvv, G_rvv);

    int mismatches = 0;
    for (int i = 0; i < N*N; i++) {
        float diff = fabsf(G_ref[i] - G_rvv[i]);
        if (diff > 1e-1f) {
            printf("MISMATCH at %d: ref=%f rvv=%f diff=%f\n", i, G_ref[i], G_rvv[i], diff);
            mismatches++;
        }
    }
    if (mismatches == 0) {
        printf("PASS: all %d elements match (3mm chained matmul verified)\n", N*N);
    } else {
        printf("FAIL: %d mismatches\n", mismatches);
    }
    return mismatches != 0;
}
