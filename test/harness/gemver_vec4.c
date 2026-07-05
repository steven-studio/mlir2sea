#include <stdio.h>
#include <stdlib.h>
#include <math.h>

extern void gemver(float* A, float* u1, float* v1, float* u2, float* v2,
                    float* y, float* z, float* x, float* w);

int main() {
    int N = 4;
    float A[16], A_ref[16];
    float u1[4], v1[4], u2[4], v2[4], y[4], z[4];
    float x[4], w[4], x_ref[4], w_ref[4];

    srand(1313);
    for (int i = 0; i < 16; i++) A[i] = (float)(rand() % 100) / 10.0f;
    for (int i = 0; i < 16; i++) A_ref[i] = A[i];
    for (int i = 0; i < 4; i++) u1[i] = (float)(rand() % 100) / 10.0f;
    for (int i = 0; i < 4; i++) v1[i] = (float)(rand() % 100) / 10.0f;
    for (int i = 0; i < 4; i++) u2[i] = (float)(rand() % 100) / 10.0f;
    for (int i = 0; i < 4; i++) v2[i] = (float)(rand() % 100) / 10.0f;
    for (int i = 0; i < 4; i++) y[i] = (float)(rand() % 100) / 10.0f;
    for (int i = 0; i < 4; i++) z[i] = (float)(rand() % 100) / 10.0f;
    for (int i = 0; i < 4; i++) x[i] = 0.0f;
    for (int i = 0; i < 4; i++) x_ref[i] = 0.0f;

    // reference
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            A_ref[i*N+j] += u1[i]*v1[j] + u2[i]*v2[j];
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            x_ref[j] += A_ref[i*N+j] * y[i];
    for (int i = 0; i < N; i++)
        x_ref[i] += z[i];
    for (int i = 0; i < N; i++) {
        float s = 0.0f;
        for (int j = 0; j < N; j++) s += A_ref[i*N+j] * x_ref[j];
        w_ref[i] = s;
    }

    gemver(A, u1, v1, u2, v2, y, z, x, w);

    int mismatches = 0;
    for (int i = 0; i < 16; i++) {
        if (fabsf(A[i] - A_ref[i]) > 1e-2f) {
            printf("A MISMATCH at %d: ref=%f rvv=%f\n", i, A_ref[i], A[i]);
            mismatches++;
        }
    }
    for (int i = 0; i < 4; i++) {
        if (fabsf(x[i] - x_ref[i]) > 1e-1f) {
            printf("x MISMATCH at %d: ref=%f rvv=%f\n", i, x_ref[i], x[i]);
            mismatches++;
        }
    }
    for (int i = 0; i < 4; i++) {
        if (fabsf(w[i] - w_ref[i]) > 1e-1f) {
            printf("w MISMATCH at %d: ref=%f rvv=%f\n", i, w_ref[i], w[i]);
            mismatches++;
        }
    }
    if (mismatches == 0) {
        printf("PASS: A, x, w all correct (gemver verified)\n");
    } else {
        printf("FAIL: %d mismatches\n", mismatches);
    }
    return mismatches != 0;
}
