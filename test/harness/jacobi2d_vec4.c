#include <stdio.h>
#include <stdlib.h>
#include <math.h>

extern void jacobi2d(float* p0, float* p1);

int main() {
    int N = 8;
    float A[64], B[64], B_ref[64];

    srand(1515);
    for (int i = 0; i < 64; i++) A[i] = (float)(rand() % 100) / 10.0f;
    for (int i = 0; i < 64; i++) { B[i] = -1.0f; B_ref[i] = -1.0f; }

    for (int i = 1; i < 7; i++) {
        for (int j = 1; j < 5; j++) {
            B_ref[i*N+j] = 0.2f * (A[i*N+j] + A[i*N+j-1] + A[i*N+j+1] + A[(i-1)*N+j] + A[(i+1)*N+j]);
        }
    }

    jacobi2d(A, B);

    int mismatches = 0;
    for (int i = 1; i < 7; i++) {
        for (int j = 1; j < 5; j++) {
            float diff = fabsf(B[i*N+j] - B_ref[i*N+j]);
            if (diff > 1e-3f) {
                printf("MISMATCH at (%d,%d): ref=%f rvv=%f\n", i, j, B_ref[i*N+j], B[i*N+j]);
                mismatches++;
            }
        }
    }
    if (mismatches == 0) {
        printf("PASS: all interior points match (jacobi2d verified)\n");
    } else {
        printf("FAIL: %d mismatches\n", mismatches);
    }
    return mismatches != 0;
}
