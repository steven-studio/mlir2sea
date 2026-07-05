#include <stdio.h>
#include <stdlib.h>
#include <math.h>

extern void gesummv(float* p0, float* p1, float* p2, float* p3);

int main() {
    float A[16], B[16], x[4], y[4], y_ref[4];

    srand(1111);
    for (int i = 0; i < 16; i++) A[i] = (float)(rand() % 100) / 10.0f;
    for (int i = 0; i < 16; i++) B[i] = (float)(rand() % 100) / 10.0f;
    for (int i = 0; i < 4; i++) x[i] = (float)(rand() % 100) / 10.0f;

    for (int i = 0; i < 4; i++) {
        float sa = 0.0f, sb = 0.0f;
        for (int j = 0; j < 4; j++) {
            sa += A[i*4+j] * x[j];
            sb += B[i*4+j] * x[j];
        }
        y_ref[i] = sa + sb;
    }

    gesummv(A, B, x, y);

    int mismatches = 0;
    for (int i = 0; i < 4; i++) {
        if (fabsf(y[i] - y_ref[i]) > 1e-2f) {
            printf("MISMATCH at %d: ref=%f rvv=%f\n", i, y_ref[i], y[i]);
            mismatches++;
        }
    }
    if (mismatches == 0) {
        printf("PASS: all 4 elements match (gesummv verified)\n");
    } else {
        printf("FAIL: %d mismatches\n", mismatches);
    }
    return mismatches != 0;
}
