#include <stdio.h>
#include <stdlib.h>
#include <math.h>

extern void atax(float* p0, float* p1, float* p2, float* p3);

int main() {
    float A[16], x[4], tmp[4], y[4];
    float tmp_ref[4], y_ref[4];

    srand(909);
    for (int i = 0; i < 16; i++) A[i] = (float)(rand() % 100) / 10.0f;
    for (int i = 0; i < 4; i++) x[i] = (float)(rand() % 100) / 10.0f;
    for (int i = 0; i < 4; i++) y[i] = (float)(rand()%50)/10.0f;
    for (int i = 0; i < 4; i++) y_ref[i] = y[i];

    // reference: tmp = A*x, y = A^T*tmp
    for (int i = 0; i < 4; i++) {
        float s = 0.0f;
        for (int j = 0; j < 4; j++) s += A[i*4+j] * x[j];
        tmp_ref[i] = s;
    }
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) y_ref[j] += A[i*4+j] * tmp_ref[i];
    }

    atax(A, x, tmp, y);

    int mismatches = 0;
    for (int i = 0; i < 4; i++) {
        if (fabsf(tmp[i] - tmp_ref[i]) > 1e-2f) {
            printf("tmp MISMATCH at %d: ref=%f rvv=%f\n", i, tmp_ref[i], tmp[i]);
            mismatches++;
        }
    }
    for (int i = 0; i < 4; i++) {
        if (fabsf(y[i] - y_ref[i]) > 1e-1f) { // 累積誤差稍大
            printf("y MISMATCH at %d: ref=%f rvv=%f\n", i, y_ref[i], y[i]);
            mismatches++;
        }
    }
    if (mismatches == 0) {
        printf("PASS: tmp and y both correct (atax verified)\n");
    } else {
        printf("FAIL: %d mismatches\n", mismatches);
    }
    return mismatches != 0;
}
