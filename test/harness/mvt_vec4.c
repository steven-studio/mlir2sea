#include <stdio.h>
#include <stdlib.h>
#include <math.h>

extern void mvt(float* p0, float* p1, float* p2, float* p3, float* p4);

int main() {
    float A[16], y1[4], y2[4], x1[4], x2[4];
    float x1_ref[4], x2_ref[4];

    srand(808);
    for (int i = 0; i < 16; i++) A[i] = (float)(rand() % 100) / 10.0f;
    for (int i = 0; i < 4; i++) y1[i] = (float)(rand() % 100) / 10.0f;
    for (int i = 0; i < 4; i++) y2[i] = (float)(rand() % 100) / 10.0f;
    for (int i = 0; i < 4; i++) { x1[i] = (float)(rand()%50)/10.0f; x2[i] = (float)(rand()%50)/10.0f; }
    for (int i = 0; i < 4; i++) { x1_ref[i] = x1[i]; x2_ref[i] = x2[i]; }

    // reference
    for (int i = 0; i < 4; i++) {
        float s = 0.0f;
        for (int j = 0; j < 4; j++) s += A[i*4+j] * y1[j];
        x1_ref[i] += s;
    }
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) x2_ref[j] += A[i*4+j] * y2[i];
    }

    mvt(A, y1, y2, x1, x2);

    int mismatches = 0;
    for (int i = 0; i < 4; i++) {
        if (fabsf(x1[i] - x1_ref[i]) > 1e-2f) {
            printf("x1 MISMATCH at %d: ref=%f rvv=%f\n", i, x1_ref[i], x1[i]);
            mismatches++;
        }
    }
    for (int i = 0; i < 4; i++) {
        if (fabsf(x2[i] - x2_ref[i]) > 1e-2f) {
            printf("x2 MISMATCH at %d: ref=%f rvv=%f\n", i, x2_ref[i], x2[i]);
            mismatches++;
        }
    }
    if (mismatches == 0) {
        printf("PASS: x1 and x2 both correct (mvt verified)\n");
    } else {
        printf("FAIL: %d mismatches\n", mismatches);
    }
    return mismatches != 0;
}
