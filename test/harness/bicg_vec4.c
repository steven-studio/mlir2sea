#include <stdio.h>
#include <stdlib.h>
#include <math.h>

extern void bicg(float* p0, float* p1, float* p2, float* p3, float* p4);

int main() {
    float A[16], p[4], r[4], s[4], q[4];
    float s_ref[4], q_ref[4];

    srand(1010);
    for (int i = 0; i < 16; i++) A[i] = (float)(rand() % 100) / 10.0f;
    for (int i = 0; i < 4; i++) p[i] = (float)(rand() % 100) / 10.0f;
    for (int i = 0; i < 4; i++) r[i] = (float)(rand() % 100) / 10.0f;
    for (int i = 0; i < 4; i++) s[i] = (float)(rand()%50)/10.0f;
    for (int i = 0; i < 4; i++) s_ref[i] = s[i];

    // reference: s = A^T*p, q = A*r
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) s_ref[j] += A[i*4+j] * p[i];
    }
    for (int i = 0; i < 4; i++) {
        float sum = 0.0f;
        for (int j = 0; j < 4; j++) sum += A[i*4+j] * r[j];
        q_ref[i] = sum;
    }

    bicg(A, p, r, s, q);

    int mismatches = 0;
    for (int i = 0; i < 4; i++) {
        if (fabsf(s[i] - s_ref[i]) > 1e-2f) {
            printf("s MISMATCH at %d: ref=%f rvv=%f\n", i, s_ref[i], s[i]);
            mismatches++;
        }
    }
    for (int i = 0; i < 4; i++) {
        if (fabsf(q[i] - q_ref[i]) > 1e-2f) {
            printf("q MISMATCH at %d: ref=%f rvv=%f\n", i, q_ref[i], q[i]);
            mismatches++;
        }
    }
    if (mismatches == 0) {
        printf("PASS: s and q both correct (bicg verified)\n");
    } else {
        printf("FAIL: %d mismatches\n", mismatches);
    }
    return mismatches != 0;
}
