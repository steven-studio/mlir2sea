#include <stdio.h>
#include <stdlib.h>
#include <math.h>

extern void fdtd2d(float* ey, float* ex, float* hz);

int main() {
    int N = 8;
    float ey[64], ex[64], hz[64];
    float ey_ref[64], ex_ref[64], hz_ref[64];

    srand(1818);
    for (int i = 0; i < 64; i++) {
        ey[i] = (float)(rand() % 100) / 10.0f; ey_ref[i] = ey[i];
        ex[i] = (float)(rand() % 100) / 10.0f; ex_ref[i] = ex[i];
        hz[i] = (float)(rand() % 100) / 10.0f; hz_ref[i] = hz[i];
    }

    // reference, same three-stage order
    for (int i = 1; i < 7; i++)
        for (int j = 0; j < 4; j++)
            ey_ref[i*N+j] -= 0.5f * (hz_ref[i*N+j] - hz_ref[(i-1)*N+j]);
    for (int i = 0; i < 8; i++)
        for (int j = 1; j < 5; j++)
            ex_ref[i*N+j] -= 0.5f * (hz_ref[i*N+j] - hz_ref[i*N+j-1]);
    for (int i = 0; i < 6; i++)
        for (int j = 0; j < 4; j++)
            hz_ref[i*N+j] -= 0.7f * ((ex_ref[i*N+j+1]-ex_ref[i*N+j]) + (ey_ref[(i+1)*N+j]-ey_ref[i*N+j]));

    fdtd2d(ey, ex, hz);

    int mismatches = 0;
    for (int i = 0; i < 64; i++) {
        if (fabsf(ey[i]-ey_ref[i]) > 1e-2f) { printf("ey MISMATCH at %d\n", i); mismatches++; }
        if (fabsf(ex[i]-ex_ref[i]) > 1e-2f) { printf("ex MISMATCH at %d\n", i); mismatches++; }
        if (fabsf(hz[i]-hz_ref[i]) > 1e-2f) { printf("hz MISMATCH at %d\n", i); mismatches++; }
    }
    if (mismatches == 0) {
        printf("PASS: ey, ex, hz all correct (fdtd2d verified)\n");
    } else {
        printf("FAIL: %d mismatches\n", mismatches);
    }
    return mismatches != 0;
}
