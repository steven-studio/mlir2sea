#include <stdio.h>
#include <stdlib.h>
#include <math.h>

extern void covariance(float* data, float* mean, float* cov);

int main() {
    int M = 4, N = 4;
    float data[16], data_ref[16];
    float mean[4], mean_ref[4];
    float cov[16], cov_ref[16];

    srand(2020);
    for (int i = 0; i < 16; i++) { data[i] = (float)(rand() % 100) / 10.0f; data_ref[i] = data[i]; }
    for (int i = 0; i < 4; i++) { mean[i] = 0.0f; mean_ref[i] = 0.0f; }
    for (int i = 0; i < 16; i++) { cov[i] = 0.0f; cov_ref[i] = 0.0f; }

    // reference
    for (int j = 0; j < N; j++) {
        float s = 0.0f;
        for (int i = 0; i < M; i++) s += data_ref[i*N+j];
        mean_ref[j] = s / M;
    }
    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++)
            data_ref[i*N+j] -= mean_ref[j];
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            float s = 0.0f;
            for (int k = 0; k < M; k++) s += data_ref[k*N+i] * data_ref[k*N+j];
            cov_ref[i*N+j] = s / (M-1);
        }
    }

    covariance(data, mean, cov);

    int mismatches = 0;
    for (int i = 0; i < 4; i++) {
        if (fabsf(mean[i] - mean_ref[i]) > 1e-2f) { printf("mean MISMATCH at %d: ref=%f rvv=%f\n", i, mean_ref[i], mean[i]); mismatches++; }
    }
    for (int i = 0; i < 16; i++) {
        if (fabsf(cov[i] - cov_ref[i]) > 1e-1f) { printf("cov MISMATCH at %d: ref=%f rvv=%f\n", i, cov_ref[i], cov[i]); mismatches++; }
    }
    if (mismatches == 0) {
        printf("PASS: mean and covariance matrix both correct (covariance verified)\n");
    } else {
        printf("FAIL: %d mismatches\n", mismatches);
    }
    return mismatches != 0;
}
