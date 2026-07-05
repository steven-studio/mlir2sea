#include <stdio.h>
#include <stdlib.h>
#include <math.h>

extern void correlation(float* data, float* mean, float* stddev, float* corr);

int main() {
    int M = 4, N = 4;
    float data[16], data_ref[16];
    float mean[4], mean_ref[4];
    float stddev[4], stddev_ref[4];
    float corr[16], corr_ref[16];

    srand(2121);
    for (int i = 0; i < 16; i++) { data[i] = (float)(rand() % 100) / 10.0f + 1.0f; data_ref[i] = data[i]; }
    for (int i = 0; i < 4; i++) { mean[i] = 0.0f; mean_ref[i] = 0.0f; }
    for (int i = 0; i < 4; i++) { stddev[i] = 0.0f; stddev_ref[i] = 0.0f; }
    for (int i = 0; i < 16; i++) { corr[i] = 0.0f; corr_ref[i] = 0.0f; }

    // reference
    for (int j = 0; j < N; j++) {
        float s = 0.0f;
        for (int i = 0; i < M; i++) s += data_ref[i*N+j];
        mean_ref[j] = s / M;
    }
    for (int j = 0; j < N; j++) {
        float s = 0.0f;
        for (int i = 0; i < M; i++) {
            float d = data_ref[i*N+j] - mean_ref[j];
            s += d*d;
        }
        stddev_ref[j] = sqrtf(s / M);
    }
    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++)
            data_ref[i*N+j] = (data_ref[i*N+j] - mean_ref[j]) / stddev_ref[j];
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            float s = 0.0f;
            for (int k = 0; k < M; k++) s += data_ref[k*N+i] * data_ref[k*N+j];
            corr_ref[i*N+j] = s / (M-1);
        }
    }

    correlation(data, mean, stddev, corr);

    int mismatches = 0;
    for (int i = 0; i < 4; i++) {
        if (fabsf(mean[i] - mean_ref[i]) > 1e-2f) { printf("mean MISMATCH at %d: ref=%f rvv=%f\n", i, mean_ref[i], mean[i]); mismatches++; }
        if (fabsf(stddev[i] - stddev_ref[i]) > 1e-2f) { printf("stddev MISMATCH at %d: ref=%f rvv=%f\n", i, stddev_ref[i], stddev[i]); mismatches++; }
    }
    for (int i = 0; i < 16; i++) {
        if (fabsf(corr[i] - corr_ref[i]) > 1e-1f) { printf("corr MISMATCH at %d: ref=%f rvv=%f\n", i, corr_ref[i], corr[i]); mismatches++; }
    }
    if (mismatches == 0) {
        printf("PASS: mean, stddev, and correlation matrix all correct (correlation verified)\n");
    } else {
        printf("FAIL: %d mismatches\n", mismatches);
    }
    return mismatches != 0;
}
