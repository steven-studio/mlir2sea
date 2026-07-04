#include <stdio.h>
#include <stdlib.h>
#include <math.h>

extern void conv1d(float* p0, float* p1, float* p2);

void conv1d_ref(float* in, float* w, float* out, int inLen, int kSize) {
    int outLen = inLen - kSize + 1;
    for (int i = 0; i < outLen; i++) {
        float acc = 0.0f;
        for (int k = 0; k < kSize; k++) {
            acc += in[i + k] * w[k];
        }
        out[i] = acc;
    }
}

int main() {
    int inLen = 10, kSize = 3, outLen = 8;
    float in[10], w[3], out_ref[8], out_rvv[8];

    srand(77);
    for (int i = 0; i < inLen; i++) in[i] = (float)(rand() % 100) / 10.0f;
    for (int i = 0; i < kSize; i++) w[i] = (float)(rand() % 100) / 10.0f;

    conv1d_ref(in, w, out_ref, inLen, kSize);
    conv1d(in, w, out_rvv);

    int mismatches = 0;
    for (int i = 0; i < outLen; i++) {
        float diff = fabsf(out_ref[i] - out_rvv[i]);
        if (diff > 1e-3f) {
            printf("MISMATCH at %d: ref=%f rvv=%f diff=%f\n", i, out_ref[i], out_rvv[i], diff);
            mismatches++;
        }
    }
    printf(mismatches == 0 ? "PASS: all %d elements match\n" : "FAIL: %d mismatches\n",
        mismatches == 0 ? outLen : mismatches);
    return mismatches != 0;
}
