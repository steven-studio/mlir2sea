#include <stdio.h>
#include <stdlib.h>
#include <math.h>

extern void conv1d_stride2(float* p0, float* p1, float* p2);

void conv1d_stride2_ref(float* in, float* w, float* out,
                         int inLen, int kSize, int cout, int stride) {
    int outLen = (inLen - kSize) / stride + 1;
    for (int op = 0; op < outLen; op++) {
        for (int c = 0; c < cout; c++) {
            float acc = 0.0f;
            for (int k = 0; k < kSize; k++) {
                acc += in[op * stride + k] * w[k * cout + c];
            }
            out[op * cout + c] = acc;
        }
    }
}

int main() {
    int inLen = 10, kSize = 3, cout = 4, stride = 2, outLen = 4;
    float in[10], w[12], out_ref[16], out_rvv[16];

    srand(202);
    for (int i = 0; i < inLen; i++) in[i] = (float)(rand() % 100) / 10.0f;
    for (int i = 0; i < kSize * cout; i++) w[i] = (float)(rand() % 100) / 10.0f;

    conv1d_stride2_ref(in, w, out_ref, inLen, kSize, cout, stride);
    conv1d_stride2(in, w, out_rvv);

    int mismatches = 0;
    for (int i = 0; i < outLen * cout; i++) {
        float diff = fabsf(out_ref[i] - out_rvv[i]);
        if (diff > 1e-3f) {
            printf("MISMATCH at %d (op=%d,c=%d): ref=%f rvv=%f diff=%f\n",
                i, i/cout, i%cout, out_ref[i], out_rvv[i], diff);
            mismatches++;
        }
    }
    printf(mismatches == 0 ? "PASS: all %d elements match\n" : "FAIL: %d mismatches\n",
        mismatches == 0 ? outLen*cout : mismatches);
    return mismatches != 0;
}
