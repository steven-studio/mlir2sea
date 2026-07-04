#include <stdio.h>
#include <stdlib.h>
#include <math.h>

extern void conv2d(float* p0, float* p1, float* p2);

void conv2d_ref(float* in, float* w, float* out, int inH, int inW, int kH, int kW) {
    int outH = inH - kH + 1;
    int outW = inW - kW + 1;
    for (int oh = 0; oh < outH; oh++) {
        for (int ow = 0; ow < outW; ow++) {
            float acc = 0.0f;
            for (int kh = 0; kh < kH; kh++) {
                for (int kw = 0; kw < kW; kw++) {
                    acc += in[(oh+kh)*inW + (ow+kw)] * w[kh*kW + kw];
                }
            }
            out[oh*outW + ow] = acc;
        }
    }
}

int main() {
    int inH = 8, inW = 8, kH = 3, kW = 3, outH = 6, outW = 6;
    float in[64], w[9], out_ref[36], out_rvv[36];

    srand(101);
    for (int i = 0; i < 64; i++) in[i] = (float)(rand() % 100) / 10.0f;
    for (int i = 0; i < 9; i++) w[i] = (float)(rand() % 100) / 10.0f;

    conv2d_ref(in, w, out_ref, inH, inW, kH, kW);
    conv2d(in, w, out_rvv);

    int mismatches = 0;
    for (int i = 0; i < outH * outW; i++) {
        float diff = fabsf(out_ref[i] - out_rvv[i]);
        if (diff > 1e-2f) {
            printf("MISMATCH at (%d,%d): ref=%f rvv=%f diff=%f\n",
                i/outW, i%outW, out_ref[i], out_rvv[i], diff);
            mismatches++;
        }
    }
    printf(mismatches == 0 ? "PASS: all %d elements match\n" : "FAIL: %d mismatches\n",
        mismatches == 0 ? outH*outW : mismatches);
    return mismatches != 0;
}
