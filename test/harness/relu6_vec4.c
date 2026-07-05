#include <stdio.h>
#include <math.h>

extern void relu6_vec4(float* p0, float* p1);

int main() {
    // 涵蓋三種 clamp 情況：負數、正常範圍、超過 6
    float in[8]  = {-3.0f, -0.5f, 0.0f, 2.5f, 4.0f, 6.0f, 7.5f, 100.0f};
    float out[8];

    relu6_vec4(in, out);

    float ref[8];
    for (int i = 0; i < 8; i++) {
        float v = in[i];
        if (v < 0.0f) v = 0.0f;
        if (v > 6.0f) v = 6.0f;
        ref[i] = v;
    }

    int mismatches = 0;
    for (int i = 0; i < 8; i++) {
        float diff = fabsf(ref[i] - out[i]);
        if (diff > 1e-4f) {
            printf("MISMATCH at %d: in=%f ref=%f rvv=%f\n", i, in[i], ref[i], out[i]);
            mismatches++;
        }
    }
    printf(mismatches == 0 ? "PASS: all 8 elements match (relu6 clamp verified)\n"
                            : "FAIL: %d mismatches\n", mismatches);
    return mismatches != 0;
}
