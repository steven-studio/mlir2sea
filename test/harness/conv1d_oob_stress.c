#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/mman.h>
#include <unistd.h>

extern void conv1d_oob_stress(float* p0, float* p1, float* p2);

int main() {
    long pagesize = sysconf(_SC_PAGESIZE);
    // 配置兩頁：第一頁尾端放輸入陣列，第二頁設成 PROT_NONE 當 guard page，
    // 任何讀到 in[8] 的操作都會立刻 SIGSEGV。
    void* region = mmap(NULL, pagesize * 2, PROT_READ | PROT_WRITE,
                         MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (region == MAP_FAILED) { perror("mmap"); return 2; }
    if (mprotect((char*)region + pagesize, pagesize, PROT_NONE) != 0) {
        perror("mprotect"); return 2;
    }

    int inLen = 8;
    float* in = (float*)((char*)region + pagesize - inLen * sizeof(float));
    // in[7] 剛好是 guard page 前最後 4 bytes；in[8] 會踩進 PROT_NONE 的那頁。

    float w[3], out[7];

    srand(42);
    for (int i = 0; i < inLen; i++) in[i] = (float)(rand() % 100) / 10.0f;
    for (int i = 0; i < 3; i++) w[i] = (float)(rand() % 100) / 10.0f;

    conv1d_oob_stress(in, w, out); // 如果有任何 op 讀了 in[8]，這裡會 SIGSEGV

    // 只有 output[0..5] 在數學上合法（inLen-kSize+1=6），
    // output[6] 需要 in[8]（不存在），故意不檢查它的值。
    float ref[6];
    for (int i = 0; i < 6; i++) {
        ref[i] = in[i]*w[0] + in[i+1]*w[1] + in[i+2]*w[2];
    }

    int mismatches = 0;
    for (int i = 0; i < 6; i++) {
        float diff = fabsf(ref[i] - out[i]);
        if (diff > 1e-3f) {
            printf("MISMATCH at %d: ref=%f rvv=%f diff=%f\n", i, ref[i], out[i], diff);
            mismatches++;
        }
    }
    printf(mismatches == 0 ? "PASS: valid outputs 0..5 match, no OOB read triggered\n"
                            : "FAIL: %d mismatches\n", mismatches);
    return mismatches != 0;
}
