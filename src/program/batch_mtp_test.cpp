#include "strata/program/batch_mtp.hpp"

#include <cstdio>
#include <cstring>

int main() {
    using strata::program::batch_mtp_refusal;
    int checks = 0;
    bool ok = true;
    auto check = [&](int batch, bool mtp, int spec, bool same, bool serve, const char* expected) {
        ++checks;
        const char* got = batch_mtp_refusal(batch, mtp, spec, same, serve);
        const bool match = (got == nullptr) == (expected == nullptr) && (got == nullptr || std::strstr(got, expected) != nullptr);
        if (!match) std::printf("FAIL batch_mtp_refusal(%d,%d,%d,%d,%d) = %s, expected %s\n", batch, (int) mtp, spec, (int) same, (int) serve,
                                got ? got : "(runs)", expected ? expected : "(runs)");
        ok &= match;
    };
    check(2, true, 2, false, true, nullptr);          // one GPU, or a layer split with a GPU per stage: runs
    check(1, true, 2, false, true, "--batch 2");
    check(2, false, 2, false, true, "--mtp");
    check(2, true, 1, false, true, "--spec");
    check(2, true, 2, true, true, "own GPU");          // stages sharing a GPU: refused
    check(2, true, 2, false, false, "--serve");
    if (!ok) return 1;
    std::printf("batch_mtp_test OK (%d checks)\n", checks);
}
