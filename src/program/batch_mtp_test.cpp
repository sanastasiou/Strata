#include "strata/program/batch_mtp.hpp"

#include <cstdio>
#include <cstring>

int main() {
    using strata::program::batch_mtp_refusal;
    using strata::program::batch_window_rows;
    int checks = 0;
    bool ok = true;
    auto check = [&](int batch, bool mtp, int spec, bool same, bool serve, int groups, bool split, const char* expected) {
        ++checks;
        const char* got = batch_mtp_refusal(batch, mtp, spec, same, serve, groups, split);
        const bool match = (got == nullptr) == (expected == nullptr) && (got == nullptr || std::strstr(got, expected) != nullptr);
        if (!match) std::printf("FAIL batch_mtp_refusal(%d,%d,%d,%d,%d,%d,%d) = %s, expected %s\n", batch, (int) mtp, spec, (int) same,
                                (int) serve, groups, (int) split, got ? got : "(runs)", expected ? expected : "(runs)");
        ok &= match;
    };
    check(2, true, 2, false, true, 1, true, nullptr);   // one GPU, or a layer split with a GPU per stage: runs
    check(2, true, 2, false, true, 1, false, nullptr);
    check(2, true, 2, false, true, 2, false, nullptr);  // groups without a split are reset to 1 by the engine
    check(1, true, 2, false, true, 1, false, "--batch 2");
    check(2, false, 2, false, true, 1, false, "--mtp");
    check(2, true, 1, false, true, 1, false, "--spec");
    check(2, true, 2, true, true, 1, true, "own GPU");   // stages sharing a GPU: refused
    check(2, true, 2, false, false, 1, false, "--serve");
    check(12, true, 2, false, true, 2, true, "--batch-groups");   // pipelined groups on a split: refused
    check(4, true, 2, false, true, 4, true, "--batch-groups");
    auto rows = [&](bool mtp, int spec, int batch, int vmax, int expected) {
        ++checks;
        const int got = batch_window_rows(mtp, spec, batch, vmax);
        if (got != expected) { std::printf("FAIL batch_window_rows(%d,%d,%d,%d) = %d, expected %d\n", (int) mtp, spec, batch, vmax, got, expected); ok = false; }
    };
    rows(true, 2, 2, 8, 8);    // every stage holds the full window when slots verify drafts (2 rows per slot)
    rows(false, 2, 2, 8, 2);
    rows(false, 4, 2, 8, 4);
    rows(false, 2, 6, 8, 6);
    if (!ok) return 1;
    std::printf("batch_mtp_test OK (%d checks)\n", checks);
}
