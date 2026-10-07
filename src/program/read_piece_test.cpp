#include "strata/program/read_piece.hpp"

#include <cstdio>

int main() {
    using strata::program::prompt_read_piece;
    int checks = 0;
    auto check = [&](int64_t chunk, bool split_idle, int64_t piece, int64_t expected) {
        ++checks;
        const int64_t got = prompt_read_piece(chunk, split_idle, piece);
        if (got != expected)
            std::printf("FAIL prompt_read_piece(%lld, %d, %lld) = %lld, expected %lld\n", (long long) chunk,
                        (int) split_idle, (long long) piece, (long long) got, (long long) expected);
        return got == expected;
    };
    bool ok = true;
    // beside decoding slots (or one GPU): one prompt chunk at a time, whatever the piece
    ok &= check(8192, false, 0, 8192);
    ok &= check(8192, false, 65536, 8192);
    ok &= check(0, false, 65536, 1);          // a chunk of 0 still advances
    // a layer split with no slot decoding: one run over the rest unless a piece is set
    ok &= check(8192, true, 0, 0);
    ok &= check(8192, true, -5, 0);
    // a piece is a whole number of chunks (the runs keep the same chunk grid), at least one
    ok &= check(8192, true, 65536, 65536);
    ok &= check(8192, true, 70000, 65536);
    ok &= check(8192, true, 100, 8192);
    ok &= check(32768, true, 65536, 65536);
    ok &= check(0, true, 3, 3);
    if (!ok) return 1;
    std::printf("read_piece_test OK (%d checks)\n", checks);
}
