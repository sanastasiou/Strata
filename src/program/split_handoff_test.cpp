#include "strata/core/split_handoff.hpp"

#include <cstdint>
#include <cstdio>
#include <thread>
#include <vector>

using namespace strata::core;

#define CHECK(cond)                                                                       \
    do {                                                                                  \
        if (!(cond)) {                                                                    \
            std::printf("split_handoff_test FAIL line %d: %s\n", __LINE__, #cond);        \
            return 1;                                                                     \
        }                                                                                 \
        ++checks;                                                                         \
    } while (0)

int main() {
    int checks = 0;

    // 1. a fresh engine starts at zero and counts the payloads it is given (the sizes the prefill peer copies and
    //    the verify stage hand-offs pass: rows x n_embd x 4, and T x handoff_floats x 4)
    CHECK(split_handoff_bytes() == 0);
    add_split_handoff((uint64_t) 512 * 4096 * 4);          // a prompt chunk's peer rows: T x N x 4
    CHECK(split_handoff_bytes() == 512 * 4096 * 4);
    add_split_handoff((uint64_t) 8 * 16416 * 4);           // a decode window's hand-off: T x handoff_floats x 4
    CHECK(split_handoff_bytes() == (uint64_t) (512 * 4096 + 8 * 16416) * 4);
    add_split_handoff(0);                                  // an empty group adds nothing
    CHECK(split_handoff_bytes() == (uint64_t) (512 * 4096 + 8 * 16416) * 4);

    // 2. the counter is the DONE/BDONE field the server differences: it only ever grows within one engine, and
    //    concurrent stages (several batch slots, the prompt path beside the decode path) never lose an add
    const uint64_t before = split_handoff_bytes();
    {
        std::vector<std::thread> th;
        for (int i = 0; i < 8; ++i)
            th.emplace_back([i] { for (int k = 0; k < 1000; ++k) add_split_handoff((uint64_t) (i + 1) * 1024); });
        for (auto& t : th) t.join();
    }
    uint64_t want = before;
    for (int i = 0; i < 8; ++i) want += 1000ull * (uint64_t) (i + 1) * 1024ull;
    CHECK(split_handoff_bytes() == want);

    // 3. the server's rule for a restart (the process starts over, the counter with it): the difference since the
    //    last line, or the whole new counter when it went backwards - never a negative delta
    auto delta = [](uint64_t prev, uint64_t cum) { return cum >= prev ? cum - prev : cum; };
    CHECK(delta(before, want) == want - before);
    CHECK(delta(want, 0) == 0);                            // a restart before any hand-off: nothing to add
    CHECK(delta(want, 4096) == 4096);                      // a restart mid-conversation: the new counter whole

    std::printf("split_handoff_test OK (%d checks)\n", checks);
}
