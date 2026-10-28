// include/strata/core/split_handoff.hpp - a layer split's GPU<->GPU hand-off bytes, one counter for the whole engine.
//
// The per-card NVML PCIe counters cannot separate the split's hand-off from the host<->GPU traffic (they count
// both), so the engine counts it itself: the prompt path adds every `cudaMemcpyPeerAsync` payload it puts on the
// peer card (src/prefill/prefill.cpp), the decode path every window's residual rows a stage hands to the next one
// (src/core/verify.cpp).  The serve loop reports the running total as the trailing field of `DONE`/`BDONE`; the
// server turns it into per-request deltas, a total and a 1 Hz rate (GET /metrics).  Header-only and atomic so it
// can be tested on the CPU, like split_ring.hpp.
#pragma once

#include <atomic>
#include <cstdint>

namespace strata::core {

/// Bytes handed between the split's GPUs since the engine started.  Relaxed: a telemetry counter, not a barrier -
/// a slightly stale reading is a slightly small rate, never a wrong token.
inline std::atomic<uint64_t>& split_handoff_counter() {
    static std::atomic<uint64_t> c{0};
    return c;
}

inline void add_split_handoff(uint64_t bytes) {
    split_handoff_counter().fetch_add(bytes, std::memory_order_relaxed);
}

inline uint64_t split_handoff_bytes() {
    return split_handoff_counter().load(std::memory_order_relaxed);
}

}  // namespace strata::core
