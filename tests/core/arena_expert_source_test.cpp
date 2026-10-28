// Telemetry G4: the serve INFO line's `arena_shared` flag comes from the arena the engine ACTUALLY attached.
// A fresh ArenaExpertSource says 0; one opened with a shared MAP_SHARED file says 1 - and so does the second
// engine that attaches the same file (the split's whole point); closing or re-opening without the shared file
// says 0 again.  No GPU needed: the CUDA registration simply fails and the arena stays locked-resident, as it
// does on any CPU-only run.
#include "strata/core/expert_source.hpp"
#include "strata/kernels/cpu/expert.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <system_error>

namespace fs = std::filesystem;

namespace {

int checks = 0;

void require(bool ok, const std::string& message) {
    if (!ok) throw std::runtime_error(message);
    ++checks;
}

struct TempDirectory {
    fs::path path;

    explicit TempDirectory(const fs::path& base = fs::temp_directory_path()) {
        const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
        path = base / ("strata-arena-shared-test-" + std::to_string(stamp));
        fs::create_directories(path);
    }

    ~TempDirectory() {
        std::error_code ignored;
        fs::remove_all(path, ignored);
    }
};

}  // namespace

int main() {
    using namespace strata::core;
    using namespace strata::kernels::cpu;
    constexpr int64_t layers = 2;
    constexpr int64_t experts = 3;
    const uint64_t total = (uint64_t) layers * (uint64_t) experts * BLOB;

    try {
        TempDirectory dir;
        std::string err;
        require(expert_layout_load(dir.path.string(), layers, experts, err), "could not load canonical layout: " + err);
        {
            std::ofstream out(dir.path / "experts.bin", std::ios::binary | std::ios::trunc);
            require((bool) out, "could not create synthetic experts.bin");
            out.seekp((std::streamoff) (total - 1));
            out.put('\0');
            require((bool) out, "could not size synthetic experts.bin");
        }

        // 1. the ordinary arena (anonymous, or hugetlb where the pool allows it): arena_shared=0
        ArenaExpertSource plain;
        require(plain.open(dir.path.string(), layers, experts, 1, err), "plain arena open failed: " + err);
        require(plain.mapped() && plain.arena_shared() == 0, "a plain arena reported itself shared");

        // 2. the shared arena, first engine: the MAP_SHARED file is created, written and attached - arena_shared=1
        const std::string shared_file = (dir.path / "shared-arena.bin").string();
        ArenaExpertSource first;
        err.clear();
        require(first.open(dir.path.string(), layers, experts, 1, err, 0, shared_file),
                "shared arena open failed: " + err);
        require(first.mapped() && first.arena_shared() == 1, "the shared arena did not report itself shared");
        require(fs::exists(shared_file), "the shared arena file was not created");

        // 3. the second engine of the split attaches the SAME file (its header names the pack): also 1
        ArenaExpertSource second;
        err.clear();
        require(second.open(dir.path.string(), layers, experts, 1, err, 0, shared_file),
                "shared arena reopen failed: " + err);
        require(second.mapped() && second.arena_shared() == 1, "the second shared-arena engine did not report itself shared");

        // 4. the flag follows the attached arena, not the process history: closed -> 0, a plain re-open -> 0
        second.close();
        require(second.arena_shared() == 0, "a closed arena still reported itself shared");
        err.clear();
        require(second.open(dir.path.string(), layers, experts, 1, err), "arena re-open without the shared file failed: " + err);
        require(second.arena_shared() == 0, "an arena re-opened without the shared file reported itself shared");

        std::printf("arena_expert_source_test OK (%d checks)\n", checks);
        return 0;
    } catch (const std::exception& e) {
        std::printf("arena_expert_source_test FAIL: %s\n", e.what());
        return 1;
    }
}
