#pragma once
#include <cstdint>
#include <cstring>

struct KagLogicalHash {
    uint64_t h = 1469598103934665603ull;
    void add_u64(uint64_t v) {
        // Mix values, not object bytes, so padding and ABI details never enter parity.
        h ^= v;
        h *= 1099511628211ull;
        h ^= h >> 32;
        h *= 0xd6e8feb86659fd93ull;
    }
    template <class T> void add_int(T v) { add_u64(static_cast<uint64_t>(static_cast<int64_t>(v))); }
    void add_bool(bool v) { add_u64(v ? 1u : 0u); }
    void add_double(double v) {
        uint64_t bits = 0;
        static_assert(sizeof(bits) == sizeof(v), "double must be 64-bit");
        std::memcpy(&bits, &v, sizeof(bits));
        add_u64(bits);
    }
};
