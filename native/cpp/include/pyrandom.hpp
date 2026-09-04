// CPython-compatible Mersenne Twister.
//
// The environment seeds `random.Random((seed * 1_000_003) ^ day)` and calls
// .random() (weed spawns) and .choice() (shop unlocks). To reproduce episodes
// bit-for-bit we must match CPython's _randommodule.c exactly: init_by_array
// seeding from the integer's 32-bit little-endian words, the 53-bit
// random() construction, and getrandbits/_randbelow rejection sampling.
#pragma once
#include <array>
#include <cstdint>
#include <cstring>

namespace kag {

class PyRandom {
public:
    explicit PyRandom(uint64_t key, bool use_seed_cache = false) {
        if (use_seed_cache) seed_cached(key);
        else seed(key);
    }

    void seed(uint64_t key) {
        uint32_t words[2];
        words[0] = static_cast<uint32_t>(key & 0xffffffffu);
        if ((key >> 32) == 0) {
            init_by_array_fixed<1>(words);
        } else {
            words[1] = static_cast<uint32_t>(key >> 32);
            init_by_array_fixed<2>(words);
        }
    }

    uint32_t genrand_uint32() {
        uint32_t y;
        if (index_ >= N) {
            for (int k = 0; k < N - M; ++k) {
                y = (mt_[k] & UPPER) | (mt_[k + 1] & LOWER);
                mt_[k] = mt_[k + M] ^ (y >> 1) ^ ((y & 1u) ? MATRIX_A : 0u);
            }
            for (int k = N - M; k < N - 1; ++k) {
                y = (mt_[k] & UPPER) | (mt_[k + 1] & LOWER);
                mt_[k] = mt_[k + (M - N)] ^ (y >> 1) ^ ((y & 1u) ? MATRIX_A : 0u);
            }
            y = (mt_[N - 1] & UPPER) | (mt_[0] & LOWER);
            mt_[N - 1] = mt_[M - 1] ^ (y >> 1) ^ ((y & 1u) ? MATRIX_A : 0u);
            index_ = 0;
        }
        y = mt_[index_++];
        y ^= (y >> 11);
        y ^= (y << 7) & 0x9d2c5680u;
        y ^= (y << 15) & 0xefc60000u;
        y ^= (y >> 18);
        return y;
    }

    double random() {
        uint32_t a = genrand_uint32() >> 5;
        uint32_t b = genrand_uint32() >> 6;
        return (a * 67108864.0 + b) * (1.0 / 9007199254740992.0);
    }

    uint32_t getrandbits(int k) { return genrand_uint32() >> (32 - k); }

    uint32_t randbelow(uint32_t n) {
        int k = 0;
        for (uint32_t v = n; v; v >>= 1) ++k;
        uint32_t r = getrandbits(k);
        while (r >= n) r = getrandbits(k);
        return r;
    }

    uint32_t choice_index(uint32_t len) { return randbelow(len); }

private:
    static constexpr int N = 624;
    static constexpr int M = 397;
    static constexpr uint32_t MATRIX_A = 0x9908b0dfu;
    static constexpr uint32_t UPPER = 0x80000000u;
    static constexpr uint32_t LOWER = 0x7fffffffu;

    uint32_t mt_[N];
    int index_ = N;

    struct SeedCacheEntry {
        uint64_t key = 0;
        bool valid = false;
        std::array<uint32_t, N> mt{};
    };

    void seed_cached(uint64_t key) {
        // Daily keys differ in their low day bits, so a 64-entry direct map is
        // collision-free for the <=30 competition days of one snapshot seed.
        static thread_local std::array<SeedCacheEntry, 64> cache{};
        SeedCacheEntry& entry = cache[static_cast<size_t>(key & 63u)];
        if (entry.valid && entry.key == key) {
            std::memcpy(mt_, entry.mt.data(), sizeof(mt_));
            index_ = N;
            return;
        }
        seed(key);
        entry.key = key;
        entry.valid = true;
        std::memcpy(entry.mt.data(), mt_, sizeof(mt_));
    }

    static const std::array<uint32_t, N>& base_seed_state() {
        static const std::array<uint32_t, N> base = [] {
            std::array<uint32_t, N> out{};
            out[0] = 19650218u;
            for (int i = 1; i < N; ++i)
                out[i] = 1812433253u * (out[i - 1] ^ (out[i - 1] >> 30)) + static_cast<uint32_t>(i);
            return out;
        }();
        return base;
    }

    void init_genrand(uint32_t s) {
        mt_[0] = s;
        for (int i = 1; i < N; ++i)
            mt_[i] = 1812433253u * (mt_[i - 1] ^ (mt_[i - 1] >> 30)) + static_cast<uint32_t>(i);
        index_ = N;
    }

    template <int KeyLen>
    void init_by_array_fixed(const uint32_t* key) {
        static_assert(KeyLen == 1 || KeyLen == 2, "uint64 RNG keys use one or two 32-bit words");
        const auto& base = base_seed_state();
        std::memcpy(mt_, base.data(), sizeof(mt_));
        index_ = N;
        int i = 1, j = 0;
        for (int k = N; k; --k) {
            mt_[i] = (mt_[i] ^ ((mt_[i - 1] ^ (mt_[i - 1] >> 30)) * 1664525u)) +
                     key[j] + static_cast<uint32_t>(j);
            ++i;
            if (i >= N) { mt_[0] = mt_[N - 1]; i = 1; }
            if constexpr (KeyLen == 2) j ^= 1;
        }
        for (int k = N - 1; k; --k) {
            mt_[i] = (mt_[i] ^ ((mt_[i - 1] ^ (mt_[i - 1] >> 30)) * 1566083941u)) - static_cast<uint32_t>(i);
            ++i;
            if (i >= N) { mt_[0] = mt_[N - 1]; i = 1; }
        }
        mt_[0] = 0x80000000u;
        index_ = N;
    }
};

}  // namespace kag
