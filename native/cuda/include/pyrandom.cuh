#pragma once
#include <cuda_runtime.h>
#include <stdint.h>

namespace kagcuda {

// CPython-compatible MT19937 for device code. The caller supplies the invariant
// 624-word init_genrand(19650218) state from DeviceTables so every daily seed
// avoids recomputing that base state.
struct PyRandom {
    static constexpr int N = 624;
    static constexpr int M = 397;
    static constexpr uint32_t MATRIX_A = 0x9908b0dfu;
    static constexpr uint32_t UPPER = 0x80000000u;
    static constexpr uint32_t LOWER = 0x7fffffffu;

    uint32_t mt[N];
    int index;

    __device__ __forceinline__ void seed(uint64_t key, const uint32_t* base) {
        #pragma unroll 4
        for (int k = 0; k < N; ++k) mt[k] = base[k];
        index = N;

        uint32_t words[2];
        words[0] = static_cast<uint32_t>(key & 0xffffffffu);
        words[1] = static_cast<uint32_t>(key >> 32);
        const int key_len = words[1] == 0 ? 1 : 2;

        int i = 1, j = 0;
        for (int k = N; k; --k) {
            mt[i] = (mt[i] ^ ((mt[i - 1] ^ (mt[i - 1] >> 30)) * 1664525u)) +
                    words[j] + static_cast<uint32_t>(j);
            ++i;
            if (i >= N) { mt[0] = mt[N - 1]; i = 1; }
            if (key_len == 2) j ^= 1;
        }
        for (int k = N - 1; k; --k) {
            mt[i] = (mt[i] ^ ((mt[i - 1] ^ (mt[i - 1] >> 30)) * 1566083941u)) -
                    static_cast<uint32_t>(i);
            ++i;
            if (i >= N) { mt[0] = mt[N - 1]; i = 1; }
        }
        mt[0] = 0x80000000u;
        index = N;
    }

    __device__ __forceinline__ uint32_t genrand_uint32() {
        uint32_t y;
        if (index >= N) {
            for (int k = 0; k < N - M; ++k) {
                y = (mt[k] & UPPER) | (mt[k + 1] & LOWER);
                mt[k] = mt[k + M] ^ (y >> 1) ^ ((y & 1u) ? MATRIX_A : 0u);
            }
            for (int k = N - M; k < N - 1; ++k) {
                y = (mt[k] & UPPER) | (mt[k + 1] & LOWER);
                mt[k] = mt[k + (M - N)] ^ (y >> 1) ^ ((y & 1u) ? MATRIX_A : 0u);
            }
            y = (mt[N - 1] & UPPER) | (mt[0] & LOWER);
            mt[N - 1] = mt[M - 1] ^ (y >> 1) ^ ((y & 1u) ? MATRIX_A : 0u);
            index = 0;
        }
        y = mt[index++];
        y ^= y >> 11;
        y ^= (y << 7) & 0x9d2c5680u;
        y ^= (y << 15) & 0xefc60000u;
        y ^= y >> 18;
        return y;
    }

    __device__ __forceinline__ double random() {
        const uint32_t a = genrand_uint32() >> 5;
        const uint32_t b = genrand_uint32() >> 6;
        return (static_cast<double>(a) * 67108864.0 + static_cast<double>(b)) *
               (1.0 / 9007199254740992.0);
    }

    __device__ __forceinline__ uint32_t getrandbits(int k) {
        return genrand_uint32() >> (32 - k);
    }

    __device__ __forceinline__ uint32_t randbelow(uint32_t n) {
        int k = 0;
        for (uint32_t v = n; v; v >>= 1) ++k;
        uint32_t r = getrandbits(k);
        while (r >= n) r = getrandbits(k);
        return r;
    }

    __device__ __forceinline__ uint32_t choice_index(uint32_t len) {
        return randbelow(len);
    }
};


// Read a precomputed day's first MT19937 output block without carrying a
// 624-word per-thread generator state. `words` contains the exact tempered
// outputs of PyRandom::genrand_uint32() after CPython seeding.
//
// randbelow() can theoretically reject indefinitely. If a day consumes beyond
// the cached 624 outputs, reconstruct the exact stream in a deliberately slow
// noinline fallback. Normal Kaggriculture daily consumption is far below this.
__device__ __noinline__ uint32_t pyrandom_word_slow(uint64_t key, const uint32_t* base, int word_index) {
    PyRandom rng;
    rng.seed(key, base);
    uint32_t out = 0;
    for (int i = 0; i <= word_index; ++i) out = rng.genrand_uint32();
    return out;
}

struct CachedDayRandom {
    const uint32_t* words;
    uint64_t key;
    const uint32_t* base;
    int pos = 0;

    __device__ __forceinline__ CachedDayRandom(const uint32_t* w, uint64_t k, const uint32_t* b)
        : words(w), key(k), base(b) {}

    __device__ __forceinline__ uint32_t genrand_uint32() {
        const int i = pos++;
        return i < PyRandom::N ? words[i] : pyrandom_word_slow(key, base, i);
    }

    __device__ __forceinline__ double random() {
        const uint32_t a = genrand_uint32() >> 5;
        const uint32_t b = genrand_uint32() >> 6;
        return (static_cast<double>(a) * 67108864.0 + static_cast<double>(b)) *
               (1.0 / 9007199254740992.0);
    }

    __device__ __forceinline__ uint32_t getrandbits(int k) {
        return genrand_uint32() >> (32 - k);
    }

    __device__ __forceinline__ uint32_t randbelow(uint32_t n) {
        int k = 0;
        for (uint32_t v = n; v; v >>= 1) ++k;
        uint32_t r = getrandbits(k);
        while (r >= n) r = getrandbits(k);
        return r;
    }

    __device__ __forceinline__ uint32_t choice_index(uint32_t len) {
        return randbelow(len);
    }
};

}  // namespace kagcuda
