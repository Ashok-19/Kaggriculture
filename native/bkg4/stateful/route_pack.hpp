#pragma once

#include "bonsai_kg_io.hpp"
#include <array>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace bonsaikg_v6 {

inline constexpr int ROUTE_COUNT = 11;
inline constexpr int MAX_SAVED_HANDS = 12;
inline constexpr int MAX_MARKET = 10;
inline constexpr int ACTION_RECORD_BYTES = 94;

#pragma pack(push,1)
struct PackHeader {
    char magic[8];
    uint16_t version;
    uint16_t route_count;
    uint32_t action_count;
    uint8_t payload_hash[32];
};
struct RouteDesc { uint16_t route_id, steps; uint32_t first; };
struct PackedUnit { uint8_t op, arg; int16_t n; };
struct PackedOrder { uint8_t op, item; uint16_t n; };
struct PackedAction {
    uint8_t saved_hands, n_market;
    PackedUnit farmer;
    PackedUnit hands[MAX_SAVED_HANDS];
    PackedOrder market[MAX_MARKET];
};
#pragma pack(pop)

static_assert(sizeof(PackHeader) == 48, "BKV6 pack header drift");
static_assert(sizeof(RouteDesc) == 8, "BKV6 route descriptor drift");
static_assert(sizeof(PackedUnit) == 4, "BKV6 packed unit drift");
static_assert(sizeof(PackedOrder) == 4, "BKV6 packed order drift");
static_assert(sizeof(PackedAction) == ACTION_RECORD_BYTES, "BKV6 packed action drift");

inline std::string hash_hex(const std::array<uint8_t,32>& h) {
    static constexpr char HEX[] = "0123456789abcdef";
    std::string out(64, '0');
    for (size_t i=0;i<h.size();++i) { out[i*2]=HEX[h[i]>>4]; out[i*2+1]=HEX[h[i]&15]; }
    return out;
}

inline std::string hash_hex(const uint8_t* h) {
    std::array<uint8_t,32> a{}; std::memcpy(a.data(), h, a.size()); return hash_hex(a);
}

inline std::string controller_hash_hex(const bonsaikg::OwnedPolicy& p) {
    return hash_hex(bonsaikg::semantic_hash(p));
}

struct RoutePack {
    std::vector<RouteDesc> routes;
    std::vector<PackedAction> actions;

    static RoutePack load(const std::string& path, const std::string& expected_sha256) {
        const uint16_t endian = 1;
        if (*reinterpret_cast<const uint8_t*>(&endian) != 1) throw std::runtime_error("BKV6 requires little-endian host");
        std::ifstream in(path, std::ios::binary);
        if (!in) throw std::runtime_error("cannot open BKV6 route pack: " + path);
        std::vector<uint8_t> raw((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
        if (raw.size() < sizeof(PackHeader)) throw std::runtime_error("short BKV6 route pack");
        const auto full = bonsaikg::sha256_bytes(raw);
        if (hash_hex(full) != expected_sha256) throw std::runtime_error("BKV6 route-pack SHA-256 mismatch");
        PackHeader h{}; std::memcpy(&h, raw.data(), sizeof(h));
        const char expected_magic[8]={'B','K','V','6','R','P','1','\0'};
        if (std::memcmp(h.magic, expected_magic, 8) || h.version != 1 || h.route_count != ROUTE_COUNT)
            throw std::runtime_error("invalid BKV6 route-pack header");
        const size_t desc_bytes = static_cast<size_t>(h.route_count) * sizeof(RouteDesc);
        const size_t expected = sizeof(PackHeader) + desc_bytes + static_cast<size_t>(h.action_count) * sizeof(PackedAction);
        if (raw.size() != expected) throw std::runtime_error("BKV6 route-pack size/count mismatch");
        std::vector<uint8_t> payload(raw.begin()+sizeof(PackHeader), raw.end());
        const auto payload_hash = bonsaikg::sha256_bytes(payload);
        if (std::memcmp(payload_hash.data(), h.payload_hash, 32)) throw std::runtime_error("BKV6 payload SHA-256 mismatch");
        RoutePack out; out.routes.resize(h.route_count); out.actions.resize(h.action_count);
        std::memcpy(out.routes.data(), raw.data()+sizeof(PackHeader), desc_bytes);
        std::memcpy(out.actions.data(), raw.data()+sizeof(PackHeader)+desc_bytes, out.actions.size()*sizeof(PackedAction));
        std::vector<uint8_t> seen(256,0), used(h.action_count,0);
        for (const RouteDesc& d : out.routes) {
            if (seen[d.route_id] || d.steps == 0 || static_cast<uint64_t>(d.first)+d.steps > h.action_count)
                throw std::runtime_error("invalid BKV6 route descriptor");
            seen[d.route_id]=1;
            for (uint32_t i=d.first;i<d.first+d.steps;++i) {
                if (used[i]) throw std::runtime_error("overlapping BKV6 route descriptor");
                used[i]=1;
            }
        }
        for (uint8_t v:used) if (!v) throw std::runtime_error("BKV6 action table not dense/used");
        for (const PackedAction& a:out.actions)
            if (a.saved_hands>MAX_SAVED_HANDS || a.n_market>MAX_MARKET) throw std::runtime_error("invalid BKV6 packed action count");
        return out;
    }

    const RouteDesc& route(uint16_t id) const {
        for (const auto& d:routes) if (d.route_id==id) return d;
        throw std::runtime_error("unknown BKV6 route id " + std::to_string(id));
    }
    const PackedAction& at(uint16_t id, int step) const {
        const RouteDesc& d=route(id); int s=step; if(s<0)s=0; if(s>=d.steps)s=d.steps-1; return actions[d.first+s];
    }
};

} // namespace bonsaikg_v6
