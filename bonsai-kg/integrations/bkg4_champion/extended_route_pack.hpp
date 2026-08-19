#pragma once

#include "../stateful_v6/route_pack.hpp"
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace bonsaikg4_champion {

inline bonsaikg_v6::RoutePack load_extended_route_pack(const std::string& path, const std::string& expected_sha256 = {}) {
    const uint16_t endian = 1;
    if (*reinterpret_cast<const uint8_t*>(&endian) != 1) throw std::runtime_error("BKV6 requires little-endian host");
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot open BKV6 route pack: " + path);
    std::vector<uint8_t> raw((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    if (raw.size() < sizeof(bonsaikg_v6::PackHeader)) throw std::runtime_error("short BKV6 route pack");
    const auto full = bonsaikg::sha256_bytes(raw);
    if (!expected_sha256.empty() && bonsaikg_v6::hash_hex(full) != expected_sha256)
        throw std::runtime_error("extended BKV6 route-pack SHA-256 mismatch");
    bonsaikg_v6::PackHeader h{};
    std::memcpy(&h, raw.data(), sizeof(h));
    const char expected_magic[8]={'B','K','V','6','R','P','1','\0'};
    if (std::memcmp(h.magic, expected_magic, 8) || h.version != 1 || h.route_count == 0)
        throw std::runtime_error("invalid extended BKV6 route-pack header");
    const size_t desc_bytes = static_cast<size_t>(h.route_count) * sizeof(bonsaikg_v6::RouteDesc);
    const size_t expected = sizeof(bonsaikg_v6::PackHeader) + desc_bytes + static_cast<size_t>(h.action_count) * sizeof(bonsaikg_v6::PackedAction);
    if (raw.size() != expected) throw std::runtime_error("extended BKV6 route-pack size/count mismatch");
    std::vector<uint8_t> payload(raw.begin()+sizeof(bonsaikg_v6::PackHeader), raw.end());
    const auto payload_hash = bonsaikg::sha256_bytes(payload);
    if (std::memcmp(payload_hash.data(), h.payload_hash, 32)) throw std::runtime_error("extended BKV6 payload SHA-256 mismatch");
    bonsaikg_v6::RoutePack out;
    out.routes.resize(h.route_count);
    out.actions.resize(h.action_count);
    std::memcpy(out.routes.data(), raw.data()+sizeof(bonsaikg_v6::PackHeader), desc_bytes);
    std::memcpy(out.actions.data(), raw.data()+sizeof(bonsaikg_v6::PackHeader)+desc_bytes, out.actions.size()*sizeof(bonsaikg_v6::PackedAction));
    std::vector<uint8_t> seen(65536,0), used(h.action_count,0);
    for (const auto& d : out.routes) {
        if (seen[d.route_id] || d.steps == 0 || static_cast<uint64_t>(d.first)+d.steps > h.action_count)
            throw std::runtime_error("invalid extended BKV6 route descriptor");
        seen[d.route_id]=1;
        for (uint32_t i=d.first;i<d.first+d.steps;++i) {
            if (used[i]) throw std::runtime_error("overlapping extended BKV6 route descriptor");
            used[i]=1;
        }
    }
    for (uint8_t v:used) if (!v) throw std::runtime_error("extended BKV6 action table not dense/used");
    for (const auto& a:out.actions)
        if (a.saved_hands>bonsaikg_v6::MAX_SAVED_HANDS || a.n_market>bonsaikg_v6::MAX_MARKET)
            throw std::runtime_error("invalid extended BKV6 packed action count");
    return out;
}

} // namespace bonsaikg4_champion
