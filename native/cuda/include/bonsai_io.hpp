#pragma once

#include "bonsai_gp.cuh"
#include <fstream>
#include <stdexcept>
#include <string>

namespace kagcuda {

// Backward-compatible loader for research programs written before the staged
// Bonsai tree was expanded beyond six rules.  The manager and rule layouts are
// unchanged; only the number of rule slots grew.
constexpr int GP_LEGACY_MAX_RULES = 6;
constexpr int GP_LEGACY12_MAX_RULES = 12;
constexpr int GP_LEGACY24_MAX_RULES = 24;
constexpr int GP_LEGACY32_MAX_RULES = 32;
constexpr int GP_LEGACY48_MAX_RULES = 48;
constexpr int GP_LEGACY64_MAX_RULES = 64;
struct LegacyGPProgram6 {
    uint8_t n_rules = 1;
    uint8_t pad[3] = {0,0,0};
    GPRule rules[GP_LEGACY_MAX_RULES];
    GPManager manager;
};

struct LegacyGPProgram12 {
    uint8_t n_rules = 1;
    uint8_t pad[3] = {0,0,0};
    GPRule rules[GP_LEGACY12_MAX_RULES];
    GPManager manager;
};

struct LegacyGPProgram24 {
    uint8_t n_rules = 1;
    uint8_t pad[3] = {0,0,0};
    GPRule rules[GP_LEGACY24_MAX_RULES];
    GPManager manager;
};

struct LegacyGPProgram32 {
    uint8_t n_rules = 1;
    uint8_t pad[3] = {0,0,0};
    GPRule rules[GP_LEGACY32_MAX_RULES];
    GPManager manager;
};

struct LegacyGPProgram48 {
    uint8_t n_rules = 1;
    uint8_t pad[3] = {0,0,0};
    GPRule rules[GP_LEGACY48_MAX_RULES];
    GPManager manager;
};

struct LegacyGPProgram64 {
    uint8_t n_rules = 1;
    uint8_t pad[3] = {0,0,0};
    GPRule rules[GP_LEGACY64_MAX_RULES];
    GPManager manager;
};

inline GPProgram load_gp_program_binary(const std::string& path) {
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    if (!in) throw std::runtime_error("cannot open program: " + path);
    const std::streamsize size = in.tellg();
    in.seekg(0);
    GPProgram out{};
    if (size == static_cast<std::streamsize>(sizeof(GPProgram))) {
        in.read(reinterpret_cast<char*>(&out), sizeof(out));
        if (!in) throw std::runtime_error("failed reading program: " + path);
        return out;
    }
    if (size == static_cast<std::streamsize>(sizeof(LegacyGPProgram6))) {
        LegacyGPProgram6 old{};
        in.read(reinterpret_cast<char*>(&old), sizeof(old));
        if (!in) throw std::runtime_error("failed reading legacy program: " + path);
        out.n_rules = old.n_rules < 1 ? 1 : (old.n_rules > GP_LEGACY_MAX_RULES ? GP_LEGACY_MAX_RULES : old.n_rules);
        for (int i = 0; i < GP_LEGACY_MAX_RULES; ++i) out.rules[i] = old.rules[i];
        out.manager = old.manager;
        return out;
    }
    if (size == static_cast<std::streamsize>(sizeof(LegacyGPProgram12))) {
        LegacyGPProgram12 old{};
        in.read(reinterpret_cast<char*>(&old), sizeof(old));
        if (!in) throw std::runtime_error("failed reading intermediate program: " + path);
        out.n_rules = old.n_rules < 1 ? 1 : (old.n_rules > GP_LEGACY12_MAX_RULES ? GP_LEGACY12_MAX_RULES : old.n_rules);
        for (int i = 0; i < GP_LEGACY12_MAX_RULES; ++i) out.rules[i] = old.rules[i];
        out.manager = old.manager;
        return out;
    }
    if (size == static_cast<std::streamsize>(sizeof(LegacyGPProgram24))) {
        LegacyGPProgram24 old{};
        in.read(reinterpret_cast<char*>(&old), sizeof(old));
        if (!in) throw std::runtime_error("failed reading 24-rule program: " + path);
        out.n_rules = old.n_rules < 1 ? 1 : (old.n_rules > GP_LEGACY24_MAX_RULES ? GP_LEGACY24_MAX_RULES : old.n_rules);
        for (int i = 0; i < GP_LEGACY24_MAX_RULES; ++i) out.rules[i] = old.rules[i];
        out.manager = old.manager;
        return out;
    }
    if (size == static_cast<std::streamsize>(sizeof(LegacyGPProgram32))) {
        LegacyGPProgram32 old{};
        in.read(reinterpret_cast<char*>(&old), sizeof(old));
        if (!in) throw std::runtime_error("failed reading 32-rule program: " + path);
        out.n_rules = old.n_rules < 1 ? 1 : (old.n_rules > GP_LEGACY32_MAX_RULES ? GP_LEGACY32_MAX_RULES : old.n_rules);
        for (int i = 0; i < GP_LEGACY32_MAX_RULES; ++i) out.rules[i] = old.rules[i];
        out.manager = old.manager;
        return out;
    }
    if (size == static_cast<std::streamsize>(sizeof(LegacyGPProgram48))) {
        LegacyGPProgram48 old{};
        in.read(reinterpret_cast<char*>(&old), sizeof(old));
        if (!in) throw std::runtime_error("failed reading 48-rule program: " + path);
        out.n_rules = old.n_rules < 1 ? 1 : (old.n_rules > GP_LEGACY48_MAX_RULES ? GP_LEGACY48_MAX_RULES : old.n_rules);
        for (int i = 0; i < GP_LEGACY48_MAX_RULES; ++i) out.rules[i] = old.rules[i];
        out.manager = old.manager;
        return out;
    }
    if (size == static_cast<std::streamsize>(sizeof(LegacyGPProgram64))) {
        LegacyGPProgram64 old{};
        in.read(reinterpret_cast<char*>(&old), sizeof(old));
        if (!in) throw std::runtime_error("failed reading 64-rule program: " + path);
        out.n_rules = old.n_rules < 1 ? 1 : (old.n_rules > GP_LEGACY64_MAX_RULES ? GP_LEGACY64_MAX_RULES : old.n_rules);
        for (int i = 0; i < GP_LEGACY64_MAX_RULES; ++i) out.rules[i] = old.rules[i];
        out.manager = old.manager;
        return out;
    }
    throw std::runtime_error("invalid program binary size: " + path + " (got " + std::to_string(size) + ")");
}

}  // namespace kagcuda
