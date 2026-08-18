#pragma once
#include "sim.cuh"
#include <cctype>
#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace kagcuda {

inline int tape_item_id(const std::string& s) {
    static constexpr const char* names[N_ITEMS] = {
        "WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON", "EGG",
        "MILK", "WOOL", "FERTILIZER", "GOOSE", "COW", "SHEEP"
    };
    for (int i = 0; i < N_ITEMS; ++i) if (s == names[i]) return i;
    return 255;
}

inline int tape_unit_op(const std::string& s) {
    static constexpr const char* names[] = {
        "PASS", "NORTH", "SOUTH", "EAST", "WEST", "PICKUP", "DROP", "PLACE",
        "PLANT", "WATER", "HARVEST", "FERTILIZE", "DIG", "BUILD_COOP", "BUILD_PASTURE",
        "FEED", "COLLECT_FERTILIZER", "CARE"
    };
    for (int i = 0; i < 18; ++i) if (s == names[i]) return i;
    return OP_INVALID;
}

inline int tape_market_op(const std::string& s) {
    if (s == "HIRE") return M_HIRE;
    if (s == "BUY_LAND") return M_BUY_LAND;
    if (s == "BUY_SEED") return M_BUY_SEED;
    if (s == "BUY_PRODUCT") return M_BUY_PRODUCT;
    if (s == "BUY_ANIMAL") return M_BUY_ANIMAL;
    if (s == "SELL") return M_SELL;
    return M_NONE;
}

class TapeJsonCursor {
public:
    explicit TapeJsonCursor(const std::string& text) : s_(text) {}

    size_t pos() const { return p_; }
    void set_pos(size_t p) { p_ = p; }

    void ws() {
        while (p_ < s_.size() && std::isspace(static_cast<unsigned char>(s_[p_]))) ++p_;
    }

    char peek() {
        ws();
        if (p_ >= s_.size()) fail("unexpected end of JSON");
        return s_[p_];
    }

    bool consume(char c) {
        ws();
        if (p_ < s_.size() && s_[p_] == c) { ++p_; return true; }
        return false;
    }

    void expect(char c) {
        ws();
        if (p_ >= s_.size() || s_[p_] != c)
            fail(std::string("expected '") + c + "'");
        ++p_;
    }

    std::string string_value() {
        ws(); expect('"');
        std::string out;
        while (p_ < s_.size()) {
            const char c = s_[p_++];
            if (c == '"') return out;
            if (c != '\\') { out.push_back(c); continue; }
            if (p_ >= s_.size()) fail("unterminated escape");
            const char e = s_[p_++];
            switch (e) {
                case '"': out.push_back('"'); break;
                case '\\': out.push_back('\\'); break;
                case '/': out.push_back('/'); break;
                case 'b': out.push_back('\b'); break;
                case 'f': out.push_back('\f'); break;
                case 'n': out.push_back('\n'); break;
                case 'r': out.push_back('\r'); break;
                case 't': out.push_back('\t'); break;
                default: fail("unsupported JSON escape in tape");
            }
        }
        fail("unterminated string");
        return {};
    }

    int64_t integer_value() {
        ws();
        bool neg = false;
        if (consume('-')) neg = true;
        ws();
        if (p_ >= s_.size() || !std::isdigit(static_cast<unsigned char>(s_[p_]))) fail("expected integer");
        int64_t v = 0;
        while (p_ < s_.size() && std::isdigit(static_cast<unsigned char>(s_[p_]))) {
            v = v * 10 + (s_[p_++] - '0');
        }
        return neg ? -v : v;
    }

    void skip_value() {
        ws();
        const char c = peek();
        if (c == '"') { (void)string_value(); return; }
        if (c == '{') {
            expect('{');
            if (consume('}')) return;
            for (;;) {
                (void)string_value(); expect(':'); skip_value();
                if (consume('}')) return;
                expect(',');
            }
        }
        if (c == '[') {
            expect('[');
            if (consume(']')) return;
            for (;;) {
                skip_value();
                if (consume(']')) return;
                expect(',');
            }
        }
        if (c == '-' || std::isdigit(static_cast<unsigned char>(c))) {
            // Metadata can contain floating rewards. Skip a generic JSON number.
            if (c == '-') ++p_;
            while (p_ < s_.size() && std::isdigit(static_cast<unsigned char>(s_[p_]))) ++p_;
            if (p_ < s_.size() && s_[p_] == '.') {
                ++p_; while (p_ < s_.size() && std::isdigit(static_cast<unsigned char>(s_[p_]))) ++p_;
            }
            if (p_ < s_.size() && (s_[p_] == 'e' || s_[p_] == 'E')) {
                ++p_; if (p_ < s_.size() && (s_[p_] == '+' || s_[p_] == '-')) ++p_;
                while (p_ < s_.size() && std::isdigit(static_cast<unsigned char>(s_[p_]))) ++p_;
            }
            return;
        }
        if (match_literal("true") || match_literal("false") || match_literal("null")) return;
        fail("unsupported JSON value");
    }

private:
    const std::string& s_;
    size_t p_ = 0;

    bool match_literal(const char* lit) {
        ws();
        size_t q = p_;
        for (const char* c = lit; *c; ++c, ++q)
            if (q >= s_.size() || s_[q] != *c) return false;
        p_ = q;
        return true;
    }

    [[noreturn]] void fail(const std::string& msg) const {
        throw std::runtime_error(msg + " at byte " + std::to_string(p_));
    }
};

inline UnitAction parse_tape_unit(TapeJsonCursor& j) {
    UnitAction out{};
    j.expect('[');
    if (j.consume(']')) return out;
    const std::string op = j.string_value();
    out.op = static_cast<uint8_t>(tape_unit_op(op));
    if (j.consume(']')) return out;
    j.expect(',');
    if (j.peek() == '"') {
        out.arg = static_cast<uint8_t>(tape_item_id(j.string_value()));
    } else {
        out.n = static_cast<int16_t>(j.integer_value());
    }
    if (j.consume(']')) return out;
    j.expect(',');
    out.n = static_cast<int16_t>(j.integer_value());
    while (!j.consume(']')) { j.expect(','); j.skip_value(); }
    return out;
}

inline Order parse_tape_order(TapeJsonCursor& j) {
    Order out{};
    j.expect('[');
    if (j.consume(']')) return out;
    const std::string op = j.string_value();
    out.op = static_cast<uint8_t>(tape_market_op(op));
    if (out.op == M_HIRE || out.op == M_BUY_LAND) {
        out.n = 1;
        while (!j.consume(']')) { j.expect(','); j.skip_value(); }
        return out;
    }
    if (j.consume(']')) { out.op = M_NONE; return out; }
    j.expect(',');
    if (j.peek() != '"') { j.skip_value(); out.op = M_NONE; while (!j.consume(']')) { j.expect(','); j.skip_value(); } return out; }
    out.item = static_cast<uint8_t>(tape_item_id(j.string_value()));
    if (j.consume(']')) { out.op = M_NONE; return out; }
    j.expect(',');
    out.n = static_cast<int32_t>(j.integer_value());
    while (!j.consume(']')) { j.expect(','); j.skip_value(); }
    return out;
}

inline Action parse_tape_action(TapeJsonCursor& j) {
    Action out{};
    out.n_units = 1;
    j.expect('{');
    if (j.consume('}')) return out;
    for (;;) {
        const std::string key = j.string_value();
        j.expect(':');
        if (key == "farmer") {
            out.units[0] = parse_tape_unit(j);
        } else if (key == "hands") {
            j.expect('[');
            int hand = 0;
            if (!j.consume(']')) {
                for (;;) {
                    UnitAction u = parse_tape_unit(j);
                    if (hand < MAX_UNITS - 1) out.units[hand + 1] = u;
                    ++hand;
                    if (j.consume(']')) break;
                    j.expect(',');
                }
            }
            out.n_units = 1 + (hand < MAX_UNITS - 1 ? hand : MAX_UNITS - 1);
        } else if (key == "market") {
            j.expect('[');
            int n = 0;
            if (!j.consume(']')) {
                for (;;) {
                    Order o = parse_tape_order(j);
                    if (n < MAX_ORDERS) out.orders[n] = o;
                    ++n;
                    if (j.consume(']')) break;
                    j.expect(',');
                }
            }
            out.n_orders = n < MAX_ORDERS ? n : MAX_ORDERS;
        } else {
            j.skip_value();
        }
        if (j.consume('}')) return out;
        j.expect(',');
    }
}

struct NativeTape {
    std::vector<Action> actions;
    uint64_t source_seed = 0;
    int source_seat = -1;
};

inline NativeTape load_native_tape_json(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot open tape: " + path);
    std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    TapeJsonCursor j(text);
    NativeTape out;

    j.ws();
    if (j.peek() == '[') {
        j.expect('[');
        if (!j.consume(']')) {
            for (;;) {
                out.actions.push_back(parse_tape_action(j));
                if (j.consume(']')) break;
                j.expect(',');
            }
        }
    } else {
        j.expect('{');
        bool found_actions = false;
        if (!j.consume('}')) {
            for (;;) {
                const std::string key = j.string_value();
                j.expect(':');
                if (key == "source_seed") {
                    // Backward-compatible legacy top-level metadata.
                    out.source_seed = static_cast<uint64_t>(j.integer_value());
                } else if (key == "source_seat") {
                    out.source_seat = static_cast<int>(j.integer_value());
                } else if (key == "source") {
                    // Canonical replay-faithful metadata is nested under source.
                    j.expect('{');
                    if (!j.consume('}')) {
                        for (;;) {
                            const std::string skey = j.string_value();
                            j.expect(':');
                            if (skey == "seed") out.source_seed = static_cast<uint64_t>(j.integer_value());
                            else if (skey == "seat") out.source_seat = static_cast<int>(j.integer_value());
                            else j.skip_value();
                            if (j.consume('}')) break;
                            j.expect(',');
                        }
                    }
                } else if (key == "actions") {
                    found_actions = true;
                    j.expect('[');
                    if (!j.consume(']')) {
                        for (;;) {
                            out.actions.push_back(parse_tape_action(j));
                            if (j.consume(']')) break;
                            j.expect(',');
                        }
                    }
                } else {
                    j.skip_value();
                }
                if (j.consume('}')) break;
                j.expect(',');
            }
        }
        if (!found_actions) throw std::runtime_error("tape JSON has no actions field: " + path);
    }
    if (out.actions.empty()) throw std::runtime_error("tape actions must not be empty: " + path);
    return out;
}

}  // namespace kagcuda
