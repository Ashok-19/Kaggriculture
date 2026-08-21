#include "bonsai_kg_io.hpp"

#include <algorithm>
#include <array>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

struct JsonCursor {
    explicit JsonCursor(const std::string& text) : s(text) {}
    const std::string& s;
    size_t p = 0;

    void ws() {
        while (p < s.size() && static_cast<unsigned char>(s[p]) <= 0x20) ++p;
    }
    char peek() {
        ws();
        if (p >= s.size()) fail("unexpected end of JSON");
        return s[p];
    }
    bool consume(char c) {
        ws();
        if (p < s.size() && s[p] == c) { ++p; return true; }
        return false;
    }
    void expect(char c) {
        ws();
        if (p >= s.size() || s[p] != c) fail(std::string("expected '") + c + "'");
        ++p;
    }
    [[noreturn]] void fail(const std::string& msg) const {
        throw std::runtime_error(msg + " at byte " + std::to_string(p));
    }

    static int hex_digit(char c) {
        if (c >= '0' && c <= '9') return c - '0';
        if (c >= 'a' && c <= 'f') return c - 'a' + 10;
        if (c >= 'A' && c <= 'F') return c - 'A' + 10;
        return -1;
    }
    uint32_t hex4() {
        uint32_t v = 0;
        for (int i = 0; i < 4; ++i) {
            if (p >= s.size()) fail("truncated unicode escape");
            int d = hex_digit(s[p++]);
            if (d < 0) fail("invalid unicode escape");
            v = (v << 4) | static_cast<uint32_t>(d);
        }
        return v;
    }
    static void append_utf8(std::string& out, uint32_t cp) {
        if (cp <= 0x7f) out.push_back(static_cast<char>(cp));
        else if (cp <= 0x7ff) {
            out.push_back(static_cast<char>(0xc0 | (cp >> 6)));
            out.push_back(static_cast<char>(0x80 | (cp & 0x3f)));
        } else if (cp <= 0xffff) {
            out.push_back(static_cast<char>(0xe0 | (cp >> 12)));
            out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3f)));
            out.push_back(static_cast<char>(0x80 | (cp & 0x3f)));
        } else {
            out.push_back(static_cast<char>(0xf0 | (cp >> 18)));
            out.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3f)));
            out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3f)));
            out.push_back(static_cast<char>(0x80 | (cp & 0x3f)));
        }
    }
    std::string string_value() {
        ws(); expect('"');
        std::string out;
        while (p < s.size()) {
            char c = s[p++];
            if (c == '"') return out;
            if (c != '\\') { out.push_back(c); continue; }
            if (p >= s.size()) fail("unterminated string escape");
            char e = s[p++];
            switch (e) {
                case '"': out.push_back('"'); break;
                case '\\': out.push_back('\\'); break;
                case '/': out.push_back('/'); break;
                case 'b': out.push_back('\b'); break;
                case 'f': out.push_back('\f'); break;
                case 'n': out.push_back('\n'); break;
                case 'r': out.push_back('\r'); break;
                case 't': out.push_back('\t'); break;
                case 'u': {
                    uint32_t cp = hex4();
                    if (cp >= 0xd800 && cp <= 0xdbff && p + 6 <= s.size() && s[p] == '\\' && s[p + 1] == 'u') {
                        p += 2;
                        uint32_t lo = hex4();
                        if (lo >= 0xdc00 && lo <= 0xdfff)
                            cp = 0x10000 + ((cp - 0xd800) << 10) + (lo - 0xdc00);
                    }
                    append_utf8(out, cp);
                    break;
                }
                default: fail("invalid string escape");
            }
        }
        fail("unterminated string");
    }
    void skip_string() {
        ws(); expect('"');
        while (p < s.size()) {
            char c = s[p++];
            if (c == '"') return;
            if (c != '\\') continue;
            if (p >= s.size()) fail("unterminated string escape");
            char e = s[p++];
            if (e == 'u') {
                if (p + 4 > s.size()) fail("truncated unicode escape");
                p += 4;
            }
        }
        fail("unterminated string");
    }

    void skip_compound() {
        ws();
        if (p >= s.size() || (s[p] != '{' && s[p] != '[')) fail("expected compound JSON value");
        int depth = 0;
        while (p < s.size()) {
            char c = s[p++];
            if (c == '"') {
                while (p < s.size()) {
                    char q = s[p++];
                    if (q == '"') break;
                    if (q == '\\') {
                        if (p >= s.size()) fail("unterminated string escape");
                        char e = s[p++];
                        if (e == 'u') {
                            if (p + 4 > s.size()) fail("truncated unicode escape");
                            p += 4;
                        }
                    }
                }
                continue;
            }
            if (c == '{' || c == '[') ++depth;
            else if (c == '}' || c == ']') {
                if (--depth == 0) return;
                if (depth < 0) fail("unbalanced compound JSON value");
            }
        }
        fail("unterminated compound JSON value");
    }

    bool literal(const char* word) {
        ws();
        size_t q = p;
        for (const char* c = word; *c; ++c, ++q)
            if (q >= s.size() || s[q] != *c) return false;
        p = q;
        return true;
    }
    std::string number_token() {
        ws();
        size_t b = p;
        if (p < s.size() && s[p] == '-') ++p;
        if (p >= s.size() || !std::isdigit(static_cast<unsigned char>(s[p]))) fail("expected JSON number");
        if (s[p] == '0') ++p;
        else while (p < s.size() && std::isdigit(static_cast<unsigned char>(s[p]))) ++p;
        if (p < s.size() && s[p] == '.') {
            ++p;
            if (p >= s.size() || !std::isdigit(static_cast<unsigned char>(s[p]))) fail("invalid JSON fraction");
            while (p < s.size() && std::isdigit(static_cast<unsigned char>(s[p]))) ++p;
        }
        if (p < s.size() && (s[p] == 'e' || s[p] == 'E')) {
            ++p;
            if (p < s.size() && (s[p] == '+' || s[p] == '-')) ++p;
            if (p >= s.size() || !std::isdigit(static_cast<unsigned char>(s[p]))) fail("invalid JSON exponent");
            while (p < s.size() && std::isdigit(static_cast<unsigned char>(s[p]))) ++p;
        }
        return s.substr(b, p - b);
    }
    void skip_value() {
        ws();
        char c = peek();
        if (c == '"') { skip_string(); return; }
        if (c == '{' || c == '[') { skip_compound(); return; }
        if (c == '-' || std::isdigit(static_cast<unsigned char>(c))) { (void)number_token(); return; }
        if (literal("true") || literal("false") || literal("null")) return;
        fail("unsupported JSON value");
    }
};

bool parse_int_string(const std::string& raw, int64_t& out) {
    size_t b = 0, e = raw.size();
    while (b < e && std::isspace(static_cast<unsigned char>(raw[b]))) ++b;
    while (e > b && std::isspace(static_cast<unsigned char>(raw[e - 1]))) --e;
    if (b == e) return false;
    std::string x = raw.substr(b, e - b);
    char* end = nullptr;
    errno = 0;
    long long v = std::strtoll(x.c_str(), &end, 10);
    if (errno || end != x.c_str() + x.size()) return false;
    out = static_cast<int64_t>(v);
    return true;
}

bool coerce_int(JsonCursor& j, int64_t& out) {
    char c = j.peek();
    if (c == '"') return parse_int_string(j.string_value(), out);
    if (c == 't') { if (!j.literal("true")) return false; out = 1; return true; }
    if (c == 'f') { if (!j.literal("false")) return false; out = 0; return true; }
    if (c == '-' || std::isdigit(static_cast<unsigned char>(c))) {
        std::string tok = j.number_token();
        char* end = nullptr;
        errno = 0;
        long double v = std::strtold(tok.c_str(), &end);
        if (errno || end != tok.c_str() + tok.size() || !std::isfinite(v)) return false;
        if (v < static_cast<long double>(std::numeric_limits<int64_t>::min()) ||
            v > static_cast<long double>(std::numeric_limits<int64_t>::max())) return false;
        out = static_cast<int64_t>(v);  // Python int(float) truncates toward zero.
        return true;
    }
    j.skip_value();
    return false;
}

struct UnitAction {
    std::string op = "PASS";
    std::string item;
    int64_t n = 1;
    bool has_item = false;
    bool has_n = false;
};
struct MarketOrder {
    std::string op;
    std::string item;
    int64_t n = 0;
    bool valid = false;
    bool has_item = false;
};
struct Action {
    UnitAction farmer;
    std::vector<UnitAction> hands;
    std::vector<MarketOrder> market;
};

bool unit_noarg(const std::string& op) {
    static const std::array<const char*, 15> ops = {
        "PASS","NORTH","SOUTH","EAST","WEST","DROP","WATER","HARVEST",
        "FERTILIZE","DIG","BUILD_COOP","BUILD_PASTURE","FEED","COLLECT_FERTILIZER","CARE"
    };
    return std::find_if(ops.begin(), ops.end(), [&](const char* x){ return op == x; }) != ops.end();
}
bool market_count(const std::string& op) {
    return op == "BUY_SEED" || op == "BUY_PRODUCT" || op == "BUY_ANIMAL" || op == "SELL";
}
void skip_array_tail(JsonCursor& j) {
    while (!j.consume(']')) { j.expect(','); j.skip_value(); }
}

UnitAction parse_unit(JsonCursor& j) {
    UnitAction out;
    if (j.peek() != '[') { j.skip_value(); return out; }
    j.expect('[');
    if (j.consume(']')) return out;
    if (j.peek() != '"') { j.skip_value(); skip_array_tail(j); return out; }
    std::string op = j.string_value();
    if (unit_noarg(op)) {
        out.op = op;
        skip_array_tail(j);
        return out;
    }
    if (op != "PLANT" && op != "PICKUP" && op != "PLACE") {
        skip_array_tail(j);
        return out;
    }
    if (j.consume(']')) return out;
    j.expect(',');
    if (j.peek() != '"') { j.skip_value(); skip_array_tail(j); return out; }
    std::string item = j.string_value();
    if (op == "PLANT") {
        out.op = op; out.item = item; out.has_item = true;
        skip_array_tail(j);
        return out;
    }
    int64_t n = 1;
    if (!j.consume(']')) {
        j.expect(',');
        if (!coerce_int(j, n)) { skip_array_tail(j); return UnitAction{}; }
        skip_array_tail(j);
    }
    out.op = op; out.item = item; out.n = n; out.has_item = true; out.has_n = true;
    return out;
}

MarketOrder parse_market(JsonCursor& j) {
    MarketOrder out;
    if (j.peek() != '[') { j.skip_value(); return out; }
    j.expect('[');
    if (j.consume(']')) return out;
    if (j.peek() != '"') { j.skip_value(); skip_array_tail(j); return out; }
    std::string op = j.string_value();
    if (op == "HIRE" || op == "BUY_LAND") {
        out.op = op; out.valid = true;
        skip_array_tail(j);
        return out;
    }
    if (!market_count(op)) { skip_array_tail(j); return out; }
    if (j.consume(']')) return out;
    j.expect(',');
    if (j.peek() != '"') { j.skip_value(); skip_array_tail(j); return out; }
    std::string item = j.string_value();
    if (j.consume(']')) return out;
    j.expect(',');
    int64_t n = 0;
    if (!coerce_int(j, n) || n <= 0) { skip_array_tail(j); return out; }
    skip_array_tail(j);
    out.op = op; out.item = item; out.n = n; out.valid = true; out.has_item = true;
    return out;
}

Action parse_action(JsonCursor& j) {
    Action out;
    if (j.peek() != '{') { j.skip_value(); return out; }
    j.expect('{');
    if (j.consume('}')) return out;
    for (;;) {
        std::string key = j.string_value();
        j.expect(':');
        if (key == "farmer") {
            out.farmer = parse_unit(j);
        } else if (key == "hands") {
            out.hands.clear();
            if (j.peek() != '[') j.skip_value();
            else {
                j.expect('[');
                if (!j.consume(']')) {
                    for (;;) {
                        out.hands.push_back(parse_unit(j));
                        if (j.consume(']')) break;
                        j.expect(',');
                    }
                }
            }
        } else if (key == "market") {
            out.market.clear();
            if (j.peek() != '[') j.skip_value();
            else {
                j.expect('[');
                if (!j.consume(']')) {
                    for (;;) {
                        out.market.push_back(parse_market(j));
                        if (j.consume(']')) break;
                        j.expect(',');
                    }
                }
            }
        } else {
            j.skip_value();
        }
        if (j.consume('}')) return out;
        j.expect(',');
    }
}

Action parse_agent(JsonCursor& j) {
    Action out;
    if (j.peek() != '{') { j.skip_value(); return out; }
    j.expect('{');
    if (j.consume('}')) return out;
    for (;;) {
        std::string key = j.string_value();
        j.expect(':');
        if (key == "action") out = parse_action(j);
        else j.skip_value();
        if (j.consume('}')) return out;
        j.expect(',');
    }
}

struct ReplayParsed {
    int64_t seed = 0;
    std::vector<Action> seats[2];
    bool saw_steps = false;
};

void parse_info(JsonCursor& j, ReplayParsed& out) {
    if (j.peek() != '{') { j.skip_value(); return; }
    j.expect('{');
    if (j.consume('}')) return;
    for (;;) {
        std::string key = j.string_value();
        j.expect(':');
        if (key == "seed") {
            int64_t v = 0;
            if (coerce_int(j, v)) out.seed = v;
        } else j.skip_value();
        if (j.consume('}')) return;
        j.expect(',');
    }
}

void parse_steps(JsonCursor& j, ReplayParsed& out) {
    out.saw_steps = true;
    if (j.peek() != '[') { j.skip_value(); return; }
    j.expect('[');
    if (j.consume(']')) return;
    size_t step_index = 0;
    for (;;) {
        Action a[2];
        if (j.peek() != '[') {
            j.skip_value();
        } else {
            j.expect('[');
            int seat = 0;
            if (!j.consume(']')) {
                for (;;) {
                    if (seat < 2) a[seat] = parse_agent(j);
                    else j.skip_value();
                    ++seat;
                    if (j.consume(']')) break;
                    j.expect(',');
                }
            }
        }
        if (step_index > 0) {
            out.seats[0].push_back(std::move(a[0]));
            out.seats[1].push_back(std::move(a[1]));
        }
        ++step_index;
        if (j.consume(']')) break;
        j.expect(',');
    }
}

ReplayParsed parse_replay(const std::string& text) {
    JsonCursor j(text);
    ReplayParsed out;
    j.expect('{');
    if (j.consume('}')) throw std::runtime_error("empty replay object");
    for (;;) {
        std::string key = j.string_value();
        j.expect(':');
        if (key == "info") parse_info(j, out);
        else if (key == "steps") parse_steps(j, out);
        else j.skip_value();
        if (j.consume('}')) break;
        j.expect(',');
    }
    if (!out.saw_steps || out.seats[0].empty() || out.seats[0].size() != out.seats[1].size())
        throw std::runtime_error("replay has no usable post-initial steps");
    return out;
}

std::string quote_ascii(const std::string& s) {
    std::string out = "\"";
    for (unsigned char c : s) {
        if (c == '"' || c == '\\') { out.push_back('\\'); out.push_back(static_cast<char>(c)); }
        else if (c < 0x20) {
            static const char* h = "0123456789abcdef";
            out += "\\u00"; out.push_back(h[c >> 4]); out.push_back(h[c & 15]);
        } else out.push_back(static_cast<char>(c));
    }
    out.push_back('"');
    return out;
}
std::string unit_json(const UnitAction& u) {
    std::string out = "[" + quote_ascii(u.op);
    if (u.has_item) out += "," + quote_ascii(u.item);
    if (u.has_n) out += "," + std::to_string(u.n);
    out += "]";
    return out;
}
std::string market_json(const MarketOrder& o) {
    if (!o.valid) return "[]";
    std::string out = "[" + quote_ascii(o.op);
    if (o.has_item) out += "," + quote_ascii(o.item) + "," + std::to_string(o.n);
    out += "]";
    return out;
}
std::string action_json(const Action& a) {
    std::string out = "{\"farmer\":" + unit_json(a.farmer) + ",\"hands\":[";
    for (size_t i = 0; i < a.hands.size(); ++i) { if (i) out += ','; out += unit_json(a.hands[i]); }
    out += "],\"market\":[";
    for (size_t i = 0; i < a.market.size(); ++i) { if (i) out += ','; out += market_json(a.market[i]); }
    out += "]}";
    return out;
}
std::string actions_json(const std::vector<Action>& actions) {
    std::string out = "[";
    for (size_t i = 0; i < actions.size(); ++i) { if (i) out += ','; out += action_json(actions[i]); }
    out += "]";
    return out;
}
std::string hex_digest(const std::array<uint8_t, 32>& d) {
    static const char* h = "0123456789abcdef";
    std::string out; out.reserve(64);
    for (uint8_t b : d) { out.push_back(h[b >> 4]); out.push_back(h[b & 15]); }
    return out;
}
std::string tape_semantic_id(const std::string& actions) {
    static const std::string ns = "KAGGRICULTURE-NATIVE-TAPE-V1";
    std::vector<uint8_t> bytes;
    bytes.reserve(ns.size() + 1 + 8 + actions.size());
    bytes.insert(bytes.end(), ns.begin(), ns.end());
    bytes.push_back(0);
    uint64_t n = static_cast<uint64_t>(actions.size());
    for (int i = 0; i < 8; ++i) bytes.push_back(static_cast<uint8_t>(n >> (8 * i)));
    bytes.insert(bytes.end(), actions.begin(), actions.end());
    return hex_digest(bonsaikg::sha256_bytes(bytes));
}

struct Args {
    fs::path spool, out_dir, metadata;
    bool use_stdin = false;
};
Args parse_args(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        std::string k = argv[i];
        if (k == "--stdin") { a.use_stdin = true; continue; }
        if (i + 1 >= argc) throw std::runtime_error("missing value for " + k);
        std::string v = argv[++i];
        if (k == "--spool") a.spool = v;
        else if (k == "--out-dir") a.out_dir = v;
        else if (k == "--metadata") a.metadata = v;
        else throw std::runtime_error("unknown argument: " + k);
    }
    if (a.out_dir.empty() || a.metadata.empty() || (a.use_stdin && !a.spool.empty()) || (!a.use_stdin && a.spool.empty()))
        throw std::runtime_error("use exactly one of --stdin or --spool, plus --out-dir and --metadata");
    return a;
}

template <typename T> bool read_exact(std::istream& in, T& v) {
    in.read(reinterpret_cast<char*>(&v), sizeof(T));
    return static_cast<bool>(in);
}

}  // namespace

int main(int argc, char** argv) {
    try {
        Args args = parse_args(argc, argv);
        fs::create_directories(args.out_dir);
        std::ifstream spool_in;
        std::istream* input = &std::cin;
        if (!args.use_stdin) {
            spool_in.open(args.spool, std::ios::binary);
            if (!spool_in) throw std::runtime_error("cannot open replay spool");
            input = &spool_in;
        }
        std::istream& in = *input;
        char magic[8]{};
        in.read(magic, 8);
        if (!in || std::string(magic, 8) != std::string("B4RSP01\0", 8)) throw std::runtime_error("bad replay stream magic");
        std::ofstream meta(args.metadata);
        if (!meta) throw std::runtime_error("cannot create converter metadata");
        meta << "episode_id\tseed\tsource_seat\ttape_semantic_id\ttape_path\n";
        size_t replay_count = 0, tape_count = 0, skipped = 0;
        for (;;) {
            int64_t episode_id = 0;
            uint64_t n = 0;
            in.read(reinterpret_cast<char*>(&episode_id), sizeof(episode_id));
            if (in.eof()) break;
            if (!in) throw std::runtime_error("truncated replay spool episode id");
            if (!read_exact(in, n)) throw std::runtime_error("truncated replay spool length");
            if (n > (1ull << 31)) throw std::runtime_error("unreasonable replay JSON length");
            std::string text(static_cast<size_t>(n), '\0');
            in.read(text.data(), static_cast<std::streamsize>(n));
            if (!in) throw std::runtime_error("truncated replay spool JSON");
            ++replay_count;
            try {
                ReplayParsed replay = parse_replay(text);
                for (int seat = 0; seat < 2; ++seat) {
                    std::string actions = actions_json(replay.seats[seat]);
                    std::string tid = tape_semantic_id(actions);
                    fs::path path = fs::absolute(args.out_dir / (std::to_string(episode_id) + "_" + std::to_string(seat) + ".json"));
                    std::ofstream out(path, std::ios::binary);
                    if (!out) throw std::runtime_error("cannot create tape file");
                    out << "{\"source\":{\"seat\":" << seat << ",\"seed\":" << replay.seed << "},\"actions\":" << actions << "}\n";
                    meta << episode_id << '\t' << replay.seed << '\t' << seat << '\t' << tid << '\t' << path.string() << '\n';
                    ++tape_count;
                }
            } catch (const std::exception& e) {
                ++skipped;
                std::cerr << "SKIP replay " << episode_id << ": " << e.what() << '\n';
            }
        }
        meta.flush();
        std::cout << "REPLAY_CONVERT replays=" << replay_count << " tapes=" << tape_count << " skipped=" << skipped << '\n';
        return tape_count ? 0 : 2;
    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << '\n';
        return 2;
    }
}
