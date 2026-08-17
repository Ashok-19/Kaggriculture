#pragma once

#include <cstddef>
#include <cstdint>
#include <climits>

#ifdef __CUDACC__
#define BKG_ABI_HD __host__ __device__ inline
#else
#define BKG_ABI_HD inline
#endif

namespace bonsaikg {

constexpr uint16_t ABI_VERSION = 1;
constexpr int MAX_NODES = 128;
constexpr int MAX_DEPTH = 24;
constexpr int MAX_CHILD_REFS = 256;
constexpr int MAX_BB = 32;
constexpr int MAX_STATE = 48;
constexpr int MAX_PRED = 128;
constexpr int MAX_LIFECYCLE = 64;
constexpr int MAX_EFFECT = 128;
constexpr int MAX_UNITS = 40;
constexpr int MAX_MARKET_INTENTS = 16;
constexpr uint16_t NONE_U16 = 0xFFFFu;
constexpr uint32_t ACTIVE_BIT = 1u << 31;
constexpr uint32_t RUNNING_MASK16 = 0xFFFFu;

enum Status : uint8_t { SUCCESS = 0, FAILURE = 1, RUNNING = 2, SKIPPED = 3 };

enum NodeType : uint8_t {
    N_SEQUENCE = 0, N_SELECTOR, N_REACTIVE_SEQUENCE, N_REACTIVE_SELECTOR, N_IF,
    N_PARALLEL, N_INVERT, N_FORCE_SUCCESS, N_FORCE_FAILURE, N_TURN_TIMEOUT,
    N_RETRY, N_REPEAT, N_KEEP_RUNNING_UNTIL_FAILURE, N_SUCCESS, N_FAILURE,
    N_CONDITION, N_SET_BLACKBOARD, N_ADD_BLACKBOARD, N_EMIT_UNIT_INTENT,
    N_EMIT_MARKET_INTENT, N_WAIT_TURNS, N_MOVE_UNIT_TO
};

enum Feature : uint16_t {
    F_ALWAYS = 0, F_STEP, F_DAY, F_HOUR, F_OWN_MONEY, F_OPP_MONEY,
    F_OWN_UNITS, F_OPP_UNITS, F_OWN_LAND, F_OPP_LAND, F_OWN_HIRES_TODAY,
    F_MARKET_PRICE, F_MARKET_INVENTORY, F_OWN_SHED, F_OWN_SHED_TOTAL,
    F_OWN_SEED, F_OWN_INVENTORY, F_SHOP_COUNT, F_SHOP_AT, F_SHOP_SEEN,
    F_OWN_WORKER_X, F_OWN_WORKER_Y, F_OPP_WORKER_X, F_OPP_WORKER_Y,
    F_OWN_TILE_KIND, F_OPP_TILE_KIND, F_OWN_TILE_ITEM, F_OPP_TILE_ITEM,
    F_OWN_TILE_YIELD, F_OPP_TILE_YIELD, F_OWN_TILE_WATERED, F_OPP_TILE_WATERED,
    F_OWN_TILE_FED, F_OPP_TILE_FED, F_OWN_TILE_CARED, F_OPP_TILE_CARED,
    F_OWN_TILE_FERT_AVAILABLE, F_OPP_TILE_FERT_AVAILABLE
};

enum Compare : uint8_t { C_EQ = 0, C_NE, C_LT, C_LE, C_GT, C_GE };
enum EffectOp : uint8_t { E_SET = 0, E_CLEAR, E_ADD };

enum UnitOp : uint8_t {
    U_PASS = 0, U_NORTH, U_SOUTH, U_EAST, U_WEST, U_PICKUP, U_DROP, U_PLACE,
    U_PLANT, U_WATER, U_HARVEST, U_FERTILIZE, U_DIG, U_BUILD_COOP,
    U_BUILD_PASTURE, U_FEED, U_COLLECT_FERTILIZER, U_CARE
};
enum MarketOp : uint8_t { M_NONE = 0, M_HIRE, M_BUY_LAND, M_BUY_SEED, M_BUY_PRODUCT, M_BUY_ANIMAL, M_SELL };

struct Header {
    char magic[4];
    uint16_t abi_version;
    uint16_t header_size;
    uint8_t hash[32];
    uint16_t root;
    uint16_t n_nodes;
    uint16_t n_child_refs;
    uint16_t n_state;
    uint16_t n_bb;
    uint16_t n_predicates;
    uint16_t n_lifecycles;
    uint16_t n_effects;
    uint16_t instruction_budget;
    uint8_t pad[6];
};
static_assert(sizeof(Header) == 64, "Bonsai-KG Header ABI drift");

struct Node {
    uint8_t type;
    uint8_t flags;
    uint16_t first_child;
    uint16_t child_count;
    uint16_t state_slot;
    uint16_t lifecycle_slot;
    int32_t p0;
    int32_t p1;
    int32_t p2;
    int32_t p3;
    uint32_t debug_id;
};
static_assert(sizeof(Node) == 32, "Bonsai-KG Node ABI drift");

struct BbDecl { uint8_t type; uint8_t pad[3]; int32_t initial; };
static_assert(sizeof(BbDecl) == 8, "Bonsai-KG BbDecl ABI drift");
struct Predicate { uint8_t source; uint8_t cmp; uint16_t id; int32_t arg; int32_t rhs; };
static_assert(sizeof(Predicate) == 12, "Bonsai-KG Predicate ABI drift");
struct Lifecycle {
    uint16_t failure_if, success_if, skip_if, while_pred;
    uint16_t on_success_start, on_success_count;
    uint16_t on_failure_start, on_failure_count;
    uint16_t on_halted_start, on_halted_count;
    uint16_t post_start, post_count;
};
static_assert(sizeof(Lifecycle) == 24, "Bonsai-KG Lifecycle ABI drift");
struct Effect { uint8_t op; uint8_t slot; uint16_t reserved; int32_t value; };
static_assert(sizeof(Effect) == 8, "Bonsai-KG Effect ABI drift");

struct PolicyView {
    const Node* nodes = nullptr;
    const uint16_t* child_refs = nullptr;
    const BbDecl* bb = nullptr;
    const Predicate* predicates = nullptr;
    const Lifecycle* lifecycles = nullptr;
    const Effect* effects = nullptr;
    uint16_t root = 0;
    uint16_t n_nodes = 0;
    uint16_t n_child_refs = 0;
    uint16_t n_state = 0;
    uint16_t n_bb = 0;
    uint16_t n_predicates = 0;
    uint16_t n_lifecycles = 0;
    uint16_t n_effects = 0;
    uint16_t instruction_budget = 0;
};

struct NodeState { uint32_t a = 0, b = 0; };
static_assert(sizeof(NodeState) == 8, "NodeState size drift");
struct UnitClaim { uint8_t present = 0, priority = 0, op = U_PASS, arg = 0; int16_t n = 1; uint16_t order = 0; };
static_assert(sizeof(UnitClaim) == 8, "UnitClaim size drift");
struct MarketClaim { uint8_t priority = 0, op = M_NONE, item = 0, pad = 0; int32_t n = 0; uint16_t order = 0, pad2 = 0; };
static_assert(sizeof(MarketClaim) == 12, "MarketClaim size drift");

struct RuntimeState {
    // Episode-persistent working state. reset_runtime() initializes these once per episode.
    int32_t blackboard[MAX_BB];
    NodeState node_state[MAX_STATE];

    // Tick-local intent/VM scratch. tick() reinitializes the active portions before root execution.
    UnitClaim unit_claims[MAX_UNITS];
    MarketClaim market_claims[MAX_MARKET_INTENTS];
    uint16_t n_market = 0;
    uint16_t emit_seq = 0;
    uint16_t visits = 0;
    uint16_t fault = 0;

    // Episode-persistent diagnostic counter; never used for policy decisions.
    uint32_t ticks = 0;
};

struct UnitIntentAction { uint8_t op = U_PASS, arg = 0; int16_t n = 1; };
static_assert(sizeof(UnitIntentAction) == 4, "UnitIntentAction size drift");
struct MarketIntentAction { uint8_t op = M_NONE, item = 0; uint16_t pad = 0; int32_t n = 0; };
static_assert(sizeof(MarketIntentAction) == 8, "MarketIntentAction size drift");
struct ActionResult {
    UnitIntentAction units[MAX_UNITS];
    int32_t n_units = 0;
    MarketIntentAction market[MAX_MARKET_INTENTS];
    int32_t n_market = 0;
};

struct TraceEvent { int16_t step = 0; uint16_t node = 0; uint8_t event = 0, status = 0; uint16_t pad = 0; };
static_assert(sizeof(TraceEvent) == 8, "TraceEvent size drift");
struct TraceBuffer { TraceEvent* events = nullptr; uint16_t capacity = 0, count = 0; };

BKG_ABI_HD bool valid_node_type(uint8_t t) { return t <= N_MOVE_UNIT_TO; }
BKG_ABI_HD bool is_decorator(uint8_t t) {
    return t == N_INVERT || t == N_FORCE_SUCCESS || t == N_FORCE_FAILURE ||
           t == N_TURN_TIMEOUT || t == N_RETRY || t == N_REPEAT || t == N_KEEP_RUNNING_UNTIL_FAILURE;
}
BKG_ABI_HD int32_t clamp_i32_add(int32_t a, int32_t b, bool& ok) {
    const int64_t v = static_cast<int64_t>(a) + static_cast<int64_t>(b);
    if (v < INT32_MIN || v > INT32_MAX) { ok = false; return 0; }
    ok = true; return static_cast<int32_t>(v);
}
BKG_ABI_HD bool compare_i32(int32_t lhs, int32_t rhs, uint8_t c) {
    switch (c) {
        case C_EQ: return lhs == rhs; case C_NE: return lhs != rhs;
        case C_LT: return lhs < rhs; case C_LE: return lhs <= rhs;
        case C_GT: return lhs > rhs; case C_GE: return lhs >= rhs;
        default: return false;
    }
}
BKG_ABI_HD void reset_runtime(const PolicyView& p, RuntimeState& s) {
    for (int i = 0; i < MAX_BB; ++i) s.blackboard[i] = (i < p.n_bb) ? p.bb[i].initial : 0;
    for (int i = 0; i < MAX_STATE; ++i) s.node_state[i] = NodeState{};
    for (int i = 0; i < MAX_UNITS; ++i) s.unit_claims[i] = UnitClaim{};
    for (int i = 0; i < MAX_MARKET_INTENTS; ++i) s.market_claims[i] = MarketClaim{};
    s.n_market = 0; s.emit_seq = 0; s.visits = 0; s.fault = 0; s.ticks = 0;
}

} // namespace bonsaikg

#undef BKG_ABI_HD
