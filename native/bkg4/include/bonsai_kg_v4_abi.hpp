#pragma once

#include <cstddef>
#include <cstdint>
#include <climits>

#ifdef __CUDACC__
#define BKG4_HD __host__ __device__ inline
#else
#define BKG4_HD inline
#endif

namespace bonsaikg4 {

constexpr uint16_t ABI_VERSION = 4;
constexpr uint16_t FEATURE_REGISTRY_VERSION = 1;
constexpr int BOARD_SIZE = 10;
constexpr int MAX_NODES = 65534;
constexpr int MAX_CHILD_REFS = 65533;
constexpr int MAX_DEPTH = 48;
constexpr int MAX_BB = 128;
constexpr int MAX_STATE = 512;
constexpr int MAX_TABLE = 65534;
constexpr int MAX_PARALLEL_CHILDREN = 16;
constexpr int MAX_UNITS = 40;
constexpr int MAX_ITEMS = 12;
constexpr int MAX_MARKET_INTENTS = 16;
constexpr int MAX_LIVE_INTENTS = 128;
constexpr int MAX_INSTRUCTION_BUDGET = 65534;
constexpr uint16_t NONE_U16 = 0xFFFFu;
constexpr uint32_t ACTIVE_BIT = 1u << 31;
constexpr uint32_t RUNNING_MASK16 = 0xFFFFu;
constexpr uint8_t INTENT_MARKET_BIT = 0x80u;

enum Status : uint8_t { SUCCESS = 0, FAILURE = 1, RUNNING = 2, SKIPPED = 3 };

enum NodeType : uint8_t {
    N_SEQUENCE = 0, N_SELECTOR, N_REACTIVE_SEQUENCE, N_REACTIVE_SELECTOR, N_IF,
    N_PARALLEL, N_INVERT, N_FORCE_SUCCESS, N_FORCE_FAILURE, N_TURN_TIMEOUT,
    N_RETRY, N_REPEAT, N_KEEP_RUNNING_UNTIL_FAILURE, N_SUCCESS, N_FAILURE,
    N_CONDITION, N_SET_BLACKBOARD, N_ADD_BLACKBOARD, N_EMIT_UNIT_INTENT,
    N_EMIT_MARKET_INTENT, N_WAIT_TURNS, N_MOVE_UNIT_TO, N_SWITCH_ENUM, N_REACTIVE_IF
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
    F_OWN_TILE_FERT_AVAILABLE, F_OPP_TILE_FERT_AVAILABLE,
    F_OPP_HIRES_TODAY, F_OWN_QUADRANT_UNLOCKED, F_OPP_QUADRANT_UNLOCKED,
    F_OWN_WORKER_TILE, F_OPP_WORKER_TILE,
    F_OWN_TILE_PLANTED_DAY, F_OPP_TILE_PLANTED_DAY,
    F_OWN_TILE_CONSECUTIVE_UNWATERED, F_OPP_TILE_CONSECUTIVE_UNWATERED,
    F_OWN_TILE_MAX_LIFESPAN_STEP, F_OPP_TILE_MAX_LIFESPAN_STEP,
    F_OWN_TILE_FERTILIZED_UNTIL_DAY, F_OPP_TILE_FERTILIZED_UNTIL_DAY,
    F_OWN_TILE_PLACED_DAY, F_OPP_TILE_PLACED_DAY,
    F_OWN_TILE_CONSECUTIVE_UNFED, F_OPP_TILE_CONSECUTIVE_UNFED,
    F_OWN_TILE_PENDING_CARE_BONUS, F_OPP_TILE_PENDING_CARE_BONUS,
    F_OWN_TILE_HAS_ANIMAL, F_OPP_TILE_HAS_ANIMAL
};
constexpr uint16_t LAST_FEATURE_V1 = F_OPP_TILE_HAS_ANIMAL;

enum Compare : uint8_t { C_EQ = 0, C_NE, C_LT, C_LE, C_GT, C_GE };
enum EffectOp : uint8_t { E_SET = 0, E_CLEAR, E_ADD };
enum BbType : uint8_t { B_BOOL = 0, B_I32, B_U16, B_ENUM8, B_TILE, B_UNIT, B_ITEM };
enum ValueKind : uint8_t { V_CONST = 0, V_BB, V_LEGAL_FEATURE };
enum FeatureArgKind : uint8_t { A_CONST = 0, A_BB = 1 };
enum Authority : uint8_t {
    AUTH_FALLBACK = 0, AUTH_TACTICAL = 1, AUTH_DOMAIN = 2,
    AUTH_COMMITTED_STRATEGY = 3, AUTH_RECOVERY = 4, AUTH_INVARIANT = 5
};
constexpr uint8_t MAX_AUTHORITY = AUTH_INVARIANT;

enum UnitOp : uint8_t {
    U_PASS = 0, U_NORTH, U_SOUTH, U_EAST, U_WEST, U_PICKUP, U_DROP, U_PLACE,
    U_PLANT, U_WATER, U_HARVEST, U_FERTILIZE, U_DIG, U_BUILD_COOP,
    U_BUILD_PASTURE, U_FEED, U_COLLECT_FERTILIZER, U_CARE
};
enum MarketOp : uint8_t { M_NONE = 0, M_HIRE, M_BUY_LAND, M_BUY_SEED, M_BUY_PRODUCT, M_BUY_ANIMAL, M_SELL };

struct Header4 {
    char magic[4];
    uint16_t abi_version;
    uint16_t header_size;
    uint8_t hash[32];
    uint16_t n_nodes;
    uint16_t n_state;
    uint16_t n_bb;
    uint16_t n_values;
    uint16_t n_feature_args;
    uint16_t n_predicates;
    uint16_t n_lifecycles;
    uint16_t n_effects;
    uint16_t instruction_budget;
    uint16_t max_live_intents;
    uint16_t max_depth;
    uint16_t feature_registry_version;
};
static_assert(sizeof(Header4) == 64, "BKG4 Header ABI drift");

struct Node4 {
    uint8_t type = 0;
    uint8_t flags = 0;
    uint16_t first_child = 0;
    uint16_t child_count = 0;
    uint16_t state_slot = NONE_U16;
    uint16_t lifecycle_slot = NONE_U16;
    uint16_t subtree_end = 0;
    int32_t p0 = 0;
    int32_t p1 = 0;
    int32_t p2 = 0;
    int32_t p3 = 0;
    uint32_t aux = 0;
};
static_assert(sizeof(Node4) == 32, "BKG4 Node ABI drift");

struct BbDecl4 {
    uint8_t type = B_I32;
    uint8_t enum_size = 0;
    uint16_t enum_tag = 0;
    int32_t initial = 0;
};
static_assert(sizeof(BbDecl4) == 8, "BKG4 BbDecl ABI drift");

struct ValueRef4 { uint8_t kind = 0, type = B_I32; uint16_t id = 0; int32_t arg = 0, value = 0; };
static_assert(sizeof(ValueRef4) == 12, "BKG4 ValueRef ABI drift");
struct FeatureArgRef4 { uint8_t kind = 0, type = B_I32; uint16_t id = 0; int32_t value = 0; };
static_assert(sizeof(FeatureArgRef4) == 8, "BKG4 FeatureArgRef ABI drift");
struct Predicate4 { uint16_t lhs = 0, rhs = 0; uint8_t cmp = 0; uint8_t pad[7] = {0,0,0,0,0,0,0}; };
static_assert(sizeof(Predicate4) == 12, "BKG4 Predicate ABI drift");
struct Lifecycle4 {
    uint16_t failure_if = NONE_U16, success_if = NONE_U16, skip_if = NONE_U16, while_pred = NONE_U16;
    uint16_t on_success_start = 0, on_success_count = 0;
    uint16_t on_failure_start = 0, on_failure_count = 0;
    uint16_t on_halted_start = 0, on_halted_count = 0;
    uint16_t post_start = 0, post_count = 0;
};
static_assert(sizeof(Lifecycle4) == 24, "BKG4 Lifecycle ABI drift");
struct Effect4 { uint8_t op = 0, slot = 0; uint16_t reserved = 0; int32_t value = 0; };
static_assert(sizeof(Effect4) == 8, "BKG4 Effect ABI drift");

struct PolicyView4 {
    const Node4* nodes = nullptr;
    const uint16_t* child_refs = nullptr;
    const BbDecl4* bb = nullptr;
    const ValueRef4* values = nullptr;
    const FeatureArgRef4* feature_args = nullptr;
    const Predicate4* predicates = nullptr;
    const Lifecycle4* lifecycles = nullptr;
    const Effect4* effects = nullptr;
    uint16_t n_nodes = 0, n_state = 0, n_bb = 0, n_values = 0, n_feature_args = 0;
    uint16_t n_predicates = 0, n_lifecycles = 0, n_effects = 0;
    uint16_t instruction_budget = 0, max_live_intents = 0, max_depth = 0;
    uint16_t feature_registry_version = FEATURE_REGISTRY_VERSION;
};

struct NodeState4 { uint32_t a = 0, b = 0; };
static_assert(sizeof(NodeState4) == 8, "BKG4 NodeState size drift");
struct RuntimeMeta4 { uint16_t fault = 0, reserved = 0; uint32_t ticks = 0; };
static_assert(sizeof(RuntimeMeta4) == 8, "BKG4 RuntimeMeta size drift");

// RuntimeView4 points at ACTUAL-sized storage.  No BKG4 per-environment structure
// contains arrays proportional to MAX_NODES, MAX_STATE or MAX_BB.
struct RuntimeView4 {
    int32_t* blackboard = nullptr;
    NodeState4* node_state = nullptr;
    RuntimeMeta4* meta = nullptr;
};

struct DeferredIntent4 {
    uint16_t priority_key = 0;
    uint16_t source = 0;
    uint16_t order = 0;
    uint16_t quantity = 0;
    uint8_t op = 0;
    uint8_t arg = 0;
    uint8_t subject_kind = 0;
    uint8_t reserved = 0;
};
static_assert(sizeof(DeferredIntent4) == 12, "BKG4 DeferredIntent size drift");
struct UnitClaim4 { uint8_t present = 0, op = U_PASS, arg = 0, pad = 0; int16_t quantity = 1; uint16_t priority_key = 0, order = 0, reserved = 0; };
static_assert(sizeof(UnitClaim4) == 12, "BKG4 UnitClaim size drift");
struct MarketClaim4 { uint16_t priority_key = 0, order = 0; uint8_t op = M_NONE, item = 0; uint16_t pad = 0; int32_t quantity = 0; };
static_assert(sizeof(MarketClaim4) == 12, "BKG4 MarketClaim size drift");

struct UnitIntentAction4 { uint8_t op = U_PASS, arg = 0; int16_t n = 1; };
static_assert(sizeof(UnitIntentAction4) == 4, "BKG4 UnitIntentAction drift");
struct MarketIntentAction4 { uint8_t op = M_NONE, item = 0; uint16_t pad = 0; int32_t n = 0; };
static_assert(sizeof(MarketIntentAction4) == 8, "BKG4 MarketIntentAction drift");
struct ActionResult4 {
    UnitIntentAction4 units[MAX_UNITS];
    int32_t n_units = 0;
    MarketIntentAction4 market[MAX_MARKET_INTENTS];
    int32_t n_market = 0;
};

struct TraceEvent4 { int16_t step = 0; uint16_t node = 0; uint8_t event = 0, status = 0; uint16_t pad = 0; };
static_assert(sizeof(TraceEvent4) == 8, "BKG4 TraceEvent drift");
struct TraceBuffer4 { TraceEvent4* events = nullptr; uint16_t capacity = 0, count = 0; };

BKG4_HD bool valid_node_type(uint8_t t) { return t <= N_REACTIVE_IF; }
BKG4_HD bool is_decorator(uint8_t t) {
    return t == N_INVERT || t == N_FORCE_SUCCESS || t == N_FORCE_FAILURE ||
           t == N_TURN_TIMEOUT || t == N_RETRY || t == N_REPEAT || t == N_KEEP_RUNNING_UNTIL_FAILURE;
}
BKG4_HD bool valid_bb_value(uint8_t type, int32_t value) {
    if (type == B_BOOL) return value == 0 || value == 1;
    if (type == B_I32) return true;
    if (type == B_U16) return value >= 0 && value <= 65535;
    if (type == B_ENUM8) return value >= 0 && value <= 255;
    if (type == B_TILE) return value >= -1 && value < BOARD_SIZE * BOARD_SIZE;
    if (type == B_UNIT) return value >= -1 && value < MAX_UNITS;
    if (type == B_ITEM) return value >= -1 && value < MAX_ITEMS;
    return false;
}
BKG4_HD uint8_t value_type_for_bb(const BbDecl4& b) {
    return b.type == B_TILE || b.type == B_UNIT || b.type == B_ITEM ? b.type : static_cast<uint8_t>(B_I32);
}
BKG4_HD uint16_t value_enum_tag(const ValueRef4& v) {
    return (v.kind == V_CONST || v.kind == V_BB) && v.arg > 0 && v.arg <= 65535 ? static_cast<uint16_t>(v.arg) : 0;
}
BKG4_HD uint16_t make_priority_key(uint8_t authority, uint8_t local_priority) {
    return static_cast<uint16_t>((static_cast<uint16_t>(authority) << 8) | local_priority);
}
BKG4_HD bool valid_priority_key(uint32_t key) { return key <= 0xFFFFu && ((key >> 8) & 0xFFu) <= MAX_AUTHORITY; }
BKG4_HD bool compare_i32(int32_t lhs, int32_t rhs, uint8_t c) {
    switch(c) {
        case C_EQ:return lhs==rhs;case C_NE:return lhs!=rhs;case C_LT:return lhs<rhs;
        case C_LE:return lhs<=rhs;case C_GT:return lhs>rhs;case C_GE:return lhs>=rhs;default:return false;
    }
}
BKG4_HD int32_t clamp_i32_add(int32_t a, int32_t b, bool& ok) {
    const int64_t v=static_cast<int64_t>(a)+static_cast<int64_t>(b);
    if(v<INT32_MIN||v>INT32_MAX){ok=false;return 0;}ok=true;return static_cast<int32_t>(v);
}
BKG4_HD void reset_runtime(const PolicyView4& p, RuntimeView4 s) {
    if(!s.meta)return;
    for(uint16_t i=0;i<p.n_bb;++i)s.blackboard[i]=p.bb[i].initial;
    for(uint16_t i=0;i<p.n_state;++i)s.node_state[i]=NodeState4{};
    s.meta->fault=0;s.meta->reserved=0;s.meta->ticks=0;
}

} // namespace bonsaikg4

#undef BKG4_HD
