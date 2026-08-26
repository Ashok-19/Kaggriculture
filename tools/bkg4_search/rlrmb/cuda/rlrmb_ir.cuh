#pragma once

#include <cstdint>

namespace rlrmb {

constexpr uint32_t IR_MAGIC = 0x524c5234u;  // "RLR4"
constexpr uint32_t IR_VERSION = 1;
constexpr int MAX_PHASES = 4;

enum Source : uint8_t { SRC_PARENT=0, SRC_DONOR_BLOCK=1, SRC_SEMANTIC_SPLICE=2, SRC_PRIMITIVE=3 };
enum FamilyMask : uint8_t { MASK_FARMER=1, MASK_HANDS=2, MASK_MARKET=4 };
enum Termination : uint8_t { TERM_FIXED_CAP=0, TERM_NEXT_EOD=1, TERM_NEXT_SHOP=2, TERM_PURCHASE=3, TERM_PUBLIC_STATE=4, TERM_PARENT_COMPATIBLE=5, TERM_MAX_STEPS=6 };
enum Primitive : uint8_t { PRIM_NONE=0, PRIM_LIQUIDITY_REBALANCE=1 };
enum Purchase : uint8_t { BUY_NONE=0, BUY_LAND=1, HIRE=2, BUY_ANIMAL=3, BUY_SEED=4, BUY_PRODUCT=5 };
enum Fault : uint16_t { FAULT_NONE=0, FAULT_VERSION=1, FAULT_PHASE_COUNT=2, FAULT_SOURCE=3, FAULT_MOTIF=4, FAULT_PRIMITIVE=5, FAULT_TERMINATION=6, FAULT_MOTIF_EXHAUSTED=7, FAULT_FEATURE=8, FAULT_HORIZON=9, FAULT_INTERNAL=10 };
enum EndReason : uint8_t { END_NONE=0, END_FIXED_CAP=1, END_EOD=2, END_SHOP=3, END_PURCHASE=4, END_PUBLIC_STATE=5, END_PARENT_COMPATIBLE=6, END_MAX_STEPS=7, END_PROGRAM_CAP=8 };

#pragma pack(push,1)
struct FileHeader {
    uint32_t magic, version, programs, phases, motifs, actions;
};
struct ProgramRecord {
    uint16_t initiation_step, max_total_steps;
    uint32_t phase_first;
    uint8_t phase_count, lane, fallback, reserved;
    uint8_t semantic_id[32];
};
struct PhaseRecord {
    uint8_t source, mask, termination, primitive;
    uint16_t max_steps, motif_index, feature, comparator;
    int32_t condition_index, condition_value;
    uint8_t purchase, primitive_item, primitive_enabled, reserved;
    int32_t target_count, primitive_quantity;
};
struct MotifRecord { uint32_t action_first; uint16_t action_count; uint8_t family_mask, reserved; };
struct UnitRecord { uint8_t op, arg; int16_t n; };
struct OrderRecord { uint8_t op, item; int32_t n; };
struct ActionRecord {
    UnitRecord units[40];
    OrderRecord orders[10];
    uint8_t n_units, n_orders;
};
struct ResultRecord {
    int32_t own_money, opp_money;
    uint16_t action_changes, fault;
    uint8_t activated, final_phase, end_reason, complete;
    uint8_t semantic_id[32];
};
#pragma pack(pop)

static_assert(sizeof(ProgramRecord)==44, "program ABI");
static_assert(sizeof(PhaseRecord)==32, "phase ABI");
static_assert(sizeof(MotifRecord)==8, "motif ABI");
static_assert(sizeof(UnitRecord)==4, "unit ABI");
static_assert(sizeof(OrderRecord)==6, "order ABI");
static_assert(sizeof(ActionRecord)==222, "action ABI");
static_assert(sizeof(ResultRecord)==48, "result ABI");

struct RuntimeState {
    uint16_t program_steps=0, phase_steps=0;
    uint8_t phase=0, active=0, complete=0, end_reason=0;
    uint8_t entry_day=0, entry_shops=0;
    int32_t entry_purchase=0;
    uint16_t fault=0;
};

} // namespace rlrmb
