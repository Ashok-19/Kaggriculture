#pragma once

#include "bonsai_kg_abi.hpp"

#ifdef __CUDACC__
#define BKG_HD __host__ __device__ inline
#define BKG_HD_NOINLINE __host__ __device__ __noinline__
#else
#define BKG_HD inline
#define BKG_HD_NOINLINE inline
#endif

namespace bonsaikg {

struct EvalFrame {
    uint16_t node = 0;
    uint16_t cursor = 0;
    uint16_t aux = 0;
    uint16_t intent_start = 0;
    uint8_t phase = 0;
    uint8_t skipped = 0;
};
static_assert(sizeof(EvalFrame) == 10, "EvalFrame size drift");

struct HaltFrame { uint16_t node = 0, cursor = 0; uint8_t entered = 0, pad[3] = {0,0,0}; };
static_assert(sizeof(HaltFrame) == 8, "HaltFrame size drift");

template <class Obs, int INTENT_CAP = MAX_DEFERRED_INTENTS>
struct Evaluator {
    const PolicyView& p;
    RuntimeState& s;
    const Obs& obs;
    TraceBuffer* trace;
    DeferredIntent intents[INTENT_CAP > 0 ? INTENT_CAP : 1];
    uint16_t n_intents = 0;
    uint16_t emit_seq = 0;
    EvalFrame stack[MAX_DEPTH];
    int sp = 0;
    Status last = FAILURE;
    bool have_last = false;

    BKG_HD Evaluator(const PolicyView& policy, RuntimeState& state, const Obs& observation, TraceBuffer* tr)
        : p(policy), s(state), obs(observation), trace(tr) {}

    BKG_HD int32_t feature(uint16_t id, int32_t arg) const { return obs.feature(id, arg); }
    BKG_HD void set_fault(uint16_t code) { if (!s.fault) s.fault = code; }

    BKG_HD void emit_trace(uint16_t node, Status status, uint8_t event) {
        if (!trace || !trace->events || trace->count >= trace->capacity) return;
        TraceEvent& e = trace->events[trace->count++];
        int32_t st = feature(F_STEP, 0);
        if (st < -32768) st = -32768;
        if (st > 32767) st = 32767;
        e.step = static_cast<int16_t>(st); e.node = static_cast<uint16_t>(p.nodes[node].debug_id);
        e.event = event; e.status = static_cast<uint8_t>(status); e.pad = 0;
    }

    BKG_HD bool active(const Node& n) const {
        return n.state_slot != NONE_U16 && (s.node_state[n.state_slot].b & ACTIVE_BIT) != 0;
    }
    BKG_HD void set_active(const Node& n, bool value) {
        if (n.state_slot == NONE_U16) return;
        if (value) s.node_state[n.state_slot].b |= ACTIVE_BIT;
        else s.node_state[n.state_slot].b &= ~ACTIVE_BIT;
    }
    BKG_HD uint16_t child(const Node& n, uint16_t i) const { return p.child_refs[n.first_child + i]; }

    BKG_HD int32_t value(uint16_t idx) const {
        if (idx >= p.n_values) return 0;
        const ValueRef& v = p.values[idx];
        if (v.kind == V_CONST) return v.value;
        if (v.kind == V_BB) return v.id < p.n_bb ? s.blackboard[v.id] : 0;
        return feature(v.id, v.arg);
    }

    BKG_HD bool pred(uint16_t idx) const {
        if (idx == NONE_U16 || idx >= p.n_predicates) return false;
        const Predicate& q = p.predicates[idx];
        return compare_i32(value(q.lhs), value(q.rhs), q.cmp);
    }

    BKG_HD void effects(uint16_t start, uint16_t count) {
        for (uint16_t k = 0; k < count; ++k) {
            const uint32_t idx = static_cast<uint32_t>(start) + k;
            if (s.fault) return;
            if (idx >= p.n_effects) { set_fault(3); return; }
            const Effect& e = p.effects[idx];
            if (e.slot >= p.n_bb) { set_fault(3); return; }
            if (e.op == E_SET) {
                const int32_t v = value(static_cast<uint16_t>(e.value));
                if (!valid_bb_value(p.bb[e.slot].type, v)) { set_fault(5); return; }
                s.blackboard[e.slot] = v;
            } else if (e.op == E_CLEAR) s.blackboard[e.slot] = p.bb[e.slot].initial;
            else if (e.op == E_ADD) {
                bool ok = false; const int32_t v = clamp_i32_add(s.blackboard[e.slot], value(static_cast<uint16_t>(e.value)), ok);
                if (!ok) { set_fault(2); return; } s.blackboard[e.slot] = v;
            } else { set_fault(3); return; }
        }
    }

    BKG_HD void halt_node(uint16_t root) {
        HaltFrame hs[MAX_DEPTH]; int hp = 0;
        hs[hp++] = HaltFrame{root, 0, 0, {0,0,0}};
        while (hp > 0) {
            HaltFrame& f = hs[hp - 1];
            if (f.node >= p.n_nodes) { set_fault(3); --hp; continue; }
            const Node& n = p.nodes[f.node];
            NodeState* st = n.state_slot != NONE_U16 && n.state_slot < p.n_state ? &s.node_state[n.state_slot] : nullptr;
            if (!f.entered) { f.entered = 1; f.cursor = 0; }
            int next = -1;
            if (n.type == N_SEQUENCE || n.type == N_SELECTOR) {
                if (f.cursor == 0 && st && st->a < n.child_count) { next = child(n, static_cast<uint16_t>(st->a)); f.cursor = 1; }
            } else if (n.type == N_REACTIVE_SEQUENCE || n.type == N_REACTIVE_SELECTOR) {
                if (f.cursor == 0 && st && st->a > 0 && st->a - 1 < n.child_count) { next = child(n, static_cast<uint16_t>(st->a - 1)); f.cursor = 1; }
            } else if (n.type == N_IF) {
                if (f.cursor == 0 && st && n.child_count == 3) {
                    const uint32_t phase = st->a; const uint16_t ci = phase == 0 ? 0 : static_cast<uint16_t>(phase);
                    if (ci < 3) next = child(n, ci);
                    f.cursor = 1;
                }
            } else if (n.type == N_PARALLEL) {
                const uint32_t running = st ? (st->b & RUNNING_MASK16) : 0;
                while (f.cursor < n.child_count && !(running & (1u << f.cursor))) ++f.cursor;
                if (f.cursor < n.child_count) next = child(n, f.cursor++);
            } else if (is_decorator(n.type)) {
                if (f.cursor == 0 && n.child_count == 1) { next = child(n, 0); f.cursor = 1; }
            }
            if (next >= 0) {
                if (hp >= MAX_DEPTH) { set_fault(1); return; }
                hs[hp++] = HaltFrame{static_cast<uint16_t>(next), 0, 0, {0,0,0}};
                continue;
            }
            const bool was = active(n);
            if (st) { st->a = 0; st->b = 0; }
            if (was && n.lifecycle_slot != NONE_U16 && n.lifecycle_slot < p.n_lifecycles) {
                const Lifecycle& lc = p.lifecycles[n.lifecycle_slot]; effects(lc.on_halted_start, lc.on_halted_count);
            }
            if (was) emit_trace(f.node, SKIPPED, 2);
            --hp;
        }
    }

    BKG_HD void finish(Status r, uint8_t event = 0) {
        const EvalFrame frame = stack[sp - 1];
        const uint16_t idx = frame.node;
        const Node& n = p.nodes[idx];
        if (r == RUNNING) set_active(n, true);
        else {
            set_active(n, false);
            if (n.lifecycle_slot != NONE_U16 && n.lifecycle_slot < p.n_lifecycles) {
                const Lifecycle& lc = p.lifecycles[n.lifecycle_slot];
                if (r == SUCCESS) { effects(lc.on_success_start, lc.on_success_count); effects(lc.post_start, lc.post_count); }
                else if (r == FAILURE) { effects(lc.on_failure_start, lc.on_failure_count); effects(lc.post_start, lc.post_count); }
            }
        }
        if (s.fault || r == FAILURE || r == SKIPPED) n_intents = frame.intent_start;
        emit_trace(idx, r, event);
        --sp; last = r; have_last = true;
    }

    BKG_HD bool push(uint16_t idx) {
        ++s.visits;
        if (s.visits > p.instruction_budget || sp >= MAX_DEPTH || idx >= p.n_nodes) {
            set_fault((idx >= p.n_nodes) ? 3 : 1);
            last = FAILURE; have_last = true; return false;
        }
        stack[sp++] = EvalFrame{idx, 0, 0, n_intents, 0, 0};
        have_last = false; return true;
    }

    BKG_HD void emit_unit(int unit, int op, int arg, int quantity, int priority) {
        if (unit < 0 || unit >= MAX_UNITS) return;
        const uint16_t order = emit_seq++;
        if (INTENT_CAP == 0 || n_intents >= INTENT_CAP) { set_fault(INTENT_CAP == 0 ? 3 : 4); return; }
        intents[n_intents++] = DeferredIntent{0, static_cast<uint8_t>(priority), static_cast<uint8_t>(op),
                                               static_cast<uint8_t>(arg), static_cast<uint16_t>(quantity),
                                               static_cast<uint8_t>(unit), static_cast<uint8_t>(order)};
    }

    BKG_HD void emit_market(int op, int item, int quantity, int priority) {
        const uint16_t order = emit_seq++;
        if (INTENT_CAP == 0 || n_intents >= INTENT_CAP) { set_fault(INTENT_CAP == 0 ? 3 : 4); return; }
        intents[n_intents++] = DeferredIntent{1, static_cast<uint8_t>(priority), static_cast<uint8_t>(op),
                                               static_cast<uint8_t>(item), static_cast<uint16_t>(quantity), 0,
                                               static_cast<uint8_t>(order)};
    }

    BKG_HD void push_parallel_next(EvalFrame& f, const Node& n) {
        NodeState& st = s.node_state[n.state_slot];
        const uint32_t success = st.a & 0xFFFFu, failure = (st.a >> 16) & 0xFFFFu;
        while (f.cursor < n.child_count) {
            const uint32_t bit = 1u << f.cursor;
            if (!(success & bit) && !(failure & bit)) { const uint16_t c = child(n, f.cursor); push(c); return; }
            ++f.cursor;
        }
        const uint32_t running = st.b & RUNNING_MASK16;
        if (running) { finish(RUNNING); return; }
        int sc = 0, fc = 0;
        for (uint16_t i = 0; i < n.child_count; ++i) { sc += (success >> i) & 1u; fc += (failure >> i) & 1u; }
        const bool all_skipped = sc == 0 && fc == 0 && f.skipped == n.child_count;
        st.a = 0; st.b = 0;
        finish(all_skipped ? SKIPPED : FAILURE);
    }

    BKG_HD Status run(uint16_t root) {
        push(root);
        while (sp > 0) {
            if (s.fault) { n_intents = 0; sp = 0; last = FAILURE; have_last = true; break; }
            EvalFrame& f = stack[sp - 1];
            const Node& n = p.nodes[f.node];
            NodeState* st = n.state_slot != NONE_U16 ? &s.node_state[n.state_slot] : nullptr;
            if (f.phase == 0) {
                if (n.lifecycle_slot != NONE_U16) {
                    if (n.lifecycle_slot >= p.n_lifecycles) { set_fault(3); finish(FAILURE); continue; }
                    const Lifecycle& lc = p.lifecycles[n.lifecycle_slot]; const bool act = active(n);
                    if (!act) {
                        if (lc.failure_if != NONE_U16 && pred(lc.failure_if)) { finish(FAILURE, 1); continue; }
                        if (lc.success_if != NONE_U16 && pred(lc.success_if)) { finish(SUCCESS, 1); continue; }
                        if (lc.skip_if != NONE_U16 && pred(lc.skip_if)) { finish(SKIPPED, 1); continue; }
                        if (lc.while_pred != NONE_U16 && !pred(lc.while_pred)) { finish(SKIPPED, 1); continue; }
                    } else if (lc.while_pred != NONE_U16 && !pred(lc.while_pred)) {
                        halt_node(f.node); finish(SKIPPED, 3); continue;
                    }
                }
                switch (n.type) {
                    case N_SEQUENCE: {
                        f.cursor = static_cast<uint16_t>(st->a); f.phase = 1;
                        if (f.cursor >= n.child_count) { st->a = 0; finish(SUCCESS); }
                        else push(child(n, f.cursor));
                        break;
                    }
                    case N_SELECTOR: {
                        f.cursor = static_cast<uint16_t>(st->a); f.phase = 2;
                        if (f.cursor >= n.child_count) { st->a = 0; finish(FAILURE); }
                        else push(child(n, f.cursor));
                        break;
                    }
                    case N_REACTIVE_SEQUENCE: f.aux = static_cast<uint16_t>(st->a); f.cursor = 0; f.phase = 3; push(child(n, 0)); break;
                    case N_REACTIVE_SELECTOR: f.aux = static_cast<uint16_t>(st->a); f.cursor = 0; f.phase = 4; push(child(n, 0)); break;
                    case N_IF: {
                        f.aux = static_cast<uint16_t>(st->a);
                        if (st->a == 0) { f.phase = 5; push(child(n, 0)); }
                        else { f.phase = 6; push(child(n, static_cast<uint16_t>(st->a))); }
                        break;
                    }
                    case N_PARALLEL: f.cursor = 0; f.skipped = 0; f.phase = 7; push_parallel_next(f, n); break;
                    case N_INVERT: case N_FORCE_SUCCESS: case N_FORCE_FAILURE:
                        f.phase = 8; push(child(n, 0)); break;
                    case N_TURN_TIMEOUT: {
                        const int32_t now = feature(F_STEP, 0); const bool act = active(n);
                        if (!act) st->a = static_cast<uint32_t>(now);
                        else if (now - static_cast<int32_t>(st->a) >= n.p0) { halt_node(child(n, 0)); st->a = 0; finish(FAILURE); break; }
                        f.phase = 9; push(child(n, 0)); break;
                    }
                    case N_RETRY: f.phase = 10; push(child(n, 0)); break;
                    case N_REPEAT: f.phase = 11; push(child(n, 0)); break;
                    case N_KEEP_RUNNING_UNTIL_FAILURE: f.phase = 12; push(child(n, 0)); break;
                    case N_SUCCESS: finish(SUCCESS); break;
                    case N_FAILURE: finish(FAILURE); break;
                    case N_CONDITION: {
                        finish(compare_i32(value(static_cast<uint16_t>(n.p0)), value(static_cast<uint16_t>(n.p1)), n.flags) ? SUCCESS : FAILURE); break;
                    }
                    case N_SET_BLACKBOARD: {
                        const int32_t v = value(static_cast<uint16_t>(n.p1));
                        if (!valid_bb_value(p.bb[n.p0].type, v)) { set_fault(5); finish(FAILURE); }
                        else { s.blackboard[n.p0] = v; finish(SUCCESS); }
                        break;
                    }
                    case N_ADD_BLACKBOARD: {
                        bool ok = false; const int32_t v = clamp_i32_add(s.blackboard[n.p0], value(static_cast<uint16_t>(n.p1)), ok);
                        if (!ok) { set_fault(2); finish(FAILURE); } else { s.blackboard[n.p0] = v; finish(SUCCESS); }
                        break;
                    }
                    case N_EMIT_UNIT_INTENT:
                        if (n.p0 >= feature(F_OWN_UNITS, 0)) finish(FAILURE);
                        else { emit_unit(n.p0, n.p1, n.p2, n.p3, n.flags); finish(SUCCESS); }
                        break;
                    case N_EMIT_MARKET_INTENT: emit_market(n.p0, n.p1, n.p2, n.p3); finish(SUCCESS); break;
                    case N_WAIT_TURNS: {
                        const int32_t now = feature(F_STEP, 0); const bool act = active(n);
                        if (!act) { st->a = static_cast<uint32_t>(now); finish(RUNNING); }
                        else if (now - static_cast<int32_t>(st->a) >= n.p0) { st->a = 0; finish(SUCCESS); }
                        else finish(RUNNING);
                        break;
                    }
                    case N_MOVE_UNIT_TO: {
                        const int unit = value(static_cast<uint16_t>(n.p0));
                        const int target = value(static_cast<uint16_t>(n.p1));
                        if (unit < 0 || target < 0 || target >= BOARD_SIZE * BOARD_SIZE || unit >= feature(F_OWN_UNITS, 0)) { finish(FAILURE); break; }
                        const int x = feature(F_OWN_WORKER_X, unit), y = feature(F_OWN_WORKER_Y, unit);
                        const int tx = target % BOARD_SIZE, ty = target / BOARD_SIZE;
                        if (x == tx && y == ty) { finish(SUCCESS); break; }
                        const int op = x < tx ? U_EAST : x > tx ? U_WEST : y < ty ? U_SOUTH : U_NORTH;
                        emit_unit(unit, op, 0, 1, n.p2); finish(RUNNING); break;
                    }
                    default: set_fault(3); finish(FAILURE); break;
                }
                continue;
            }
            if (!have_last) { set_fault(3); finish(FAILURE); continue; }
            const Status r = last; have_last = false;
            switch (f.phase) {
                case 1: // Sequence
                    if (r == SUCCESS || r == SKIPPED) {
                        ++f.cursor; st->a = f.cursor;
                        if (f.cursor >= n.child_count) { st->a = 0; finish(SUCCESS); }
                        else push(child(n, f.cursor));
                    } else if (r == RUNNING) { st->a = f.cursor; finish(RUNNING); }
                    else { st->a = 0; finish(FAILURE); }
                    break;
                case 2: // Selector
                    if (r == FAILURE || r == SKIPPED) {
                        ++f.cursor; st->a = f.cursor;
                        if (f.cursor >= n.child_count) { st->a = 0; finish(FAILURE); }
                        else push(child(n, f.cursor));
                    } else if (r == RUNNING) { st->a = f.cursor; finish(RUNNING); }
                    else { st->a = 0; finish(SUCCESS); }
                    break;
                case 3: { // Reactive Sequence
                    const int prev = f.aux ? static_cast<int>(f.aux) - 1 : -1;
                    if (r == SUCCESS || r == SKIPPED) {
                        ++f.cursor;
                        if (f.cursor >= n.child_count) { if (prev >= 0) halt_node(child(n, static_cast<uint16_t>(prev))); st->a = 0; finish(SUCCESS); }
                        else push(child(n, f.cursor));
                    } else {
                        if (prev >= 0 && prev != f.cursor) halt_node(child(n, static_cast<uint16_t>(prev)));
                        st->a = r == RUNNING ? static_cast<uint32_t>(f.cursor + 1) : 0; finish(r);
                    }
                    break;
                }
                case 4: { // Reactive Selector
                    const int prev = f.aux ? static_cast<int>(f.aux) - 1 : -1;
                    if (r == FAILURE || r == SKIPPED) {
                        ++f.cursor;
                        if (f.cursor >= n.child_count) { if (prev >= 0) halt_node(child(n, static_cast<uint16_t>(prev))); st->a = 0; finish(FAILURE); }
                        else push(child(n, f.cursor));
                    } else {
                        if (prev >= 0 && prev != f.cursor) halt_node(child(n, static_cast<uint16_t>(prev)));
                        st->a = r == RUNNING ? static_cast<uint32_t>(f.cursor + 1) : 0; finish(r == RUNNING ? RUNNING : SUCCESS);
                    }
                    break;
                }
                case 5: // If condition
                    if (r == RUNNING) finish(RUNNING);
                    else if (r == SKIPPED) { st->a = 0; finish(SKIPPED); }
                    else { st->a = r == SUCCESS ? 1u : 2u; f.phase = 6; push(child(n, static_cast<uint16_t>(st->a))); }
                    break;
                case 6: if (r != RUNNING) st->a = 0; finish(r); break;
                case 7: { // Parallel
                    const uint32_t bit = 1u << f.cursor;
                    uint32_t success = st->a & 0xFFFFu, failure = (st->a >> 16) & 0xFFFFu, running = st->b & RUNNING_MASK16;
                    if (r == SUCCESS) { success |= bit; running &= ~bit; }
                    else if (r == FAILURE) { failure |= bit; running &= ~bit; }
                    else if (r == RUNNING) running |= bit;
                    else { ++f.skipped; running &= ~bit; }
                    st->a = success | (failure << 16); st->b = (st->b & ACTIVE_BIT) | running;
                    int sc = 0, fc = 0; for (uint16_t i = 0; i < n.child_count; ++i) { sc += (success >> i) & 1u; fc += (failure >> i) & 1u; }
                    const int possible_successes = static_cast<int>(n.child_count) - fc - static_cast<int>(f.skipped);
                    if (sc >= n.p0 || fc >= n.p1 || possible_successes < n.p0) {
                        for (uint16_t i = 0; i < n.child_count; ++i) if (running & (1u << i)) halt_node(child(n, i));
                        const Status out = sc >= n.p0 ? SUCCESS : FAILURE; st->a = 0; st->b = 0; finish(out);
                    } else { ++f.cursor; push_parallel_next(f, n); }
                    break;
                }
                case 8:
                    if (n.type == N_INVERT) finish(r == SUCCESS ? FAILURE : r == FAILURE ? SUCCESS : r);
                    else if (r == RUNNING || r == SKIPPED) finish(r);
                    else finish(n.type == N_FORCE_SUCCESS ? SUCCESS : FAILURE);
                    break;
                case 9: if (r != RUNNING) st->a = 0; finish(r); break;
                case 10:
                    if (r == SUCCESS) { st->a = 0; finish(SUCCESS); }
                    else if (r == FAILURE) { ++st->a; if (st->a >= static_cast<uint32_t>(n.p0)) { st->a = 0; finish(FAILURE); } else finish(RUNNING); }
                    else finish(r);
                    break;
                case 11:
                    if (r == FAILURE) { st->a = 0; finish(FAILURE); }
                    else if (r == SUCCESS) { ++st->a; if (st->a >= static_cast<uint32_t>(n.p0)) { st->a = 0; finish(SUCCESS); } else finish(RUNNING); }
                    else finish(r);
                    break;
                case 12: finish(r == FAILURE ? SUCCESS : r == SUCCESS ? RUNNING : r); break;
                default: set_fault(3); finish(FAILURE); break;
            }
        }
        return last;
    }
};

template <class Obs>
BKG_HD Status tick_no_intents(const PolicyView& p, RuntimeState& s, const Obs& obs,
                              ActionResult& out, TraceBuffer* trace) {
    int n_units = obs.feature(F_OWN_UNITS, 0);
    if (n_units < 0) n_units = 0;
    if (n_units > MAX_UNITS) n_units = MAX_UNITS;
    out.n_units = n_units;
    for (int i = 0; i < n_units; ++i) out.units[i] = UnitIntentAction{};
    out.n_market = 0;
    s.visits = 0;
    if (trace) trace->count = 0;
    Evaluator<Obs, 0> ev(p, s, obs, trace);
    Status raw = FAILURE;
    if (!s.fault) raw = ev.run(p.root);
    if (ev.n_intents != 0) ev.set_fault(3);
    const Status status = s.fault ? FAILURE : raw == SKIPPED ? SUCCESS : raw;
    ++s.ticks;
    return status;
}

template <class Obs>
BKG_HD_NOINLINE Status tick_with_intents(const PolicyView& p, RuntimeState& s, const Obs& obs,
                                         int max_market_orders, ActionResult& out, TraceBuffer* trace) {
    int n_units = obs.feature(F_OWN_UNITS, 0);
    if (n_units < 0) n_units = 0;
    if (n_units > MAX_UNITS) n_units = MAX_UNITS;
    out.n_units = n_units;
    for (int i = 0; i < n_units; ++i) out.units[i] = UnitIntentAction{};
    out.n_market = 0;
    s.visits = 0;
    if (trace) trace->count = 0;
    Evaluator<Obs, MAX_DEFERRED_INTENTS> ev(p, s, obs, trace);
    Status raw = FAILURE;
    if (!s.fault) raw = ev.run(p.root);
    if (s.fault || raw == FAILURE || raw == SKIPPED) ev.n_intents = 0;

    UnitClaim unit_claims[MAX_UNITS]{};
    MarketClaim market_claims[MAX_MARKET_INTENTS]{};
    uint16_t n_market = 0;
    if (!s.fault && (raw == SUCCESS || raw == RUNNING)) {
        for (uint16_t i = 0; i < ev.n_intents; ++i) {
            const DeferredIntent& intent = ev.intents[i];
            if (intent.kind == 0) {
                if (intent.subject >= MAX_UNITS) continue;
                UnitClaim& old = unit_claims[intent.subject];
                if (!old.present || intent.priority > old.priority ||
                    (intent.priority == old.priority && intent.order < old.order)) {
                    old.present = 1; old.priority = intent.priority; old.op = intent.op; old.arg = intent.arg;
                    old.n = static_cast<int16_t>(intent.n); old.order = intent.order;
                }
            } else {
                const MarketClaim candidate{intent.priority, intent.op, intent.arg, 0, intent.n, intent.order, 0};
                int pos = n_market;
                if (pos < MAX_MARKET_INTENTS) ++n_market;
                else {
                    const MarketClaim& worst = market_claims[MAX_MARKET_INTENTS - 1];
                    if (candidate.priority < worst.priority ||
                        (candidate.priority == worst.priority && candidate.order >= worst.order)) continue;
                    pos = MAX_MARKET_INTENTS - 1;
                }
                while (pos > 0) {
                    const MarketClaim& prev = market_claims[pos - 1];
                    if (candidate.priority < prev.priority ||
                        (candidate.priority == prev.priority && candidate.order >= prev.order)) break;
                    market_claims[pos] = prev; --pos;
                }
                market_claims[pos] = candidate;
            }
        }
    }
    for (int i = 0; i < n_units; ++i) if (unit_claims[i].present) {
        const UnitClaim& c = unit_claims[i]; out.units[i] = UnitIntentAction{c.op, c.arg, c.n};
    }
    int cap = max_market_orders; if (cap < 0) cap = 0; if (cap > MAX_MARKET_INTENTS) cap = MAX_MARKET_INTENTS;
    out.n_market = n_market < cap ? n_market : cap;
    for (int i = 0; i < out.n_market; ++i) {
        const MarketClaim& c = market_claims[i]; out.market[i] = MarketIntentAction{c.op, c.item, 0, c.n};
    }
    const Status status = s.fault ? FAILURE : raw == SKIPPED ? SUCCESS : raw;
    ++s.ticks;
    return status;
}

template <class Obs>
BKG_HD Status tick(const PolicyView& p, RuntimeState& s, const Obs& obs, int max_market_orders,
                   ActionResult& out, TraceBuffer* trace = nullptr) {
    if (p.max_intents == 0) return tick_no_intents(p, s, obs, out, trace);
    return tick_with_intents(p, s, obs, max_market_orders, out, trace);
}

} // namespace bonsaikg

#undef BKG_HD
#undef BKG_HD_NOINLINE
