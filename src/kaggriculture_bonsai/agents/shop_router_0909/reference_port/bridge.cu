// SPDX-License-Identifier: Apache-2.0
#include "native.hpp"
#include "bonsai_kg_v4_io.hpp"
#include "extended_route_pack.hpp"
#include <memory>
#include <stdexcept>
#ifdef __CUDACC__
#include <cuda_runtime.h>
#endif

namespace shop_router_0909 {
thread_local std::string error;
#ifdef __CUDACC__
void check(cudaError_t e) { if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e)); }
__global__ void run(PolicyView4 policy, PackView pack, const Observation* obs, State* states, Result* out) {
    tick(policy, pack, *obs, states[obs->player], *out);
}
#endif
struct Handle {
    OwnedPolicy4 policy;
    bonsaikg_v6::RoutePack pack;
    std::vector<uint16_t> arity;
    std::unique_ptr<State[]> states;
    PolicyView4 device_policy{};
    PackView device_pack{};
    State* device_states = nullptr;
    Observation* device_obs = nullptr;
    Result* device_result = nullptr;
    std::vector<void*> allocations;
    bool gpu = false;
    template <class T> T* upload(const T* source, size_t count) {
#ifdef __CUDACC__
        if (!count) return nullptr;
        T* dest = nullptr; check(cudaMalloc(&dest, count * sizeof(T))); allocations.push_back(dest);
        if (source) check(cudaMemcpy(dest, source, count * sizeof(T), cudaMemcpyHostToDevice));
        else check(cudaMemset(dest, 0, count * sizeof(T)));
        return dest;
#else
        (void)source; (void)count; throw std::runtime_error("CUDA not compiled");
#endif
    }
    ~Handle() {
#ifdef __CUDACC__
        for (void* p : allocations) cudaFree(p);
#endif
    }
    void initialize(const char* controller, const char* routes, const char* pack_hash, const char* arity_path, bool use_gpu) {
        gpu = use_gpu;
        policy = load_compiled4(controller);
        // One slot is the exact parent; two adds the candidate's endgame gate.
        if (policy.header.n_bb < 1 || policy.header.n_bb > 2 || policy.header.n_state != 0)
            throw std::runtime_error("unexpected shop-router controller footprint");
        pack = bonsaikg4_champion::load_extended_route_pack(routes, pack_hash);
        const size_t actions = static_cast<size_t>(ROUTES) * STEPS;
        if (pack.routes.size() != ROUTES || pack.actions.size() != actions)
            throw std::runtime_error("unexpected shop-router routes");
        for (int i = 0; i < ROUTES; ++i)
            if (pack.routes[i].route_id != i || pack.routes[i].steps != STEPS || pack.routes[i].first != i * STEPS)
                throw std::runtime_error("unexpected shop-router route descriptor");
        arity.resize(actions);
        std::ifstream f(arity_path, std::ios::binary);
        f.read(reinterpret_cast<char*>(arity.data()), static_cast<std::streamsize>(actions * 2));
        if (!f || f.peek() != EOF) throw std::runtime_error("invalid arity table");
        states = std::make_unique<State[]>(2);
        if (gpu) {
            device_policy = policy.view();
#define UPLOAD(field) device_policy.field = upload(policy.field.data(), policy.field.size())
            UPLOAD(nodes); UPLOAD(child_refs); UPLOAD(bb); UPLOAD(values);
            UPLOAD(feature_args); UPLOAD(predicates); UPLOAD(lifecycles); UPLOAD(effects);
#undef UPLOAD
            device_pack = {upload(pack.actions.data(), pack.actions.size()), upload(arity.data(), arity.size())};
            device_states = upload(states.get(), 2);
            device_obs = upload<Observation>(nullptr, 1);
            device_result = upload<Result>(nullptr, 1);
        }
    }
    void act(const Observation& o, Result& out) {
        if (o.player < 0 || o.player > 1 || o.n_positions < 1 || o.n_positions > TAPE_UNITS)
            throw std::runtime_error("unsupported observation dimensions");
        if (!gpu) { tick(policy.view(), {pack.actions.data(), arity.data()}, o, states[o.player], out); return; }
#ifdef __CUDACC__
        check(cudaMemcpy(device_obs, &o, sizeof(o), cudaMemcpyHostToDevice));
        run<<<1, 1>>>(device_policy, device_pack, device_obs, device_states, device_result);
        check(cudaGetLastError());
        check(cudaMemcpy(&out, device_result, sizeof(out), cudaMemcpyDeviceToHost));
#else
        throw std::runtime_error("CUDA not compiled");
#endif
    }
};
}  // namespace shop_router_0909

extern "C" {
const char* shop_error() { return shop_router_0909::error.c_str(); }
void* shop_create(const char* controller, const char* routes, const char* pack_hash, const char* arity, int gpu) {
    try {
        auto h = std::make_unique<shop_router_0909::Handle>();
        h->initialize(controller, routes, pack_hash, arity, gpu != 0);
        return h.release();
    } catch (const std::exception& e) { shop_router_0909::error = e.what(); return nullptr; }
}
void shop_destroy(void* p) { delete static_cast<shop_router_0909::Handle*>(p); }
int shop_tick(void* p, const shop_router_0909::Observation* o, shop_router_0909::Result* out) {
    try { static_cast<shop_router_0909::Handle*>(p)->act(*o, *out); return 0; }
    catch (const std::exception& e) { shop_router_0909::error = e.what(); return -1; }
}
int shop_queues(void* p, int player, int* buffer, int capacity) {
    try {
        auto& h = *static_cast<shop_router_0909::Handle*>(p);
        if (player < 0 || player > 1) throw std::runtime_error("invalid player");
        auto snapshot = std::make_unique<shop_router_0909::State>();
        const shop_router_0909::State* s = &h.states[player];
#ifdef __CUDACC__
        if (h.gpu) { shop_router_0909::check(cudaMemcpy(snapshot.get(), h.device_states + player, sizeof(*snapshot), cudaMemcpyDeviceToHost)); s = snapshot.get(); }
#endif
        int size = 0;
        for (int i = 0; i < shop_router_0909::UNITS; ++i)
            for (int j = 0; j < s->counts[i]; ++j) {
                if (size + 4 > capacity) throw std::runtime_error("queue snapshot buffer too small");
                const auto& u = s->queue[i][(s->heads[i] + j) % shop_router_0909::QUEUE];
                for (int v : {u.op, u.arg, u.n, u.arity}) buffer[size++] = v;
            }
        return size;
    } catch (const std::exception& e) { shop_router_0909::error = e.what(); return -1; }
}
int shop_size(int which) {
    if (which == 0) return sizeof(shop_router_0909::Observation);
    if (which == 1) return sizeof(shop_router_0909::Result);
    return sizeof(shop_router_0909::State);
}
}
