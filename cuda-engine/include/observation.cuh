#pragma once
#include "sim.cuh"

namespace kagcuda {

// Zero-copy legal observation boundary for device policies. These view classes
// deliberately do not expose a State* or Farm* publicly, preventing accidental
// access to opponent-private shed/seeds/unit inventories. Numeric item/op ids
// are the exact enum ids used by the simulator and correspond to the official
// string names materialized by the CPU bridge.

class PublicFarmView {
public:
    __device__ __forceinline__ double money() const { return farm_->money; }
    __device__ __forceinline__ int n_units() const { return farm_->n_units; }
    __device__ __forceinline__ int n_quadrants() const { return farm_->n_quadrants; }
    __device__ __forceinline__ int hires_today() const { return farm_->hires_today; }
    __device__ __forceinline__ int worker_x(int u) const { return farm_->pos_x[u]; }
    __device__ __forceinline__ int worker_y(int u) const { return farm_->pos_y[u]; }
    __device__ __forceinline__ const Tile& tile(int x, int y) const { return farm_->tiles[y][x]; }

private:
    const Farm* farm_;
    __device__ __forceinline__ explicit PublicFarmView(const Farm* f) : farm_(f) {}
    friend class ObservationView;
};

class PrivateFarmView {
public:
    __device__ __forceinline__ int shed(int item) const { return farm_->shed[item]; }
    __device__ __forceinline__ int shed_total() const { return farm_->shed_total; }
    __device__ __forceinline__ int seed(int crop) const { return farm_->seeds[crop]; }
    __device__ __forceinline__ int inventory(int unit, int item) const { return farm_->inv[unit][item]; }
    __device__ __forceinline__ int inventory_nkeys(int unit) const { return farm_->inv_nkeys[unit]; }
    __device__ __forceinline__ int inventory_key(int unit, int k) const { return farm_->inv_keys[unit][k]; }
    __device__ __forceinline__ int n_units() const { return farm_->n_units; }

private:
    const Farm* farm_;
    __device__ __forceinline__ explicit PrivateFarmView(const Farm* f) : farm_(f) {}
    friend class ObservationView;
};

class MarketView {
public:
    __device__ __forceinline__ int inventory(int product) const { return market_->inventory[product]; }
    __device__ __forceinline__ int price(int product) const { return market_->prices[product]; }

private:
    const Market* market_;
    __device__ __forceinline__ explicit MarketView(const Market* m) : market_(m) {}
    friend class ObservationView;
};

class TownView {
public:
    __device__ __forceinline__ int n_shops() const { return state_->n_shops; }
    __device__ __forceinline__ int shop(int i) const { return state_->shops[i]; }

private:
    const State* state_;
    __device__ __forceinline__ explicit TownView(const State* s) : state_(s) {}
    friend class ObservationView;
};

class ObservationView;
__device__ __forceinline__ ObservationView make_observation(const Sim& sim, int player);

class ObservationView {
public:
    __device__ __forceinline__ int step() const { return state_->step; }
    __device__ __forceinline__ int player() const { return player_; }
    __device__ __forceinline__ int day() const { return state_->day; }
    __device__ __forceinline__ int hour() const { return state_->hour; }
    __device__ __forceinline__ PublicFarmView farm(int p) const { return PublicFarmView(&state_->farms[p]); }
    __device__ __forceinline__ MarketView market() const { return MarketView(&state_->market); }
    __device__ __forceinline__ TownView town() const { return TownView(state_); }
    __device__ __forceinline__ PrivateFarmView private_farm() const { return PrivateFarmView(&state_->farms[player_]); }

private:
    const State* state_;
    int player_;
    __device__ __forceinline__ ObservationView(const State* s, int p) : state_(s), player_(p) {}
    friend struct Sim;
    friend __device__ __forceinline__ ObservationView make_observation(const Sim& sim, int player);
};

__device__ __forceinline__ ObservationView make_observation(const Sim& sim, int player) {
    return ObservationView(&sim.st, player);
}

}  // namespace kagcuda
