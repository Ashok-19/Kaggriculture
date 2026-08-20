#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include "kaggriculture.h"

static void init_env(KaggricultureEnv& env, uint64_t seed, int env_id, int opponent,
                     float* obs, float* acts, float* rewards, float* terminals) {
    std::memset(&env, 0, sizeof(env));
    env.observations = obs;
    env.actions = acts;
    env.rewards = rewards;
    env.terminals = terminals;
    env.num_agents = 1;
    env.env_id = env_id;
    env.base_seed = seed;
    env.opponent_kind = opponent;
    env.sim = new kag::Sim();
    c_reset(&env);
}

static void fill_actions(KaggricultureEnv& env, int t, uint64_t salt) {
    for (int h = 0; h < KAG_NUM_ATNS; ++h) {
        const int size = h < KAG_UNIT_HEADS ? KAG_UNIT_CODE_SIZE : KAG_MARKET_CODE_SIZE;
        uint64_t x = kag_mix_seed(salt + static_cast<uint64_t>(t + 1) * 131U + h * 257U);
        env.actions[h] = static_cast<float>(x % size);
    }
}

static void check_obs(const float* obs) {
    for (int i = 0; i < KAG_OBS_SIZE; ++i) {
        assert(std::isfinite(obs[i]));
        assert(obs[i] >= -1.0001f && obs[i] <= 1.0001f);
    }
}

int main() {
    int episodes = 0;
    int transition_mismatches = 0;
    for (int ep = 0; ep < 8; ++ep) {
        float obs[KAG_OBS_SIZE]{};
        float acts[KAG_NUM_ATNS]{};
        float rewards[1]{};
        float terminals[1]{};
        KaggricultureEnv env{};
        init_env(env, 1234567, ep, ep & 1, obs, acts, rewards, terminals);
        check_obs(obs);
        for (int t = 0; t < 719; ++t) {
            fill_actions(env, t, 1000 + ep);
            kag::Sim ref = *env.sim;
            const int learner_player = env.learner_player;
            const kag::Action learner = kag_decode_learner_action(&env);
            const kag::Action opponent = kag_make_opponent_action(&env);
            if (learner_player == 0) ref.step(learner, opponent);
            else ref.step(opponent, learner);
            c_step(&env);
            if (ref.st.done) {
                const double margin = ref.st.farms[learner_player].money - ref.st.farms[1 - learner_player].money;
                const float sign = margin > 0 ? 1.0f : (margin < 0 ? -1.0f : 0.0f);
                const float expected = 0.9f * sign + 0.1f * std::tanh(static_cast<float>(margin / 5000.0));
                assert(terminals[0] == 1.0f);
                assert(std::fabs(rewards[0] - expected) < 1e-6f);
                episodes++;
                assert(env.pending_reset == 1);
                for (int h = 0; h < KAG_NUM_ATNS; ++h) env.actions[h] = 0.0f;
                c_step(&env);  // terminal flush/dummy slot: action ignored, next episode reset
                assert(terminals[0] == 0.0f);
                assert(rewards[0] == 0.0f);
                assert(env.pending_reset == 0);
                assert(env.sim->st.step == 0);
                break;
            }
            if (std::memcmp(&ref.st, &env.sim->st, sizeof(kag::State)) != 0) {
                transition_mismatches++;
                break;
            }
            assert(terminals[0] == 0.0f);
            check_obs(obs);
        }
        c_close(&env);
    }
    assert(episodes == 8);
    assert(transition_mismatches == 0);
    std::printf("PASS adapter parity: episodes=%d transition_mismatches=%d obs_size=%d heads=%d\n",
                episodes, transition_mismatches, KAG_OBS_SIZE, KAG_NUM_ATNS);
    return 0;
}
