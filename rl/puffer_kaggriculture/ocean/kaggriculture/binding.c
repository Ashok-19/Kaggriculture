#include "kaggriculture.h"

#define OBS_SIZE KAG_OBS_SIZE
#define NUM_ATNS KAG_NUM_ATNS
#define ACT_SIZES { \
    212, 212, 212, 212, 212, 212, 212, 212, 212, 212, \
    212, 212, 212, 212, 212, 212, 212, 212, 212, 212, \
    212, 212, 212, 212, 212, 212, 212, 212, 212, 212, \
    212, 212, 212, 212, 212, 212, 212, 212, 212, 212, \
    155, 155, 155, 155 \
}
#define OBS_TENSOR_T FloatTensor

#define Env KaggricultureEnv
extern "C" {
#include "vecenv.h"
}

void my_init(Env* env, Dict* kwargs) {
    env->num_agents = 1;
    env->env_id = static_cast<int>(env->rng);
    env->base_seed = static_cast<uint64_t>(dict_get(kwargs, "seed")->value);
    env->opponent_kind = static_cast<int>(dict_get(kwargs, "opponent")->value);
    env->episode_index = 0;
    env->sim = new kag::Sim();
    (void)kag::market_price(kag::WHEAT, 10000);
}

void my_log(Log* log, Dict* out) {
    dict_set(out, "perf", log->perf);
    dict_set(out, "score", log->score);
    dict_set(out, "episode_return", log->episode_return);
    dict_set(out, "episode_length", log->episode_length);
    dict_set(out, "win", log->win);
    dict_set(out, "tie", log->tie);
    dict_set(out, "margin", log->margin);
    dict_set(out, "unit_nonpass_rate", log->unit_nonpass_rate);
    dict_set(out, "market_nonempty_rate", log->market_nonempty_rate);
}
