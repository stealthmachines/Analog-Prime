// ============================================================================
// HDGL Critic v33 — Header
// ============================================================================
#pragma once
#include <stdint.h>

#define CRITIC_IN    5
// Feature indices for critic_observe / critic_forward
// 0: residue    (LL-lite normalised [0,1])
// 1: coherence  (|sum_sin| / 4, phase coupling proxy)
// 2: amp        (local amplitude)
// 3: r_h_norm   (r_harmonic / 1000, normalised)
// 4: acc_norm   (reward_accum / 10, normalised)

#ifdef __cplusplus
extern "C" {
#endif

void  critic_init(void);
float critic_forward(const float s[CRITIC_IN]);
void  critic_observe(const float s[CRITIC_IN], float target);
void  critic_update(void);
float critic_td_target(float observed_reward, const float s_next[CRITIC_IN]);
int   critic_weight_count(void);
void  critic_pack_weights(float *out);
void  critic_print_stats(void);
int   critic_save(const char *path);
int   critic_load(const char *path);

#ifdef __cplusplus
}
#endif
