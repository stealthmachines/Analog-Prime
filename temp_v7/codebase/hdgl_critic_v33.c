// ============================================================================
// HDGL Learned Critic v33 — Tiny Host-Side Policy Reward Network
// ============================================================================
//
// [ROADMAP ITEM 10] Replaces the hand-tuned reward formula in v32:
//   OLD: reward = 1/(1+residue) + 0.25*coherence + 0.15*amp
//   NEW: reward = critic_forward(state_features)
//
// Architecture: 2-layer MLP, pure C, no external dependencies
//   Input  dim = 5: [residue, coherence, amp, r_h_norm, acc_norm]
//   Hidden dim = 8: fully-connected, ReLU activation
//   Output dim = 1: linear, scalar reward estimate
//
// Training: online TD(0) with SGD
//   Target: r_observed + GAMMA_TD * V(s')
//   Loss:   MSE (critic_forward(s) - target)^2
//   Update: every CRITIC_UPDATE_INTERVAL reward injections
//
// Thread safety: single-threaded (called from host loop only).
// The critic weights are pushed to the GPU as a __constant__ symbol
// by hdgl_host_v33.c each time they are updated.
//
// Compile: included as part of nvcc compilation of hdgl_host_v33.c
// ============================================================================

#include <math.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include "hdgl_critic_v33.h"

// ============================================================================
// Constants
// ============================================================================

#define CRITIC_IN    5
#define CRITIC_HIDE  8
#define CRITIC_OUT   1

#define CRITIC_LR         0.001f  /* reduced from 0.01: prevents over-convergence to negative prior */
#define CRITIC_GAMMA_TD   0.99f
#define CRITIC_CLIP       5.0f     // gradient clipping
#define CRITIC_UPDATE_INTERVAL 32  // reward samples before a weight update

// ============================================================================
// Weight storage (single global instance, host-only)
// ============================================================================

typedef struct {
    // Layer 1: [CRITIC_HIDE × CRITIC_IN] weights + [CRITIC_HIDE] bias
    float w1[CRITIC_HIDE][CRITIC_IN];
    float b1[CRITIC_HIDE];
    // Layer 2: [CRITIC_OUT × CRITIC_HIDE] weights + [CRITIC_OUT] bias
    float w2[CRITIC_OUT][CRITIC_HIDE];
    float b2[CRITIC_OUT];
    // Gradient accumulators
    float dw1[CRITIC_HIDE][CRITIC_IN];
    float db1[CRITIC_HIDE];
    float dw2[CRITIC_OUT][CRITIC_HIDE];
    float db2[CRITIC_OUT];
    // Running normalisation stats (online Welford)
    float feat_mean[CRITIC_IN];
    float feat_M2[CRITIC_IN];
    long long feat_n;
    // Replay mini-buffer for batch updates
    float  replay_s[CRITIC_UPDATE_INTERVAL][CRITIC_IN];
    float  replay_target[CRITIC_UPDATE_INTERVAL];
    int    replay_head;
    int    replay_count;
    // Training step counter
    long long steps;
} Critic;

static Critic g_critic;

// ============================================================================
// Activation
// ============================================================================

static inline float relu(float x) { return x > 0.0f ? x : 0.0f; }
static inline float relu_d(float x) { return x > 0.0f ? 1.0f : 0.0f; }

// ============================================================================
// Forward pass — stores activations in caller's buffers for backprop
// ============================================================================

static float critic_forward_full(
    const float s[CRITIC_IN],
    float h1[CRITIC_HIDE])   // hidden activations (output for backprop)
{
    for (int j = 0; j < CRITIC_HIDE; j++) {
        float z = g_critic.b1[j];
        for (int i = 0; i < CRITIC_IN; i++) z += g_critic.w1[j][i] * s[i];
        h1[j] = relu(z);
    }
    float out = g_critic.b2[0];
    for (int j = 0; j < CRITIC_HIDE; j++) out += g_critic.w2[0][j] * h1[j];
    return out;
}

// ============================================================================
// Public: forward-only (no stored activations)
// ============================================================================

float critic_forward(const float s[CRITIC_IN]) {
    float h1[CRITIC_HIDE];
    return critic_forward_full(s, h1);
}

// ============================================================================
// Online feature normalisation (Welford one-pass)
// ============================================================================

static void normalise_inplace(float s[CRITIC_IN]) {
    g_critic.feat_n++;
    for (int i = 0; i < CRITIC_IN; i++) {
        float delta  = s[i] - g_critic.feat_mean[i];
        g_critic.feat_mean[i] += delta / (float)g_critic.feat_n;
        float delta2 = s[i] - g_critic.feat_mean[i];
        g_critic.feat_M2[i] += delta * delta2;
        float var = (g_critic.feat_n > 1)
                    ? g_critic.feat_M2[i] / (float)(g_critic.feat_n - 1)
                    : 1.0f;
        float std = sqrtf(var + 1e-8f);
        s[i] = (s[i] - g_critic.feat_mean[i]) / std;
        // Clamp to [-3, 3] to prevent extreme inputs
        if (s[i] >  3.0f) s[i] =  3.0f;
        if (s[i] < -3.0f) s[i] = -3.0f;
    }
}

// ============================================================================
// Push a (state, target) sample into the replay mini-buffer
// ============================================================================

void critic_observe(const float raw_s[CRITIC_IN], float target) {
    // Normalise features
    float s[CRITIC_IN];
    memcpy(s, raw_s, CRITIC_IN * sizeof(float));
    normalise_inplace(s);

    int pos = g_critic.replay_head % CRITIC_UPDATE_INTERVAL;
    memcpy(g_critic.replay_s[pos], s, CRITIC_IN * sizeof(float));
    g_critic.replay_target[pos] = target;
    g_critic.replay_head++;
    if (g_critic.replay_count < CRITIC_UPDATE_INTERVAL) g_critic.replay_count++;

    // Trigger a mini-batch update when buffer is full
    if (g_critic.replay_count == CRITIC_UPDATE_INTERVAL &&
        g_critic.replay_head % CRITIC_UPDATE_INTERVAL == 0) {
        critic_update();
    }
}

// ============================================================================
// Mini-batch gradient update
// ============================================================================

void critic_update(void) {
    if (g_critic.replay_count == 0) return;

    // Zero accumulators
    memset(g_critic.dw1, 0, sizeof(g_critic.dw1));
    memset(g_critic.db1, 0, sizeof(g_critic.db1));
    memset(g_critic.dw2, 0, sizeof(g_critic.dw2));
    memset(g_critic.db2, 0, sizeof(g_critic.db2));

    int n = g_critic.replay_count;
    float inv_n = 1.0f / (float)n;

    for (int b = 0; b < n; b++) {
        const float *s     = g_critic.replay_s[b];
        float        tgt   = g_critic.replay_target[b];
        float h1[CRITIC_HIDE];

        float pred  = critic_forward_full(s, h1);
        float err   = pred - tgt;                   // MSE gradient = 2*err/N ≈ err
        float dout  = err * inv_n;

        // Clip
        if (dout >  CRITIC_CLIP) dout =  CRITIC_CLIP;
        if (dout < -CRITIC_CLIP) dout = -CRITIC_CLIP;

        // Layer 2 gradients
        for (int j = 0; j < CRITIC_HIDE; j++) {
            g_critic.dw2[0][j] += dout * h1[j];
        }
        g_critic.db2[0] += dout;

        // Layer 1 gradients (backprop through ReLU)
        for (int j = 0; j < CRITIC_HIDE; j++) {
            float dh = dout * g_critic.w2[0][j];
            // ReLU derivative: pass through only if h1 > 0
            // h1[j] = relu(z1[j]) so dz = dh * (h1[j] > 0 ? 1 : 0)
            float dz = dh * (h1[j] > 0.0f ? 1.0f : 0.0f);
            for (int i = 0; i < CRITIC_IN; i++) {
                g_critic.dw1[j][i] += dz * s[i];
            }
            g_critic.db1[j] += dz;
        }
    }

    // SGD step
    for (int j = 0; j < CRITIC_HIDE; j++) {
        for (int i = 0; i < CRITIC_IN; i++) {
            g_critic.w1[j][i] -= CRITIC_LR * g_critic.dw1[j][i];
        }
        g_critic.b1[j] -= CRITIC_LR * g_critic.db1[j];
    }
    for (int j = 0; j < CRITIC_HIDE; j++) {
        g_critic.w2[0][j] -= CRITIC_LR * g_critic.dw2[0][j];
    }
    g_critic.b2[0] -= CRITIC_LR * g_critic.db2[0];

    g_critic.steps++;
    g_critic.replay_count = 0;
    g_critic.replay_head  = 0;
}

// ============================================================================
// TD target construction
//   observed_reward: immediate reward from the environment (LL result etc.)
//   s_next:          normalised features at s'
//   Returns TD(0) target: r + γ * V(s')
// ============================================================================

float critic_td_target(float observed_reward, const float raw_s_next[CRITIC_IN]) {
    float s_next[CRITIC_IN];
    memcpy(s_next, raw_s_next, CRITIC_IN * sizeof(float));
    normalise_inplace(s_next);
    float h1[CRITIC_HIDE];
    float v_next = critic_forward_full(s_next, h1);
    return observed_reward + CRITIC_GAMMA_TD * v_next;
}

// ============================================================================
// Serialise weights for GPU upload (packed float array)
// Layout: w1 (HIDE×IN), b1 (HIDE), w2 (OUT×HIDE), b2 (OUT)
//   = 5*8 + 8 + 8*1 + 1 = 57 floats
// ============================================================================

int critic_weight_count(void) {
    return CRITIC_HIDE * CRITIC_IN  // w1
         + CRITIC_HIDE              // b1
         + CRITIC_OUT * CRITIC_HIDE // w2
         + CRITIC_OUT;              // b2
}

void critic_pack_weights(float *out) {
    int p = 0;
    for (int j = 0; j < CRITIC_HIDE; j++)
        for (int i = 0; i < CRITIC_IN; i++)
            out[p++] = g_critic.w1[j][i];
    for (int j = 0; j < CRITIC_HIDE; j++) out[p++] = g_critic.b1[j];
    for (int j = 0; j < CRITIC_HIDE; j++) out[p++] = g_critic.w2[0][j];
    out[p++] = g_critic.b2[0];
}

// ============================================================================
// Init — Xavier/Glorot init
// ============================================================================

void critic_init(void) {
    memset(&g_critic, 0, sizeof(g_critic));
    srand(42);
    // Xavier: std = sqrt(2 / (fan_in + fan_out))
    float std1 = sqrtf(2.0f / (float)(CRITIC_IN + CRITIC_HIDE));
    float std2 = sqrtf(2.0f / (float)(CRITIC_HIDE + CRITIC_OUT));
    for (int j = 0; j < CRITIC_HIDE; j++)
        for (int i = 0; i < CRITIC_IN; i++)
            g_critic.w1[j][i] = std1 * ((float)rand()/(float)RAND_MAX * 2.0f - 1.0f);
    for (int j = 0; j < CRITIC_HIDE; j++)
        g_critic.w2[0][j]  = std2 * ((float)rand()/(float)RAND_MAX * 2.0f - 1.0f);
    // biases initialised to 0 (memset above)
    for (int i = 0; i < CRITIC_IN; i++) g_critic.feat_M2[i] = 1.0f;  // avoid div-by-zero
}

// ============================================================================
// Debug print
// ============================================================================

void critic_print_stats(void) {
    printf("[critic] steps=%lld  feat_n=%lld\n",
           g_critic.steps, g_critic.feat_n);
    printf("[critic] w2: ");
    for (int j = 0; j < CRITIC_HIDE; j++) printf("%.4f ", g_critic.w2[0][j]);
    printf("\n[critic] b2: %.4f\n", g_critic.b2[0]);
}

// ============================================================================
// Checkpoint I/O — persist weights + normalisation stats across runs
// Format: 4-byte magic (0x48443333) + 4-byte version (1) + raw Critic payload
// Only the stable fields are saved: weights, Welford stats, step counter.
// The replay mini-buffer is transient and is NOT persisted.
// ============================================================================

#define CKPT_MAGIC   0x48443333u   /* "HD33" */
#define CKPT_VERSION 1u

int critic_save(const char *path) {
    FILE *f = fopen(path, "wb");
    if (!f) {
        fprintf(stderr, "[critic] save: cannot open '%s'\n", path);
        return -1;
    }
    uint32_t hdr[2] = { CKPT_MAGIC, CKPT_VERSION };
    if (fwrite(hdr, sizeof(hdr), 1, f) != 1) goto fail;

    /* weights */
    if (fwrite(g_critic.w1, sizeof(g_critic.w1), 1, f) != 1) goto fail;
    if (fwrite(g_critic.b1, sizeof(g_critic.b1), 1, f) != 1) goto fail;
    if (fwrite(g_critic.w2, sizeof(g_critic.w2), 1, f) != 1) goto fail;
    if (fwrite(g_critic.b2, sizeof(g_critic.b2), 1, f) != 1) goto fail;
    /* Welford normalisation stats */
    if (fwrite(g_critic.feat_mean, sizeof(g_critic.feat_mean), 1, f) != 1) goto fail;
    if (fwrite(g_critic.feat_M2,   sizeof(g_critic.feat_M2),   1, f) != 1) goto fail;
    if (fwrite(&g_critic.feat_n,   sizeof(g_critic.feat_n),    1, f) != 1) goto fail;
    /* step counter */
    if (fwrite(&g_critic.steps, sizeof(g_critic.steps), 1, f) != 1) goto fail;

    fclose(f);
    return 0;
fail:
    fprintf(stderr, "[critic] save: write error\n");
    fclose(f);
    return -1;
}

int critic_load(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) return -1;   /* not an error: first run, no checkpoint yet */

    uint32_t hdr[2];
    if (fread(hdr, sizeof(hdr), 1, f) != 1) goto fail;
    if (hdr[0] != CKPT_MAGIC || hdr[1] != CKPT_VERSION) {
        fprintf(stderr, "[critic] load: bad magic/version in '%s'\n", path);
        goto fail;
    }

    /* weights */
    if (fread(g_critic.w1, sizeof(g_critic.w1), 1, f) != 1) goto fail;
    if (fread(g_critic.b1, sizeof(g_critic.b1), 1, f) != 1) goto fail;
    if (fread(g_critic.w2, sizeof(g_critic.w2), 1, f) != 1) goto fail;
    if (fread(g_critic.b2, sizeof(g_critic.b2), 1, f) != 1) goto fail;
    /* Welford normalisation stats */
    if (fread(g_critic.feat_mean, sizeof(g_critic.feat_mean), 1, f) != 1) goto fail;
    if (fread(g_critic.feat_M2,   sizeof(g_critic.feat_M2),   1, f) != 1) goto fail;
    if (fread(&g_critic.feat_n,   sizeof(g_critic.feat_n),    1, f) != 1) goto fail;
    /* step counter */
    if (fread(&g_critic.steps, sizeof(g_critic.steps), 1, f) != 1) goto fail;

    fclose(f);
    printf("[critic] loaded checkpoint '%s'  steps=%lld  feat_n=%lld\n",
           path, g_critic.steps, g_critic.feat_n);
    return 0;
fail:
    fprintf(stderr, "[critic] load: read error in '%s' — starting fresh\n", path);
    fclose(f);
    return -1;
}
