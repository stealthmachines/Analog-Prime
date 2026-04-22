// ============================================================================
// HDGL Analog v35 — Full U-field Bridge (Ev3 Long complete)
// ============================================================================
//
// CHANGES FROM v34:
//
//   [U-FIELD BRIDGE — Ev3 Long, COMPLETE]
//
//     In v34, S(U) was computed directly from r_harmonic via phi_resonance_score:
//       Lambda_phi = log(r_h * LN2_LNPHI) / LN_PHI - INV_2PHI
//     r_h is only a proxy for p — the signal flows:
//       p → r_h (a float accumulator) → Lambda_phi → S(U)
//
//     In v35, the full derivation closes the intended loop:
//       p → r_h (KAPPA·log(p) injection in dphvel)
//            → phvel → Feistel bias → ph (field state)
//            → phj_n (Feistel coupling, already read for Feistel map)
//            → u_inner = φ^(0.5·phj_n + κ·log(r_h+1))   [inner φ-exponent]
//            → M_inner = warp_mean(u_inner)               [warp reduce, 32 lanes]
//            → u_mid   = φ^(M_inner)                      [middle φ-exponent]
//            → M_U     = block_mean(u_mid)                [block reduce, 8 warps]
//            → Λ_φ^(U) = log(M_U) / ln(φ) − 1/(2φ)
//            → S(U) = |e^(iπΛ_φ^(U)) + 1_eff|
//
//   The 3-level nested structure implements:
//     U^(p) = φ^( Σ_i φ^( Σ_j φ^(interaction(U_i, U_j) + κ·log(p)) ) )
//   where interaction(U_i, U_j) = 0.5 · phj_n  (Feistel coupling, already computed).
//
//   Prime invariant test (from roadmap):
//     When all slots are phase-locked (prime signal):
//       phj_n ≈ ph_n for all threads (coherent field)
//       u_inner ≈ φ^(0.5·ph_n + κ·log(r_h+1))  [uniform across warp]
//       M_inner = u_inner  [mean of equal values]
//       u_mid   = φ^(u_inner)
//       M_U     = u_mid    [mean of equal values]
//       Λ_φ^(U) = log(φ^(u_inner)) / ln(φ) − 1/(2φ) = u_inner − 1/(2φ)
//     For a typical prime candidate (ph_n≈0.5, r_h≈50):
//       u_inner ≈ φ^(0.25 + 0.02·ln51) ≈ φ^0.328 ≈ 1.189
//       Λ_φ^(U) ≈ 1.189 − 0.309 ≈ 0.880  →  S(U) near its prime minimum
//     For composite (scattered phases):
//       warp u_inner values vary → M_inner < coherent case → S(U) pushed up
//
//   IMPLEMENTATION DETAILS:
//     • phj_n hoisted to kernel scope (was block-local in v34 Feistel scope)
//     • New shared memory: sh_u_mid[FIELD_BLOCK/32] = 8 floats = 32 bytes
//       (total smem per block still ~12 KB — no occupancy change)
//     • New device function: phi_resonance_from_lambda(float lambda_phi)
//       Takes Λ_φ directly; same formula as old phi_resonance_score body.
//     • phi_resonance_score(r_h, ...) retained for reference / fallback.
//     • Two new top-level constants: LN_PHI_F, INV_2PHI_F
//       (were inner-function locals in v33/v34; now shared by the U-field block
//       and phi_resonance_from_lambda)
//
//   All v34 items preserved:
//     • Feistel phase map on T² (STRIDE_A=89, golden torus)
//     • KAPPA·log(p) correctly wired before integration
//     • Markov trit verdict gate (phi+/phi-/R threshold)
//     • Slot4096 slow-sync (SYNC_GAIN=0.08, every 16 steps)
//     • Warp majority-vote sigma correction
//     • 4-scale wavelet spectral basis (Morlet-like, Hebbian)
//     • MLP critic (5→8→1, TD(0))
//     • Candidate gate with cluster quorum
//
// Compile (bench):
//   nvcc -O3 -arch=sm_75 -allow-unsupported-compiler ^
//        hdgl_analog_v35.cu hdgl_warp_ll_v33.cu hdgl_sieve_v34.cu ^
//        hdgl_critic_v33.c hdgl_bench_v33.cu -o hdgl_bench_v35.exe
//
// Compile (host):
//   nvcc -O3 -arch=sm_75 -allow-unsupported-compiler ^
//        hdgl_analog_v35.cu hdgl_warp_ll_v33.cu ^
//        hdgl_host_v33.c hdgl_critic_v33.c -o hdgl_v35.exe
// ============================================================================

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdint.h>
#include <stdio.h>
#include <math.h>

// ============================================================================
// Constants
// ============================================================================

#define SPECTRAL_N        4
#define LL_LITE_ITERS     4
#define MAX_CAND_BUF      256
#define FIELD_BLOCK       256
#define POOL_ALPHA        0.7f
#define CLUSTER_QUORUM    2

#define GAMMA_F           0.02f
#define LAMBDA_F          0.05f
#define SAT_LIMIT_F       1e6f
#define NOISE_SIGMA_F     0.01f
#define K_COUPLING_F      1.0f
#define BASE_GRA_F        0.18f
#define GRA_PLASTIC_F     0.008f
#define R_MAX_F           1000.0f

#define CAND_ACCUM_THRESH 5.0f
#define CAND_RESIDUE_MAX  0.02f
#define CAND_AMP_MIN      0.6f
#define REWARD_DECAY      0.9f
#define LR                1e-3f
#define W_CLAMP           2.0f
#define SYNC_GAIN         0.08f   /* Slot4096 slow-sync blend coefficient      */

/* Feistel phase update constants */
#define PHI_F             1.6180339887f     /* golden ratio                    */
#define INV_2PI_F         0.1591549430f     /* 1 / (2π) — phase normalisation  */
#define TWO_PI_F          6.2831853071f
#define STRIDE_A          89                /* Fibonacci — Feistel partner gap  */

/* U-field bridge constants — promoted from inner-function locals in v33/v34  */
#define LN_PHI_F          0.4812118251f     /* ln(φ)                           */
#define INV_2PHI_F        0.3090169944f     /* 1/(2φ)                          */

/* U-field injection: κ·log(p) bias into phvel                                */
#define KAPPA_PHI         0.02f

// ============================================================================
// Shared types  (identical to v33/v34 — binary-compatible with host_v33.c)
// ============================================================================

typedef struct {
    int   slot_idx;
    float score;
    float ll_seed_f;
    float r_harmonic;
    float phase;
    float coherence;   /* |sum_sin| at gate time — critic feature 1 */
    float amp;         /* local_amp at gate time — critic feature 2 */
    float acc;         /* reward_accum at gate time — critic feature 4 */
} Candidate;

typedef struct {
    float    *A_re;
    float    *A_im;
    float    *phase;
    float    *phase_vel;
    float    *r_harmonic;
    uint32_t *ll_state;
    float    *reward_accum;
    float    *w_cos;         /* [N * SPECTRAL_N] — wavelet cosine weights      */
    float    *w_sin;         /* [N * SPECTRAL_N] — wavelet sine weights        */
    float    *w_sigma;       /* [N * SPECTRAL_N] — learnable envelope widths   */
    int8_t   *ll_verified;
    Candidate *candidates;
    int      *cand_count;
    int       N;
    int       S;
    float     omega;
    float     dt;
    float     w_amp_self;
    float     w_amp_neigh;
} DevSoA;

__device__ __constant__ DevSoA g_soa;

// ============================================================================
// Critic weights in __constant__ memory
// Layout: w1[HIDE×IN] + b1[HIDE] + w2[HIDE] + b2[1]
// HIDE=8, IN=5 → 40+8+8+1 = 57 floats
// ============================================================================

#define CRITIC_IN_D    5
#define CRITIC_HIDE_D  8
#define CRITIC_W_TOTAL 57

__device__ __constant__ float g_critic_w[CRITIC_W_TOTAL];

// ============================================================================
// Device helpers
// ============================================================================

__device__ __forceinline__ uint32_t ll_step32(uint32_t s) {
    return (uint32_t)((uint64_t)s * (uint64_t)s) - 2u;
}

__device__ __forceinline__ float residue_from_ll(uint32_t s) {
    return fabsf((float)s * 2.3283064365386963e-10f);
}

// ============================================================================
// Wavelet spectral evaluation  (unchanged from v33/v34)
// ============================================================================

__device__ __forceinline__ float wavelet_spectral_eval(
    float dphi,
    float local_amp, float neigh_amp,
    const float * __restrict__ wc,
    const float * __restrict__ ws,
    const float * __restrict__ wsig,
    float w_self, float w_neigh)
{
    float spec = 0.0f;
    float phi2 = dphi * dphi;

    #pragma unroll
    for (int k = 0; k < SPECTRAL_N; k++) {
        float freq   = (float)(1 << k);
        float sigma  = wsig[k];
        float sigma2 = sigma * sigma + 1e-6f;
        float gauss  = expf(-phi2 / (2.0f * sigma2));
        float kd     = freq * dphi;
        spec += gauss * (wc[k] * cosf(kd) + ws[k] * sinf(kd));
    }
    spec += w_self  * local_amp;
    spec += w_neigh * neigh_amp;
    return spec;
}

// ============================================================================
// Phi-resonance from Λ_φ — v35 primary path
//
// Takes Lambda_phi computed from the U-field warp-reduce (M_U → Λ_φ^(U))
// rather than directly from r_harmonic.  This is the full U-field bridge.
//
// S(Λ) = |e^(iπΛ) + 1_eff|,  primes → S near minimum,  composites → S near 2
// ============================================================================

__device__ __forceinline__ float phi_resonance_from_lambda(float lambda_phi)
{
    const float PI = 3.1415926536f;

    if (lambda_phi < 0.1f) return 2.0f;   /* guard degenerate M_U */

    float n_f  = floorf(lambda_phi);
    float beta = lambda_phi - n_f;

    float n_val  = n_f + 1.0f;
    float delta  = fabsf(cosf(PI * beta * PHI_F))
                 * logf(n_val + 2.0f)
                 / expf((n_val + beta) * LN_PHI_F);

    float one_eff = 1.0f + delta;

    float re = cosf(PI * lambda_phi) + one_eff;
    float im = sinf(PI * lambda_phi);
    return sqrtf(re * re + im * im);
}

// ============================================================================
// Phi-resonance from r_harmonic — retained for reference / fallback
// (used in v33/v34; superseded in v35 by phi_resonance_from_lambda above)
// ============================================================================

__device__ __forceinline__ float phi_resonance_score(float r_harmonic, float /*local_amp*/)
{
    const float LN2_LNPHI = 1.4404200904f;
    const float PI        = 3.1415926536f;

    if (r_harmonic < 2.0f) return 2.0f;

    float M_U_proxy  = r_harmonic * LN2_LNPHI;
    float lambda_phi = logf(M_U_proxy) / LN_PHI_F - INV_2PHI_F;

    float n_f  = floorf(lambda_phi);
    float beta = lambda_phi - n_f;

    float n_val  = n_f + 1.0f;
    float delta  = fabsf(cosf(PI * beta * PHI_F))
                 * logf(n_val + 2.0f)
                 / expf((n_val + beta) * LN_PHI_F);

    float one_eff = 1.0f + delta;

    float re = cosf(PI * lambda_phi) + one_eff;
    float im = sinf(PI * lambda_phi);
    return sqrtf(re * re + im * im);
}

// ============================================================================
// MLP critic reward  (unchanged from v33/v34)
// ============================================================================

__device__ float critic_reward(
    float residue, float coherence, float amp, float r_h, float acc)
{
    float s[CRITIC_IN_D] = {
        residue,
        coherence * 0.25f,
        amp,
        r_h * 1e-3f,
        acc * 0.1f
    };

    const float *w1 = g_critic_w;
    const float *b1 = g_critic_w + CRITIC_IN_D * CRITIC_HIDE_D;
    const float *w2 = b1 + CRITIC_HIDE_D;
    float        b2 = g_critic_w[CRITIC_W_TOTAL - 1];

    float h[CRITIC_HIDE_D];
    for (int j = 0; j < CRITIC_HIDE_D; j++) {
        float z = b1[j];
        for (int i = 0; i < CRITIC_IN_D; i++) {
            z += w1[j * CRITIC_IN_D + i] * s[i];
        }
        h[j] = z > 0.0f ? z : 0.0f;
    }

    float out = b2;
    for (int j = 0; j < CRITIC_HIDE_D; j++) out += w2[j] * h[j];
    return fmaxf(0.0f, fminf(5.0f, out));
}

// ============================================================================
// Main field evolution kernel v35
//
// KEY CHANGE — U-field bridge (full Λ_φ^(U) from field state, not r_harmonic):
//
//   After GRA plasticity (r_h updated), before the critic reward, we compute:
//
//     u_inner = φ^(0.5·phj_n + κ·log(r_h+1))      [per-thread, uses Feistel partner]
//     M_inner = (1/32) · Σ_{warp lanes} u_inner     [warp reduce via __shfl]
//     u_mid   = φ^(M_inner)                          [middle φ-layer, block-uniform]
//     M_U     = (1/8) · Σ_{warps in block} u_mid     [block reduce via sh_u_mid[8]]
//     Λ_φ^(U) = log(M_U) / ln(φ) − 1/(2φ)
//
//   resonance = phi_resonance_from_lambda(Λ_φ^(U))   [S(U), full U-field result]
//
//   phj_n is hoisted to kernel scope (was block-local in v34 Feistel block).
//   sh_u_mid[FIELD_BLOCK/32] = sh_u_mid[8] added to shared memory (+32 bytes).
// ============================================================================

__global__ void hdgl_field_step_kernel(int step_count)
{
    const int i   = blockIdx.x * blockDim.x + threadIdx.x;
    const int tid = threadIdx.x;
    const int N   = g_soa.N;
    const int S   = g_soa.S;
    if (i >= N) return;

    /* Shared: wavelet Hebbian gradient pooling */
    __shared__ float sh_gc [SPECTRAL_N][FIELD_BLOCK];
    __shared__ float sh_gs [SPECTRAL_N][FIELD_BLOCK];
    __shared__ float sh_gsg[SPECTRAL_N][FIELD_BLOCK];

    /* [Phase 2] Markov trit block-level aggregation */
    __shared__ float s_phi_pos;
    __shared__ float s_phi_neg;
    __shared__ float s_gamma_sum;

    /* [U-FIELD BRIDGE] Block-level warp-mean staging for M_U             */
    /* One entry per warp: 256/32 = 8 floats = 32 bytes                   */
    __shared__ float sh_u_mid[FIELD_BLOCK / 32];

    /* ------------------------------------------------------------------ */
    /* Load SoA                                                             */
    /* ------------------------------------------------------------------ */
    float A_re   = g_soa.A_re[i];
    float A_im   = g_soa.A_im[i];
    float ph     = g_soa.phase[i];
    float phvel  = g_soa.phase_vel[i];
    float r_h    = g_soa.r_harmonic[i];
    uint32_t s   = g_soa.ll_state[i];
    float acc    = g_soa.reward_accum[i];
    float dt     = g_soa.dt;
    float omega  = g_soa.omega;

    /* [FEISTEL] Read Fibonacci-stride partner phase BEFORE any writes.   */
    /* phj_n hoisted to kernel scope (used by both Feistel map and        */
    /* U-field bridge below).                                             */
    float ph_j  = g_soa.phase[(i + STRIDE_A) % N];
    float phj_n = ph_j * INV_2PI_F;    /* normalised partner phase ∈ [0,1) */

    float wc_local[SPECTRAL_N];
    float ws_local[SPECTRAL_N];
    float wsig_local[SPECTRAL_N];
    #pragma unroll
    for (int k = 0; k < SPECTRAL_N; k++) {
        wc_local[k]   = g_soa.w_cos  [i * SPECTRAL_N + k];
        ws_local[k]   = g_soa.w_sin  [i * SPECTRAL_N + k];
        wsig_local[k] = g_soa.w_sigma[i * SPECTRAL_N + k];
    }

    float local_amp = sqrtf(A_re * A_re + A_im * A_im);

    /* LL-lite proxy */
    #pragma unroll
    for (int k = 0; k < LL_LITE_ITERS; k++) s = ll_step32(s);
    float residue = residue_from_ll(s);

    /* ------------------------------------------------------------------ */
    /* 4-neighbour von Neumann coupling                                    */
    /* ------------------------------------------------------------------ */
    int ni[4] = {
        (i - 1 + N) % N,
        (i + 1)     % N,
        (i - S + N) % N,
        (i + S)     % N
    };

    float dA_re   = -GAMMA_F * A_re;
    float dA_im   = -GAMMA_F * A_im;
    float sum_sin = 0.0f;
    float gra_sum = 0.0f;

    float dphi_hebb = g_soa.phase[ni[1]] - ph;

    #pragma unroll
    for (int j = 0; j < 4; j++) {
        int   n      = ni[j];
        float dphi   = g_soa.phase[n] - ph;
        float n_A_re = g_soa.A_re[n];
        float n_A_im = g_soa.A_im[n];
        float n_amp  = sqrtf(n_A_re * n_A_re + n_A_im * n_A_im);

        float spec = wavelet_spectral_eval(dphi, local_amp, n_amp,
                                           wc_local, ws_local, wsig_local,
                                           g_soa.w_amp_self, g_soa.w_amp_neigh);

        float combined = sqrtf(r_h * r_h + g_soa.r_harmonic[n] * g_soa.r_harmonic[n]);
        float gra_fac  = BASE_GRA_F * combined / (1.0f + combined);
        float factor   = spec + gra_fac;

        sum_sin += sinf(dphi);
        dA_re   += factor * cosf(dphi);
        dA_im   += factor * sinf(dphi);
        gra_sum += gra_fac;
    }

    /* κ·log(p) injection in dphvel — closes U^(p) bias loop (active since v34) */
    float dphvel = omega
                 + K_COUPLING_F * sum_sin
                 + 0.15f * gra_sum
                 + KAPPA_PHI * logf(r_h + 1.0f);

    /* ------------------------------------------------------------------ */
    /* Field amplitude Euler step                                          */
    /* ------------------------------------------------------------------ */
    A_re  += dt * dA_re;
    A_im  += dt * dA_im;

    /* Velocity Euler step */
    phvel += dt * dphvel;

    /* ------------------------------------------------------------------ */
    /* [FEISTEL] Phase update on T²  (from v34, uses phj_n now in scope)  */
    /* ------------------------------------------------------------------ */
    {
        float ph_n = ph * INV_2PI_F;
        float bias = 1.0f + phvel * dt * INV_2PI_F;

        ph_n = fmodf(PHI_F * (ph_n + 0.5f * phj_n + bias), 1.0f);
        if (ph_n < 0.0f) ph_n += 1.0f;

        ph = ph_n * TWO_PI_F;
    }

    /* ------------------------------------------------------------------ */
    /* Amplitude normalisation, noise, saturation  (unchanged from v34)   */
    /* ------------------------------------------------------------------ */
    float A = sqrtf(A_re * A_re + A_im * A_im);
    A      *= expf(-LAMBDA_F * dt);
    if (A > SAT_LIMIT_F) A = SAT_LIMIT_F;

    uint32_t noise_s = s ^ (uint32_t)(i * 2654435761u);
    noise_s = noise_s * 1664525u + 1013904223u;
    float noise = ((float)(noise_s >> 16) * 1.52587890625e-5f - 1.0f) * NOISE_SIGMA_F;
    A += noise;
    if (A < 0.0f) A = 0.0f;

    float norm = sqrtf(A_re * A_re + A_im * A_im);
    if (norm > 1e-8f) { float inv = A / norm; A_re *= inv; A_im *= inv; }

    /* GRA plasticity */
    float plastic = GRA_PLASTIC_F * (local_amp - 0.5f) * (r_h > 50.0f ? 0.05f : 1.0f);
    r_h += plastic;
    r_h  = fmaxf(1.0f, fminf(R_MAX_F, r_h));

    /* ------------------------------------------------------------------ */
    /* [U-FIELD BRIDGE] Compute Λ_φ^(U) from field state                  */
    /*                                                                     */
    /* Level 0 — innermost φ-exponent:                                    */
    /*   interaction(U_i, U_j) = 0.5 · phj_n  (Feistel coupling)         */
    /*   u_inner = φ^(interaction + κ·log(r_h+1))                        */
    /*                                                                     */
    /* Level 1 — warp reduce (32 lanes → scalar M_inner):                 */
    /*   M_inner = (1/32) · Σ_{lane=0}^{31} u_inner_lane                 */
    /*   u_mid   = φ^(M_inner)  [block-uniform within warp]              */
    /*                                                                     */
    /* Level 2 — block reduce (8 warps → scalar M_U):                    */
    /*   sh_u_mid[warp_id] = u_mid   (lane 0 of each warp writes)        */
    /*   M_U = (1/8) · Σ_{w=0}^{7} sh_u_mid[w]   (thread 0 computes)    */
    /*                                                                     */
    /* Output:                                                             */
    /*   Λ_φ^(U) = log(M_U) / ln(φ) − 1/(2φ)                            */
    /*   resonance = S(Λ_φ^(U)) via phi_resonance_from_lambda()           */
    /* ------------------------------------------------------------------ */
    float u_inner;
    {
        float inner_exp = 0.5f * phj_n + KAPPA_PHI * logf(r_h + 1.0f);
        u_inner = expf(inner_exp * LN_PHI_F);   /* = φ^(inner_exp) */
    }

    /* Warp reduce: sum u_inner across all 32 lanes */
    float ui_sum = u_inner;
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        ui_sum += __shfl_down_sync(0xffffffffu, ui_sum, off);
    float M_inner = __shfl_sync(0xffffffffu, ui_sum, 0) / 32.0f;

    float u_mid = expf(M_inner * LN_PHI_F);   /* = φ^(M_inner) */

    /* Block reduce: lane 0 of each warp writes u_mid to sh_u_mid */
    if ((tid & 31) == 0)
        sh_u_mid[tid >> 5] = u_mid;
    __syncthreads();

    /* Thread 0 sums across warps and broadcasts via sh_u_mid[0] */
    if (tid == 0) {
        float su = 0.0f;
        #pragma unroll
        for (int w = 0; w < FIELD_BLOCK / 32; w++) su += sh_u_mid[w];
        sh_u_mid[0] = su * (1.0f / (float)(FIELD_BLOCK / 32));   /* mean */
    }
    __syncthreads();

    float M_U          = sh_u_mid[0];
    float lambda_phi_U = logf(M_U + 1e-6f) / LN_PHI_F - INV_2PHI_F;
    if (lambda_phi_U < 0.0f) lambda_phi_U = 0.0f;   /* guard degenerate */

    /* ------------------------------------------------------------------ */
    /* [ITEM 10] Critic-predicted reward                                   */
    /* ------------------------------------------------------------------ */
    float coherence = fabsf(sum_sin);
    float reward    = critic_reward(residue, coherence, local_amp, r_h, acc);

    /* [X+1=0] Phi-resonance from U-field Λ_φ^(U)  (v35: full bridge)   */
    float resonance = phi_resonance_from_lambda(lambda_phi_U);
    float res_bonus = fmaxf(0.0f, 1.0f - resonance * 0.5f);
    reward = reward + res_bonus;

    /* ------------------------------------------------------------------ */
    /* [ITEM 11] Wavelet Hebbian update with block pooling                 */
    /* ------------------------------------------------------------------ */
    float phi2 = dphi_hebb * dphi_hebb;

    float grad_cos[SPECTRAL_N], grad_sin[SPECTRAL_N], grad_sig[SPECTRAL_N];
    #pragma unroll
    for (int k = 0; k < SPECTRAL_N; k++) {
        float freq   = (float)(1 << k);
        float sigma  = wsig_local[k];
        float sigma2 = sigma * sigma + 1e-6f;
        float gauss  = expf(-phi2 / (2.0f * sigma2));
        float kd     = freq * dphi_hebb;
        float coskd  = cosf(kd);
        float sinkd  = sinf(kd);

        grad_cos[k] = LR * reward * gauss * coskd;
        grad_sin[k] = LR * reward * gauss * sinkd;

        float dgauss_dsig = gauss * phi2 / (sigma2 * sigma + 1e-8f);
        grad_sig[k] = LR * reward * dgauss_dsig
                    * (wc_local[k] * coskd + ws_local[k] * sinkd);

        sh_gc[k][tid]  = grad_cos[k];
        sh_gs[k][tid]  = grad_sin[k];
        sh_gsg[k][tid] = grad_sig[k];
    }
    __syncthreads();

    for (int stride = FIELD_BLOCK / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            #pragma unroll
            for (int k = 0; k < SPECTRAL_N; k++) {
                sh_gc[k][tid]  += sh_gc[k][tid + stride];
                sh_gs[k][tid]  += sh_gs[k][tid + stride];
                sh_gsg[k][tid] += sh_gsg[k][tid + stride];
            }
        }
        __syncthreads();
    }

    const float inv_b = 1.0f / (float)FIELD_BLOCK;
    #pragma unroll
    for (int k = 0; k < SPECTRAL_N; k++) {
        float pool_gc  = sh_gc[k][0]  * inv_b;
        float pool_gs  = sh_gs[k][0]  * inv_b;
        float pool_gsg = sh_gsg[k][0] * inv_b;

        wc_local[k]   += POOL_ALPHA * grad_cos[k] + (1.0f - POOL_ALPHA) * pool_gc;
        ws_local[k]   += POOL_ALPHA * grad_sin[k] + (1.0f - POOL_ALPHA) * pool_gs;
        wsig_local[k] += POOL_ALPHA * grad_sig[k] + (1.0f - POOL_ALPHA) * pool_gsg;

        wc_local[k]   = fmaxf(-W_CLAMP, fminf(W_CLAMP, wc_local[k]));
        ws_local[k]   = fmaxf(-W_CLAMP, fminf(W_CLAMP, ws_local[k]));
        wsig_local[k] = fmaxf(0.19635f, fminf(3.14159f, wsig_local[k]));
    }

    /* Temporal accumulator */
    acc = REWARD_DECAY * acc + reward;

    /* ------------------------------------------------------------------ */
    /* Slot4096 slow-sync correction (unchanged from v33/v34)              */
    /* ------------------------------------------------------------------ */
    {
        float rh_sum = r_h;
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            rh_sum += __shfl_down_sync(0xffffffffu, rh_sum, off);
        float rh_slow = __shfl_sync(0xffffffffu, rh_sum, 0) * (1.0f / 32.0f);
        if ((step_count & 15) == 0)
            r_h += SYNC_GAIN * (rh_slow - r_h);
    }

    /* ------------------------------------------------------------------ */
    /* Write back                                                           */
    /* ------------------------------------------------------------------ */
    g_soa.A_re[i]         = A_re;
    g_soa.A_im[i]         = A_im;
    g_soa.phase[i]        = ph;
    g_soa.phase_vel[i]    = phvel;
    g_soa.r_harmonic[i]   = r_h;
    g_soa.ll_state[i]     = s;
    g_soa.reward_accum[i] = acc;

    #pragma unroll
    for (int k = 0; k < SPECTRAL_N; k++) {
        g_soa.w_cos  [i * SPECTRAL_N + k] = wc_local[k];
        g_soa.w_sin  [i * SPECTRAL_N + k] = ws_local[k];
        g_soa.w_sigma[i * SPECTRAL_N + k] = wsig_local[k];
    }

    /* ------------------------------------------------------------------ */
    /* [Phase 2] Markov trit verdict gate  (unchanged from v33/v34)        */
    /* ------------------------------------------------------------------ */

    float lambda_k = local_amp;

    float lk_sum = lambda_k;
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        lk_sum += __shfl_down_sync(0xffffffffu, lk_sum, off);
    float lambda_bar = __shfl_sync(0xffffffffu, lk_sum, 0) * (1.0f / 32.0f);

    float l_neg  = -1.0f * lambda_k;
    float l_zero = -1.0f * fabsf(lambda_k - lambda_bar);
    float l_pos  =  1.0f * (lambda_bar - lambda_k);
    float m_max  = fmaxf(l_neg, fmaxf(l_zero, l_pos));
    float e_neg  = __expf(l_neg  - m_max);
    float e_zero = __expf(l_zero - m_max);
    float e_pos  = __expf(l_pos  - m_max);
    float inv_z  = 1.0f / (e_neg + e_zero + e_pos);
    e_neg  *= inv_z;
    e_zero *= inv_z;

    uint32_t lcg = s ^ __float_as_uint(acc);
    lcg = lcg * 1664525u + 1013904223u;
    float u_lcg  = (float)(lcg >> 8) * (1.0f / 16777216.0f);
    int sigma_trit;
    if      (u_lcg < e_neg)            sigma_trit = -1;
    else if (u_lcg < e_neg + e_zero)   sigma_trit =  0;
    else                               sigma_trit = +1;

    unsigned b_pos_w = __ballot_sync(0xffffffffu, sigma_trit > 0);
    unsigned b_neg_w = __ballot_sync(0xffffffffu, sigma_trit < 0);
    if      (__popc(b_pos_w) > 16) sigma_trit = +1;
    else if (__popc(b_neg_w) > 16) sigma_trit = -1;

    if (tid == 0) { s_phi_pos = 0.0f; s_phi_neg = 0.0f; s_gamma_sum = 0.0f; }
    __syncthreads();

    atomicAdd(&s_phi_pos,   (sigma_trit ==  1) ? 1.0f : 0.0f);
    atomicAdd(&s_phi_neg,   (sigma_trit == -1) ? 1.0f : 0.0f);
    atomicAdd(&s_gamma_sum, fabsf(lambda_k - lambda_bar));
    __syncthreads();

    float inv_n   = 1.0f / (float)FIELD_BLOCK;
    float phi_pos = s_phi_pos   * inv_n;
    float phi_neg = s_phi_neg   * inv_n;
    float gamma_v = s_gamma_sum * inv_n;

    int verdict;
    if (phi_neg > 0.45f) {
        verdict = 2;
    } else {
        float R = 1.2f * phi_neg + 0.8f * gamma_v - phi_pos;
        if      (R       > 0.6f)  verdict = 2;
        else if (phi_pos > 0.35f) verdict = 1;
        else                      verdict = 0;
    }

    int meets_gate = (verdict == 1 && acc > CAND_ACCUM_THRESH) ? 1 : 0;

    unsigned resonance_mask = __ballot_sync(0xffffffff, meets_gate);
    int      cluster_size   = __popc(resonance_mask);

    if (meets_gate && cluster_size >= CLUSTER_QUORUM) {
        int lane_in_warp  = tid & 31;
        int elected_lane  = __ffs((int)resonance_mask) - 1;
        if (lane_in_warp == elected_lane) {
            float boosted_score = (2.0f - resonance) * (1.0f + 0.1f * (float)cluster_size);
            int pos = atomicAdd(g_soa.cand_count, 1);
            if (pos < MAX_CAND_BUF) {
                g_soa.candidates[pos].slot_idx   = i;
                g_soa.candidates[pos].score      = boosted_score;
                g_soa.candidates[pos].ll_seed_f  = (float)s;
                g_soa.candidates[pos].r_harmonic = r_h;
                g_soa.candidates[pos].phase      = ph;
                g_soa.candidates[pos].coherence  = coherence;
                g_soa.candidates[pos].amp        = local_amp;
                g_soa.candidates[pos].acc        = acc;
            }
        }
        g_soa.reward_accum[i] = acc * 0.5f;
    }
}

// ============================================================================
// Reward injection kernel  (identical to v33/v34)
// ============================================================================

__global__ void hdgl_reward_inject_kernel(void)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= g_soa.N) return;

    int8_t flag = g_soa.ll_verified[i];
    if (flag == 0) return;

    float r_h = g_soa.r_harmonic[i];
    float acc = g_soa.reward_accum[i];

    if (flag == 1) { acc += 1.0f;  r_h = fminf(r_h * 1.05f, R_MAX_F); }
    else           { acc *= 0.7f;  r_h = fmaxf(1.0f, r_h * 0.98f);    }

    g_soa.reward_accum[i] = acc;
    g_soa.r_harmonic[i]   = r_h;
    g_soa.ll_verified[i]  = 0;
}

// ============================================================================
// Weight sync kernel  (identical to v33/v34)
// ============================================================================

__device__ float g_global_w_cos[SPECTRAL_N];
__device__ float g_global_w_sin[SPECTRAL_N];
__device__ float g_global_w_sigma[SPECTRAL_N];

__global__ void hdgl_weight_sync_kernel(void)
{
    __shared__ float sh_cos[SPECTRAL_N][256];
    __shared__ float sh_sin[SPECTRAL_N][256];
    __shared__ float sh_sig[SPECTRAL_N][256];

    int tid = threadIdx.x;
    int N   = g_soa.N;

    float lc[SPECTRAL_N] = {0}, ls[SPECTRAL_N] = {0}, lsg[SPECTRAL_N] = {0};
    for (int i = tid; i < N; i += 256) {
        #pragma unroll
        for (int k = 0; k < SPECTRAL_N; k++) {
            lc[k]  += g_soa.w_cos  [i * SPECTRAL_N + k];
            ls[k]  += g_soa.w_sin  [i * SPECTRAL_N + k];
            lsg[k] += g_soa.w_sigma[i * SPECTRAL_N + k];
        }
    }
    #pragma unroll
    for (int k = 0; k < SPECTRAL_N; k++) {
        sh_cos[k][tid] = lc[k];
        sh_sin[k][tid] = ls[k];
        sh_sig[k][tid] = lsg[k];
    }
    __syncthreads();

    for (int stride = 128; stride > 0; stride >>= 1) {
        if (tid < stride) {
            #pragma unroll
            for (int k = 0; k < SPECTRAL_N; k++) {
                sh_cos[k][tid] += sh_cos[k][tid + stride];
                sh_sin[k][tid] += sh_sin[k][tid + stride];
                sh_sig[k][tid] += sh_sig[k][tid + stride];
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        float inv = (N > 0) ? (1.0f / (float)N) : 0.0f;
        #pragma unroll
        for (int k = 0; k < SPECTRAL_N; k++) {
            g_global_w_cos[k]   = sh_cos[k][0] * inv;
            g_global_w_sin[k]   = sh_sin[k][0] * inv;
            g_global_w_sigma[k] = sh_sig[k][0] * inv;
        }
    }
}

// ============================================================================
// Host-callable launchers  (same signature as v33/v34 — drop-in compatible)
// ============================================================================

extern "C" {

void hdgl_v33_upload_soa(const DevSoA *h) {
    cudaMemcpyToSymbol(g_soa, h, sizeof(DevSoA));
}

void hdgl_v33_upload_critic(const float *weights, int count) {
    cudaMemcpyToSymbol(g_critic_w, weights, count * sizeof(float));
}

void hdgl_v33_field_step(int N, int block, int step_count, cudaStream_t s) {
    hdgl_field_step_kernel<<<(N + block - 1) / block, block, 0, s>>>(step_count);
}

void hdgl_v33_reward_inject(int N, int block, cudaStream_t s) {
    hdgl_reward_inject_kernel<<<(N + block - 1) / block, block, 0, s>>>();
}

void hdgl_v33_weight_sync(cudaStream_t s) {
    hdgl_weight_sync_kernel<<<1, 256, 0, s>>>();
}

void hdgl_v33_read_global_weights(float oc[SPECTRAL_N], float os[SPECTRAL_N],
                                   float osg[SPECTRAL_N]) {
    cudaMemcpyFromSymbol(oc,  g_global_w_cos,   SPECTRAL_N * sizeof(float));
    cudaMemcpyFromSymbol(os,  g_global_w_sin,   SPECTRAL_N * sizeof(float));
    cudaMemcpyFromSymbol(osg, g_global_w_sigma, SPECTRAL_N * sizeof(float));
}

} // extern "C"
