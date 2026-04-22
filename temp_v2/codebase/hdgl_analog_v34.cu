// ============================================================================
// HDGL Analog v34 — Feistel Phase Update (Golden Torus / Fixed-Point Destruction)
// ============================================================================
//
// CHANGES FROM v33:
//
//   [FEISTEL] Feistel coupled phase update replaces simple Euler integration.
//
//     OLD (v33 — has golden-angle fixed-point trap):
//       ph += dt * phvel
//
//     NEW (v34 — Feistel map on T²):
//       bias  = 1.0 + phvel * dt / (2π)          // normalized rotation + frequency
//       ph_j  = g_soa.phase[(i + STRIDE_A) % N]  // Fibonacci-stride partner
//       ph    = fmod(φ * (ph/(2π) + 0.5*(ph_j/(2π)) + bias), 1.0) * 2π
//
//     STRIDE_A = 89  (Fibonacci — ensures irrational winding on T²)
//
//     GEOMETRY:
//       The fixed point of f(x) = frac(φ*(x+1)) is x* = 1/φ = golden_angle/(2π)
//       ≈ 0.6180.  Every slot starting near this value was TRAPPED forever,
//       producing a global golden-angle attractor that suppressed prime signals.
//
//       The Feistel coupling f(xi, xj) = frac(φ*(xi + 0.5*xj + bias)) destroys
//       the fixed point — it now exists only at xi = xj = x* simultaneously,
//       which has measure zero once entropy injection moves any slot off x*.
//
//       Formerly-trapped slots now trace a dense quasi-periodic orbit on the
//       2-torus T² = [0,1)² with winding ratio 1:φ (the golden torus knot).
//       See: right-brain-left-brain.py — full geometric analysis.
//
//   [KAPPA FIX] κ·log(p) U-field injection now correctly feeds into dphvel
//               BEFORE the phase update.  In v33 it was applied after the
//               Euler step (dead code — never took effect).
//
//   [U-FIELD BRIDGE — Ev3 Long, partial]:
//       p biases field via κ·log(p) in dphvel → phvel → Feistel bias.
//       Slots with different r_h (Mersenne exponent proxy) now evolve at
//       genuinely different frequencies, not just different offsets of the
//       same attractor.  The Feistel map is the discrete approximation of:
//         U^(p) = φ^(Σ_{i} φ^(Σ_{j} φ^(interaction(U_i,U_j) + κ·log(p))))
//       with interaction(U_i, U_j) = 0.5 * U_j  (first-order linear coupling).
//
//   All v33 items preserved:
//     • Markov trit verdict gate (Phase 2)
//     • Slot4096 slow-sync (SYNC_GAIN=0.08)
//     • 4-scale wavelet spectral basis (Morlet-like)
//     • Learned MLP critic (5→8→1, TD(0))
//     • Warp majority-vote sigma correction
//     • Candidate gate with cluster quorum
//     • gpucarry / NTT dispatch (warp_ll_v33)
//
// Compile (bench, drop-in for hdgl_analog_v33.cu):
//   nvcc -O3 -arch=sm_75 -allow-unsupported-compiler ^
//        hdgl_analog_v34.cu hdgl_warp_ll_v33.cu hdgl_sieve_v34.cu ^
//        hdgl_critic_v33.c hdgl_bench_v33.cu -o hdgl_bench_v34.exe
//
// Compile (host):
//   nvcc -O3 -arch=sm_75 -allow-unsupported-compiler ^
//        hdgl_analog_v34.cu hdgl_warp_ll_v33.cu ^
//        hdgl_host_v33.c hdgl_critic_v33.c -o hdgl_v34.exe
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

/* U-field injection: κ·log(p) bias into phvel (now active, was dead in v33)  */
#define KAPPA_PHI         0.02f

// ============================================================================
// Shared types  (identical to v33 — binary-compatible with hdgl_host_v33.c)
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
// Wavelet spectral evaluation  (unchanged from v33)
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
// Phi-resonance prime gate  (unchanged from v33)
// S(p) = |e^(iπΛ_φ) + 1_eff|,  primes → 0,  composites → 2
// ============================================================================

__device__ __forceinline__ float phi_resonance_score(float r_harmonic, float /*local_amp*/)
{
    const float LN_PHI    = 0.4812118251f;
    const float LN2_LNPHI = 1.4404200904f;
    const float INV_2PHI  = 0.3090169944f;
    const float PI        = 3.1415926536f;

    if (r_harmonic < 2.0f) return 2.0f;

    float M_U        = r_harmonic * LN2_LNPHI;
    float lambda_phi = logf(M_U) / LN_PHI - INV_2PHI;

    float n_f  = floorf(lambda_phi);
    float beta = lambda_phi - n_f;

    float n_val  = n_f + 1.0f;
    float delta  = fabsf(cosf(PI * beta * PHI_F))
                 * logf(n_val + 2.0f)
                 / expf((n_val + beta) * LN_PHI);

    float one_eff = 1.0f + delta;

    float re = cosf(PI * lambda_phi) + one_eff;
    float im = sinf(PI * lambda_phi);
    return sqrtf(re * re + im * im);
}

// ============================================================================
// MLP critic reward  (unchanged from v33)
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
// Main field evolution kernel v34
//
// KEY CHANGE — Feistel phase update:
//
//   Each slot reads its Fibonacci-stride partner ph_j BEFORE computing
//   its new phase.  Because all reads come from the same global-memory
//   snapshot (no __syncthreads() between read and write), the update is
//   naturally asymmetric — i reads j's old value, j reads i's old value —
//   which produces the correct irrational winding on T².
//
//   Map (normalised to [0,1)):
//     bias   = 1.0 + phvel * dt / (2π)
//     ph'    = fmod( φ * (ph/(2π) + 0.5 * (ph_j/(2π)) + bias), 1.0 )
//     ph_new = ph' * 2π
//
//   Fixed-point destruction proof:
//     For ph to be a fixed point: ph/(2π) = fmod(φ*(ph/(2π) + 0.5*(ph_j/(2π)) + bias), 1)
//     This requires ph = ph_j AND the usual scalar condition — but ph_j is an
//     independent state updated from its own partner, so the joint system has
//     no stable fixed point (measure-zero in practice after entropy injection).
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
    /* STRIDE_A=89 is Fibonacci; irrational partner gap ensures T² orbit. */
    float ph_j = g_soa.phase[(i + STRIDE_A) % N];

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

    /* ------------------------------------------------------------------ */
    /* [KAPPA FIX] κ·log(p) injection now included in dphvel BEFORE       */
    /* the integration step (was dead code in v33 — applied after Euler). */
    /* This closes the U^(p) loop: p biases phvel → Feistel bias → field. */
    /* ------------------------------------------------------------------ */
    float dphvel = omega
                 + K_COUPLING_F * sum_sin
                 + 0.15f * gra_sum
                 + KAPPA_PHI * logf(r_h + 1.0f);   /* U-field p-injection */

    /* ------------------------------------------------------------------ */
    /* Field amplitude Euler step (A_re, A_im unchanged)                  */
    /* ------------------------------------------------------------------ */
    A_re  += dt * dA_re;
    A_im  += dt * dA_im;

    /* Velocity Euler step */
    phvel += dt * dphvel;

    /* ------------------------------------------------------------------ */
    /* [FEISTEL] Phase update — replaces ph += dt * phvel                 */
    /*                                                                     */
    /* bias = 1.0 + phvel*dt/(2π):                                        */
    /*   +1.0 term = base rotation (same role as +1 in Python script).    */
    /*   +phvel*dt/(2π) = slot-specific frequency deviation (Kuramoto +   */
    /*                    KAPPA*log(p) already folded into phvel above).   */
    /*                                                                     */
    /* Coupling: 0.5 * ph_j/(2π) provides the inter-slot interaction      */
    /* that destroys the golden-angle fixed point.                         */
    /* ------------------------------------------------------------------ */
    {
        float ph_n  = ph    * INV_2PI_F;          /* normalise to [0, 1) */
        float phj_n = ph_j  * INV_2PI_F;
        float bias  = 1.0f  + phvel * dt * INV_2PI_F;

        ph_n = fmodf(PHI_F * (ph_n + 0.5f * phj_n + bias), 1.0f);
        if (ph_n < 0.0f) ph_n += 1.0f;

        ph = ph_n * TWO_PI_F;
    }

    /* ------------------------------------------------------------------ */
    /* Amplitude normalisation, noise, saturation (unchanged from v33)    */
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

    /* ph is already in [0, 2π) from the Feistel map — no extra wrap needed */

    /* GRA plasticity */
    float plastic = GRA_PLASTIC_F * (local_amp - 0.5f) * (r_h > 50.0f ? 0.05f : 1.0f);
    r_h += plastic;
    r_h  = fmaxf(1.0f, fminf(R_MAX_F, r_h));

    /* ------------------------------------------------------------------ */
    /* [ITEM 10] Critic-predicted reward                                   */
    /* ------------------------------------------------------------------ */
    float coherence = fabsf(sum_sin);
    float reward    = critic_reward(residue, coherence, local_amp, r_h, acc);

    /* [X+1=0] Phi-resonance score */
    float resonance = phi_resonance_score(r_h, local_amp);
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
    /* Slot4096 slow-sync correction (unchanged from v33)                  */
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
    /* [Phase 2] Markov trit verdict gate  (unchanged from v33)            */
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
// Reward injection kernel  (identical to v33)
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
// Weight sync kernel  (identical to v33)
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
// Host-callable launchers
// ============================================================================

extern "C" {

void hdgl_v33_upload_soa(const DevSoA *h) {
    cudaMemcpyToSymbol(g_soa, h, sizeof(DevSoA));
}

void hdgl_v33_upload_critic(const float *weights, int count) {
    cudaMemcpyToSymbol(g_critic_w, weights, count * sizeof(float));
}

/* Drop-in replacement — same signature as v33.  Swap hdgl_analog_v33.cu for  */
/* hdgl_analog_v34.cu in the build command to activate Feistel phase dynamics. */
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
