// ============================================================================
// HDGL Analog v33 — Wavelet Spectral Basis + Learned Critic Integration
// ============================================================================
//
// CHANGES FROM v32:
//   [ITEM 11] Wavelet spectral basis replaces 4-harmonic Fourier
//             4-scale Morlet-like wavelets: ψ_k(φ) = cos(2^k φ) · exp(−φ²/2σ_k²)
//             σ_k = π / 2^k  →  scale 0 catches global phase, scale 3 catches fine detail
//   [ITEM 10] Learned critic reward replaces hand-tuned formula
//             GPU kernel computes raw features; reward comes from packed NN weights
//             uploaded as __constant__ symbol CriticWeights g_critic_w
//
//   All v32 items (8,9) preserved: warp spectral pooling, resonance clustering
//
// Compile:
//   nvcc -O3 -arch=sm_75 -lineinfo hdgl_analog_v33.cu hdgl_warp_ll_v33.cu \
//        hdgl_host_v33.c hdgl_critic_v33.c -o hdgl_v33 -lm
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

// ============================================================================
// Shared types
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
    float    *w_cos;         // [N * SPECTRAL_N] — wavelet cosine weights
    float    *w_sin;         // [N * SPECTRAL_N] — wavelet sine weights
    float    *w_sigma;       // [N * SPECTRAL_N] — per-slot per-scale envelope width σ_k (learnable)
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
#define CRITIC_W_TOTAL 57  // 5*8 + 8 + 8 + 1

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
// [ITEM 11] Wavelet spectral evaluation
//
// Replaces: spec += wc[k]*cos(k*dphi) + ws[k]*sin(k*dphi)
// With:     spec += wc[k]*cos(2^k * dphi) * gauss_k(dphi)
//                 + ws[k]*sin(2^k * dphi) * gauss_k(dphi)
//
// where gauss_k(φ) = exp(−φ² / (2 σ_k²)),  σ_k = π / 2^k
//
// Learnable σ_k: w_sigma[k] starts at σ_k and is Hebbian-adjusted,
// allowing each slot to tune its frequency-scale sensitivity.
//
// The Gaussian envelope makes the wavelet localised in phase space:
//   scale 0 (σ=π)      → broad, detects global phase alignment
//   scale 1 (σ=π/2)    → medium, typical Kuramoto coupling
//   scale 2 (σ=π/4)    → fine, detects sharp phase transitions
//   scale 3 (σ=π/8)    → very fine, high-frequency resonance
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
        float freq  = (float)(1 << k);          // 2^k
        float sigma = wsig[k];                  // learnable per-slot per-scale
        float sigma2 = sigma * sigma + 1e-6f;   // avoid div-by-zero
        float gauss  = expf(-phi2 / (2.0f * sigma2));
        float kd     = freq * dphi;
        spec += gauss * (wc[k] * cosf(kd) + ws[k] * sinf(kd));
    }
    spec += w_self  * local_amp;
    spec += w_neigh * neigh_amp;
    return spec;
}

// ============================================================================
// [X+1=0] Phi-resonance prime gate  (ref: zchg.org/t/x-1-0/955)
//
// ── Physical identity (Step 2 of the framework) ──────────────────────────────
//   X = e^(iπ) = 1/φ − φ = Ω·C² − 1    →    X + 1 = 0
//   Ω = Ohms (impedance), C = Coulombs (charge)
//   |Ω·C²| = 1   →   Ω·C² ∈ U(1)          ← U(1) normalization constraint
//
//   Since Ω ∈ ℝ  (defined as [1+sin(π{Λ_φ}φ)]/2 below), U(1) normalization
//   requires  C²(Λ_φ) = 1/Ω(Λ_φ),  so  Ω·C² = 1  (real, unit magnitude).
//
//   Consequence:  S(p) = |Ω·C²·e^(iπΛ_φ) + 1_eff|
//                       = |e^(iπΛ_φ) + 1_eff|          ← C² cancels exactly
//
// ── Λ_φ projection (Step B) ──────────────────────────────────────────────────
//   Λ_φ = log(p·ln2/lnφ) / lnφ − 1/(2φ)
//   Mersenne bridge: p·ln2/lnφ maps the Mersenne exponent into φ-log space
//   p ← r_harmonic (GRA plasticity tracks the Mersenne exponent per slot)
//
// ── 1_eff correction (Step 3 / Step 5) ───────────────────────────────────────
//   1_eff(i) = 1 + δ(i)
//   δ(i) = |cos(π·β·φ)| · ln(n+2) / φ^(n+β)    n=⌊Λ_φ⌋, β={Λ_φ}
//   (P_n approximated by n+2 under log — avoids prime table on device)
//   δ → 0 as n → ∞ (large p):  classical Euler identity recovered at macro scale
//   δ ≠ 0 for small n:  lattice correction gives small but nonzero prime residue
//
// ── Resonance (Step D) ───────────────────────────────────────────────────────
//   S(p) = |e^(iπΛ_φ) + 1_eff|
//        = √((cos(πΛ_φ) + 1_eff)² + sin²(πΛ_φ))
//   Minimum at Λ_φ = odd integer:  S ≈ δ ≈ 0   (prime: destructive interference)
//   Maximum at Λ_φ = even integer: S ≈ 2 + δ   (composite: constructive)
//
// Range ≈ [0, 2]:  primes → 0,  composites → 2
// ============================================================================
__device__ __forceinline__ float phi_resonance_score(float r_harmonic, float /*local_amp*/)
{
    const float PHI       = 1.6180339887f;
    const float LN_PHI    = 0.4812118251f;  // ln(φ)
    const float LN2_LNPHI = 1.4404200904f;  // ln2/lnφ — Mersenne bridge
    const float INV_2PHI  = 0.3090169944f;  // 1/(2φ)
    const float PI        = 3.1415926536f;

    if (r_harmonic < 2.0f) return 2.0f;     // guard: exponent must be ≥ 2

    // (B) Λ_φ = log(p·ln2/lnφ) / lnφ − 1/(2φ)
    float M_U        = r_harmonic * LN2_LNPHI;
    float lambda_phi = logf(M_U) / LN_PHI - INV_2PHI;

    float n_f = floorf(lambda_phi);          // n = ⌊Λ_φ⌋
    float beta = lambda_phi - n_f;           // β = {Λ_φ}

    // (Step 3) δ(i) = |cos(π·β·φ)| · ln(n+2) / φ^(n+β)
    // φ^(n+β) grows rapidly → δ → 0 at large scale (macro limit)
    float n_val  = n_f + 1.0f;              // shift to avoid log/exp edge at n=0
    float delta  = fabsf(cosf(PI * beta * PHI))
                 * logf(n_val + 2.0f)
                 / expf((n_val + beta) * LN_PHI);

    float one_eff = 1.0f + delta;           // 1_eff(i) = 1 + δ(i)

    // (D) S(p) = |e^(iπΛ_φ) + 1_eff|    (C²=1/Ω cancels with Ω → U(1) = 1)
    float re = cosf(PI * lambda_phi) + one_eff;
    float im = sinf(PI * lambda_phi);
    return sqrtf(re * re + im * im);         // ≈ [0, 2]; primes → δ ≈ 0
}

//
// Computes reward = MLP_forward(features) using g_critic_w.
// Features: [residue, coherence, amp, r_h_norm, acc_norm]
// ============================================================================

__device__ float critic_reward(
    float residue, float coherence, float amp, float r_h, float acc)
{
    // Pack features (rough online normalisation via fixed scale factors)
    float s[CRITIC_IN_D] = {
        residue,
        coherence * 0.25f,
        amp,
        r_h * 1e-3f,     // / R_MAX_F
        acc * 0.1f        // / 10
    };

    // Unpack weights from g_critic_w
    // w1: [HIDE × IN] starting at 0
    // b1: [HIDE] starting at HIDE*IN = 40
    // w2: [HIDE] starting at 48
    // b2: [1] at 56
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
        h[j] = z > 0.0f ? z : 0.0f;  // ReLU
    }

    float out = b2;
    for (int j = 0; j < CRITIC_HIDE_D; j++) out += w2[j] * h[j];

    // Clamp to reasonable reward range [0, 5]
    return fmaxf(0.0f, fminf(5.0f, out));
}

// ============================================================================
// Main field evolution kernel v33
// [ITEM 11] Wavelet spectral basis
// [ITEM 10] Critic-predicted reward
// [ITEM 8]  Warp spectral pooling (preserved from v32)
// [ITEM 9]  Resonance clustering (preserved from v32)
// ============================================================================

__global__ void hdgl_field_step_kernel(void)
{
    const int i   = blockIdx.x * blockDim.x + threadIdx.x;
    const int tid = threadIdx.x;
    const int N   = g_soa.N;
    const int S   = g_soa.S;
    if (i >= N) return;

    // Shared memory: wavelet Hebbian gradient pooling [ITEM 8 + ITEM 11]
    // We pool gradients for both w_cos and w_sin (4 each) + w_sigma (4 each) = 12 arrays
    __shared__ float sh_gc[SPECTRAL_N][FIELD_BLOCK];
    __shared__ float sh_gs[SPECTRAL_N][FIELD_BLOCK];
    __shared__ float sh_gsg[SPECTRAL_N][FIELD_BLOCK];  // sigma gradients

    // [Phase 2] Markov trit block-level aggregation (phi_pos, phi_neg, gamma)
    __shared__ float s_phi_pos;
    __shared__ float s_phi_neg;
    __shared__ float s_gamma_sum;

    // Load SoA
    float A_re   = g_soa.A_re[i];
    float A_im   = g_soa.A_im[i];
    float ph     = g_soa.phase[i];
    float phvel  = g_soa.phase_vel[i];
    float r_h    = g_soa.r_harmonic[i];
    uint32_t s   = g_soa.ll_state[i];
    float acc    = g_soa.reward_accum[i];
    float dt     = g_soa.dt;
    float omega  = g_soa.omega;

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

    // LL-lite
    #pragma unroll
    for (int k = 0; k < LL_LITE_ITERS; k++) s = ll_step32(s);
    float residue = residue_from_ll(s);

    // 4-neighbour von Neumann
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

        // [ITEM 11] wavelet spectral coupling
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

    float dphvel = omega + K_COUPLING_F * sum_sin + 0.15f * gra_sum;

    // Euler
    A_re  += dt * dA_re;
    A_im  += dt * dA_im;
    ph    += dt * phvel;
    phvel += dt * dphvel;

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

    ph = fmodf(ph, 6.28318530717958647f);
    if (ph < 0.0f) ph += 6.28318530717958647f;

    // GRA plasticity
    float plastic = GRA_PLASTIC_F * (local_amp - 0.5f) * (r_h > 50.0f ? 0.05f : 1.0f);
    r_h += plastic;
    r_h  = fmaxf(1.0f, fminf(R_MAX_F, r_h));

    // [ITEM 10] Critic-predicted reward (replaces hand formula)
    float coherence = fabsf(sum_sin);
    float reward    = critic_reward(residue, coherence, local_amp, r_h, acc);

    // [X+1=0] Phi-resonance score: S(p) = |Ω·C²·e^(iπΛ_φ)+1|, primes → 0
    // r_h is our Mersenne exponent proxy; local_amp reserved for future C²(Λ_φ)
    float resonance = phi_resonance_score(r_h, local_amp);  // ∈ [0,2]
    // Reward bonus for destructive interference: max at S=0, zero at S=1
    float res_bonus = fmaxf(0.0f, 1.0f - resonance * 0.5f);
    reward = reward + res_bonus;

    // [X+1=0 §3] κ·log(p) field injection — closes U^(p) loop:
    //   U^(p) = φ^(Σ φ^(Σ φ^(interaction(U_i,U_j) + κ·log(p))))
    //   Adding κ·log(r_h) to dphvel biases each slot's phase dynamics
    //   toward the resonant configuration for exponent p ≈ r_h
    const float KAPPA_PHI = 0.02f;  // tunable; small enough not to overwhelm spectral coupling
    dphvel += KAPPA_PHI * logf(r_h + 1.0f);

    // =========================================================================
    // [ITEM 11 + ITEM 8] Wavelet Hebbian update with block pooling
    //
    // For each scale k, the gradient has three components:
    //   ∂L/∂wc_k  = reward * gauss_k(dphi) * cos(2^k * dphi)
    //   ∂L/∂ws_k  = reward * gauss_k(dphi) * sin(2^k * dphi)
    //   ∂L/∂σ_k   = reward * wc_k * cos * gauss * (φ²/σ_k³)    (envelope stretch)
    //             + reward * ws_k * sin * gauss * (φ²/σ_k³)
    //
    // Block-pooled blend (POOL_ALPHA=0.7): 70% local, 30% block average
    // =========================================================================
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

        // σ gradient: d/dσ [gauss] = gauss * (φ² / σ³)
        float dgauss_dsig = gauss * phi2 / (sigma2 * sigma + 1e-8f);
        grad_sig[k] = LR * reward * dgauss_dsig
                    * (wc_local[k] * coskd + ws_local[k] * sinkd);

        sh_gc[k][tid]  = grad_cos[k];
        sh_gs[k][tid]  = grad_sin[k];
        sh_gsg[k][tid] = grad_sig[k];
    }
    __syncthreads();

    // Tree reduce for block average [ITEM 8]
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
        // σ clamped to [π/16, π]: allow narrow but prevent degenerate envelopes
        wsig_local[k] = fmaxf(0.19635f, fminf(3.14159f, wsig_local[k]));
    }

    // Temporal accumulator
    acc = REWARD_DECAY * acc + reward;

    // Write back
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

    // =========================================================================
    // [Phase 2] Markov trit verdict gate — replaces threshold gate
    //
    // lambda_k = local amplitude (spectral resonance proxy).
    // Warp-reduce → lambda_bar.  Softmax logits → LCG sample → sigma_trit ∈ {-1,0,+1}.
    // Warp majority-vote correction (>16/32 agree → override minority).
    // Block-aggregate phi_pos, phi_neg, gamma; compute_verdict replaces threshold.
    //
    // Verdict rule (from conscious_fused_engine.cu):
    //   phi_neg > 0.45         → REJECT
    //   R = 1.2·φ- + 0.8·γ - φ+ > 0.6 → REJECT
    //   phi_pos > 0.35         → ACCEPT  (prime lock signal)
    //   else                   → UNCERTAIN
    // =========================================================================

    float lambda_k = local_amp;

    // Warp-reduce lambda_bar (5-stage shuffle, broadcast from lane 0)
    float lk_sum = lambda_k;
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        lk_sum += __shfl_down_sync(0xffffffffu, lk_sum, off);
    float lambda_bar = __shfl_sync(0xffffffffu, lk_sum, 0) * (1.0f / 32.0f);

    // Numerically-stable softmax over three logits
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

    // LCG sample (uses ll_state + acc for per-step variation)
    uint32_t lcg = s ^ __float_as_uint(acc);
    lcg = lcg * 1664525u + 1013904223u;
    float u_lcg  = (float)(lcg >> 8) * (1.0f / 16777216.0f);
    int sigma_trit;
    if      (u_lcg < e_neg)            sigma_trit = -1;
    else if (u_lcg < e_neg + e_zero)   sigma_trit =  0;
    else                               sigma_trit = +1;

    // Warp majority-vote correction (>16/32 agree → override)
    unsigned b_pos_w = __ballot_sync(0xffffffffu, sigma_trit > 0);
    unsigned b_neg_w = __ballot_sync(0xffffffffu, sigma_trit < 0);
    if      (__popc(b_pos_w) > 16) sigma_trit = +1;
    else if (__popc(b_neg_w) > 16) sigma_trit = -1;

    // Block-level phi accumulation — initialize in thread 0, then sync
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

    // compute_verdict
    int verdict;  /* 0=UNCERTAIN 1=ACCEPT 2=REJECT */
    if (phi_neg > 0.45f) {
        verdict = 2;
    } else {
        float R = 1.2f * phi_neg + 0.8f * gamma_v - phi_pos;
        if      (R       > 0.6f)  verdict = 2;
        else if (phi_pos > 0.35f) verdict = 1;
        else                      verdict = 0;
    }

    /* Temporal stability guard: acc > CAND_ACCUM_THRESH ensures the field has
     * settled (sustained signal, not transient noise) before applying the verdict.
     * The trit verdict provides spatial coherence; acc threshold is temporal. */
    int meets_gate = (verdict == 1 && acc > CAND_ACCUM_THRESH) ? 1 : 0;

    unsigned resonance_mask = __ballot_sync(0xffffffff, meets_gate);
    int      cluster_size   = __popc(resonance_mask);

    if (meets_gate && cluster_size >= CLUSTER_QUORUM) {
        int lane_in_warp = tid & 31;
        int elected_lane = __ffs((int)resonance_mask) - 1;
        if (lane_in_warp == elected_lane) {
            // Score = (2 − S(p)) × cluster_boost
            // Higher score = stronger destructive interference = better prime candidate
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
// Reward injection kernel (identical to v32)
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
// Weight sync kernel (identical to v32 logic, extended to include w_sigma)
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

// Upload packed critic weights (57 floats) to __constant__ g_critic_w
void hdgl_v33_upload_critic(const float *weights, int count) {
    cudaMemcpyToSymbol(g_critic_w, weights, count * sizeof(float));
}

void hdgl_v33_field_step(int N, int block, cudaStream_t s) {
    hdgl_field_step_kernel<<<(N + block - 1) / block, block, 0, s>>>();
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
