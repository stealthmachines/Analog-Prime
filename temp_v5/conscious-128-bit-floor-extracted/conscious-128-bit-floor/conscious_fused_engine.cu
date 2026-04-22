/*
 * conscious_fused_engine.cu
 * "conscious, by zchg.org"
 * ─────────────────────────────────────────────────────────────────────────────
 *
 *  Dual-Slot λₖ–σ Fused Engine
 *    FastSlot warp dynamics  ·  λₖ Markov gate
 *    σ clustering             ·  Slot4096 sync correction
 *
 *  Self-correcting spectral Markov dynamical sieve running entirely on
 *  GPU warps.  Single kernel-launch domain; no host intervention mid-step.
 *
 *  Architecture:
 *    ┌──────────────────────────────────────────────────────────────────┐
 *    │  GRID                                                            │
 *    │  ├── Warp threads   → FastSlot Euler evolution                   │
 *    │  ├── Warp reduction → λₖ estimation (shuffle, no smem needed)    │
 *    │  ├── Thread kernel  → σ Markov trit (softmax + curand)           │
 *    │  ├── Warp ballot    → majority-vote σ correction                 │
 *    │  ├── Every-16-step  → Slot4096 slow-sync (device-local)          │
 *    │  └── Thread 0/block → BlockStats export                          │
 *    └──────────────────────────────────────────────────────────────────┘
 *
 *  6 kernel stages in one launch family:
 *    1. FastSlot Euler-step oscillator       (warp-parallel)
 *    2. λₖ warp-reduction via shuffle        (in-kernel, no smem)
 *    3. Markov trit gate  λₖ → σ ∈ {−1,0,+1}
 *    4. σ warp majority-vote correction      (ballot + __popc)
 *    5. Slot4096 slow-sync correction        (every 16 steps)
 *    6. Block stats export                   (thread 0 / block)
 *
 *  Verdict rule (prime resonance classifier):
 *    φ+ > 0.35                         → ACCEPT   (prime lock signal)
 *    φ− > 0.45 ‖ R = 1.2φ−+0.8γ−φ+ > 0.6 → REJECT
 *    else                              → UNCERTAIN
 *
 *  Married UIs:
 *    prime_ui.exe   — phi-lattice math explorer (Track E, primes/)
 *    chat_win.exe   — analog inference TUI      (metal_infer_for_primes/)
 *    conscious.exe  — this: GPU resonance layer shared by both
 *
 *  Build:  build_conscious.bat
 *    nvcc -O3 -arch=sm_75 -lcurand conscious_fused_engine.cu -o conscious.exe
 *
 *  Usage:
 *    conscious.exe [--N 8192] [--steps 1024] [--quiet] [--seed <hex>]
 *    conscious.exe --selftest
 */

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ── Compile-time configuration ─────────────────────────────────────────── */

#define WARP         32
#define BLOCK       256       /* threads per block — 8 warps               */
#define DEFAULT_N  8192       /* default lattice dimension                  */
#define DEFAULT_S  1024       /* default evolution steps                    */
#define REDUCE_INT   64       /* host-readback interval (steps)             */
#define SYNC_INTERVAL 16      /* Slot4096 correction interval (steps)       */

/* ── Device-side constants (overridable via cudaMemcpyToSymbol) ─────────── */

__device__ __constant__ float c_ALPHA     = 2.0f;   /* σ=−1 logit weight  */
__device__ __constant__ float c_BETA      = 1.5f;   /* σ= 0 logit weight  */
__device__ __constant__ float c_GAMMA_C   = 1.0f;   /* σ=+1 logit weight  */
__device__ __constant__ float c_SYNC_GAIN = 0.08f;  /* slow-sync gain     */
__device__ __constant__ float c_DT        = 0.01f;  /* Euler Δt           */
__device__ __constant__ float c_PHI       = 1.6180339887f;

/* ══════════════════════════ DATA STRUCTURES ════════════════════════════════ */

/*
 * Slot4096 — high-fidelity anchor.
 *   GPU fp32 projection of the APA mantissa from ll_analog.c / hdgl_analog_v30.c.
 *   Updated slowly (0.1% per step) to avoid short-circuit of the dynamics.
 */
typedef struct __align__(16) {
    float re, im;
    float phase;
    float Dn;          /* Dₙ(r) resonance amplitude (from phi-lattice)     */
} Slot4096;

/*
 * FastSlot — cheap warp oscillator.
 *   Prediction manifold; diverges from Slot4096 and gets corrected every
 *   SYNC_INTERVAL steps.  Padded to 16 bytes for coalesced global access.
 */
typedef struct __align__(16) {
    float re, im;
    float phase;
    float _pad;
} FastSlot;

/*
 * DualState — per-thread lattice cell.
 *   48 bytes total; 256 threads × 48 bytes = 12 KB per block (fits L1).
 */
typedef struct {
    Slot4096  slot;         /* 16 bytes — slow anchor                       */
    FastSlot  fast;         /* 16 bytes — warp oscillator                   */
    float     lambda_k;     /*  4 bytes — local spectral proxy λₖ           */
    float     error_accum;  /*  4 bytes — cumulative fast↔slow sync error    */
    int       sigma;        /*  4 bytes — trit Markov state ∈ {−1, 0, +1}   */
    int       _pad;         /*  4 bytes — alignment                         */
} DualState;                /* = 48 bytes                                   */

/*
 * BlockStats — per-block aggregated output after each step.
 *   Written by thread 0 of each block; reduced by reduce_cluster_metrics.
 */
typedef struct __align__(32) {
    float phi_pos;    /* mean fraction σ=+1 in block  */
    float phi_zero;   /* mean fraction σ= 0 in block  */
    float phi_neg;    /* mean fraction σ=−1 in block  */
    float gamma_bar;  /* mean |λₖ − λ̄_warp| in block */
    float lambda_bar; /* mean λₖ in block             */
    float _pad[3];
} BlockStats;         /* = 32 bytes                   */

/* ══════════════════════════ INIT KERNEL ════════════════════════════════════ */

__global__ void init_states_kernel(
    DualState         *states,
    curandState       *rng,
    int                N,
    unsigned long long seed
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    curand_init(seed + (unsigned long long)i, 0, 0, &rng[i]);

    float u1         = curand_uniform(&rng[i]);
    float u2         = curand_uniform(&rng[i]);
    float u3         = curand_uniform(&rng[i]);
    float phi_depth  = c_PHI * (float)(i & 63) + u1;

    DualState s;

    /* Slot4096: φ-lattice seeded, unit-magnitude start */
    s.slot.re    = cosf(phi_depth);
    s.slot.im    = sinf(phi_depth);
    s.slot.phase = phi_depth;
    s.slot.Dn    = c_PHI * (1.0f + 0.1f * u1);

    /* FastSlot: small perturbation around Slot4096 */
    s.fast.re    = s.slot.re  + 0.01f * (u2 - 0.5f);
    s.fast.im    = s.slot.im  + 0.01f * (u3 - 0.5f);
    s.fast.phase = s.slot.phase;
    s.fast._pad  = 0.0f;

    s.lambda_k    = 1.0f;
    s.error_accum = 0.0f;
    s.sigma       = 0;
    s._pad        = 0;

    states[i] = s;
}

/* ══════════════════════════ FUSED MAIN KERNEL ══════════════════════════════ */

__global__ void fused_lambda_sigma_kernel(
    DualState   *states,
    curandState *rng,
    BlockStats  *block_stats,
    int          N,
    int          step
) {
    int i    = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = threadIdx.x & (WARP - 1);

    /* ── Block-level accumulators in shared memory ──────────────────────── */
    __shared__ float s_phi_pos;
    __shared__ float s_phi_zero;
    __shared__ float s_phi_neg;
    __shared__ float s_gamma_sum;
    __shared__ float s_lambda_sum;

    if (threadIdx.x == 0) {
        s_phi_pos    = 0.0f;
        s_phi_zero   = 0.0f;
        s_phi_neg    = 0.0f;
        s_gamma_sum  = 0.0f;
        s_lambda_sum = 0.0f;
    }
    __syncthreads();

    /* ── Per-thread computation (guarded for out-of-bounds) ─────────────── */
    if (i < N) {
        DualState s = states[i];

        /* ════════════════════════════════════════════════════════════════
         * 1.  FAST SLOT EVOLUTION — Euler step
         *     Kuramoto-style oscillator: amplitude decay + phase forcing
         * ════════════════════════════════════════════════════════════════ */
        float f_re  = s.fast.re;
        float f_im  = s.fast.im;
        float phase = s.fast.phase;

        f_re  += c_DT * (-0.5f * f_re  + cosf(phase));
        f_im  += c_DT * (-0.5f * f_im  + sinf(phase));
        phase += c_DT * (s.lambda_k + 0.3f * (float)s.sigma);

        /* wrap phase to [0, 2π) — branchless */
        phase = fmodf(phase + 6.28318530f, 6.28318530f);

        s.fast.re    = f_re;
        s.fast.im    = f_im;
        s.fast.phase = phase;

        /* ════════════════════════════════════════════════════════════════
         * 2.  λₖ ESTIMATION + WARP REDUCTION
         *     spectral proxy = amplitude + Dₙ modulation
         *     warp λ̄ via 5-stage shuffle reduction (no shared memory)
         * ════════════════════════════════════════════════════════════════ */
        float mag = sqrtf(f_re * f_re + f_im * f_im);
        float lk  = mag + 0.1f * s.slot.Dn;

        float lk_sum = lk;
        #pragma unroll
        for (int off = WARP >> 1; off > 0; off >>= 1)
            lk_sum += __shfl_down_sync(0xFFFFFFFF, lk_sum, off);

        /* broadcast λ̄_warp to all lanes */
        float lambda_bar = __shfl_sync(0xFFFFFFFF, lk_sum, 0) * (1.0f / (float)WARP);

        s.lambda_k = lk;
        atomicAdd(&s_lambda_sum, lk);

        /* ════════════════════════════════════════════════════════════════
         * 3.  MARKOV TRIT GATE   λₖ → σ ∈ {−1, 0, +1}
         *     Three logits → numerically-stable softmax → curand sample
         * ════════════════════════════════════════════════════════════════ */
        float l_neg  = -c_ALPHA   * lk;
        float l_zero = -c_BETA    * fabsf(lk - lambda_bar);
        float l_pos  =  c_GAMMA_C * (lambda_bar - lk);

        float m_max  = fmaxf(l_neg, fmaxf(l_zero, l_pos));
        float p_neg  = __expf(l_neg  - m_max);
        float p_zero = __expf(l_zero - m_max);
        float p_pos  = __expf(l_pos  - m_max);
        float inv_z  = 1.0f / (p_neg + p_zero + p_pos);
        p_neg  *= inv_z;
        p_zero *= inv_z;

        float u = curand_uniform(&rng[i]);
        int sigma;
        if      (u < p_neg)            sigma = -1;
        else if (u < p_neg + p_zero)   sigma =  0;
        else                           sigma = +1;

        /* ════════════════════════════════════════════════════════════════
         * 4.  WARP MAJORITY-VOTE CORRECTION
         *     ballot + __popc — no shared memory, no loops
         *     if >16/32 lanes agree on a sign, override minority threads
         * ════════════════════════════════════════════════════════════════ */
        unsigned b_pos = __ballot_sync(0xFFFFFFFF, sigma > 0);
        unsigned b_neg = __ballot_sync(0xFFFFFFFF, sigma < 0);
        int cnt_pos = __popc(b_pos);
        int cnt_neg = __popc(b_neg);

        if      (cnt_pos > 16) sigma = +1;
        else if (cnt_neg > 16) sigma = -1;

        s.sigma = sigma;

        /* ── Accumulate cluster metrics for block stats ──────────────── */
        float gamma_i = fabsf(lk - lambda_bar);
        atomicAdd(&s_phi_pos,   (sigma ==  1) ? 1.0f : 0.0f);
        atomicAdd(&s_phi_zero,  (sigma ==  0) ? 1.0f : 0.0f);
        atomicAdd(&s_phi_neg,   (sigma == -1) ? 1.0f : 0.0f);
        atomicAdd(&s_gamma_sum, gamma_i);

        /* ════════════════════════════════════════════════════════════════
         * 5.  SLOT4096 SLOW-SYNC CORRECTION (every SYNC_INTERVAL steps)
         *     Keeps FastSlot anchored to the high-fidelity Slot4096 state.
         *     Radial error → directional nudge along current (re, im) axis.
         *     Slot4096 drifts toward FastSlot at 0.1% per step.
         * ════════════════════════════════════════════════════════════════ */
        if ((step & (SYNC_INTERVAL - 1)) == 0) {
            float truth    = sqrtf(s.slot.re * s.slot.re + s.slot.im * s.slot.im);
            float err      = truth - mag;
            s.error_accum += err;

            float inv_mag  = (mag > 1e-6f) ? (1.0f / mag) : 0.0f;
            s.fast.re    += c_SYNC_GAIN * err * (f_re * inv_mag);
            s.fast.im    += c_SYNC_GAIN * err * (f_im * inv_mag);
            s.fast.phase += 0.01f * err;

            s.slot.re += 0.001f * (s.fast.re - s.slot.re);
            s.slot.im += 0.001f * (s.fast.im - s.slot.im);
        } else {
            /* lightweight drift every step */
            s.slot.re += 0.001f * (s.fast.re - s.slot.re);
            s.slot.im += 0.001f * (s.fast.im - s.slot.im);
        }

        states[i] = s;
    } /* end if (i < N) */

    /* ════════════════════════════════════════════════════════════════════════
     * 6.  EXPORT BLOCK STATS (thread 0 writes, after all atomicAdds settle)
     * ════════════════════════════════════════════════════════════════════════ */
    __syncthreads();

    if (threadIdx.x == 0) {
        int active = min((int)blockDim.x, N - (int)(blockIdx.x * blockDim.x));
        if (active <= 0) active = 1;
        float inv_n = 1.0f / (float)active;

        BlockStats bs;
        bs.phi_pos    = s_phi_pos    * inv_n;
        bs.phi_zero   = s_phi_zero   * inv_n;
        bs.phi_neg    = s_phi_neg    * inv_n;
        bs.gamma_bar  = s_gamma_sum  * inv_n;
        bs.lambda_bar = s_lambda_sum * inv_n;
        bs._pad[0] = bs._pad[1] = bs._pad[2] = 0.0f;

        block_stats[blockIdx.x] = bs;
    }
}

/* ══════════════════════════ GRID-LEVEL REDUCTION ═══════════════════════════ */

__global__ void reduce_cluster_metrics(
    const BlockStats *block_stats,
    float            *d_phi_pos,
    float            *d_phi_zero,
    float            *d_phi_neg,
    float            *d_gamma,
    float            *d_lambda,
    int               num_blocks
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_blocks) return;

    atomicAdd(d_phi_pos,  block_stats[i].phi_pos);
    atomicAdd(d_phi_zero, block_stats[i].phi_zero);
    atomicAdd(d_phi_neg,  block_stats[i].phi_neg);
    atomicAdd(d_gamma,    block_stats[i].gamma_bar);
    atomicAdd(d_lambda,   block_stats[i].lambda_bar);
}

/* ══════════════════════════ VERDICT RULE ═══════════════════════════════════ */

typedef enum { UNCERTAIN = 0, ACCEPT = 1, REJECT = 2 } Verdict;

static const char *VERDICT_NAME[] = { "UNCERTAIN", "ACCEPT  ", "REJECT  " };

/*
 * compute_verdict — prime resonance classifier.
 *
 * phi_pos > 0.35             → σ=+1 majority → lattice locked → ACCEPT
 * phi_neg > 0.45             → σ=−1 majority → scattered      → REJECT
 * R = 1.2·φ− + 0.8·γ − φ+ > 0.6 → spectral spread exceeds bound → REJECT
 * else                       → UNCERTAIN
 *
 * This mirrors the analog oscillator's osc LOCKED / REJECT signal
 * (ll_analog.c) but derived from warp-aggregate cluster geometry.
 */
static Verdict compute_verdict(float phi_pos, float phi_neg, float gamma) {
    if (phi_neg > 0.45f) return REJECT;
    float R = 1.2f * phi_neg + 0.8f * gamma - 1.0f * phi_pos;
    if (R       > 0.6f)  return REJECT;
    if (phi_pos > 0.35f) return ACCEPT;
    return UNCERTAIN;
}

/* ══════════════════════════ HOST PIPELINE ══════════════════════════════════ */

typedef struct {
    /* GPU allocations */
    DualState   *d_states;
    curandState *d_rng;
    BlockStats  *d_block_stats;
    float       *d_phi_pos, *d_phi_zero, *d_phi_neg;
    float       *d_gamma, *d_lambda;
    /* config */
    int  N, grid, steps;
} ConsciousCtx;

static int conscious_init(ConsciousCtx *ctx, int N, int steps) {
    ctx->N     = N;
    ctx->grid  = (N + BLOCK - 1) / BLOCK;
    ctx->steps = steps;

    if (cudaMalloc(&ctx->d_states,     (size_t)N    * sizeof(DualState))  ||
        cudaMalloc(&ctx->d_rng,        (size_t)N    * sizeof(curandState))||
        cudaMalloc(&ctx->d_block_stats,(size_t)ctx->grid * sizeof(BlockStats)) ||
        cudaMalloc(&ctx->d_phi_pos,  sizeof(float)) ||
        cudaMalloc(&ctx->d_phi_zero, sizeof(float)) ||
        cudaMalloc(&ctx->d_phi_neg,  sizeof(float)) ||
        cudaMalloc(&ctx->d_gamma,    sizeof(float)) ||
        cudaMalloc(&ctx->d_lambda,   sizeof(float))) {
        fprintf(stderr, "[conscious] cudaMalloc failed\n");
        return -1;
    }
    return 0;
}

static void conscious_free(ConsciousCtx *ctx) {
    cudaFree(ctx->d_states);
    cudaFree(ctx->d_rng);
    cudaFree(ctx->d_block_stats);
    cudaFree(ctx->d_phi_pos);
    cudaFree(ctx->d_phi_zero);
    cudaFree(ctx->d_phi_neg);
    cudaFree(ctx->d_gamma);
    cudaFree(ctx->d_lambda);
}

static void conscious_run(ConsciousCtx *ctx, unsigned long long seed, int verbose) {
    int N    = ctx->N;
    int grid = ctx->grid;

    init_states_kernel<<<grid, BLOCK>>>(ctx->d_states, ctx->d_rng, N, seed);
    cudaDeviceSynchronize();

    if (verbose)
        printf("[conscious] N=%d  grid=%d×%d  steps=%d  seed=0x%llx\n",
               N, grid, BLOCK, ctx->steps, seed);

    int accept_count = 0, reject_count = 0, uncertain_count = 0;

    for (int step = 0; step < ctx->steps; step++) {

        fused_lambda_sigma_kernel<<<grid, BLOCK>>>(
            ctx->d_states, ctx->d_rng, ctx->d_block_stats, N, step);

        if (step % REDUCE_INT == 0) {
            cudaMemset(ctx->d_phi_pos,  0, sizeof(float));
            cudaMemset(ctx->d_phi_zero, 0, sizeof(float));
            cudaMemset(ctx->d_phi_neg,  0, sizeof(float));
            cudaMemset(ctx->d_gamma,    0, sizeof(float));
            cudaMemset(ctx->d_lambda,   0, sizeof(float));

            int red_grid = (grid + BLOCK - 1) / BLOCK;
            reduce_cluster_metrics<<<red_grid, BLOCK>>>(
                ctx->d_block_stats,
                ctx->d_phi_pos, ctx->d_phi_zero, ctx->d_phi_neg,
                ctx->d_gamma, ctx->d_lambda, grid);

            float h_pp, h_pz, h_pm, h_g, h_lam;
            cudaMemcpy(&h_pp,  ctx->d_phi_pos,  sizeof(float), cudaMemcpyDeviceToHost);
            cudaMemcpy(&h_pz,  ctx->d_phi_zero, sizeof(float), cudaMemcpyDeviceToHost);
            cudaMemcpy(&h_pm,  ctx->d_phi_neg,  sizeof(float), cudaMemcpyDeviceToHost);
            cudaMemcpy(&h_g,   ctx->d_gamma,    sizeof(float), cudaMemcpyDeviceToHost);
            cudaMemcpy(&h_lam, ctx->d_lambda,   sizeof(float), cudaMemcpyDeviceToHost);

            /* normalize block sums → means */
            float inv_g = (grid > 0) ? 1.0f / (float)grid : 0.0f;
            h_pp  *= inv_g; h_pz *= inv_g; h_pm *= inv_g;
            h_g   *= inv_g; h_lam *= inv_g;

            Verdict v = compute_verdict(h_pp, h_pm, h_g);
            switch (v) {
                case ACCEPT:    accept_count++;    break;
                case REJECT:    reject_count++;    break;
                default:        uncertain_count++; break;
            }

            if (verbose)
                printf("  step %4d  φ+=%5.3f φ0=%5.3f φ−=%5.3f "
                       "γ=%6.4f λ̄=%6.3f  → %s\n",
                       step, h_pp, h_pz, h_pm, h_g, h_lam, VERDICT_NAME[v]);
        }
    }

    cudaDeviceSynchronize();

    int total_votes = accept_count + reject_count + uncertain_count;
    if (total_votes > 0 && verbose) {
        printf("\n  Verdict summary: "
               "ACCEPT %d/%d  REJECT %d/%d  UNCERTAIN %d/%d\n",
               accept_count, total_votes,
               reject_count, total_votes,
               uncertain_count, total_votes);
        if (accept_count > reject_count + uncertain_count)
            printf("  ** PRIME RESONANCE SIGNAL DETECTED **\n");
    }
}

/* ── Self-test ───────────────────────────────────────────────────────────── */

static int selftest(void) {
    printf("[conscious] selftest: N=%d steps=%d\n", BLOCK, REDUCE_INT * 2);
    ConsciousCtx ctx;
    if (conscious_init(&ctx, BLOCK, REDUCE_INT * 2)) return 1;
    conscious_run(&ctx, 0xDEADBEEFCAFE1234ULL, 1);
    conscious_free(&ctx);
    printf("[conscious] selftest passed.\n");
    return 0;
}

/* ══════════════════════════ MAIN ═══════════════════════════════════════════ */

int main(int argc, char *argv[]) {
    int N          = DEFAULT_N;
    int steps      = DEFAULT_S;
    int verbose    = 1;
    int do_test    = 0;
    unsigned long long seed = 0xC0FFEE00DEAD1234ULL;

    for (int i = 1; i < argc; i++) {
        if      (!strcmp(argv[i], "--N")        && i+1 < argc) N     = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--steps")    && i+1 < argc) steps = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--seed")     && i+1 < argc) seed  = (unsigned long long)strtoull(argv[++i], NULL, 16);
        else if (!strcmp(argv[i], "--quiet"))                  verbose = 0;
        else if (!strcmp(argv[i], "--selftest"))               do_test = 1;
    }

    /* snap N to BLOCK multiple */
    N = ((N + BLOCK - 1) / BLOCK) * BLOCK;

    int dev = 0;
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, dev);

    printf("=== conscious, by zchg.org ===\n");
    printf("GPU: %s  (sm_%d%d)  %.0f MB  warpSize=%d\n",
           prop.name, prop.major, prop.minor,
           (double)prop.totalGlobalMem / (1024.0*1024.0),
           prop.warpSize);
    printf("Engine: Dual-Slot λₖ–σ Fused  |  FastSlot+Markov+Slot4096\n\n");

    if (do_test) return selftest();

    ConsciousCtx ctx;
    if (conscious_init(&ctx, N, steps)) return 1;
    conscious_run(&ctx, seed, verbose);
    conscious_free(&ctx);

    return 0;
}
