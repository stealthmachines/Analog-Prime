// ============================================================================
// HDGL Host v33 — CPU Orchestration (wavelet + learned critic)
// ============================================================================
//
// Extends hdgl_host_v32.c:
//   • DevSoA has w_sigma field (learnable wavelet envelope widths)
//   • After each weight sync: critic update → pack weights → upload to GPU
//   • Critic builds TD targets from LL residue observations
//   • Otherwise identical 3-stream async pattern
//
// Compile:
//   nvcc -O3 -arch=sm_75 -lineinfo hdgl_analog_v33.cu hdgl_warp_ll_v33.cu \
//        hdgl_host_v33.c hdgl_critic_v33.c -o hdgl_v33 -lm
// ============================================================================

#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <signal.h>
#include "hdgl_critic_v33.h"
#include "hdgl_psi_filter_v35.h"
#include "hdgl_predictor_seed.h"
#include "hdgl_prismatic_v35.h"

// Windows-compatible high-resolution timer
#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
typedef struct { LARGE_INTEGER v; } v33_timespec_t;
static LARGE_INTEGER g_qpf;  // counts per second
static void v33_clock_init(void) { QueryPerformanceFrequency(&g_qpf); }
static void v33_gettime(v33_timespec_t *t) { QueryPerformanceCounter(&t->v); }
static double v33_elapsed(const v33_timespec_t *t0, const v33_timespec_t *t1) {
    return (double)(t1->v.QuadPart - t0->v.QuadPart) / (double)g_qpf.QuadPart;
}
#else
#include <time.h>
typedef struct timespec v33_timespec_t;
static void v33_clock_init(void) {}
static void v33_gettime(v33_timespec_t *t) { clock_gettime(CLOCK_MONOTONIC, t); }
static double v33_elapsed(const v33_timespec_t *t0, const v33_timespec_t *t1) {
    return (t1->tv_sec - t0->tv_sec) + (t1->tv_nsec - t0->tv_nsec) * 1e-9;
}
#endif

// ============================================================================
// Forward declarations from CUDA translation units
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
    float    *w_cos;
    float    *w_sin;
    float    *w_sigma;    // new in v33: learnable wavelet envelope widths
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

#define SPECTRAL_N      4
#define MAX_CAND_BUF    256
#define CRITIC_W_TOTAL  57

/* How often (main cycles) to run the sieve pipeline */
#define SIEVE_PIPELINE_PERIOD  10
/* Top-K sieve survivors to submit to exact LL */
#define SIEVE_LL_TOP_K  4

/* How often to inject a known-prime "scent strip" into critic training.
 * Without this, the critic only ever sees composite (reward=-1) examples
 * because warp_ll at p_bits=127 always returns residue~0.5 (NTT proxy,
 * not exact LL). One positive per INJECT_PERIOD composite batches prevents
 * the critic from collapsing to an all-negative prior.
 * N=32 per cycle vs ~256 composites/cycle = 1:8 ratio — keeps b2 calibrated. */
#define CRITIC_PRIME_INJECT_N       32   /* prime examples injected each cycle */

// hdgl_analog_v33.cu
void hdgl_v33_upload_soa(const DevSoA *h);
void hdgl_v33_upload_critic(const float *w, int n);
void hdgl_v33_field_step(int N, int block, int step_count, cudaStream_t s);
void hdgl_v33_reward_inject(int N, int block, cudaStream_t s);
void hdgl_v33_weight_sync(cudaStream_t s);
void hdgl_v33_read_global_weights(float oc[SPECTRAL_N], float os[SPECTRAL_N],
                                   float osg[SPECTRAL_N]);

// hdgl_warp_ll_v33.cu
typedef struct { uint64_t S[64]; uint64_t C[64]; } CSSState;
void hdgl_warp_ll_v33_alloc(CSSState **out);
void hdgl_warp_ll_v33_free(CSSState *p);
void hdgl_warp_ll_v33_seed(CSSState *d, int n, cudaStream_t s);
void hdgl_warp_ll_v33_launch(const Candidate *d_cands, int n_cands,
                              int p_bits, int iters, CSSState *d_css,
                              float *d_residue, int8_t *d_verified,
                              float eps, cudaStream_t s);
int  hdgl_gpucarry_ll(uint32_t p);

// hdgl_sieve_v34.cu
typedef struct {
    uint32_t *d_assigned_p, *d_sieve_state;
    float    *d_r_harmonic, *d_phase, *d_reward_accum;
    uint32_t *d_prime_found;
    int      *d_prime_count;
    uint32_t *h_prime_found;
    int      *h_prime_count;
    int N, S;
    float dt;
} SieveHostState;
int  hdgl_sieve_alloc(SieveHostState *st, int N, int S, float dt);
void hdgl_sieve_free(SieveHostState *st);
void hdgl_sieve_step(const SieveHostState *st, cudaStream_t s);
int  hdgl_sieve_harvest(SieveHostState *st, uint32_t *out, int max, cudaStream_t s);
void hdgl_sieve_seed_priority(SieveHostState *st, const uint32_t *p_list, int n);

// ============================================================================
// Configuration
// ============================================================================

#define DEFAULT_N           (1 << 20)
#define DEFAULT_S           1024
#define STEPS_PER_SYNC      64
#define BLOCK_SIZE          256
#define SYNC_CYCLES         1000

#define LL_P_BITS           127
#define LL_ITERS            100
#define LL_RESIDUE_EPS      1e-6f

#define WEIGHT_MOMENTUM     0.95f

#define CRITIC_CKPT_FILE    "critic_checkpoint.bin"
#define CRITIC_CKPT_PERIOD  100   /* save checkpoint every N cycles */

// Initial wavelet envelope widths: σ_k = π / 2^k
static const float SIGMA_INIT[SPECTRAL_N] = {
    3.14159265f,   // scale 0: π
    1.57079633f,   // scale 1: π/2
    0.78539816f,   // scale 2: π/4
    0.39269908f    // scale 3: π/8
};

// ============================================================================
// Global running flag
// ============================================================================

static volatile int g_running = 1;
static void signal_handler(int sig) { (void)sig; g_running = 0; }

// ============================================================================
// CUDA error check
// ============================================================================

#define CUDA_CHECK(c) do {                                               \
    cudaError_t _e = (c);                                                \
    if (_e != cudaSuccess) {                                             \
        fprintf(stderr, "[v33] CUDA error %s:%d: %s\n",                 \
                __FILE__, __LINE__, cudaGetErrorString(_e));             \
        exit(EXIT_FAILURE);                                              \
    }                                                                    \
} while (0)

// ============================================================================
// Host state
// ============================================================================

typedef struct {
    // Device arrays (owned, allocated by alloc_device_arrays)
    float    *d_A_re, *d_A_im;
    float    *d_phase, *d_phase_vel;
    float    *d_r_harmonic;
    uint32_t *d_ll_state;
    float    *d_reward_accum;
    float    *d_w_cos, *d_w_sin, *d_w_sigma;
    int8_t   *d_ll_verified;
    Candidate *d_candidates;
    int      *d_cand_count;
    float    *d_ll_residue;
    CSSState *d_css_states;

    // Pinned host buffers
    Candidate *h_candidates;
    int       *h_cand_count;
    float     *h_ll_residue;

    // CUDA streams and events
    cudaStream_t stream0, stream1, stream2;
    cudaEvent_t  ev_start, ev_field_done, ev_ll_done;

    // Configuration
    int   N, S, p_bits;
    float dt, omega;

    // Host-side momentum weights
    float gw_c[SPECTRAL_N], gw_s[SPECTRAL_N], gw_sg[SPECTRAL_N];

    // Critic weight buffer
    float critic_packed[CRITIC_W_TOTAL];

    // Counters
    long long total_cycles;
    long long total_candidates;
    long long total_steps;     /* cumulative field steps for slow-sync dispatch */
    int       verified_primes;

    // Sieve pipeline (Ev2 Medium)
    SieveHostState sieve;
    PsiFilterState psi_filter;
    int            sieve_active;
    long long      sieve_candidates_total;
    long long      sieve_psi_survivors;
    long long      sieve_ll_verified;
} HostState;

// ============================================================================
// Allocate device arrays
// ============================================================================

static void alloc_device_arrays(HostState *st) {
    int N = st->N;
    CUDA_CHECK(cudaMalloc(&st->d_A_re,        N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&st->d_A_im,        N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&st->d_phase,       N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&st->d_phase_vel,   N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&st->d_r_harmonic,  N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&st->d_ll_state,    N * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&st->d_reward_accum,N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&st->d_w_cos,  N * SPECTRAL_N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&st->d_w_sin,  N * SPECTRAL_N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&st->d_w_sigma,N * SPECTRAL_N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&st->d_ll_verified, N * sizeof(int8_t)));
    CUDA_CHECK(cudaMalloc(&st->d_candidates,  MAX_CAND_BUF * sizeof(Candidate)));
    CUDA_CHECK(cudaMalloc(&st->d_cand_count,  sizeof(int)));
    CUDA_CHECK(cudaMalloc(&st->d_ll_residue,  MAX_CAND_BUF * sizeof(float)));
    hdgl_warp_ll_v33_alloc(&st->d_css_states);
}

// ============================================================================
// Allocate pinned host buffers
// ============================================================================

static void alloc_pinned_buffers(HostState *st) {
    CUDA_CHECK(cudaMallocHost(&st->h_candidates, MAX_CAND_BUF * sizeof(Candidate)));
    CUDA_CHECK(cudaMallocHost(&st->h_cand_count, sizeof(int)));
    CUDA_CHECK(cudaMallocHost(&st->h_ll_residue, MAX_CAND_BUF * sizeof(float)));
}

// ============================================================================
// Initialise device state
// ============================================================================

static void init_device_state(HostState *st) {
    int N = st->N;
    float *tmp_f = (float *)malloc(N * SPECTRAL_N * sizeof(float));
    float *tmp_u = (float *)malloc(N * sizeof(float));

    // Phase: uniform [0, 2π], phase_vel = omega, A = 1, r_harmonic = 10
    for (int i = 0; i < N; i++) {
        float ph = 6.28318530f * (float)i / (float)N;
        ((float *)tmp_u)[i] = ph;
    }
    CUDA_CHECK(cudaMemcpy(st->d_phase, tmp_u, N * sizeof(float), cudaMemcpyHostToDevice));

    for (int i = 0; i < N; i++) ((float *)tmp_u)[i] = st->omega;
    CUDA_CHECK(cudaMemcpy(st->d_phase_vel, tmp_u, N * sizeof(float), cudaMemcpyHostToDevice));

    for (int i = 0; i < N; i++) ((float *)tmp_u)[i] = 1.0f;
    CUDA_CHECK(cudaMemcpy(st->d_A_re, tmp_u, N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(st->d_A_im, 0, N * sizeof(float)));

    for (int i = 0; i < N; i++) ((float *)tmp_u)[i] = 10.0f;
    CUDA_CHECK(cudaMemcpy(st->d_r_harmonic, tmp_u, N * sizeof(float), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemset(st->d_reward_accum, 0, N * sizeof(float)));
    CUDA_CHECK(cudaMemset(st->d_ll_verified, 0, N * sizeof(int8_t)));
    CUDA_CHECK(cudaMemset(st->d_cand_count, 0, sizeof(int)));

    // Randomise LL states (simple LCG seeded by index)
    uint32_t *ll_tmp = (uint32_t *)malloc(N * sizeof(uint32_t));
    for (int i = 0; i < N; i++) {
        uint32_t v = (uint32_t)(i * 2654435761u + 1013904223u);
        ll_tmp[i] = (v == 0 || v == 2u) ? 3u : v;
    }
    CUDA_CHECK(cudaMemcpy(st->d_ll_state, ll_tmp, N * sizeof(uint32_t), cudaMemcpyHostToDevice));
    free(ll_tmp);

    // Wavelet weights: small random init ±0.1 for w_cos/w_sin; σ from table
    for (int i = 0; i < N; i++) {
        for (int k = 0; k < SPECTRAL_N; k++) {
            int idx = i * SPECTRAL_N + k;
            tmp_f[idx] = 0.1f * (((float)rand() / (float)RAND_MAX) * 2.0f - 1.0f);
        }
    }
    CUDA_CHECK(cudaMemcpy(st->d_w_cos, tmp_f, N * SPECTRAL_N * sizeof(float), cudaMemcpyHostToDevice));

    for (int i = 0; i < N; i++) {
        for (int k = 0; k < SPECTRAL_N; k++) {
            int idx = i * SPECTRAL_N + k;
            tmp_f[idx] = 0.1f * (((float)rand() / (float)RAND_MAX) * 2.0f - 1.0f);
        }
    }
    CUDA_CHECK(cudaMemcpy(st->d_w_sin, tmp_f, N * SPECTRAL_N * sizeof(float), cudaMemcpyHostToDevice));

    // Sigma initialised to SIGMA_INIT for all slots
    for (int i = 0; i < N; i++) {
        for (int k = 0; k < SPECTRAL_N; k++) {
            tmp_f[i * SPECTRAL_N + k] = SIGMA_INIT[k];
        }
    }
    CUDA_CHECK(cudaMemcpy(st->d_w_sigma, tmp_f, N * SPECTRAL_N * sizeof(float), cudaMemcpyHostToDevice));

    free(tmp_f);
    free(tmp_u);
}

// ============================================================================
// Upload SoA struct to device __constant__
// ============================================================================

static void upload_soa(HostState *st) {
    DevSoA soa;
    soa.A_re         = st->d_A_re;
    soa.A_im         = st->d_A_im;
    soa.phase        = st->d_phase;
    soa.phase_vel    = st->d_phase_vel;
    soa.r_harmonic   = st->d_r_harmonic;
    soa.ll_state     = st->d_ll_state;
    soa.reward_accum = st->d_reward_accum;
    soa.w_cos        = st->d_w_cos;
    soa.w_sin        = st->d_w_sin;
    soa.w_sigma      = st->d_w_sigma;
    soa.ll_verified  = st->d_ll_verified;
    soa.candidates   = st->d_candidates;
    soa.cand_count   = st->d_cand_count;
    soa.N            = st->N;
    soa.S            = st->S;
    soa.omega        = st->omega;
    soa.dt           = st->dt;
    soa.w_amp_self   = 1.0f;
    soa.w_amp_neigh  = 0.5f;
    hdgl_v33_upload_soa(&soa);
}

// ============================================================================
// One sync cycle: field × STEPS_PER_SYNC → LL → reward → weight sync → critic
// ============================================================================

static void run_sync_cycle(HostState *st) {
    const int N   = st->N;
    const int blk = BLOCK_SIZE;

    // stream0: field steps
    CUDA_CHECK(cudaEventRecord(st->ev_start, st->stream0));
    for (int s = 0; s < STEPS_PER_SYNC; s++)
        hdgl_v33_field_step(N, blk, (int)(st->total_steps + s), st->stream0);
    st->total_steps += STEPS_PER_SYNC;
    CUDA_CHECK(cudaEventRecord(st->ev_field_done, st->stream0));

    // stream1: harvest candidates → LL
    CUDA_CHECK(cudaStreamWaitEvent(st->stream1, st->ev_start, 0));
    CUDA_CHECK(cudaMemcpyAsync(st->h_cand_count, st->d_cand_count,
                               sizeof(int), cudaMemcpyDeviceToHost, st->stream1));
    CUDA_CHECK(cudaStreamSynchronize(st->stream1));

    int n_cands = *st->h_cand_count;
    if (n_cands > MAX_CAND_BUF) n_cands = MAX_CAND_BUF;

    if (n_cands > 0) {
        CUDA_CHECK(cudaMemcpyAsync(st->h_candidates, st->d_candidates,
                                   n_cands * sizeof(Candidate),
                                   cudaMemcpyDeviceToHost, st->stream1));
        hdgl_warp_ll_v33_seed(st->d_css_states, n_cands, st->stream1);
        hdgl_warp_ll_v33_launch(st->d_candidates, n_cands, st->p_bits,
                                 st->p_bits - 2, st->d_css_states,
                                 st->d_ll_residue, st->d_ll_verified,
                                 LL_RESIDUE_EPS, st->stream1);
        CUDA_CHECK(cudaMemcpyAsync(st->h_ll_residue, st->d_ll_residue,
                                   n_cands * sizeof(float),
                                   cudaMemcpyDeviceToHost, st->stream1));
        st->total_candidates += n_cands;
    }
    CUDA_CHECK(cudaEventRecord(st->ev_ll_done, st->stream1));

    // stream2: reward inject + weight sync (waits for both field and LL)
    CUDA_CHECK(cudaStreamWaitEvent(st->stream2, st->ev_field_done, 0));
    CUDA_CHECK(cudaStreamWaitEvent(st->stream2, st->ev_ll_done, 0));
    hdgl_v33_reward_inject(N, blk, st->stream2);
    hdgl_v33_weight_sync(st->stream2);

    // stream0: reset candidate counter
    CUDA_CHECK(cudaStreamWaitEvent(st->stream0, st->ev_field_done, 0));
    CUDA_CHECK(cudaMemsetAsync(st->d_cand_count, 0, sizeof(int), st->stream0));

    // Sync all before critic update
    CUDA_CHECK(cudaStreamSynchronize(st->stream2));

    // Read global weights and blend with host momentum
    float new_c[SPECTRAL_N], new_s[SPECTRAL_N], new_sg[SPECTRAL_N];
    hdgl_v33_read_global_weights(new_c, new_s, new_sg);
    for (int k = 0; k < SPECTRAL_N; k++) {
        st->gw_c[k]  = WEIGHT_MOMENTUM * st->gw_c[k]  + (1.0f - WEIGHT_MOMENTUM) * new_c[k];
        st->gw_s[k]  = WEIGHT_MOMENTUM * st->gw_s[k]  + (1.0f - WEIGHT_MOMENTUM) * new_s[k];
        st->gw_sg[k] = WEIGHT_MOMENTUM * st->gw_sg[k] + (1.0f - WEIGHT_MOMENTUM) * new_sg[k];
    }

    // Feed LL results to critic for online learning
    CUDA_CHECK(cudaStreamSynchronize(st->stream1));
    if (n_cands > 0) {
        for (int c = 0; c < n_cands; c++) {
            float residue   = st->h_ll_residue[c];
            float rh_norm   = st->h_candidates[c].r_harmonic / 1000.0f;
            float coherence = st->h_candidates[c].coherence;
            float amp       = st->h_candidates[c].amp;
            float acc_norm  = st->h_candidates[c].acc / 10.0f;
            float raw_s[CRITIC_IN] = {
                residue, coherence, amp, rh_norm, acc_norm
            };
            float raw_s_next[CRITIC_IN] = {
                residue * 0.95f, coherence, amp, rh_norm, acc_norm
            };
            // Binary reward: LL passed (residue=0) → +1, failed → -1
            float obs_reward = (residue < LL_RESIDUE_EPS) ? 1.0f : -1.0f;
            float td_target  = critic_td_target(obs_reward, raw_s_next);
            static int s_first_ll_update = 1;
            if (s_first_ll_update) {
                printf("[critic] first LL-residue→reward: residue=%.2e reward=%.1f "
                       "coherence=%.3f amp=%.3f rh_norm=%.3f acc_norm=%.3f\n",
                       (double)residue, (double)obs_reward,
                       (double)coherence, (double)amp,
                       (double)rh_norm, (double)acc_norm);
                fflush(stdout);
                s_first_ll_update = 0;
            }
            critic_observe(raw_s, td_target);

            // Log potential prime discoveries with phi-lattice coordinate
            if (residue < LL_RESIDUE_EPS) {
                st->verified_primes++;
                /* phi-lattice n-coord: n(2^p) = log(p*ln2/ln_phi)/ln_phi - 1/(2*phi) */
                double ln2     = 0.6931471805599453;
                double ln_phi  = 0.4812118250596035;
                double inv2phi = 0.3090169943749474;   /* 1/(2*phi) */
                double n_coord = log((double)st->p_bits * ln2 / ln_phi) / ln_phi - inv2phi;
                double frac_n  = n_coord - (long long)n_coord;
                if (frac_n < 0.0) frac_n += 1.0;
                printf("[cycle %6lld] PRIME CANDIDATE  p=%d  residue=%.2e  score=%.2f"
                       "  n=%.6f  frac(n)=%.4f  (<0.5: %s)\n",
                       st->total_cycles, st->p_bits, (double)residue,
                       (double)st->h_candidates[c].score,
                       n_coord, frac_n,
                       (frac_n < 0.5) ? "YES" : "NO");
                fflush(stdout);
            }
        }
        // Push updated critic weights to GPU
        critic_pack_weights(st->critic_packed);
        hdgl_v33_upload_critic(st->critic_packed, CRITIC_W_TOTAL);
    }

    /* ------------------------------------------------------------------ */
    /* Known-prime corpus injection ("bloodhound scent strip")             */
    /* ------------------------------------------------------------------ */
    /* warp_ll at p_bits=127 is a proxy (NTT only valid for p<4000);      */
    /* every candidate gets reward=-1.  Inject CRITIC_PRIME_INJECT_N      */
    /* prime-like observations every cycle to balance the 1:8 ratio.      */
    /* Feature vector mirrors what a real gpucarry LL prime looks like:   */
    /*   residue=0   coherence=0.92   amp=0.90   r_h_norm=0.10   acc=3.0  */
    {
        const float prime_s[CRITIC_IN]      = { 0.00f, 0.92f, 0.90f, 0.10f, 0.30f };
        const float prime_s_next[CRITIC_IN] = { 0.00f, 0.93f, 0.91f, 0.10f, 0.32f };
        float td = critic_td_target(+1.0f, prime_s_next);
        for (int pi = 0; pi < CRITIC_PRIME_INJECT_N; pi++)
            critic_observe(prime_s, td);
        /* Re-upload so the field sees the refreshed gate */
        critic_pack_weights(st->critic_packed);
        hdgl_v33_upload_critic(st->critic_packed, CRITIC_W_TOTAL);
    }

    st->total_cycles++;
}

// ============================================================================
// Sieve pipeline: harvest → psi filter → prismatic sort → gpucarry LL
// ============================================================================

#define SIEVE_N       4096
#define SIEVE_S       64
#define SIEVE_STEPS   50     /* sieve steps per pipeline call */

static void run_sieve_pipeline(HostState *st)
{
    if (!st->sieve_active) return;

    /* Step the sieve */
    for (int i = 0; i < SIEVE_STEPS; i++)
        hdgl_sieve_step(&st->sieve, 0);

    /* Harvest candidate exponents */
    uint32_t raw[256];
    int n_raw = hdgl_sieve_harvest(&st->sieve, raw, 256, 0);
    if (n_raw <= 0) return;
    st->sieve_candidates_total += n_raw;

    /* Phi lower-half pre-filter (free CPU, kills ~33%) */
    uint32_t phi_pass[256];
    int n_phi = 0;
    for (int i = 0; i < n_raw; i++) {
        if (hdgl_phi_lower_half(raw[i]))
            phi_pass[n_phi++] = raw[i];
    }
    if (n_phi == 0) n_phi = n_raw, memcpy(phi_pass, raw, n_raw * sizeof(uint32_t));

    /* Psi filter (GPU, 3-pass Riemann spike detector) */
    uint32_t psi_out[256];
    float    psi_scores[256];
    int n_psi = hdgl_psi_filter_run(&st->psi_filter,
                                     phi_pass, n_phi,
                                     psi_out, psi_scores, 0);
    if (n_psi <= 0) return;
    st->sieve_psi_survivors += n_psi;

    /* Prismatic sort: build r_h proxy from psi score, sort descending */
    float r_hs[256];
    for (int i = 0; i < n_psi; i++) r_hs[i] = psi_scores[i] + 1.0f;
    hdgl_prismatic_sort(psi_out, r_hs, n_psi);

    /* Submit top-K to exact gpucarry LL */
    int top_k = (n_psi < SIEVE_LL_TOP_K) ? n_psi : SIEVE_LL_TOP_K;
    for (int i = 0; i < top_k; i++) {
        uint32_t p = psi_out[i];
        int result = hdgl_gpucarry_ll(p);
        if (result) {
            st->sieve_ll_verified++;
            st->verified_primes++;
            printf("[SIEVE-PRIME] p=%u  (M_%u = 2^%u - 1 is PRIME!)\n", p, p, p);
            fflush(stdout);
        }
    }
}

// ============================================================================
// Free resources
// ============================================================================

static void free_host_state(HostState *st) {
    cudaFree(st->d_A_re);      cudaFree(st->d_A_im);
    cudaFree(st->d_phase);     cudaFree(st->d_phase_vel);
    cudaFree(st->d_r_harmonic);cudaFree(st->d_ll_state);
    cudaFree(st->d_reward_accum);
    cudaFree(st->d_w_cos);     cudaFree(st->d_w_sin);
    cudaFree(st->d_w_sigma);   cudaFree(st->d_ll_verified);
    cudaFree(st->d_candidates);cudaFree(st->d_cand_count);
    cudaFree(st->d_ll_residue);
    hdgl_warp_ll_v33_free(st->d_css_states);
    cudaFreeHost(st->h_candidates);
    cudaFreeHost(st->h_cand_count);
    cudaFreeHost(st->h_ll_residue);
    cudaStreamDestroy(st->stream0);
    cudaStreamDestroy(st->stream1);
    cudaStreamDestroy(st->stream2);
    cudaEventDestroy(st->ev_start);
    cudaEventDestroy(st->ev_field_done);
    cudaEventDestroy(st->ev_ll_done);
    if (st->sieve_active) {
        hdgl_sieve_free(&st->sieve);
        hdgl_psi_filter_free(&st->psi_filter);
        st->sieve_active = 0;
    }
}

// ============================================================================
// main
// ============================================================================

int main(int argc, char **argv) {
    signal(SIGINT,  signal_handler);
    signal(SIGTERM, signal_handler);

    int   N       = (argc > 1) ? atoi(argv[1]) : DEFAULT_N;
    int   S       = (argc > 2) ? atoi(argv[2]) : DEFAULT_S;
    int   cycles  = (argc > 3) ? atoi(argv[3]) : SYNC_CYCLES;
    int   p_bits  = (argc > 4) ? atoi(argv[4]) : LL_P_BITS;
    float dt      = 0.001f;
    float omega   = 0.1f * 3.14159265f;

    printf("[hdgl_v33] N=%d  S=%d  cycles=%d  p_bits=%d\n", N, S, cycles, p_bits);
    fflush(stdout);

    HostState st;
    memset(&st, 0, sizeof(st));
    st.N      = N;
    st.S      = S;
    st.p_bits = p_bits;
    st.dt     = dt;
    st.omega  = omega;

    srand(42);
    critic_init();
    if (critic_load(CRITIC_CKPT_FILE) == 0) {
        printf("[host] critic weights restored from '%s'\n", CRITIC_CKPT_FILE);
    } else {
        printf("[host] no checkpoint found — starting with fresh critic weights\n");
    }
    fflush(stdout);
    critic_pack_weights(st.critic_packed);

    alloc_device_arrays(&st);
    alloc_pinned_buffers(&st);
    init_device_state(&st);

    CUDA_CHECK(cudaStreamCreate(&st.stream0));
    CUDA_CHECK(cudaStreamCreate(&st.stream1));
    CUDA_CHECK(cudaStreamCreate(&st.stream2));
    CUDA_CHECK(cudaEventCreateWithFlags(&st.ev_start,       cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&st.ev_field_done,  cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&st.ev_ll_done,     cudaEventDisableTiming));

    upload_soa(&st);
    hdgl_v33_upload_critic(st.critic_packed, CRITIC_W_TOTAL);

    /* Sieve pipeline: init sieve, seed with top-20 phi-lattice predictions */
    {
        uint32_t prio[20];
        int n_prio = hdgl_predictor_top20(prio, 20);
        if (hdgl_sieve_alloc(&st.sieve, SIEVE_N, SIEVE_S, 0.001f) == 0) {
            if (n_prio > 0)
                hdgl_sieve_seed_priority(&st.sieve, prio, n_prio);
            hdgl_psi_filter_alloc(&st.psi_filter, 256);
            st.sieve_active = 1;
            printf("[sieve] active: N=%d  priority_seeds=%d\n", SIEVE_N, n_prio);
            fflush(stdout);
        }
    }

    v33_clock_init();
    v33_timespec_t t0, t1;
    v33_gettime(&t0);

    int done = 0;
    while (!done && g_running) {
        run_sync_cycle(&st);
        if (cycles > 0 && st.total_cycles >= cycles) done = 1;

        /* Run sieve pipeline every SIEVE_PIPELINE_PERIOD main cycles */
        if (st.sieve_active && (st.total_cycles % SIEVE_PIPELINE_PERIOD == 0))
            run_sieve_pipeline(&st);

        if (st.total_cycles % 100 == 0) {
            v33_gettime(&t1);
            double elapsed = v33_elapsed(&t0, &t1);
            double mslots_per_sec =
                (double)(st.total_cycles) * STEPS_PER_SYNC * (double)N / elapsed / 1e6;
            printf("[cycle %6lld] cands=%lld  verified=%d  %.1f MSlots/s\n",
                   st.total_cycles, st.total_candidates,
                   st.verified_primes, mslots_per_sec);
            critic_print_stats();
            fflush(stdout);
        }
        if (st.total_cycles % CRITIC_CKPT_PERIOD == 0) {
            if (critic_save(CRITIC_CKPT_FILE) == 0) {
                printf("[host] checkpoint saved  cycle=%lld\n", st.total_cycles);
                fflush(stdout);
            }
        }
    }

    v33_gettime(&t1);
    double elapsed = v33_elapsed(&t0, &t1);
    double mslots_per_sec = (double)(st.total_cycles) * STEPS_PER_SYNC * (double)N
                            / elapsed / 1e6;

    printf("\n[hdgl_v33] FINAL RESULTS\n");
    printf("  cycles        = %lld\n", st.total_cycles);
    printf("  total_cands   = %lld\n", st.total_candidates);
    printf("  verified primes = %d\n", st.verified_primes);
    printf("  wall_time     = %.2f s\n", elapsed);
    printf("  throughput    = %.1f MSlots/s  (%.2f GSlots/s)\n",
           mslots_per_sec, mslots_per_sec / 1000.0);
    if (st.sieve_active) {
        printf("  sieve_cands_total = %lld\n", st.sieve_candidates_total);
        printf("  sieve_psi_survivors = %lld\n", st.sieve_psi_survivors);
        printf("  sieve_ll_verified = %lld\n", st.sieve_ll_verified);
    }
    fflush(stdout);

    /* Final checkpoint — preserve all learning before exit */
    if (critic_save(CRITIC_CKPT_FILE) == 0)
        printf("[host] final checkpoint saved to '%s'\n", CRITIC_CKPT_FILE);
    fflush(stdout);

    free_host_state(&st);
    return 0;
}
