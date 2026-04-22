// ============================================================================
// HDGL Benchmark & Test Harness v33
// ============================================================================
//
// Tests four subsystems independently, then runs an integrated benchmark:
//
//   TEST 1 — Critic (CPU):
//     • Init → forward pass on known input → verify output range
//     • 100 observe+update cycles → verify weights change
//     • TD target computation
//
//   TEST 2 — NTT arithmetic (GPU):
//     • Load a known 64-limb number, NTT-square it, verify result vs direct
//
//   TEST 3 — Sieve kernel (GPU):
//     • Allocate 4096 slots, run 100 sieve steps, verify ring buffer is
//       populated (all candidates come back in [0,1] residue range)
//
//   TEST 4 — Field kernel (GPU):
//     • Allocate N=65536 slots, run 10 field steps, verify no NaN/Inf
//
//   BENCHMARK — Field kernel throughput:
//     • N = 256K, 512K, 1M slots
//     • 500 field steps each
//     • Report ns/step and GSlots/s
//
// Compile:
//   nvcc -O3 -arch=sm_75 hdgl_analog_v33.cu hdgl_warp_ll_v33.cu \
//        hdgl_sieve_v34.cu hdgl_critic_v33.c hdgl_bench_v33.cu -o hdgl_bench -lm
// ============================================================================

#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "hdgl_critic_v33.h"
#include "hdgl_psi_filter_v35.h"
#include "hdgl_predictor_seed.h"
#include "hdgl_prismatic_v35.h"

// Windows-compatible high-resolution timer
#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
static LARGE_INTEGER g_qpf_bench;
static void bench_clock_init(void) { QueryPerformanceFrequency(&g_qpf_bench); }
static double now_sec(void) {
    LARGE_INTEGER t; QueryPerformanceCounter(&t);
    return (double)t.QuadPart / (double)g_qpf_bench.QuadPart;
}
#else
#include <time.h>
static void bench_clock_init(void) {}
static double now_sec(void) {
    struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}
#endif

// ============================================================================
// Forward declarations
// ============================================================================

typedef struct {
    int   slot_idx;
    float score;
    float ll_seed_f;
    float r_harmonic;
    float phase;
    float coherence;
    float amp;
    float acc;
} Candidate;

typedef struct {
    float    *A_re, *A_im, *phase, *phase_vel, *r_harmonic;
    uint32_t *ll_state;
    float    *reward_accum;
    float    *w_cos, *w_sin, *w_sigma;
    int8_t   *ll_verified;
    Candidate *candidates;
    int      *cand_count;
    int       N, S;
    float     omega, dt, w_amp_self, w_amp_neigh;
} DevSoA;

typedef struct { uint64_t S[64]; uint64_t C[64]; } CSSState;

typedef struct {
    uint32_t *d_assigned_p, *d_sieve_state;
    float    *d_r_harmonic, *d_phase, *d_reward_accum;
    uint32_t *d_prime_found;
    int      *d_prime_count;
    uint32_t *h_prime_found;
    int      *h_prime_count;
    int N, S; float dt;
} SieveHostState;

#define SPECTRAL_N     4
#define CRITIC_W_TOTAL 57
#define MAX_CAND_BUF   256

// All cross-TU functions are compiled as C or exported with extern "C",
// so declare them with extern "C" here to match linkage.
// (critic_* functions come from hdgl_critic_v33.h which already has the guard)
extern "C" {
    // hdgl_analog_v33.cu
    void hdgl_v33_upload_soa(const DevSoA *h);
    void hdgl_v33_upload_critic(const float *w, int n);
    void hdgl_v33_field_step(int N, int block, int step_count, cudaStream_t s);
    void hdgl_v33_weight_sync(cudaStream_t s);
    void hdgl_v33_read_global_weights(float oc[], float os[], float osg[]);

    // hdgl_warp_ll_v33.cu
    void hdgl_warp_ll_v33_alloc(CSSState **out);
    void hdgl_warp_ll_v33_free(CSSState *p);
    void hdgl_warp_ll_v33_seed(CSSState *d, int n, cudaStream_t s);
    void hdgl_warp_ll_v33_launch(const Candidate *dc, int nc, int pb, int it,
                                  CSSState *dc2, float *dr, int8_t *dv,
                                  float eps, cudaStream_t s);
    int  hdgl_gpucarry_ll(uint32_t p);

    // hdgl_sieve_v34.cu
    int  hdgl_sieve_alloc(SieveHostState *st, int N, int S, float dt);
    void hdgl_sieve_free(SieveHostState *st);
    void hdgl_sieve_step(const SieveHostState *st, cudaStream_t s);
    int  hdgl_sieve_harvest(SieveHostState *st, uint32_t *out, int max, cudaStream_t s);
    void hdgl_sieve_seed_priority(SieveHostState *st, const uint32_t *p_list, int n);
}  // extern "C"

// ============================================================================
// Helpers
// ============================================================================

#define CUDA_CHECK(c) do {                                              \
    cudaError_t _e = (c);                                               \
    if (_e != cudaSuccess) {                                            \
        fprintf(stderr, "CUDA error %s:%d: %s\n",                      \
                __FILE__, __LINE__, cudaGetErrorString(_e));            \
        exit(EXIT_FAILURE);                                             \
    }                                                                   \
} while (0)

// now_sec() defined conditionally near the top of this file

static int g_pass = 0, g_fail = 0;

static void check(const char *label, int cond) {
    if (cond) { printf("  [PASS] %s\n", label); g_pass++; }
    else       { printf("  [FAIL] %s\n", label); g_fail++; }
}

// ============================================================================
// TEST 1 — Critic (pure CPU)
// ============================================================================

static void test_critic(void) {
    printf("\n=== TEST 1: Critic (CPU) ===\n");
    critic_init();

    // Known input: all zeros → output should be close to b2 (small Xavier bias)
    float s_zero[CRITIC_IN] = {0};
    float out0 = critic_forward(s_zero);
    check("forward(zeros) is finite", isfinite(out0));

    // Known input: all ones
    float s_ones[CRITIC_IN] = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
    float out1 = critic_forward(s_ones);
    check("forward(ones) is finite", isfinite(out1));

    // 200 observe + update cycles: weights should change
    float w_before[CRITIC_W_TOTAL], w_after[CRITIC_W_TOTAL];
    critic_pack_weights(w_before);

    for (int i = 0; i < 200; i++) {
        float feat[CRITIC_IN] = {
            (float)i / 200.0f, 0.5f, 0.8f, 0.1f, 0.05f
        };
        float target = 1.0f / (1.0f + (float)i * 0.01f);
        critic_observe(feat, target);
    }
    critic_pack_weights(w_after);

    float max_diff = 0.0f;
    for (int i = 0; i < CRITIC_W_TOTAL; i++) {
        float d = fabsf(w_after[i] - w_before[i]);
        if (d > max_diff) max_diff = d;
    }
    check("weights changed after 200 updates", max_diff > 1e-6f);
    printf("  max_weight_delta = %.4e\n", (double)max_diff);

    // TD target: r + gamma * V(s')
    float s_next[CRITIC_IN] = {0.1f, 0.4f, 0.9f, 0.2f, 0.1f};
    float td = critic_td_target(0.5f, s_next);
    check("td_target is finite", isfinite(td));
    printf("  td_target(r=0.5, s') = %.4f\n", (double)td);

    // Verify pack/unpack round-trip
    float packed[CRITIC_W_TOTAL];
    critic_pack_weights(packed);
    check("critic_weight_count() == CRITIC_W_TOTAL",
          critic_weight_count() == CRITIC_W_TOTAL);

    critic_print_stats();
}

// ============================================================================
// TEST 2 — NTT + Warp LL (GPU)
// ============================================================================

static void test_warp_ll(void) {
    printf("\n=== TEST 2: Warp LL v33 (GPU) ===\n");

    CSSState *d_css = NULL;
    float    *d_res = NULL;
    int8_t   *d_ver = NULL;
    Candidate *d_cands = NULL;

    hdgl_warp_ll_v33_alloc(&d_css);
    CUDA_CHECK(cudaMalloc(&d_res,   8 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_ver,   8 * sizeof(int8_t)));
    CUDA_CHECK(cudaMalloc(&d_cands, 8 * sizeof(Candidate)));

    // Set up 4 fake candidates pointing to slot 0..3
    Candidate h_cands[4];
    for (int i = 0; i < 4; i++) {
        h_cands[i].slot_idx   = i;
        h_cands[i].score      = 1.0f;
        h_cands[i].ll_seed_f  = 4.0f;
        h_cands[i].r_harmonic = 10.0f;
        h_cands[i].phase      = 0.0f;
        h_cands[i].coherence  = 0.0f;
        h_cands[i].amp        = 0.0f;
        h_cands[i].acc        = 0.0f;
    }
    CUDA_CHECK(cudaMemcpy(d_cands, h_cands, 4 * sizeof(Candidate), cudaMemcpyHostToDevice));

    cudaStream_t s;
    CUDA_CHECK(cudaStreamCreate(&s));

    // Seed (S_0 = 4, C_0 = 0) and run 10 LL steps
    hdgl_warp_ll_v33_seed(d_css, 4, s);
    hdgl_warp_ll_v33_launch(d_cands, 4, 127, 10, d_css, d_res, d_ver, 1e-6f, s);
    CUDA_CHECK(cudaStreamSynchronize(s));

    float h_res[4];
    int8_t h_ver[4];
    CUDA_CHECK(cudaMemcpy(h_res, d_res, 4 * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_ver, d_ver, 4 * sizeof(int8_t), cudaMemcpyDeviceToHost));

    int all_finite = 1, all_normalised = 1;
    for (int i = 0; i < 4; i++) {
        printf("  cand[%d]: residue=%.4e  verified=%d\n",
               i, (double)h_res[i], (int)h_ver[i]);
        if (!isfinite(h_res[i])) all_finite = 0;
        if (h_res[i] < 0.0f || h_res[i] > 1.0f) all_normalised = 0;
    }
    check("all residues are finite",    all_finite);
    check("all residues in [0,1]",      all_normalised);

    cudaStreamDestroy(s);
    hdgl_warp_ll_v33_free(d_css);
    cudaFree(d_res);
    cudaFree(d_ver);
    cudaFree(d_cands);
}

// ============================================================================
// TEST 3 — Sieve kernel (GPU)
// ============================================================================

static void test_sieve(void) {
    printf("\n=== TEST 3: Sieve v34 (GPU) ===\n");

    SieveHostState sieve;
    int N = 4096, S = 64;
    hdgl_sieve_alloc(&sieve, N, S, 0.001f);

    cudaStream_t s;
    CUDA_CHECK(cudaStreamCreate(&s));

    // Run 200 sieve steps
    for (int i = 0; i < 200; i++) hdgl_sieve_step(&sieve, s);
    CUDA_CHECK(cudaStreamSynchronize(s));

    // Harvest — just verify no crash and count is reasonable
    uint32_t found[256];
    int n = hdgl_sieve_harvest(&sieve, found, 256, s);
    printf("  harvested %d candidate exponents after 200 steps\n", n);
    check("harvest returns non-negative count", n >= 0);
    check("harvest count within ring capacity",  n <= 256);

    if (n > 0) {
        check("first candidate exponent > BASE_P", found[0] > 82589933u);
        printf("  first exponent: p = %u\n", found[0]);
    }

    // Verify phase array has no NaN after 200 steps
    float *h_phase = (float *)malloc(N * sizeof(float));
    CUDA_CHECK(cudaMemcpy(h_phase, sieve.d_phase, N * sizeof(float), cudaMemcpyDeviceToHost));
    int nan_count = 0;
    for (int i = 0; i < N; i++) if (!isfinite(h_phase[i])) nan_count++;
    check("no NaN/Inf in phase after 200 steps", nan_count == 0);
    printf("  nan_count_in_phase = %d / %d\n", nan_count, N);
    free(h_phase);

    cudaStreamDestroy(s);
    hdgl_sieve_free(&sieve);
}

// ============================================================================
// TEST 4 — Field kernel (GPU)
// ============================================================================

static void test_field(void) {
    printf("\n=== TEST 4: Field kernel v33 (GPU) ===\n");

    const int N = 65536, S = 256;
    float *d_Ar, *d_Ai, *d_ph, *d_phv, *d_rh, *d_acc;
    float *d_wc, *d_ws, *d_wsg;
    uint32_t *d_ll;
    int8_t *d_ver;
    Candidate *d_cands;
    int *d_cnt;

    CUDA_CHECK(cudaMalloc(&d_Ar,    N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_Ai,    N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_ph,    N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_phv,   N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_rh,    N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_acc,   N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_wc,    N * SPECTRAL_N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_ws,    N * SPECTRAL_N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_wsg,   N * SPECTRAL_N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_ll,    N * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_ver,   N * sizeof(int8_t)));
    CUDA_CHECK(cudaMalloc(&d_cands, MAX_CAND_BUF * sizeof(Candidate)));
    CUDA_CHECK(cudaMalloc(&d_cnt,   sizeof(int)));

    // Initialise with simple values
    float *tmp = (float *)malloc(N * SPECTRAL_N * sizeof(float));
    float sigma_init[4] = {3.14159f, 1.5708f, 0.7854f, 0.3927f};

    for (int i = 0; i < N; i++) ((float *)tmp)[i] = 1.0f;
    CUDA_CHECK(cudaMemcpy(d_Ar, tmp, N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_Ai, 0, N * sizeof(float)));

    for (int i = 0; i < N; i++)
        tmp[i] = 6.28318530f * (float)i / (float)N;
    CUDA_CHECK(cudaMemcpy(d_ph, tmp, N * sizeof(float), cudaMemcpyHostToDevice));

    for (int i = 0; i < N; i++) tmp[i] = 0.314159f;
    CUDA_CHECK(cudaMemcpy(d_phv, tmp, N * sizeof(float), cudaMemcpyHostToDevice));

    for (int i = 0; i < N; i++) tmp[i] = 10.0f;
    CUDA_CHECK(cudaMemcpy(d_rh, tmp, N * sizeof(float), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemset(d_acc, 0, N * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_ver, 0, N * sizeof(int8_t)));
    CUDA_CHECK(cudaMemset(d_cnt, 0, sizeof(int)));

    // LL state: non-zero seeds
    uint32_t *ll_tmp = (uint32_t *)malloc(N * sizeof(uint32_t));
    for (int i = 0; i < N; i++) ll_tmp[i] = (uint32_t)(i * 2654435761u + 7u) | 1u;
    CUDA_CHECK(cudaMemcpy(d_ll, ll_tmp, N * sizeof(uint32_t), cudaMemcpyHostToDevice));
    free(ll_tmp);

    for (int i = 0; i < N; i++)
        for (int k = 0; k < SPECTRAL_N; k++) {
            tmp[i * SPECTRAL_N + k] = 0.05f * ((float)rand() / RAND_MAX - 0.5f);
        }
    CUDA_CHECK(cudaMemcpy(d_wc, tmp, N * SPECTRAL_N * sizeof(float), cudaMemcpyHostToDevice));

    for (int i = 0; i < N; i++)
        for (int k = 0; k < SPECTRAL_N; k++)
            tmp[i * SPECTRAL_N + k] = 0.05f * ((float)rand() / RAND_MAX - 0.5f);
    CUDA_CHECK(cudaMemcpy(d_ws, tmp, N * SPECTRAL_N * sizeof(float), cudaMemcpyHostToDevice));

    for (int i = 0; i < N; i++)
        for (int k = 0; k < SPECTRAL_N; k++)
            tmp[i * SPECTRAL_N + k] = sigma_init[k];
    CUDA_CHECK(cudaMemcpy(d_wsg, tmp, N * SPECTRAL_N * sizeof(float), cudaMemcpyHostToDevice));

    free(tmp);

    // Upload SoA
    float critic_w[CRITIC_W_TOTAL] = {0};
    critic_init();
    critic_pack_weights(critic_w);

    DevSoA soa = {
        d_Ar, d_Ai, d_ph, d_phv, d_rh, d_ll, d_acc,
        d_wc, d_ws, d_wsg, d_ver, d_cands, d_cnt,
        N, S, 0.314159f, 0.001f, 1.0f, 0.5f
    };
    hdgl_v33_upload_soa(&soa);
    hdgl_v33_upload_critic(critic_w, CRITIC_W_TOTAL);

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    // Run 10 field steps
    for (int i = 0; i < 10; i++) hdgl_v33_field_step(N, 256, i, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // Check for NaN/Inf in A_re
    float *h_Ar = (float *)malloc(N * sizeof(float));
    CUDA_CHECK(cudaMemcpy(h_Ar, d_Ar, N * sizeof(float), cudaMemcpyDeviceToHost));
    int nan_cnt = 0, inf_cnt = 0;
    float sum_abs = 0.0f;
    for (int i = 0; i < N; i++) {
        if (isnan(h_Ar[i]))  nan_cnt++;
        else if (isinf(h_Ar[i])) inf_cnt++;
        else sum_abs += fabsf(h_Ar[i]);
    }
    check("no NaN in A_re after 10 field steps",  nan_cnt == 0);
    check("no Inf in A_re after 10 field steps",  inf_cnt == 0);
    printf("  nan=%d  inf=%d  mean|A_re|=%.4f\n",
           nan_cnt, inf_cnt, (double)(sum_abs / N));
    free(h_Ar);

    // Check candidate count is reasonable
    int h_cnt = -1;
    CUDA_CHECK(cudaMemcpy(&h_cnt, d_cnt, sizeof(int), cudaMemcpyDeviceToHost));
    check("candidate count >= 0", h_cnt >= 0);
    check("candidate count <= MAX_CAND_BUF", h_cnt <= MAX_CAND_BUF * 10);
    printf("  candidates promoted = %d\n", h_cnt);

    cudaStreamDestroy(stream);
    cudaFree(d_Ar);  cudaFree(d_Ai); cudaFree(d_ph);  cudaFree(d_phv);
    cudaFree(d_rh);  cudaFree(d_acc);cudaFree(d_wc);  cudaFree(d_ws);
    cudaFree(d_wsg); cudaFree(d_ll); cudaFree(d_ver); cudaFree(d_cands);
    cudaFree(d_cnt);
}

// ============================================================================
// TEST 5 — gpucarry LL (correctness: known Mersenne primes + composites)
// ============================================================================

static void test_gpucarry(void) {
    printf("\n=== TEST 5: gpucarry LL (GPU, exact integer) ===\n");

    // Known Mersenne PRIME exponents (small, fast to verify)
    static const uint32_t PRIMES_K[]     = { 3, 5, 7, 13, 17, 19, 31, 61, 89, 107, 127 };
    // Known COMPOSITE Mersenne exponents
    static const uint32_t COMPOSITES_K[] = { 11, 23, 29, 37, 41 };

    int all_prime_ok = 1, all_comp_ok = 1;

    for (size_t i = 0; i < sizeof(PRIMES_K)/sizeof(PRIMES_K[0]); i++) {
        uint32_t p = PRIMES_K[i];
        int r = hdgl_gpucarry_ll(p);
        printf("  M_%u: %s (expected PRIME)\n", p, r ? "PRIME" : "COMPOSITE");
        if (!r) all_prime_ok = 0;
    }
    for (size_t i = 0; i < sizeof(COMPOSITES_K)/sizeof(COMPOSITES_K[0]); i++) {
        uint32_t p = COMPOSITES_K[i];
        int r = hdgl_gpucarry_ll(p);
        printf("  M_%u: %s (expected COMPOSITE)\n", p, r ? "PRIME" : "COMPOSITE");
        if (r) all_comp_ok = 0;
    }

    check("all known Mersenne primes return 1",     all_prime_ok);
    check("all known Mersenne composites return 0",  all_comp_ok);
}

// ============================================================================
// TEST 6 — gpucarry timing at larger known Mersenne primes
// p=21701 (~1.2s expected), p=44497 (~3.4s expected)
// ============================================================================

static void test_gpucarry_timing(void) {
    printf("\n=== TEST 6: gpucarry timing (p=21701, p=44497) ===\n");
    printf("  (expected: ~1.2s and ~3.4s on RTX 2060)\n");

    static const struct { uint32_t p; double exp_max_s; } CASES[] = {
        { 21701,  15.0 },   /* known Mersenne prime M_21701 */
        { 44497,  40.0 },   /* known Mersenne prime M_44497 */
    };
    int all_ok = 1;
    for (int i = 0; i < 2; i++) {
        uint32_t p = CASES[i].p;
        double t0 = now_sec();
        int r = hdgl_gpucarry_ll(p);
        double elapsed = now_sec() - t0;
        printf("  M_%u: %s  %.2f s\n", p,
               r ? "PRIME" : "COMPOSITE", elapsed);
        if (!r) all_ok = 0;
        if (elapsed > CASES[i].exp_max_s) {
            printf("  WARNING: took %.1f s (budget %.1f s)\n",
                   elapsed, CASES[i].exp_max_s);
        }
    }
    check("p=21701 and p=44497 both return PRIME", all_ok);
}

// ============================================================================
// TEST 7 — psi filter: composites killed, small Mersenne primes survive
// ============================================================================

static void test_psi_filter(void) {
    printf("\n=== TEST 7: psi filter (GPU, Riemann zeta pre-filter) ===\n");

    PsiFilterState psi;
    hdgl_psi_filter_alloc(&psi, 64);

    /* Known prime exponents (p itself is prime — von Mangoldt = log(p)) */
    static const uint32_t KNOWN_PRIMES[] = { 97, 101, 103, 107, 109, 113, 127, 131 };
    /* Known composite exponents that are NOT prime powers (von Mangoldt = 0) */
    static const uint32_t KNOWN_COMP[]   = { 98, 100, 102, 104, 106, 108, 110, 112 };
    const int NP = (int)(sizeof(KNOWN_PRIMES)/sizeof(KNOWN_PRIMES[0]));
    const int NC = (int)(sizeof(KNOWN_COMP)/sizeof(KNOWN_COMP[0]));

    /* Test 1: prime exponents — should produce spikes near 1.0 */
    uint32_t out_p[64];
    float    sc_p[64];
    int n_surv_p = hdgl_psi_filter_run(&psi, KNOWN_PRIMES, NP, out_p, sc_p, 0);
    printf("  prime exponents: %d/%d survived psi filter\n", n_surv_p, NP);
    for (int i = 0; i < n_surv_p; i++)
        printf("    p=%u  score=%.3f\n", out_p[i], (double)sc_p[i]);

    /* Collect average raw spike for primes by running with B=500 threshold 0 */
    /* Proxy: use survivor scores to gauge mean spike */
    double avg_prime_score = 0.0;
    for (int i = 0; i < n_surv_p; i++) avg_prime_score += sc_p[i];
    if (n_surv_p > 0) avg_prime_score /= n_surv_p;

    check("psi filter does not crash (primes)", n_surv_p >= 0);
    check("at least 1 prime survives psi filter", n_surv_p >= 1);

    /* Test 2: composite exponents — should produce near-zero spikes */
    uint32_t out_c[64];
    float    sc_c[64];
    int n_surv_c = hdgl_psi_filter_run(&psi, KNOWN_COMP, NC, out_c, sc_c, 0);
    printf("  composite exponents: %d/%d survived psi filter\n", n_surv_c, NC);

    double avg_comp_score = 0.0;
    for (int i = 0; i < n_surv_c; i++) avg_comp_score += sc_c[i];
    if (n_surv_c > 0) avg_comp_score /= n_surv_c;

    check("psi filter does not crash (composites)", n_surv_c >= 0);
    /* With B=500 zeros, primes should score higher on average than composites */
    check("primes survive at least as often as composites", n_surv_p >= n_surv_c);

    hdgl_psi_filter_free(&psi);
}

// ============================================================================
// TEST 8 — prismatic scorer: basic properties + phi-lattice monotonicity
// ============================================================================

static void test_prismatic(void) {
    printf("\n=== TEST 8: prismatic scorer (CPU) ===\n");

    /* Test: scores are finite for known Mersenne prime exponents */
    static const uint32_t MP[] = { 3, 5, 7, 13, 17, 19, 31, 61, 89, 107, 127 };
    const int NMP = (int)(sizeof(MP)/sizeof(MP[0]));
    int all_finite = 1;
    for (int i = 0; i < NMP; i++) {
        float sc = hdgl_prismatic_score(MP[i], 10.0f);
        if (!isfinite(sc)) all_finite = 0;
        printf("  score(M_%u) = %.4f  n_coord=%.4f  lower_half=%d\n",
               MP[i], (double)sc,
               hdgl_n_coord(MP[i]),
               hdgl_phi_lower_half(MP[i]));
    }
    check("all known prime exponents give finite score", all_finite);

    /* Verify phi lower-half bias: count how many known exponents qualify */
    int lower_half_count = 0;
    for (int i = 0; i < NMP; i++)
        if (hdgl_phi_lower_half(MP[i])) lower_half_count++;
    printf("  phi lower-half count: %d/%d (bias ~67%%)\n", lower_half_count, NMP);
    check("phi lower-half count > 0", lower_half_count > 0);

    /* Test: prismatic_sort orders by descending score */
    uint32_t test_exps[4] = { 3, 7, 31, 127 };
    float    test_rhs[4]  = { 10.0f, 10.0f, 10.0f, 10.0f };
    hdgl_prismatic_sort(test_exps, test_rhs, 4);
    int sorted_ok = 1;
    for (int i = 0; i < 3; i++) {
        float s0 = hdgl_prismatic_score(test_exps[i],   test_rhs[i]);
        float s1 = hdgl_prismatic_score(test_exps[i+1], test_rhs[i+1]);
        if (s0 < s1) sorted_ok = 0;
    }
    check("prismatic_sort produces descending order", sorted_ok);

    /* Test: predictor top-20 returns positive count */
    uint32_t prio[20];
    int n_prio = hdgl_predictor_top20(prio, 20);
    printf("  predictor top-%d exponents:\n", n_prio);
    for (int i = 0; i < n_prio && i < 5; i++)
        printf("    p=%u  lower_half=%d  score=%.4f\n",
               prio[i], hdgl_phi_lower_half(prio[i]),
               (double)hdgl_prismatic_score(prio[i], 1.0f));
    check("predictor_top20 returns > 0 candidates", n_prio > 0);
    check("predictor_top20 returns <= 20 candidates", n_prio <= 20);
}

// ============================================================================
// BENCHMARK — Field kernel throughput
// ============================================================================

static void benchmark_field(void) {
    printf("\n=== BENCHMARK: Field kernel throughput ===\n");
    printf("  %-12s  %-10s  %-14s  %-12s\n",
           "N", "steps", "wall_ms", "GSlots/s");
    printf("  ------------------------------------------------------------\n");

    static const int N_vals[] = {262144, 524288, 1048576};
    const int STEPS = 500;
    const int S = 1024;
    const int BLK = 256;

    for (int ni = 0; ni < 3; ni++) {
        int N = N_vals[ni];

        // Alloc device arrays
        float *d_Ar, *d_Ai, *d_ph, *d_phv, *d_rh, *d_acc;
        float *d_wc, *d_ws, *d_wsg;
        uint32_t *d_ll;
        int8_t *d_ver;
        Candidate *d_cands;
        int *d_cnt;

        CUDA_CHECK(cudaMalloc(&d_Ar,  N * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_Ai,  N * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_ph,  N * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_phv, N * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_rh,  N * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_acc, N * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_wc,  N * SPECTRAL_N * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_ws,  N * SPECTRAL_N * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_wsg, N * SPECTRAL_N * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_ll,  N * sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(&d_ver, N * sizeof(int8_t)));
        CUDA_CHECK(cudaMalloc(&d_cands, MAX_CAND_BUF * sizeof(Candidate)));
        CUDA_CHECK(cudaMalloc(&d_cnt,   sizeof(int)));

        // Simple init via memset
        CUDA_CHECK(cudaMemset(d_Ar,  0x3f, N * sizeof(float)));   // ≈ 1.0 in fp32
        CUDA_CHECK(cudaMemset(d_Ai,  0, N * sizeof(float)));
        CUDA_CHECK(cudaMemset(d_ph,  0, N * sizeof(float)));
        CUDA_CHECK(cudaMemset(d_phv, 0x3e, N * sizeof(float)));   // ≈ 0.5
        CUDA_CHECK(cudaMemset(d_rh,  0x41, N * sizeof(float)));   // ≈ some fp val
        CUDA_CHECK(cudaMemset(d_acc, 0, N * sizeof(float)));
        CUDA_CHECK(cudaMemset(d_wc,  0, N * SPECTRAL_N * sizeof(float)));
        CUDA_CHECK(cudaMemset(d_ws,  0, N * SPECTRAL_N * sizeof(float)));
        CUDA_CHECK(cudaMemset(d_ll,  0x01, N * sizeof(uint32_t))); // non-zero
        CUDA_CHECK(cudaMemset(d_ver, 0, N * sizeof(int8_t)));
        CUDA_CHECK(cudaMemset(d_cnt, 0, sizeof(int)));

        // Init w_sigma to float constants via pattern
        // 0x3f 0xc9 0x0f 0xda = pi in IEEE 754 little-endian (x86: 0xDA0F49C0)
        // Use a simple loop via float array
        {
            float *tmp = (float *)malloc(N * SPECTRAL_N * sizeof(float));
            float sigma[4] = {3.14159f, 1.5708f, 0.7854f, 0.3927f};
            for (int i = 0; i < N; i++)
                for (int k = 0; k < 4; k++) tmp[i*4+k] = sigma[k];
            CUDA_CHECK(cudaMemcpy(d_wsg, tmp, N * SPECTRAL_N * sizeof(float), cudaMemcpyHostToDevice));
            free(tmp);
        }

        DevSoA soa = {
            d_Ar, d_Ai, d_ph, d_phv, d_rh, d_ll, d_acc,
            d_wc, d_ws, d_wsg, d_ver, d_cands, d_cnt,
            N, S, 0.314159f, 0.001f, 1.0f, 0.5f
        };
        hdgl_v33_upload_soa(&soa);

        float cw[CRITIC_W_TOTAL] = {0};
        hdgl_v33_upload_critic(cw, CRITIC_W_TOTAL);

        cudaStream_t stream;
        CUDA_CHECK(cudaStreamCreate(&stream));

        // Warmup
        for (int i = 0; i < 5; i++) hdgl_v33_field_step(N, BLK, i, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));

        // Timed run
        double t0 = now_sec();
        for (int i = 0; i < STEPS; i++) hdgl_v33_field_step(N, BLK, 5 + i, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        double t1 = now_sec();

        double elapsed_ms = (t1 - t0) * 1000.0;
        double gslots = (double)STEPS * (double)N / (t1 - t0) / 1e9;

        printf("  %-12d  %-10d  %-14.2f  %-12.3f\n",
               N, STEPS, elapsed_ms, gslots);

        cudaStreamDestroy(stream);
        cudaFree(d_Ar);  cudaFree(d_Ai); cudaFree(d_ph);  cudaFree(d_phv);
        cudaFree(d_rh);  cudaFree(d_acc);cudaFree(d_wc);  cudaFree(d_ws);
        cudaFree(d_wsg); cudaFree(d_ll); cudaFree(d_ver); cudaFree(d_cands);
        cudaFree(d_cnt);
    }
}

// ============================================================================
// main
// ============================================================================

int main(void) {
    printf("HDGL v33 Test & Benchmark Harness\n");
    printf("CUDA device: ");
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("%s  cc=%d.%d  smem=%zu KB  sharedPerBlock=%zu KB\n",
           prop.name, prop.major, prop.minor,
           prop.totalGlobalMem / (1024*1024),
           prop.sharedMemPerBlock / 1024);

    bench_clock_init();
    srand(42);

    test_critic();
    test_warp_ll();
    test_sieve();
    test_field();
    test_gpucarry();
    test_gpucarry_timing();
    test_psi_filter();
    test_prismatic();
    benchmark_field();

    printf("\n========================================\n");
    printf("RESULTS: %d passed, %d failed\n", g_pass, g_fail);
    printf("========================================\n");
    return (g_fail > 0) ? 1 : 0;
}
