// ============================================================================
// HDGL Host v32 — CPU Orchestration with Three Async CUDA Streams
// ============================================================================
//
// Manages:
//   stream0 — field evolution  (hdgl_field_step_kernel)
//   stream1 — candidate I/O + warp LL engine
//   stream2 — reward injection + weight sync
//
// Async overlap pattern (per sync cycle):
//   stream0 runs STEPS_PER_SYNC field steps continuously.
//   stream1 D2H's candidate buffer while field runs, then launches warp LL.
//   stream2 injects rewards + syncs weights after field is done.
//
// Compile with:
//   nvcc -O3 -arch=sm_86 -lineinfo hdgl_analog_v32.cu hdgl_warp_ll_v32.cu \
//        hdgl_host_v32.c -o hdgl_v32 -lm
// ============================================================================

#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <signal.h>
#include <time.h>

// ============================================================================
// Forward declarations from CUDA TUs
// ============================================================================

typedef struct {
    int   slot_idx;
    float score;
    float ll_seed_f;
    float r_harmonic;
    float phase;
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

// Declared in hdgl_analog_v32.cu
void hdgl_v32_upload_soa(const DevSoA *host_soa);
void hdgl_v32_field_step(int N, int block, cudaStream_t stream);
void hdgl_v32_reward_inject(int N, int block, cudaStream_t stream);
void hdgl_v32_weight_sync(cudaStream_t stream);
void hdgl_v32_read_global_weights(float out_cos[SPECTRAL_N], float out_sin[SPECTRAL_N]);

// Declared in hdgl_warp_ll_v32.cu
void hdgl_warp_ll_alloc(uint64_t **d_ll_limbs_out);
void hdgl_warp_ll_free(uint64_t *d_ll_limbs);
void hdgl_warp_ll_seed(uint64_t *d_ll_limbs, int n_cands, cudaStream_t stream);
void hdgl_warp_ll_launch(
    const Candidate *d_cands, int n_cands, int p_bits, int iters,
    uint64_t *d_ll_limbs, float *d_ll_residue, int8_t *d_ll_verified,
    float residue_eps, cudaStream_t stream);

// ============================================================================
// Configuration
// ============================================================================

#define DEFAULT_N           (1 << 20)  // 1M slots
#define DEFAULT_S           1024       // slots_per_instance
#define STEPS_PER_SYNC      64         // field steps between candidate harvests
#define BLOCK_SIZE          256        // threads per block for field kernel
#define SYNC_CYCLES         1000       // main loop iteration count (0 = infinite)

#define LL_P_BITS           127        // target Mersenne exponent (M127 for testing)
#define LL_ITERS            100        // LL iterations per candidate (increase for exact)
#define LL_RESIDUE_EPS      1e-6f      // residue threshold for confirmation

#define WEIGHT_MOMENTUM     0.95f      // host-side momentum for global weights
#define PHI_F               1.6180339887498948f

// ============================================================================
// Global running flag (SIGINT / SIGTERM handler)
// ============================================================================

static volatile int g_running = 1;
static void signal_handler(int sig) { (void)sig; g_running = 0; }

// ============================================================================
// CUDA error check macro
// ============================================================================

#define CUDA_CHECK(call) do {                                             \
    cudaError_t _err = (call);                                            \
    if (_err != cudaSuccess) {                                            \
        fprintf(stderr, "[v32 CUDA] %s:%d  %s\n",                       \
                __FILE__, __LINE__, cudaGetErrorString(_err));            \
        exit(EXIT_FAILURE);                                               \
    }                                                                     \
} while (0)

// ============================================================================
// Host state
// ============================================================================

typedef struct {
    // Device arrays
    DevSoA  dev;

    // Warp LL device buffers
    uint64_t *d_ll_limbs;
    float    *d_ll_residue;  // [MAX_CAND_BUF]

    // CUDA streams
    cudaStream_t stream0;  // field evolution
    cudaStream_t stream1;  // candidate I/O + warp LL
    cudaStream_t stream2;  // reward inject + weight sync

    // CUDA events
    cudaEvent_t ev_field_start;
    cudaEvent_t ev_field_done;
    cudaEvent_t ev_ll_done;

    // Pinned host buffers
    Candidate *h_candidates;   // MAX_CAND_BUF entries
    int       *h_cand_count;   // 1 int
    int8_t    *h_ll_verified;  // [N] per-slot flags
    float     *h_ll_residue;   // [MAX_CAND_BUF]

    // Host-side global weights (momentum state)
    float  gw_cos[SPECTRAL_N];
    float  gw_sin[SPECTRAL_N];

    int N;
    int S;
    long long total_steps;
    long long total_candidates;
} HostState;

// ============================================================================
// Allocation helpers
// ============================================================================

static void alloc_device_arrays(HostState *hs) {
    int N = hs->N;
    CUDA_CHECK(cudaMalloc(&hs->dev.A_re,         N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&hs->dev.A_im,         N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&hs->dev.phase,        N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&hs->dev.phase_vel,    N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&hs->dev.r_harmonic,   N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&hs->dev.ll_state,     N * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&hs->dev.reward_accum, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&hs->dev.w_cos,        N * SPECTRAL_N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&hs->dev.w_sin,        N * SPECTRAL_N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&hs->dev.ll_verified,  N * sizeof(int8_t)));
    CUDA_CHECK(cudaMalloc(&hs->dev.candidates,   MAX_CAND_BUF * sizeof(Candidate)));
    CUDA_CHECK(cudaMalloc(&hs->dev.cand_count,   sizeof(int)));
    CUDA_CHECK(cudaMalloc(&hs->d_ll_residue,     MAX_CAND_BUF * sizeof(float)));
    hdgl_warp_ll_alloc(&hs->d_ll_limbs);
}

static void free_device_arrays(HostState *hs) {
    cudaFree(hs->dev.A_re);
    cudaFree(hs->dev.A_im);
    cudaFree(hs->dev.phase);
    cudaFree(hs->dev.phase_vel);
    cudaFree(hs->dev.r_harmonic);
    cudaFree(hs->dev.ll_state);
    cudaFree(hs->dev.reward_accum);
    cudaFree(hs->dev.w_cos);
    cudaFree(hs->dev.w_sin);
    cudaFree(hs->dev.ll_verified);
    cudaFree(hs->dev.candidates);
    cudaFree(hs->dev.cand_count);
    cudaFree(hs->d_ll_residue);
    hdgl_warp_ll_free(hs->d_ll_limbs);
}

static void alloc_pinned_buffers(HostState *hs) {
    CUDA_CHECK(cudaHostAlloc(&hs->h_candidates,  MAX_CAND_BUF * sizeof(Candidate), cudaHostAllocDefault));
    CUDA_CHECK(cudaHostAlloc(&hs->h_cand_count,  sizeof(int),                      cudaHostAllocDefault));
    CUDA_CHECK(cudaHostAlloc(&hs->h_ll_verified, hs->N * sizeof(int8_t),           cudaHostAllocDefault));
    CUDA_CHECK(cudaHostAlloc(&hs->h_ll_residue,  MAX_CAND_BUF * sizeof(float),     cudaHostAllocDefault));
}

static void free_pinned_buffers(HostState *hs) {
    cudaFreeHost(hs->h_candidates);
    cudaFreeHost(hs->h_cand_count);
    cudaFreeHost(hs->h_ll_verified);
    cudaFreeHost(hs->h_ll_residue);
}

// ============================================================================
// Initialise device field state
// ============================================================================

static void init_device_state(HostState *hs) {
    int N = hs->N;

    // Temporary host buffer for initialisation
    float    *h_f  = (float *)   malloc(N * sizeof(float));
    uint32_t *h_u  = (uint32_t *)malloc(N * sizeof(uint32_t));
    float    *h_wc = (float *)   malloc(N * SPECTRAL_N * sizeof(float));

    srand((unsigned)time(NULL));
    for (int i = 0; i < N; i++) {
        h_f[i] = (float)rand() / (float)RAND_MAX * 0.5f;  // A_re
    }
    CUDA_CHECK(cudaMemcpy(hs->dev.A_re, h_f, N * sizeof(float), cudaMemcpyHostToDevice));

    for (int i = 0; i < N; i++) h_f[i] = (float)rand() / (float)RAND_MAX * 0.1f;
    CUDA_CHECK(cudaMemcpy(hs->dev.A_im, h_f, N * sizeof(float), cudaMemcpyHostToDevice));

    for (int i = 0; i < N; i++)
        h_f[i] = (float)(rand() % 1000) / 1000.0f * 6.283185307f;
    CUDA_CHECK(cudaMemcpy(hs->dev.phase, h_f, N * sizeof(float), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemset(hs->dev.phase_vel,  0, N * sizeof(float)));
    CUDA_CHECK(cudaMemset(hs->dev.reward_accum, 0, N * sizeof(float)));
    CUDA_CHECK(cudaMemset(hs->dev.ll_verified,  0, N * sizeof(int8_t)));

    for (int i = 0; i < N; i++) h_f[i] = 1.0f + (float)(i % 16);
    CUDA_CHECK(cudaMemcpy(hs->dev.r_harmonic, h_f, N * sizeof(float), cudaMemcpyHostToDevice));

    for (int i = 0; i < N; i++) h_u[i] = (uint32_t)rand() | 1u;
    CUDA_CHECK(cudaMemcpy(hs->dev.ll_state, h_u, N * sizeof(uint32_t), cudaMemcpyHostToDevice));

    // Spectral weights: small uniform init
    for (int i = 0; i < N * SPECTRAL_N; i++) h_wc[i] = 0.05f;
    CUDA_CHECK(cudaMemcpy(hs->dev.w_cos, h_wc, N * SPECTRAL_N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(hs->dev.w_sin, 0, N * SPECTRAL_N * sizeof(float)));

    CUDA_CHECK(cudaMemset(hs->dev.cand_count, 0, sizeof(int)));

    free(h_f); free(h_u); free(h_wc);

    // Initialise host global weights
    for (int k = 0; k < SPECTRAL_N; k++) {
        hs->gw_cos[k] = 0.05f;
        hs->gw_sin[k] = 0.00f;
    }

    // Configure DevSoA scalar fields
    hs->dev.N          = N;
    hs->dev.S          = hs->S;
    hs->dev.omega      = 1.0f;
    hs->dev.dt         = 1.0f / 32768.0f;
    hs->dev.w_amp_self = 0.05f;
    hs->dev.w_amp_neigh= 0.05f;

    hdgl_v32_upload_soa(&hs->dev);
}

// ============================================================================
// Main async loop — one sync cycle
// ============================================================================

static void run_sync_cycle(HostState *hs) {
    // ---- stream0: run STEPS_PER_SYNC field steps ----
    CUDA_CHECK(cudaEventRecord(hs->ev_field_start, hs->stream0));
    for (int step = 0; step < STEPS_PER_SYNC; step++) {
        hdgl_v32_field_step(hs->N, BLOCK_SIZE, hs->stream0);
    }
    CUDA_CHECK(cudaEventRecord(hs->ev_field_done, hs->stream0));
    hs->total_steps += STEPS_PER_SYNC;

    // ---- stream1: transfer candidate count (overlap with field) ----
    CUDA_CHECK(cudaStreamWaitEvent(hs->stream1, hs->ev_field_start, 0));
    CUDA_CHECK(cudaMemcpyAsync(hs->h_cand_count, hs->dev.cand_count,
                               sizeof(int), cudaMemcpyDeviceToHost, hs->stream1));
    CUDA_CHECK(cudaStreamSynchronize(hs->stream1));

    int n_cands = *hs->h_cand_count;
    if (n_cands > MAX_CAND_BUF) n_cands = MAX_CAND_BUF;

    if (n_cands > 0) {
        // D2H candidate descriptors
        CUDA_CHECK(cudaMemcpyAsync(hs->h_candidates, hs->dev.candidates,
                                   n_cands * sizeof(Candidate),
                                   cudaMemcpyDeviceToHost, hs->stream1));
        CUDA_CHECK(cudaStreamSynchronize(hs->stream1));

        // Seed warp LL for these candidates
        hdgl_warp_ll_seed(hs->d_ll_limbs, n_cands, hs->stream1);

        // Launch warp LL
        hdgl_warp_ll_launch(
            hs->dev.candidates, n_cands,
            LL_P_BITS, LL_ITERS,
            hs->d_ll_limbs, hs->d_ll_residue, hs->dev.ll_verified,
            LL_RESIDUE_EPS, hs->stream1);

        // D2H residue for logging
        CUDA_CHECK(cudaMemcpyAsync(hs->h_ll_residue, hs->d_ll_residue,
                                   n_cands * sizeof(float),
                                   cudaMemcpyDeviceToHost, hs->stream1));

        CUDA_CHECK(cudaEventRecord(hs->ev_ll_done, hs->stream1));
        CUDA_CHECK(cudaStreamSynchronize(hs->stream1));

        hs->total_candidates += n_cands;

        // Log candidates
        printf("[v32] cycle %lld | %d candidate(s):\n", hs->total_steps / STEPS_PER_SYNC, n_cands);
        for (int c = 0; c < n_cands && c < 8; c++) {
            printf("  slot=%6d  score=%.4f  residue=%.4e  r_h=%.2f  phase=%.4f\n",
                   hs->h_candidates[c].slot_idx,
                   hs->h_candidates[c].score,
                   hs->h_ll_residue[c],
                   hs->h_candidates[c].r_harmonic,
                   hs->h_candidates[c].phase);
        }
    }

    // ---- stream2: wait for field done, then inject rewards + sync weights ----
    CUDA_CHECK(cudaStreamWaitEvent(hs->stream2, hs->ev_field_done, 0));

    if (n_cands > 0) {
        // H2D the updated ll_verified flags (written by warp LL kernel)
        // The warp LL kernel writes directly to d_ll_verified on the device;
        // no explicit H2D needed here — just ensure stream2 runs after stream1.
        if (n_cands > 0) {
            CUDA_CHECK(cudaStreamWaitEvent(hs->stream2, hs->ev_ll_done, 0));
        }
        hdgl_v32_reward_inject(hs->N, BLOCK_SIZE, hs->stream2);
    }

    hdgl_v32_weight_sync(hs->stream2);
    CUDA_CHECK(cudaStreamSynchronize(hs->stream2));

    // Apply host-side momentum to global weights and push back to device
    {
        float new_cos[SPECTRAL_N], new_sin[SPECTRAL_N];
        hdgl_v32_read_global_weights(new_cos, new_sin);
        for (int k = 0; k < SPECTRAL_N; k++) {
            hs->gw_cos[k] = WEIGHT_MOMENTUM * hs->gw_cos[k]
                          + (1.0f - WEIGHT_MOMENTUM) * new_cos[k];
            hs->gw_sin[k] = WEIGHT_MOMENTUM * hs->gw_sin[k]
                          + (1.0f - WEIGHT_MOMENTUM) * new_sin[k];
        }
        // Re-upload g_soa scalar fields if omega/dt adapted
        // (simple: no adaptation here; use hdgl_v32_upload_soa if needed)
    }

    // Reset candidate count for next cycle (stream0 reset after field is done)
    CUDA_CHECK(cudaStreamWaitEvent(hs->stream0, hs->ev_field_done, 0));
    CUDA_CHECK(cudaMemsetAsync(hs->dev.cand_count, 0, sizeof(int), hs->stream0));
}

// ============================================================================
// main
// ============================================================================

int main(int argc, char **argv) {
    signal(SIGINT,  signal_handler);
    signal(SIGTERM, signal_handler);

    int N            = DEFAULT_N;
    int S            = DEFAULT_S;
    int max_cycles   = SYNC_CYCLES;
    int p_bits_arg   = LL_P_BITS;

    // Simple arg parsing: hdgl_v32 [N] [S] [cycles] [p_bits]
    if (argc > 1) N          = atoi(argv[1]);
    if (argc > 2) S          = atoi(argv[2]);
    if (argc > 3) max_cycles = atoi(argv[3]);
    if (argc > 4) p_bits_arg = atoi(argv[4]);
    (void)p_bits_arg;  // used via LL_P_BITS constant in run_sync_cycle

    printf("[v32] Starting: N=%d S=%d steps_per_sync=%d cycles=%d p_bits=%d\n",
           N, S, STEPS_PER_SYNC, max_cycles, p_bits_arg);

    // Validate GPU
    int dev_count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&dev_count));
    if (dev_count == 0) { fprintf(stderr, "No CUDA devices.\n"); return 1; }
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("[v32] GPU: %s  SM=%d.%d  mem=%.1f GB\n",
           prop.name, prop.major, prop.minor,
           (double)prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));

    // Build host state
    HostState hs;
    memset(&hs, 0, sizeof(hs));
    hs.N = N;
    hs.S = S;

    alloc_device_arrays(&hs);
    alloc_pinned_buffers(&hs);

    CUDA_CHECK(cudaStreamCreateWithFlags(&hs.stream0, cudaStreamNonBlocking));
    CUDA_CHECK(cudaStreamCreateWithFlags(&hs.stream1, cudaStreamNonBlocking));
    CUDA_CHECK(cudaStreamCreateWithFlags(&hs.stream2, cudaStreamNonBlocking));
    CUDA_CHECK(cudaEventCreateWithFlags(&hs.ev_field_start, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&hs.ev_field_done,  cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&hs.ev_ll_done,     cudaEventDisableTiming));

    init_device_state(&hs);

    printf("[v32] Allocation complete.  Starting main loop.\n");

    // Main loop
    for (int cycle = 0; (max_cycles == 0 || cycle < max_cycles) && g_running; cycle++) {
        run_sync_cycle(&hs);

        if (cycle % 100 == 0) {
            printf("[v32] cycle=%d  total_steps=%lld  total_cands=%lld\n",
                   cycle, hs.total_steps, hs.total_candidates);
        }
    }

    // Graceful shutdown
    CUDA_CHECK(cudaDeviceSynchronize());
    printf("[v32] Shutdown.  total_steps=%lld  total_candidates=%lld\n",
           hs.total_steps, hs.total_candidates);

    cudaEventDestroy(hs.ev_field_start);
    cudaEventDestroy(hs.ev_field_done);
    cudaEventDestroy(hs.ev_ll_done);
    cudaStreamDestroy(hs.stream0);
    cudaStreamDestroy(hs.stream1);
    cudaStreamDestroy(hs.stream2);
    free_pinned_buffers(&hs);
    free_device_arrays(&hs);

    return 0;
}
