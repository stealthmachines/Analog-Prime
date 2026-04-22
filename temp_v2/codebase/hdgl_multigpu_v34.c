// ============================================================================
// HDGL Multi-GPU Orchestrator v34
// ============================================================================
//
// [ROADMAP ITEM 16] Domain decomposition across G GPUs.
//
// Decomposition strategy:
//   GPU k handles slots [k*N/G, (k+1)*N/G).
//   Each GPU owns its DevSoA (or SieveDevState for the sieve path).
//   Boundary exchange: 1-slot-wide "halo" at each GPU boundary is copied
//   P2P before each step so neighbour coupling is correct.
//
// Weight synchronisation:
//   After each hdgl_weight_sync_kernel on each GPU, the per-GPU global weight
//   averages are reduced across GPUs:
//     - If NCCL available: ncclAllReduce over (G) float arrays.
//     - Fallback (no NCCL): manual ring-reduce via cudaMemcpyPeerAsync.
//
// Prime candidate merge:
//   Each GPU produces its own d_prime_found ring.
//   Host merges all rings, deduplicates, and logs verified primes.
//
// Usage:
//   hdgl_multigpu_v34  [N] [S] [cycles] [p_bits] [n_gpu]
//   Defaults: N=4194304, S=2048, cycles=1000, p_bits=127, n_gpu=auto
//
// Compile (without NCCL):
//   nvcc -O3 -arch=sm_86 -lineinfo hdgl_analog_v33.cu hdgl_warp_ll_v33.cu \
//        hdgl_sieve_v34.cu hdgl_host_v34.c hdgl_critic_v33.c \
//        hdgl_multigpu_v34.c -o hdgl_v34_mg -lm
//
// Compile (with NCCL):
//   nvcc -O3 -arch=sm_86 -lineinfo ... -lnccl -DUSE_NCCL
// ============================================================================

#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef USE_NCCL
#include <nccl.h>
#endif

// ============================================================================
// Forward declarations matching the extern "C" signatures in v33 kernels
// ============================================================================

typedef struct {
    float *A_re, *A_im, *phase, *phase_vel;
    float *r_harmonic;
    uint32_t *ll_state;
    float *reward_accum;
    float *w_cos, *w_sin, *w_sigma;
    int8_t *ll_verified;
    void  *candidates;
    int   *cand_count;
    int    N, S;
    float  omega, dt;
    float  w_amp_self, w_amp_neigh;
} DevSoA;

typedef struct {
    uint32_t *d_assigned_p;
    uint32_t *d_sieve_state;
    float    *d_r_harmonic;
    float    *d_phase;
    float    *d_reward_accum;
    uint32_t *d_prime_found;
    int      *d_prime_count;
    uint32_t *h_prime_found;
    int      *h_prime_count;
    int N, S;
    float dt;
} SieveHostState;

extern "C" {
    // Field kernels
    void hdgl_v33_upload_soa(const DevSoA *h);
    void hdgl_v33_upload_critic(const float *w, int n);
    void hdgl_v33_field_step(int N, int block, cudaStream_t s);
    void hdgl_v33_reward_inject(int N, int block, cudaStream_t s);
    void hdgl_v33_weight_sync(cudaStream_t s);
    void hdgl_v33_read_global_weights(float oc[], float os[], float osg[]);
    // Sieve
    int  hdgl_sieve_alloc(SieveHostState *st, int N, int S, float dt);
    void hdgl_sieve_free(SieveHostState *st);
    void hdgl_sieve_step(const SieveHostState *st, cudaStream_t s);
    int  hdgl_sieve_harvest(SieveHostState *st, uint32_t *out, int max, cudaStream_t s);
    // Critic
    void critic_init(void);
    void critic_pack_weights(float *out);
    int  critic_weight_count(void);
    void critic_print_stats(void);
}

// ============================================================================
// Constants
// ============================================================================

#define MAX_GPUS          8
#define FIELD_BLOCK       256
#define SPECTRAL_N        4
#define PRIME_RING        256
#define HALO_SLOTS        1     // one slot per boundary
#define WEIGHT_TOTAL      57    // critic weights

// ============================================================================
// Per-GPU state
// ============================================================================

typedef struct {
    int gpu_id;
    int slot_start;        // global slot index of first slot on this GPU
    int slot_count;        // number of slots on this GPU (N/G or N/G+1)
    int S;                 // row width (same across all GPUs)

    // CUDA primitives
    cudaStream_t stream0;  // field evolution
    cudaStream_t stream1;  // candidate LL + I/O
    cudaStream_t stream2;  // reward + weight sync
    cudaEvent_t  ev_field_done;
    cudaEvent_t  ev_ll_done;

    // Field-path device arrays
    DevSoA       soa;          // device arrays for this GPU's slot partition
    float       *d_critic_w;   // critic weights on this GPU (WEIGHT_TOTAL floats)
    float       *d_global_wc;  // SPECTRAL_N global cos weights
    float       *d_global_ws;  // SPECTRAL_N global sin weights
    float       *d_global_wsg; // SPECTRAL_N global sigma weights

    // Sieve-path state
    SieveHostState sieve;

    // Halo buffers for P2P boundary exchange [field path]
    // Left boundary: last halo slots from previous GPU (inbound)
    float *d_halo_r_in;   // r_harmonic of HALO_SLOTS slots
    float *d_halo_ph_in;  // phase of HALO_SLOTS slots
    // Right boundary: first halo slots from next GPU (outbound)
    float *d_halo_r_out;
    float *d_halo_ph_out;

    // Harvest buffer
    uint32_t h_primes[PRIME_RING];
    int      n_primes;

#ifdef USE_NCCL
    ncclComm_t nccl_comm;
#endif
} GpuState;

static GpuState g_gpus[MAX_GPUS];
static int      g_n_gpus = 0;
static int      g_N      = 0;
static int      g_S      = 0;
static float    g_dt     = 0.001f;
static int      g_mode   = 0;   // 0 = field path, 1 = sieve path

// Host-side momentum weights (field path)
static float g_host_gw_c[SPECTRAL_N];
static float g_host_gw_s[SPECTRAL_N];
static float g_host_gw_sg[SPECTRAL_N];

// Critic weights
static float g_critic_packed[WEIGHT_TOTAL];

// ============================================================================
// Helper: check if P2P is available between two GPUs
// ============================================================================

static int check_p2p(int src, int dst) {
    int can = 0;
    cudaDeviceCanAccessPeer(&can, src, dst);
    return can;
}

// ============================================================================
// Alloc device arrays for one GPU (field path)
// ============================================================================

static int alloc_gpu_field(GpuState *g) {
    cudaSetDevice(g->gpu_id);
    int n = g->slot_count;

    cudaMalloc(&g->soa.A_re,        n * sizeof(float));
    cudaMalloc(&g->soa.A_im,        n * sizeof(float));
    cudaMalloc(&g->soa.phase,       n * sizeof(float));
    cudaMalloc(&g->soa.phase_vel,   n * sizeof(float));
    cudaMalloc(&g->soa.r_harmonic,  n * sizeof(float));
    cudaMalloc(&g->soa.ll_state,    n * sizeof(uint32_t));
    cudaMalloc(&g->soa.reward_accum,n * sizeof(float));
    cudaMalloc(&g->soa.w_cos,       n * SPECTRAL_N * sizeof(float));
    cudaMalloc(&g->soa.w_sin,       n * SPECTRAL_N * sizeof(float));
    cudaMalloc(&g->soa.w_sigma,     n * SPECTRAL_N * sizeof(float));
    cudaMalloc(&g->soa.ll_verified, n * sizeof(int8_t));
    cudaMalloc(&g->soa.cand_count,  sizeof(int));

    cudaMalloc(&g->d_critic_w,   WEIGHT_TOTAL * sizeof(float));
    cudaMalloc(&g->d_global_wc,  SPECTRAL_N   * sizeof(float));
    cudaMalloc(&g->d_global_ws,  SPECTRAL_N   * sizeof(float));
    cudaMalloc(&g->d_global_wsg, SPECTRAL_N   * sizeof(float));

    // Halo buffers (2 directions × HALO_SLOTS × {r_h, phase})
    cudaMalloc(&g->d_halo_r_in,   HALO_SLOTS * sizeof(float));
    cudaMalloc(&g->d_halo_ph_in,  HALO_SLOTS * sizeof(float));
    cudaMalloc(&g->d_halo_r_out,  HALO_SLOTS * sizeof(float));
    cudaMalloc(&g->d_halo_ph_out, HALO_SLOTS * sizeof(float));

    // Create streams and events
    cudaStreamCreate(&g->stream0);
    cudaStreamCreate(&g->stream1);
    cudaStreamCreate(&g->stream2);
    cudaEventCreate(&g->ev_field_done);
    cudaEventCreate(&g->ev_ll_done);

    return 0;
}

// ============================================================================
// Free device arrays for one GPU
// ============================================================================

static void free_gpu_field(GpuState *g) {
    cudaSetDevice(g->gpu_id);
    cudaFree(g->soa.A_re);        cudaFree(g->soa.A_im);
    cudaFree(g->soa.phase);       cudaFree(g->soa.phase_vel);
    cudaFree(g->soa.r_harmonic);  cudaFree(g->soa.ll_state);
    cudaFree(g->soa.reward_accum);
    cudaFree(g->soa.w_cos);       cudaFree(g->soa.w_sin);
    cudaFree(g->soa.w_sigma);     cudaFree(g->soa.ll_verified);
    cudaFree(g->soa.cand_count);
    cudaFree(g->d_critic_w);
    cudaFree(g->d_global_wc);     cudaFree(g->d_global_ws);
    cudaFree(g->d_global_wsg);
    cudaFree(g->d_halo_r_in);     cudaFree(g->d_halo_ph_in);
    cudaFree(g->d_halo_r_out);    cudaFree(g->d_halo_ph_out);
    cudaStreamDestroy(g->stream0);
    cudaStreamDestroy(g->stream1);
    cudaStreamDestroy(g->stream2);
    cudaEventDestroy(g->ev_field_done);
    cudaEventDestroy(g->ev_ll_done);
}

// ============================================================================
// P2P halo exchange (field path)
// Exchange right boundary of GPU k with left boundary of GPU k+1 (ring)
// Called before field step.
// ============================================================================

static void exchange_halos(void) {
    for (int k = 0; k < g_n_gpus; k++) {
        GpuState *cur  = &g_gpus[k];
        GpuState *next = &g_gpus[(k + 1) % g_n_gpus];
        int n = cur->slot_count;

        if (g_n_gpus == 1) continue;  // no exchange needed

        // Export: copy last HALO_SLOTS from cur.r_harmonic to cur.d_halo_r_out
        cudaSetDevice(cur->gpu_id);
        cudaMemcpyAsync(cur->d_halo_r_out,
                        cur->soa.r_harmonic + (n - HALO_SLOTS),
                        HALO_SLOTS * sizeof(float),
                        cudaMemcpyDeviceToDevice,
                        cur->stream0);
        cudaMemcpyAsync(cur->d_halo_ph_out,
                        cur->soa.phase + (n - HALO_SLOTS),
                        HALO_SLOTS * sizeof(float),
                        cudaMemcpyDeviceToDevice,
                        cur->stream0);
        cudaStreamSynchronize(cur->stream0);

        // P2P transfer to next GPU's incoming halo
        if (check_p2p(cur->gpu_id, next->gpu_id)) {
            cudaMemcpyPeerAsync(next->d_halo_r_in,  next->gpu_id,
                                cur->d_halo_r_out,  cur->gpu_id,
                                HALO_SLOTS * sizeof(float),
                                next->stream0);
            cudaMemcpyPeerAsync(next->d_halo_ph_in, next->gpu_id,
                                cur->d_halo_ph_out, cur->gpu_id,
                                HALO_SLOTS * sizeof(float),
                                next->stream0);
        } else {
            // Fallback: staged through host
            float tmp_r[HALO_SLOTS], tmp_ph[HALO_SLOTS];
            cudaMemcpy(tmp_r,  cur->d_halo_r_out,  HALO_SLOTS * sizeof(float), cudaMemcpyDeviceToHost);
            cudaMemcpy(tmp_ph, cur->d_halo_ph_out, HALO_SLOTS * sizeof(float), cudaMemcpyDeviceToHost);
            cudaSetDevice(next->gpu_id);
            cudaMemcpy(next->d_halo_r_in,  tmp_r,  HALO_SLOTS * sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(next->d_halo_ph_in, tmp_ph, HALO_SLOTS * sizeof(float), cudaMemcpyHostToDevice);
            cudaSetDevice(cur->gpu_id);
        }

        // Inject incoming halo: write into slot 0 of next GPU's phase/r arrays
        // (The kernel will read neighbours[i-1] which wraps to the halo for slot 0)
        cudaSetDevice(next->gpu_id);
        cudaMemcpyAsync(next->soa.r_harmonic, next->d_halo_r_in,
                        HALO_SLOTS * sizeof(float),
                        cudaMemcpyDeviceToDevice, next->stream0);
        cudaMemcpyAsync(next->soa.phase, next->d_halo_ph_in,
                        HALO_SLOTS * sizeof(float),
                        cudaMemcpyDeviceToDevice, next->stream0);
    }
    // Sync all GPUs before field step
    for (int k = 0; k < g_n_gpus; k++) {
        cudaSetDevice(g_gpus[k].gpu_id);
        cudaStreamSynchronize(g_gpus[k].stream0);
    }
}

// ============================================================================
// Weight synchronisation across GPUs
//
// After each GPU computes its per-slot weight averages (via hdgl_v33_weight_sync),
// we need a cross-GPU reduce to produce a truly global weight vector.
//
// NCCL path:   ncclAllReduce on (g_global_wc, g_global_ws, g_global_wsg) per GPU
// Fallback:    ring-reduce via P2P or host staging
// ============================================================================

static void sync_weights_across_gpus(void) {
    // Read per-GPU averages
    float wc[MAX_GPUS][SPECTRAL_N];
    float ws[MAX_GPUS][SPECTRAL_N];
    float wsg[MAX_GPUS][SPECTRAL_N];

    for (int k = 0; k < g_n_gpus; k++) {
        cudaSetDevice(g_gpus[k].gpu_id);
        hdgl_v33_read_global_weights(wc[k], ws[k], wsg[k]);
    }

#ifdef USE_NCCL
    // NCCL AllReduce — each GPU contributes SPECTRAL_N floats
    for (int k = 0; k < g_n_gpus; k++) {
        cudaSetDevice(g_gpus[k].gpu_id);
        cudaMemcpyAsync(g_gpus[k].d_global_wc, wc[k],
                        SPECTRAL_N * sizeof(float), cudaMemcpyHostToDevice,
                        g_gpus[k].stream2);
    }
    // AllReduce (sum) then divide by g_n_gpus
    ncclGroupStart();
    for (int k = 0; k < g_n_gpus; k++) {
        ncclAllReduce(g_gpus[k].d_global_wc,  g_gpus[k].d_global_wc,
                      SPECTRAL_N, ncclFloat, ncclSum, g_gpus[k].nccl_comm,
                      g_gpus[k].stream2);
        ncclAllReduce(g_gpus[k].d_global_ws,  g_gpus[k].d_global_ws,
                      SPECTRAL_N, ncclFloat, ncclSum, g_gpus[k].nccl_comm,
                      g_gpus[k].stream2);
        ncclAllReduce(g_gpus[k].d_global_wsg, g_gpus[k].d_global_wsg,
                      SPECTRAL_N, ncclFloat, ncclSum, g_gpus[k].nccl_comm,
                      g_gpus[k].stream2);
    }
    ncclGroupEnd();
    for (int k = 0; k < g_n_gpus; k++) {
        cudaSetDevice(g_gpus[k].gpu_id);
        cudaStreamSynchronize(g_gpus[k].stream2);
    }
    // Read back and divide
    float inv = 1.0f / (float)g_n_gpus;
    for (int i = 0; i < SPECTRAL_N; i++) {
        float sum_c = 0, sum_s = 0, sum_sg = 0;
        for (int k = 0; k < g_n_gpus; k++) { sum_c += wc[k][i]; sum_s += ws[k][i]; sum_sg += wsg[k][i]; }
        g_host_gw_c[i]  = sum_c  * inv;
        g_host_gw_s[i]  = sum_s  * inv;
        g_host_gw_sg[i] = sum_sg * inv;
    }
#else
    // Manual ring reduce: compute average on host, broadcast back
    float inv = 1.0f / (float)g_n_gpus;
    for (int i = 0; i < SPECTRAL_N; i++) {
        float sum_c = 0, sum_s = 0, sum_sg = 0;
        for (int k = 0; k < g_n_gpus; k++) {
            sum_c  += wc[k][i];
            sum_s  += ws[k][i];
            sum_sg += wsg[k][i];
        }
        g_host_gw_c[i]  = sum_c  * inv;
        g_host_gw_s[i]  = sum_s  * inv;
        g_host_gw_sg[i] = sum_sg * inv;
    }
#endif

    // Momentum blend and broadcast critic weights to all GPUs
    critic_pack_weights(g_critic_packed);
    for (int k = 0; k < g_n_gpus; k++) {
        cudaSetDevice(g_gpus[k].gpu_id);
        hdgl_v33_upload_critic(g_critic_packed, WEIGHT_TOTAL);
    }
}

// ============================================================================
// Prime candidate merge with deduplication
// ============================================================================

static void merge_primes(int cycle) {
    // Harvest from all GPUs
    uint32_t all[MAX_GPUS * PRIME_RING];
    int total = 0;

    for (int k = 0; k < g_n_gpus; k++) {
        if (g_mode == 1) {
            // Sieve path
            int n = hdgl_sieve_harvest(&g_gpus[k].sieve, all + total,
                                        PRIME_RING, g_gpus[k].stream1);
            total += n;
        }
    }

    if (total == 0) return;

    // Deduplicate (simple sort + unique)
    // Insertion sort for small arrays
    for (int i = 1; i < total; i++) {
        uint32_t key = all[i];
        int j = i - 1;
        while (j >= 0 && all[j] > key) { all[j+1] = all[j]; j--; }
        all[j+1] = key;
    }

    uint32_t last = 0;
    for (int i = 0; i < total; i++) {
        if (all[i] == 0 || all[i] == last) continue;
        last = all[i];
        printf("[cycle %5d] CANDIDATE MERSENNE PRIME EXPONENT: p = %u  "
               "(2^%u - 1 needs full LL verification)\n",
               cycle, all[i], all[i]);
        fflush(stdout);
    }
}

// ============================================================================
// Initialise multi-GPU system
// ============================================================================

int hdgl_multigpu_init(int N, int S, float dt, int n_gpu_request, int mode) {
    int device_count = 0;
    cudaGetDeviceCount(&device_count);
    if (device_count == 0) {
        fprintf(stderr, "[multigpu] No CUDA devices found.\n");
        return -1;
    }

    g_n_gpus = (n_gpu_request > 0 && n_gpu_request <= device_count)
               ? n_gpu_request : device_count;
    if (g_n_gpus > MAX_GPUS) g_n_gpus = MAX_GPUS;
    g_N    = N;
    g_S    = S;
    g_dt   = dt;
    g_mode = mode;

    printf("[multigpu] Using %d GPU(s), N=%d, S=%d, mode=%s\n",
           g_n_gpus, N, S, mode == 1 ? "sieve" : "field");

    // Enable P2P where possible
    for (int k = 0; k < g_n_gpus; k++) {
        for (int j = 0; j < g_n_gpus; j++) {
            if (k != j && check_p2p(k, j)) cudaDeviceEnablePeerAccess(j, 0);
        }
    }

#ifdef USE_NCCL
    // Initialise NCCL communicators
    ncclUniqueId nccl_id;
    ncclGetUniqueId(&nccl_id);
    ncclGroupStart();
    for (int k = 0; k < g_n_gpus; k++) {
        cudaSetDevice(k);
        ncclCommInitRank(&g_gpus[k].nccl_comm, g_n_gpus, nccl_id, k);
    }
    ncclGroupEnd();
#endif

    // Init critic (single instance on host)
    critic_init();
    critic_pack_weights(g_critic_packed);

    // Distribute slots
    int base = N / g_n_gpus;
    int rem  = N % g_n_gpus;
    int start = 0;

    for (int k = 0; k < g_n_gpus; k++) {
        GpuState *g = &g_gpus[k];
        g->gpu_id     = k;
        g->slot_start = start;
        g->slot_count = base + (k < rem ? 1 : 0);
        g->S          = S;

        cudaSetDevice(k);

        if (mode == 0) {
            alloc_gpu_field(g);
            // Upload critic weights to this GPU
            cudaMemcpy(g->d_critic_w, g_critic_packed,
                       WEIGHT_TOTAL * sizeof(float), cudaMemcpyHostToDevice);
            hdgl_v33_upload_critic(g_critic_packed, WEIGHT_TOTAL);
        } else {
            hdgl_sieve_alloc(&g->sieve, g->slot_count, S, dt);
            cudaStreamCreate(&g->stream0);
            cudaStreamCreate(&g->stream1);
            cudaStreamCreate(&g->stream2);
        }

        printf("[multigpu] GPU %d: slots [%d, %d)\n",
               k, start, start + g->slot_count);
        start += g->slot_count;
    }

    memset(g_host_gw_c,  0, sizeof(g_host_gw_c));
    memset(g_host_gw_s,  0, sizeof(g_host_gw_s));
    memset(g_host_gw_sg, 0, sizeof(g_host_gw_sg));

    return 0;
}

// ============================================================================
// Run one multi-GPU cycle
// ============================================================================

void hdgl_multigpu_cycle(int cycle) {
    if (g_mode == 0) {
        // Field path: halo exchange → parallel field steps → weight sync
        exchange_halos();

        for (int k = 0; k < g_n_gpus; k++) {
            GpuState *g = &g_gpus[k];
            cudaSetDevice(g->gpu_id);
            // Upload SoA for this GPU's partition
            hdgl_v33_upload_soa(&g->soa);
            // Launch field step (stream0)
            hdgl_v33_field_step(g->slot_count, FIELD_BLOCK, g->stream0);
            cudaEventRecord(g->ev_field_done, g->stream0);
            // Reward inject + weight sync (stream2, waits for field)
            cudaStreamWaitEvent(g->stream2, g->ev_field_done, 0);
            hdgl_v33_reward_inject(g->slot_count, FIELD_BLOCK, g->stream2);
            hdgl_v33_weight_sync(g->stream2);
        }

        // Wait for all GPUs
        for (int k = 0; k < g_n_gpus; k++) {
            cudaSetDevice(g_gpus[k].gpu_id);
            cudaStreamSynchronize(g_gpus[k].stream2);
        }

        sync_weights_across_gpus();

    } else {
        // Sieve path: parallel sieve steps → harvest
        for (int k = 0; k < g_n_gpus; k++) {
            GpuState *g = &g_gpus[k];
            cudaSetDevice(g->gpu_id);
            hdgl_sieve_step(&g->sieve, g->stream0);
        }
        for (int k = 0; k < g_n_gpus; k++) {
            cudaSetDevice(g_gpus[k].gpu_id);
            cudaStreamSynchronize(g_gpus[k].stream0);
        }
        merge_primes(cycle);
    }
}

// ============================================================================
// Shutdown
// ============================================================================

void hdgl_multigpu_shutdown(void) {
    for (int k = 0; k < g_n_gpus; k++) {
        cudaSetDevice(g_gpus[k].gpu_id);
        if (g_mode == 0) {
            free_gpu_field(&g_gpus[k]);
        } else {
            hdgl_sieve_free(&g_gpus[k].sieve);
            cudaStreamDestroy(g_gpus[k].stream0);
            cudaStreamDestroy(g_gpus[k].stream1);
            cudaStreamDestroy(g_gpus[k].stream2);
        }
#ifdef USE_NCCL
        ncclCommDestroy(g_gpus[k].nccl_comm);
#endif
    }
    printf("[multigpu] Shutdown complete.\n");
}

// ============================================================================
// main
// ============================================================================

int main(int argc, char **argv) {
    int   N       = (argc > 1) ? atoi(argv[1]) : 4194304;
    int   S       = (argc > 2) ? atoi(argv[2]) : 2048;
    int   cycles  = (argc > 3) ? atoi(argv[3]) : 1000;
    int   p_bits  = (argc > 4) ? atoi(argv[4]) : 127;
    int   n_gpu   = (argc > 5) ? atoi(argv[5]) : 0;
    int   mode    = (argc > 6) ? atoi(argv[6]) : 1;  // 1=sieve by default
    float dt      = 0.001f;
    (void)p_bits;

    if (hdgl_multigpu_init(N, S, dt, n_gpu, mode) != 0) return 1;

    printf("[multigpu] Running %d cycles...\n", cycles);
    for (int c = 0; c < cycles; c++) {
        hdgl_multigpu_cycle(c);
        if (c % 100 == 0) {
            printf("[multigpu] Cycle %5d complete.\n", c);
            critic_print_stats();
            fflush(stdout);
        }
    }

    hdgl_multigpu_shutdown();
    return 0;
}
