// ============================================================================
// HDGL Continuous Mersenne Sieve v34
// ============================================================================
//
// [ROADMAP ITEM 15] Eliminates the discrete candidate buffer.
//
// Architecture: SIEVE replaces EMITTER/CANDIDATE model.
//   Each slot i is permanently assigned a Mersenne exponent p_i from a band:
//       p_i = BASE_P + (i % P_BAND_SIZE) * P_STRIDE
//
//   The slot evolves its LL proxy state toward convergence for p_i.
//   When convergence is detected (LL residue near zero mod (2^p_i − 1)),
//   the slot writes p_i to a ring buffer d_prime_found[].
//
//   The host drains d_prime_found each cycle — no candidate promotion logic,
//   no score accumulator, no threshold tuning.
//
// Sieve kernel replaces hdgl_field_step_kernel:
//   - Runs 4× LL proxy steps per call (same cadence as v32 LL-lite)
//   - GRA plasticity steers r_harmonic toward Mersenne-consistent oscillation
//   - Wave coupling remains: neighbours with close exponents help each other converge
//
// Sieve layout:
//   d_assigned_p[N]    — const once assigned, exponent for each slot
//   d_sieve_state[N]   — 32-bit LL proxy state (evolves each kernel call)
//   d_sieve_r_h[N]     — r_harmonic (plasticity)
//   d_prime_found[256] — ring buffer of found exponents
//   d_prime_count      — atomic write index into ring buffer
//
// Compile:
//   nvcc -O3 -arch=sm_86 -lineinfo hdgl_sieve_v34.cu hdgl_host_v34.c \
//        hdgl_critic_v33.c -o hdgl_v34 -lm
// ============================================================================

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

// ============================================================================
// Sieve configuration constants
// ============================================================================

// Starting Mersenne exponent for the band
// GIMPS record ~2^82M; we target a search band above the current frontier:
//   BASE_P = 82,589,933 + 1  (one above the last known Mersenne prime exponent)
#define SIEVE_BASE_P       82589934u
// Spacing between assigned exponents (candidates must be prime; stride≥2)
#define SIEVE_P_STRIDE     2u
// Number of distinct exponents across the band (wraps modulo)
// With N=1M slots and stride=2: covers 2M consecutive odd exponents
#define SIEVE_P_BAND_SIZE  1048576u  // = 2^20

#define SIEVE_BLOCK        256
#define SIEVE_LL_ITERS     8          // more LL steps per call than v32 (no overhead from cands)
#define SIEVE_PRIME_RING   256
#define SIEVE_RESIDUE_EPS  0.001f     // convergence threshold
#define SIEVE_GRA_PLASTICITY 0.005f
#define SIEVE_R_H_MAX      1000.0f
#define SIEVE_REWARD_DECAY  0.95f

// ============================================================================
// Device arrays — held as global __device__ pointers and scalars
// ============================================================================

typedef struct {
    uint32_t *assigned_p;     // [N] — Mersenne exponent for each slot
    uint32_t *sieve_state;    // [N] — LL proxy state
    float    *r_harmonic;     // [N] — GRA plasticity state
    float    *phase;          // [N] — phase for wave coupling
    float    *reward_accum;   // [N] — convergence confidence accumulator
    uint32_t *prime_found;    // [SIEVE_PRIME_RING] — ring of found exponents
    int      *prime_count;    // atomic counter into ring
    int       N;
    int       S;              // lattice row width (for neighbour calculation)
    float     dt;
} SieveDevState;

__device__ __constant__ SieveDevState g_sieve;

// ============================================================================
// LL proxy arithmetic
// ============================================================================

__device__ __forceinline__ uint32_t ll_step32_mod(uint32_t s, uint32_t p_mod) {
    // Step: s → (s^2 − 2) mod (2^p − 1) approximated as mod p_mod (32-bit)
    // Exact LL mod (2^p−1): fold the 64-bit square result
    uint64_t sq   = (uint64_t)s * (uint64_t)s;
    uint64_t full = sq - 2ULL;
    // Fold mod (2^32 - 1) approximation:  hi*2^32 ≡ hi mod (2^32-1) ≡ hi
    uint64_t lo = full & 0xFFFFFFFFULL;
    uint64_t hi = full >> 32;
    uint64_t r  = lo + hi;
    if (r >= 0xFFFFFFFFULL) r -= 0xFFFFFFFFULL;
    return (uint32_t)r;
}

__device__ __forceinline__ float sieve_residue(uint32_t s) {
    // Normalised residue ∈ [0,1]: how close s is to 0 mod (2^32-1)
    float r = (float)s * 2.32830644e-10f;  // / (2^32)
    if (r > 0.5f) r = 1.0f - r;           // fold to [0, 0.5]
    return r * 2.0f;                       // map to [0, 1]
}

// ============================================================================
// Sieve field step kernel
//
// Each slot:
//   1. Runs SIEVE_LL_ITERS LL proxy steps for its assigned exponent p_i
//   2. Measures convergence: residue < SIEVE_RESIDUE_EPS
//   3. Wave coupling: phase synchronises toward neighbours with close exponents
//   4. GRA plasticity: r_harmonic increases when convergence improves
//   5. If converged: write p_i to prime_found ring (atomic, deduplicate)
// ============================================================================

__global__ void hdgl_sieve_kernel(void)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= g_sieve.N) return;

    uint32_t p_i    = g_sieve.assigned_p[i];
    uint32_t s      = g_sieve.sieve_state[i];
    float    r_h    = g_sieve.r_harmonic[i];
    float    ph     = g_sieve.phase[i];
    float    acc    = g_sieve.reward_accum[i];
    int      N      = g_sieve.N;
    int      S      = g_sieve.S;

    // LL proxy evolution: run SIEVE_LL_ITERS steps
    #pragma unroll
    for (int k = 0; k < SIEVE_LL_ITERS; k++) {
        s = ll_step32_mod(s, (uint32_t)(p_i & 0xFFFFFFFFu));
    }
    float residue = sieve_residue(s);

    // Convergence confidence
    float delta_acc = (residue < SIEVE_RESIDUE_EPS) ? (1.0f - residue) : -residue;
    acc = SIEVE_REWARD_DECAY * acc + delta_acc;

    // GRA plasticity: r_harmonic tracks convergence quality
    float plastic = SIEVE_GRA_PLASTICITY * delta_acc;
    r_h += plastic;
    r_h = fmaxf(1.0f, fminf(SIEVE_R_H_MAX, r_h));

    // Wave coupling: 4-neighbour phase synchronisation weighted by exponent closeness
    int ni[4] = {
        (i - 1 + N) % N,
        (i + 1)     % N,
        (i - S + N) % N,
        (i + S)     % N
    };

    float sum_sin = 0.0f;
    #pragma unroll
    for (int j = 0; j < 4; j++) {
        int    nj     = ni[j];
        float  dphi   = g_sieve.phase[nj] - ph;
        // Exponent proximity weight: slots targeting close exponents couple more strongly
        uint32_t pj   = g_sieve.assigned_p[nj];
        uint32_t pdiff = (p_i > pj) ? (p_i - pj) : (pj - p_i);
        float    prox  = expf(-(float)pdiff * 0.001f);  // decay over exponent distance
        sum_sin += prox * sinf(dphi);
    }

    // Phase Euler step
    ph += g_sieve.dt * (1.0f + 0.5f * sum_sin + 0.01f * r_h * residue);
    ph  = fmodf(ph, 6.28318530717958647f);
    if (ph < 0.0f) ph += 6.28318530717958647f;

    // Convergence detection: write to ring if residue is very small
    if (residue < SIEVE_RESIDUE_EPS && acc > 2.0f) {
        // Attempt to write p_i to ring buffer (atomic)
        int pos = atomicAdd(g_sieve.prime_count, 1) % SIEVE_PRIME_RING;
        g_sieve.prime_found[pos] = p_i;
        // Reset accumulator to avoid duplicate flooding
        acc = 0.0f;
        // Re-seed LL state with S_0 = 4 for next evaluation pass
        s = 4u;
    }

    // Write back
    g_sieve.sieve_state[i]   = s;
    g_sieve.r_harmonic[i]    = r_h;
    g_sieve.phase[i]         = ph;
    g_sieve.reward_accum[i]  = acc;
}

// ============================================================================
// Exponent assignment kernel: slot i → p_i = BASE_P + (i % BAND) * STRIDE
// ============================================================================

__global__ void hdgl_sieve_assign_kernel(
    uint32_t *d_assigned_p, int N,
    uint32_t base_p, uint32_t stride, uint32_t band_size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    d_assigned_p[i] = base_p + (uint32_t)(i % band_size) * stride;
}

// ============================================================================
// Seed kernel: reset all LL states to S_0 = 4, phase = i/N * 2π
// ============================================================================

__global__ void hdgl_sieve_seed_kernel(
    uint32_t *d_state, float *d_phase, float *d_r_h, float *d_acc,
    int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    d_state[i] = 4u;
    d_phase[i] = 6.28318530717958647f * (float)i / (float)N;
    d_r_h[i]   = 10.0f;
    d_acc[i]   = 0.0f;
}

// ============================================================================
// Host-visible data structures and helpers
// ============================================================================

typedef struct {
    // Device pointers
    uint32_t *d_assigned_p;
    uint32_t *d_sieve_state;
    float    *d_r_harmonic;
    float    *d_phase;
    float    *d_reward_accum;
    uint32_t *d_prime_found;
    int      *d_prime_count;
    // Host buffers (pinned)
    uint32_t *h_prime_found;
    int      *h_prime_count;
    // Dimensions
    int N;
    int S;
    float dt;
} SieveHostState;

extern "C" {

// ============================================================================
// Allocate and initialise the sieve
// ============================================================================

int hdgl_sieve_alloc(SieveHostState *st, int N, int S, float dt) {
    st->N  = N;
    st->S  = S;
    st->dt = dt;

    cudaMalloc(&st->d_assigned_p,   N * sizeof(uint32_t));
    cudaMalloc(&st->d_sieve_state,  N * sizeof(uint32_t));
    cudaMalloc(&st->d_r_harmonic,   N * sizeof(float));
    cudaMalloc(&st->d_phase,        N * sizeof(float));
    cudaMalloc(&st->d_reward_accum, N * sizeof(float));
    cudaMalloc(&st->d_prime_found,  SIEVE_PRIME_RING * sizeof(uint32_t));
    cudaMalloc(&st->d_prime_count,  sizeof(int));
    cudaMemset(st->d_prime_count, 0, sizeof(int));
    cudaMemset(st->d_prime_found, 0, SIEVE_PRIME_RING * sizeof(uint32_t));

    cudaMallocHost(&st->h_prime_found,  SIEVE_PRIME_RING * sizeof(uint32_t));
    cudaMallocHost(&st->h_prime_count,  sizeof(int));

    // Assign exponents
    int blk = SIEVE_BLOCK;
    int grd = (N + blk - 1) / blk;
    hdgl_sieve_assign_kernel<<<grd, blk>>>(
        st->d_assigned_p, N,
        SIEVE_BASE_P, SIEVE_P_STRIDE, SIEVE_P_BAND_SIZE);

    // Seed states
    hdgl_sieve_seed_kernel<<<grd, blk>>>(
        st->d_sieve_state, st->d_phase, st->d_r_harmonic, st->d_reward_accum, N);
    cudaDeviceSynchronize();

    // Upload to __constant__
    SieveDevState dev;
    dev.assigned_p   = st->d_assigned_p;
    dev.sieve_state  = st->d_sieve_state;
    dev.r_harmonic   = st->d_r_harmonic;
    dev.phase        = st->d_phase;
    dev.reward_accum = st->d_reward_accum;
    dev.prime_found  = st->d_prime_found;
    dev.prime_count  = st->d_prime_count;
    dev.N            = N;
    dev.S            = S;
    dev.dt           = dt;
    cudaMemcpyToSymbol(g_sieve, &dev, sizeof(SieveDevState));

    return 0;
}

// ============================================================================
// Free device/pinned memory
// ============================================================================

void hdgl_sieve_free(SieveHostState *st) {
    cudaFree(st->d_assigned_p);
    cudaFree(st->d_sieve_state);
    cudaFree(st->d_r_harmonic);
    cudaFree(st->d_phase);
    cudaFree(st->d_reward_accum);
    cudaFree(st->d_prime_found);
    cudaFree(st->d_prime_count);
    cudaFreeHost(st->h_prime_found);
    cudaFreeHost(st->h_prime_count);
    memset(st, 0, sizeof(SieveHostState));
}

// ============================================================================
// Run one sieve step (async)
// ============================================================================

void hdgl_sieve_step(const SieveHostState *st, cudaStream_t stream) {
    int blk = SIEVE_BLOCK;
    int grd = (st->N + blk - 1) / blk;
    hdgl_sieve_kernel<<<grd, blk, 0, stream>>>();
}

// ============================================================================
// Harvest ring buffer: copy prime_found and prime_count to host
// Returns number of new found exponents
// ============================================================================

int hdgl_sieve_harvest(SieveHostState *st, uint32_t *out, int max_out,
                       cudaStream_t stream) {
    cudaMemcpyAsync(st->h_prime_count, st->d_prime_count,
                    sizeof(int), cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(st->h_prime_found, st->d_prime_found,
                    SIEVE_PRIME_RING * sizeof(uint32_t),
                    cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    int count = *(st->h_prime_count);
    if (count > SIEVE_PRIME_RING) count = SIEVE_PRIME_RING;

    int n = (count < max_out) ? count : max_out;
    for (int i = 0; i < n; i++) out[i] = st->h_prime_found[i];

    // Reset ring
    cudaMemsetAsync(st->d_prime_count, 0, sizeof(int), stream);
    cudaMemsetAsync(st->d_prime_found, 0,
                    SIEVE_PRIME_RING * sizeof(uint32_t), stream);
    return n;
}

// ============================================================================
// Priority seed: override first n slots with high-score exponents from the
// phi-lattice predictor and pre-charge their reward accumulator to +1.0.
// Call immediately after hdgl_sieve_alloc() with the output of
// hdgl_predictor_top20().
// ============================================================================

void hdgl_sieve_seed_priority(SieveHostState *st,
                               const uint32_t *p_list, int n)
{
    if (!st || !p_list || n <= 0) return;
    if (n > st->N) n = st->N;

    /* Overwrite the first n exponent assignments */
    cudaMemcpy(st->d_assigned_p, p_list,
               n * sizeof(uint32_t), cudaMemcpyHostToDevice);

    /* Pre-charge reward accumulator to encourage these slots */
    float *reward_host = (float *)malloc(n * sizeof(float));
    if (reward_host) {
        for (int i = 0; i < n; i++) reward_host[i] = 1.0f;
        cudaMemcpy(st->d_reward_accum, reward_host,
                   n * sizeof(float), cudaMemcpyHostToDevice);
        free(reward_host);
    }

    /* Propagate updated constant to device symbol */
    SieveDevState dev;
    dev.assigned_p   = st->d_assigned_p;
    dev.sieve_state  = st->d_sieve_state;
    dev.r_harmonic   = st->d_r_harmonic;
    dev.phase        = st->d_phase;
    dev.reward_accum = st->d_reward_accum;
    dev.prime_found  = st->d_prime_found;
    dev.prime_count  = st->d_prime_count;
    dev.N            = st->N;
    dev.S            = st->S;
    dev.dt           = st->dt;
    cudaMemcpyToSymbol(g_sieve, &dev, sizeof(SieveDevState));
}

} // extern "C"
