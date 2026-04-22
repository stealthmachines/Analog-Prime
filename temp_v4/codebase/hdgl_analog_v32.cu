// ============================================================================
// HDGL Analog v32 — GPU Field Evolution Kernel
// ============================================================================
//
// GPU PORT OF: hdgl_analog_v31b.c (SoA FastLattice, Euler, LL-lite 32-bit)
//
// MEMORY LAYOUT : Structure-of-Arrays (SoA) for coalesced warp reads
// INTEGRATION   : Euler, 1 thread = 1 slot
// LL PROXY      : 32-bit LL-lite (4 iters per step per slot)
// SPECTRAL      : 4-harmonic per-slot learned weights (Hebbian)
// CANDIDATES    : atomicAdd promotion buffer → host via async D2H
// REWARD INJECT : From warp LL exact results (d_ll_verified flags)
// STREAMS       : Exposed launch wrappers; host assigns stream handles
//
// Compile:
//   nvcc -O3 -arch=sm_86 -lineinfo hdgl_analog_v32.cu hdgl_warp_ll_v32.cu \
//        hdgl_host_v32.c -o hdgl_v32 -lm
// ============================================================================

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdint.h>
#include <stdio.h>
#include <math.h>

// ============================================================================
// Constants
// ============================================================================

#define SPECTRAL_N        4          // harmonics per slot
#define LL_LITE_ITERS     4          // squaring iters per step (32-bit proxy)
#define MAX_CAND_BUF      256        // candidate ring buffer capacity (power-of-2)

#define PHI_F             1.6180339887498948f
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

#define LR                1e-3f      // Hebbian learning rate
#define W_CLAMP           2.0f       // weight clamp
#define POOL_ALPHA        0.7f       // [ITEM 8] local vs block-pooled gradient blend
#define CLUSTER_QUORUM    2          // [ITEM 9] min warp-mates resonating to emit candidate
// N must be a multiple of FIELD_BLOCK for __syncthreads() correctness
#define FIELD_BLOCK       256        // must match block arg passed to hdgl_v32_field_step()

// ============================================================================
// Shared types — must match hdgl_warp_ll_v32.cu and hdgl_host_v32.c
// ============================================================================

typedef struct {
    int   slot_idx;
    float score;
    float ll_seed_f;   // float encoding of ll_state for host re-seeding
    float r_harmonic;
    float phase;
} Candidate;

// ============================================================================
// Device SoA pointer block
// All pointers are allocated by the host and set via cudaMemcpyToSymbol.
// Using a single struct as symbol keeps the kernel signature clean.
// ============================================================================

typedef struct {
    float    *A_re;
    float    *A_im;
    float    *phase;
    float    *phase_vel;
    float    *r_harmonic;
    uint32_t *ll_state;
    float    *reward_accum;
    float    *w_cos;         // [N * SPECTRAL_N]
    float    *w_sin;         // [N * SPECTRAL_N]
    int8_t   *ll_verified;   // per-slot: +1 confirmed, -1 rejected, 0 pending
    Candidate *candidates;
    int      *cand_count;
    int       N;             // total slots
    int       S;             // slots_per_instance (for neighbour wrap)
    float     omega;
    float     dt;
    float     w_amp_self;
    float     w_amp_neigh;
} DevSoA;

__device__ __constant__ DevSoA g_soa;

// ============================================================================
// Device helpers
// ============================================================================

__device__ __forceinline__ uint32_t ll_step32(uint32_t s) {
    // s_{n+1} = s_n^2 - 2  mod 2^32
    uint64_t sq = (uint64_t)s * (uint64_t)s;
    return (uint32_t)(sq) - 2u;
}

__device__ __forceinline__ float residue_from_ll(uint32_t s) {
    return fabsf((float)s * 2.3283064365386963e-10f); // / 4294967296.0f
}

// 4-harmonic spectral coupling kernel (per-slot weights)
__device__ __forceinline__ float spectral_eval(
    float dphi,
    float local_amp, float neigh_amp,
    const float * __restrict__ wc,
    const float * __restrict__ ws,
    float w_self, float w_neigh)
{
    float spec = 0.0f;
    #pragma unroll
    for (int k = 0; k < SPECTRAL_N; k++) {
        float kd = (float)(k + 1) * dphi;
        spec += wc[k] * cosf(kd);
        spec += ws[k] * sinf(kd);
    }
    spec += w_self  * local_amp;
    spec += w_neigh * neigh_amp;
    return spec;
}

// ============================================================================
// Main field evolution kernel — 1 thread per slot
// grid = N/FIELD_BLOCK, block = FIELD_BLOCK  (N must be a multiple of FIELD_BLOCK)
// [ITEM 8] Warp-level spectral pooling: block-averaged Hebbian gradient blend
// [ITEM 9] Warp-level resonance clustering: __ballot_sync candidate gating
// ============================================================================

__global__ void hdgl_field_step_kernel(void)
{
    const int i   = blockIdx.x * blockDim.x + threadIdx.x;
    const int tid = threadIdx.x;
    const int N   = g_soa.N;
    const int S   = g_soa.S;
    if (i >= N) return;

    // Shared memory for per-block spectral gradient pooling [ITEM 8]
    // Layout: sh_gc[harmonic][thread], sh_gs[harmonic][thread]
    __shared__ float sh_gc[SPECTRAL_N][FIELD_BLOCK];
    __shared__ float sh_gs[SPECTRAL_N][FIELD_BLOCK];

    // --- Load SoA (coalesced reads) ---
    float A_re    = g_soa.A_re[i];
    float A_im    = g_soa.A_im[i];
    float ph      = g_soa.phase[i];
    float phvel   = g_soa.phase_vel[i];
    float r_h     = g_soa.r_harmonic[i];
    uint32_t s    = g_soa.ll_state[i];
    float acc     = g_soa.reward_accum[i];
    float dt      = g_soa.dt;
    float omega   = g_soa.omega;

    // Per-slot spectral weight local copies
    float wc_local[SPECTRAL_N];
    float ws_local[SPECTRAL_N];
    #pragma unroll
    for (int k = 0; k < SPECTRAL_N; k++) {
        wc_local[k] = g_soa.w_cos[i * SPECTRAL_N + k];
        ws_local[k] = g_soa.w_sin[i * SPECTRAL_N + k];
    }

    float local_amp = sqrtf(A_re * A_re + A_im * A_im);

    // --- LL-lite: 4× squaring ---
    #pragma unroll
    for (int k = 0; k < LL_LITE_ITERS; k++) s = ll_step32(s);
    float residue = residue_from_ll(s);

    // --- 4-neighbour von Neumann coupling ---
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

    // Use right-neighbour phase diff for Hebbian update
    float dphi_hebb = g_soa.phase[ni[1]] - ph;

    #pragma unroll
    for (int j = 0; j < 4; j++) {
        int   n      = ni[j];
        float dphi   = g_soa.phase[n] - ph;
        float n_A_re = g_soa.A_re[n];
        float n_A_im = g_soa.A_im[n];
        float n_amp  = sqrtf(n_A_re * n_A_re + n_A_im * n_A_im);

        float spec = spectral_eval(dphi, local_amp, n_amp,
                                   wc_local, ws_local,
                                   g_soa.w_amp_self, g_soa.w_amp_neigh);

        // GRA coupling — bounded: BASE_GRA * combined / (1 + combined)
        float combined = sqrtf(r_h * r_h + g_soa.r_harmonic[n] * g_soa.r_harmonic[n]);
        float gra_fac  = BASE_GRA_F * combined / (1.0f + combined);
        float factor   = spec + gra_fac;

        sum_sin += sinf(dphi);
        dA_re   += factor * cosf(dphi);
        dA_im   += factor * sinf(dphi);
        gra_sum += gra_fac;
    }

    float dphvel = omega + K_COUPLING_F * sum_sin + 0.15f * gra_sum;

    // --- Euler update ---
    A_re  += dt * dA_re;
    A_im  += dt * dA_im;
    ph    += dt * phvel;
    phvel += dt * dphvel;

    // Amplitude damping + saturation
    float A = sqrtf(A_re * A_re + A_im * A_im);
    A      *= expf(-LAMBDA_F * dt);
    if (A > SAT_LIMIT_F) A = SAT_LIMIT_F;

    // LCG noise (deterministic, no curand overhead per-thread)
    uint32_t noise_s = s ^ (uint32_t)(i * 2654435761u);
    noise_s = noise_s * 1664525u + 1013904223u;
    float noise = ((float)(noise_s >> 16) * 1.52587890625e-5f - 1.0f) * NOISE_SIGMA_F;
    A += noise;
    if (A < 0.0f) A = 0.0f;

    float norm = sqrtf(A_re * A_re + A_im * A_im);
    if (norm > 1e-8f) {
        float inv = A / norm;
        A_re *= inv;
        A_im *= inv;
    }

    // Phase wrap [0, 2π)
    ph = fmodf(ph, 6.28318530717958647f);
    if (ph < 0.0f) ph += 6.28318530717958647f;

    // --- GRA Plasticity (open-recursive) ---
    float plastic = GRA_PLASTIC_F * (local_amp - 0.5f) * (r_h > 50.0f ? 0.05f : 1.0f);
    r_h += plastic;
    r_h  = fmaxf(1.0f, fminf(R_MAX_F, r_h));

    // --- Reward ---
    float reward = 1.0f / (1.0f + residue)
                 + 0.25f * fabsf(sum_sin)
                 + 0.15f * local_amp;

    // =========================================================================
    // [ITEM 8] Warp-level spectral pooling — block-averaged Hebbian gradient
    //
    // Each thread computes its local gradient and writes to shared memory.
    // A tree reduction computes the block average.  Each slot's weight update
    // blends its local gradient with the block average:
    //   Δw = POOL_ALPHA * local_grad + (1 - POOL_ALPHA) * block_avg_grad
    //
    // Effect: slots that detect a pattern weakly still learn from block-mates
    // that detect it strongly — collective intelligence without global sync.
    // =========================================================================
    float grad_cos[SPECTRAL_N], grad_sin[SPECTRAL_N];
    #pragma unroll
    for (int k = 0; k < SPECTRAL_N; k++) {
        float kd = (float)(k + 1) * dphi_hebb;
        grad_cos[k] = LR * reward * cosf(kd);
        grad_sin[k] = LR * reward * sinf(kd);
        sh_gc[k][tid] = grad_cos[k];
        sh_gs[k][tid] = grad_sin[k];
    }
    __syncthreads();

    // Tree reduce to block sum
    for (int stride = FIELD_BLOCK / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            #pragma unroll
            for (int k = 0; k < SPECTRAL_N; k++) {
                sh_gc[k][tid] += sh_gc[k][tid + stride];
                sh_gs[k][tid] += sh_gs[k][tid + stride];
            }
        }
        __syncthreads();
    }

    // Blend local gradient with block average and apply
    const float inv_b = 1.0f / (float)FIELD_BLOCK;
    #pragma unroll
    for (int k = 0; k < SPECTRAL_N; k++) {
        float pool_gc = sh_gc[k][0] * inv_b;
        float pool_gs = sh_gs[k][0] * inv_b;
        wc_local[k] += POOL_ALPHA * grad_cos[k] + (1.0f - POOL_ALPHA) * pool_gc;
        ws_local[k] += POOL_ALPHA * grad_sin[k] + (1.0f - POOL_ALPHA) * pool_gs;
        wc_local[k]  = fmaxf(-W_CLAMP, fminf(W_CLAMP, wc_local[k]));
        ws_local[k]  = fmaxf(-W_CLAMP, fminf(W_CLAMP, ws_local[k]));
    }

    // --- Temporal accumulator ---
    acc = REWARD_DECAY * acc + reward;

    // --- Write back SoA ---
    g_soa.A_re[i]         = A_re;
    g_soa.A_im[i]         = A_im;
    g_soa.phase[i]        = ph;
    g_soa.phase_vel[i]    = phvel;
    g_soa.r_harmonic[i]   = r_h;
    g_soa.ll_state[i]     = s;
    g_soa.reward_accum[i] = acc;

    // Write updated weights back
    #pragma unroll
    for (int k = 0; k < SPECTRAL_N; k++) {
        g_soa.w_cos[i * SPECTRAL_N + k] = wc_local[k];
        g_soa.w_sin[i * SPECTRAL_N + k] = ws_local[k];
    }

    // =========================================================================
    // [ITEM 9] Warp-level resonance clustering — __ballot_sync candidate gating
    //
    // Each 32-thread warp votes on who meets the resonance gate.
    // A candidate is only emitted if cluster_size >= CLUSTER_QUORUM warp-mates
    // also resonate (suppresses isolated noise spikes).
    // Only the elected lane (lowest qualifying lane in the warp) fires atomicAdd,
    // carrying a score boosted by cluster_size.  All qualifying slots decay.
    // =========================================================================
    int meets_gate = (acc > CAND_ACCUM_THRESH &&
                      residue < CAND_RESIDUE_MAX &&
                      local_amp > CAND_AMP_MIN) ? 1 : 0;

    unsigned resonance_mask = __ballot_sync(0xffffffff, meets_gate);
    int      cluster_size   = __popc(resonance_mask);

    if (meets_gate && cluster_size >= CLUSTER_QUORUM) {
        int lane_in_warp = tid & 31;
        int elected_lane = __ffs((int)resonance_mask) - 1;  // lowest set bit, 0-indexed

        // Elected lane emits for the whole cluster; score boosted by cluster size
        if (lane_in_warp == elected_lane) {
            float boosted_score = acc * (1.0f + 0.1f * (float)cluster_size);
            int pos = atomicAdd(g_soa.cand_count, 1);
            if (pos < MAX_CAND_BUF) {
                g_soa.candidates[pos].slot_idx   = i;
                g_soa.candidates[pos].score      = boosted_score;
                g_soa.candidates[pos].ll_seed_f  = (float)s;
                g_soa.candidates[pos].r_harmonic = r_h;
                g_soa.candidates[pos].phase      = ph;
            }
        }
        // All qualifying slots decay accumulator after cluster promotion
        g_soa.reward_accum[i] = acc * 0.5f;
    }
}

// ============================================================================
// Reward injection kernel — applies warp LL verdicts from host
// Called on stream1 after warp LL has completed and h_ll_verified is D2H'd
// ============================================================================

__global__ void hdgl_reward_inject_kernel(void)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= g_soa.N) return;

    int8_t flag = g_soa.ll_verified[i];
    if (flag == 0) return;

    float r_h = g_soa.r_harmonic[i];
    float acc = g_soa.reward_accum[i];

    if (flag == 1) {
        // Confirmed Mersenne candidate: reinforce
        acc += 1.0f;
        r_h  = fminf(r_h * 1.05f, R_MAX_F);
    } else {
        // Rejected: dampen
        acc *= 0.7f;
        r_h *= 0.98f;
        r_h  = fmaxf(1.0f, r_h);
    }

    g_soa.reward_accum[i] = acc;
    g_soa.r_harmonic[i]   = r_h;
    g_soa.ll_verified[i]  = 0;  // clear flag
}

// ============================================================================
// Weight sync kernel — aggregate per-slot weights into global averages
// Launch: <<<1, 256, 0, stream2>>>
// Writes g_global_w_cos / g_global_w_sin  (4 floats each)
// Host reads back and applies momentum before cudaMemcpyToSymbol to g_soa
// ============================================================================

__device__ float g_global_w_cos[SPECTRAL_N];
__device__ float g_global_w_sin[SPECTRAL_N];

__global__ void hdgl_weight_sync_kernel(void)
{
    // Shared reduction buffers
    __shared__ float sh_cos[SPECTRAL_N][256];
    __shared__ float sh_sin[SPECTRAL_N][256];

    int tid = threadIdx.x;
    int N   = g_soa.N;

    // Each thread strides over all slots (grid-stride inside single block)
    float lc[SPECTRAL_N] = {0};
    float ls[SPECTRAL_N] = {0};

    for (int i = tid; i < N; i += 256) {
        const float *wc = g_soa.w_cos + i * SPECTRAL_N;
        const float *ws = g_soa.w_sin + i * SPECTRAL_N;
        #pragma unroll
        for (int k = 0; k < SPECTRAL_N; k++) {
            lc[k] += wc[k];
            ls[k] += ws[k];
        }
    }

    #pragma unroll
    for (int k = 0; k < SPECTRAL_N; k++) {
        sh_cos[k][tid] = lc[k];
        sh_sin[k][tid] = ls[k];
    }
    __syncthreads();

    // Tree reduction
    for (int stride = 128; stride > 0; stride >>= 1) {
        if (tid < stride) {
            #pragma unroll
            for (int k = 0; k < SPECTRAL_N; k++) {
                sh_cos[k][tid] += sh_cos[k][tid + stride];
                sh_sin[k][tid] += sh_sin[k][tid + stride];
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        float inv = (N > 0) ? (1.0f / (float)N) : 0.0f;
        #pragma unroll
        for (int k = 0; k < SPECTRAL_N; k++) {
            g_global_w_cos[k] = sh_cos[k][0] * inv;
            g_global_w_sin[k] = sh_sin[k][0] * inv;
        }
    }
}

// ============================================================================
// Host-callable launcher wrappers (extern C for host_v32.c linkage)
// ============================================================================

extern "C" {

// Upload the DevSoA pointer struct to the __constant__ symbol
void hdgl_v32_upload_soa(const DevSoA *host_soa) {
    cudaMemcpyToSymbol(g_soa, host_soa, sizeof(DevSoA));
}

// Launch field evolution kernel
void hdgl_v32_field_step(int N, int block, cudaStream_t stream) {
    int grid = (N + block - 1) / block;
    hdgl_field_step_kernel<<<grid, block, 0, stream>>>();
}

// Launch reward injection kernel
void hdgl_v32_reward_inject(int N, int block, cudaStream_t stream) {
    int grid = (N + block - 1) / block;
    hdgl_reward_inject_kernel<<<grid, block, 0, stream>>>();
}

// Launch weight sync kernel
void hdgl_v32_weight_sync(cudaStream_t stream) {
    hdgl_weight_sync_kernel<<<1, 256, 0, stream>>>();
}

// Read back global averaged weights after sync kernel completes
void hdgl_v32_read_global_weights(float out_cos[SPECTRAL_N], float out_sin[SPECTRAL_N]) {
    cudaMemcpyFromSymbol(out_cos, g_global_w_cos, SPECTRAL_N * sizeof(float));
    cudaMemcpyFromSymbol(out_sin, g_global_w_sin, SPECTRAL_N * sizeof(float));
}

} // extern "C"
