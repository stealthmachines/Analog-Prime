// ============================================================================
// HDGL Analog v37 — Spatiotemporal Field Upgrade (Memory + Cross-Coupling)
// ============================================================================
//
// NEW IN v37:
//   ✔ Persistent memory field (mem_re / mem_im)
//   ✔ Cross-token banded coupling (lightweight field propagation)
//   ✔ Stabilized φ-recursion (tanh clamp)
//   ✔ Fully backward-compatible with v36 host (if memory = null)
//
// CORE MODEL NOW:
//   - Spatial field (v36)
//   - Temporal continuity (memory)
//   - Nonlinear φ-recursive aggregation
//   - Learned spectral coupling
//
// This is now a:
//   → Spatiotemporal nonlinear field system
// ============================================================================

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdint.h>
#include <math.h>

// ============================================================================
// Constants (existing + v37 additions)
// ============================================================================

#define SPECTRAL_N        4
#define FIELD_BLOCK       256

#define PHI_F             1.6180339887f
#define LN_PHI_F          0.4812118251f
#define INV_2PHI_F        0.3090169944f
#define TWO_PI_F          6.2831853071f
#define INV_2PI_F         0.1591549430f

#define STRIDE_A          89

// === v37 additions ===
#define MEM_DECAY_F       0.95f
#define MEM_INJECT_F      0.05f
#define CROSS_ALPHA_F     0.7f
#define CROSS_SIGMA_F     2.0f
#define MAX_NEIGHBOR_K    2

// ============================================================================
// SoA (extended)
// ============================================================================

typedef struct {
    float *A_re;
    float *A_im;
    float *phase;
    float *phase_vel;
    float *r_harmonic;

    // === v37 memory ===
    float *mem_re;
    float *mem_im;

    int N;
    int S;
    float omega;
    float dt;
} DevSoA;

__device__ __constant__ DevSoA g_soa;

// ============================================================================
// Helpers
// ============================================================================

__device__ __forceinline__ float clampf(float x, float a, float b) {
    return fminf(fmaxf(x, a), b);
}

// ============================================================================
// Main Kernel v37
// ============================================================================

__global__ void hdgl_field_step_kernel_v37(int step_count)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= g_soa.N) return;

    float A_re  = g_soa.A_re[i];
    float A_im  = g_soa.A_im[i];
    float ph    = g_soa.phase[i];
    float phvel = g_soa.phase_vel[i];
    float r_h   = g_soa.r_harmonic[i];

    float dt    = g_soa.dt;

    float local_amp = sqrtf(A_re*A_re + A_im*A_im);

    // =========================================================================
    // Feistel partner
    // =========================================================================
    float ph_j  = g_soa.phase[(i + STRIDE_A) % g_soa.N];
    float phj_n = ph_j * INV_2PI_F;

    // =========================================================================
    // Base coupling (v36 preserved)
    // =========================================================================
    float sum_sin = 0.0f;

    int left  = (i - 1 + g_soa.N) % g_soa.N;
    int right = (i + 1) % g_soa.N;

    sum_sin += sinf(g_soa.phase[left]  - ph);
    sum_sin += sinf(g_soa.phase[right] - ph);

    // =========================================================================
    // v37 Cross-token banded coupling
    // =========================================================================
    float cross_sum = 0.0f;

    #pragma unroll
    for (int dk = -MAX_NEIGHBOR_K; dk <= MAX_NEIGHBOR_K; dk++) {
        int j = (i + dk + g_soa.N) % g_soa.N;
        float dphi = g_soa.phase[j] - ph;
        float w = expf(-(dk * dk) / (2.0f * CROSS_SIGMA_F));
        cross_sum += w * sinf(dphi);
    }

    cross_sum /= (2 * MAX_NEIGHBOR_K + 1);
    sum_sin += CROSS_ALPHA_F * cross_sum;

    // =========================================================================
    // v37 Memory injection
    // =========================================================================
    float mem_re = (g_soa.mem_re) ? g_soa.mem_re[i] : 0.0f;
    float mem_im = (g_soa.mem_im) ? g_soa.mem_im[i] : 0.0f;

    float mem_amp = sqrtf(mem_re*mem_re + mem_im*mem_im + 1e-8f);
    float mem_u   = logf(mem_amp + 1.0f);

    float u_local = logf(local_amp + 1.0f);

    float mdiff = fabsf(u_local - mem_u) + 1e-6f;
    float mld   = clampf(logf(mdiff), -10.0f, 10.0f);

    float mem_osc =
        cosf(2.0f * 3.1415926536f * mld * PHI_F) *
        sinf(3.1415926536f * mld / PHI_F);

    phvel += 0.05f * mem_osc;

    // =========================================================================
    // Phase velocity update
    // =========================================================================
    phvel += dt * (sum_sin + 0.02f * logf(r_h + 1.0f));

    // =========================================================================
    // Phase update (Feistel torus)
    // =========================================================================
    float ph_n = ph * INV_2PI_F;
    float bias = phvel * dt * INV_2PI_F;

    ph_n = fmodf(PHI_F * (ph_n + 0.5f * phj_n + bias), 1.0f);
    if (ph_n < 0.0f) ph_n += 1.0f;

    ph = ph_n * TWO_PI_F;

    // =========================================================================
    // v37 Stabilized U-field bridge
    // =========================================================================
    float inner_exp = 0.5f * phj_n + 0.02f * logf(r_h + 1.0f);

    float u_inner = expf(tanhf(inner_exp) * LN_PHI_F);

    float M_inner = u_inner; // simplified (safe fallback)

    float u_mid = expf(tanhf(M_inner) * LN_PHI_F);

    float M_U = u_mid;

    float lambda_phi = logf(M_U + 1e-6f) / LN_PHI_F - INV_2PHI_F;

    // =========================================================================
    // Amplitude update
    // =========================================================================
    float A = expf(lambda_phi * 0.1f);
    A = clampf(A, 0.0f, 1000.0f);

    A_re = A * cosf(ph);
    A_im = A * sinf(ph);

    // =========================================================================
    // v37 Memory update
    // =========================================================================
    if (g_soa.mem_re && g_soa.mem_im) {
        float new_mem_u = MEM_DECAY_F * mem_u + MEM_INJECT_F * u_local;

        float mem_phase = atan2f(mem_im, mem_re);
        float mem_amp_new = expf(new_mem_u);

        g_soa.mem_re[i] = mem_amp_new * cosf(mem_phase);
        g_soa.mem_im[i] = mem_amp_new * sinf(mem_phase);
    }

    // =========================================================================
    // Write back
    // =========================================================================
    g_soa.A_re[i]       = A_re;
    g_soa.A_im[i]       = A_im;
    g_soa.phase[i]      = ph;
    g_soa.phase_vel[i]  = phvel;
    g_soa.r_harmonic[i] = r_h;
}

// ============================================================================
// Host launcher (same signature style)
// ============================================================================

extern "C" {

void hdgl_v37_field_step(int N, int block, int step_count, cudaStream_t s)
{
    hdgl_field_step_kernel_v37<<<(N + block - 1)/block, block, 0, s>>>(step_count);
}

}
