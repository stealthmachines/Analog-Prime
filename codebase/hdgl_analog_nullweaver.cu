// ============================================================================
// Nullweaver Ω BEAST MODE — Pure CUDA (No Python)
// Ultimate fusion of HDGL v36 + Matured Phi-Pi Supergate
// ============================================================================
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuComplex.h>
#include <stdio.h>
#include <math.h>

#define FIELD_BLOCK     256
#define STRIDE_A        89                    // Fibonacci Feistel stride
#define MAX_STEPS       6
#define PHI_F           1.618033988749895f
#define LN_PHI_F        0.48121182505960347f
#define INV_2PHI_F      0.3090169943749474f
#define KAPPA_BASE      2.0780869212350273f   // 1/ln(φ) — more accurate
#define ALPHA_INV_F     137.035999f

__device__ __forceinline__ float safe_log(float x) {
    return logf(fmaxf(x, 1e-8f));
}

__device__ __forceinline__ cuFloatComplex c_mul(cuFloatComplex a, cuFloatComplex b) {
    return make_cuFloatComplex(a.x*b.x - a.y*b.y, a.x*b.y + a.y*b.x);
}

__device__ __forceinline__ float c_abs(cuFloatComplex z) {
    return sqrtf(z.x*z.x + z.y*z.y + 1e-12f);
}

// ============================================================================
// BEAST MODE KERNEL
// ============================================================================
__global__ void nullweaver_beast_kernel(
    cuFloatComplex* __restrict__ field,     // in/out [N]
    float* __restrict__ one_eff_out,        // [N]
    float* __restrict__ null_score_out,     // [N]
    float* __restrict__ coherence_out,      // [N]
    int N,
    int steps,
    float kappa_scale,      // learnable / tunable
    float coupling,         // learnable
    float anchor_base)      // learnable
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int tid = threadIdx.x;
    if (i >= N) return;

    __shared__ float sh_u_mid[FIELD_BLOCK / 32];

    // Load field
    cuFloatComplex z = field[i];
    float amp = c_abs(z);
    float phase = atan2f(z.y, z.x);
    float r_h = amp + 1e-8f;

    // Feistel partner (HDGL v36 exact style)
    int j = (i + STRIDE_A) % N;
    cuFloatComplex zj = field[j];
    float ph_j = atan2f(zj.y, zj.x);
    float phj_n = fmodf(ph_j / 6.283185307179586f + 1.0f, 1.0f);

    // === Nested U-Field Beast Evolution ===
    float inner_exp = 0.5f * phj_n + kappa_scale * KAPPA_BASE * safe_log(r_h + 1.0f);
    float u = expf(inner_exp * LN_PHI_F);

    for (int s = 0; s < steps; ++s) {
        // Warp collective mean
        float u_sum = u;
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            u_sum += __shfl_down_sync(0xffffffffu, u_sum, off);

        float u_mean = __shfl_sync(0xffffffffu, u_sum, 0) / 32.0f;

        float diff = fabsf(u - u_mean) + 1e-8f;
        float log_diff = safe_log(diff);

        // Beast oscillatory kernel — full phi-pi resonance
        float osc = cosf(2.0f * M_PI_F * log_diff * PHI_F) * 
                    sinf(M_PI_F * log_diff / PHI_F) * 
                    cosf(0.5f * log_diff);   // extra harmonic

        float gauss = expf(-diff * diff / (2.0f * PHI_F * PHI_F));
        float interaction = osc * gauss * coupling;

        float anchor = anchor_base / sqrtf((float)s + 1.618f);   // φ-flavored decay

        u = expf(u * LN_PHI_F) + interaction * anchor;
        u = fmaxf(1e-6f, fminf(600.0f, u));   // controlled explosion prevention
    }

    // Block reduce → M_U
    if ((tid & 31) == 0)
        sh_u_mid[tid >> 5] = u;

    __syncthreads();

    if (tid == 0) {
        float mu = 0.0f;
        #pragma unroll
        for (int w = 0; w < FIELD_BLOCK/32; ++w)
            mu += sh_u_mid[w];
        sh_u_mid[0] = mu / (FIELD_BLOCK / 32.0f);
    }
    __syncthreads();

    float M_U = fmaxf(sh_u_mid[0], 1e-6f);
    float lambda_phi = safe_log(M_U) / LN_PHI_F - INV_2PHI_F;

    // === Final Phi-Pi Alignment & Null Gate ===
    float beta = lambda_phi - floorf(lambda_phi);
    float delta = fabsf(cosf(M_PI_F * beta * PHI_F)) *
                  (logf(ALPHA_INV_F) / powf(PHI_F, 2.0f + 0.618f * beta)) + 0.15f;

    float one_eff = fmaxf(0.6f, fminf(4.2f, 1.0f + delta));

    // Drive phase toward π (negative real axis)
    float phase_raw = M_PI_F * lambda_phi;
    float correction = M_PI_F - fmodf(phase_raw + 6.283185307179586f, 6.283185307179586f);

    // Apply beast modulation
    cuFloatComplex mod = make_cuFloatComplex(one_eff * cosf(correction),
                                             one_eff * sinf(correction));
    field[i] = c_mul(z, mod);

    // Diagnostics
    if (one_eff_out)     one_eff_out[i] = one_eff;

    float null_score = fabsf(cosf(phase_raw) + one_eff) + fabsf(sinf(phase_raw));
    if (null_score_out)  null_score_out[i] = null_score;

    float coherence = __expf(-null_score * 15.0f);   // extremely sharp
    if (coherence_out)   coherence_out[i] = coherence;
}

// ============================================================================
// Host interface
// ============================================================================
extern "C" {

void launch_beast_omega(
    cuFloatComplex* field,
    float* one_eff_out,
    float* null_score_out,
    float* coherence_out,
    int N,
    int steps,
    float kappa_scale,
    float coupling,
    float anchor_base,
    cudaStream_t stream = 0)
{
    int blockSize = FIELD_BLOCK;
    int gridSize = (N + blockSize - 1) / blockSize;

    nullweaver_beast_kernel<<<gridSize, blockSize, 0, stream>>>(
        field, one_eff_out, null_score_out, coherence_out,
        N, steps, kappa_scale, coupling, anchor_base);
}

} // extern "C"
