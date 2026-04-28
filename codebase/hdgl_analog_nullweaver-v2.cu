// ============================================================================
// Nullweaver Ω BEAST MODE — FULL EDITION
// Single fused kernel with multi-stack, residuals, wavelet basis, and batched layout
// ============================================================================
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuComplex.h>
#include <stdio.h>
#include <math.h>

#define FIELD_BLOCK     256
#define STRIDE_A        89
#define MAX_STACKS      6
#define SPECTRAL_N      4
#define PHI_F           1.618033988749895f
#define LN_PHI_F        0.48121182505960347f
#define INV_2PHI_F      0.3090169943749474f
#define KAPPA_BASE      2.0780869212350273f   // 1 / ln(φ)

struct OmegaParams {
    float kappa_scale;
    float coupling_strength;
    float anchor_base;
    float null_sharpness;
    float wavelet_self;
    float wavelet_neigh;
    float mix_weights[MAX_STACKS + 1];   // for residual blending
};

__constant__ OmegaParams c_params;

// ============================================================================
// Wavelet Spectral Basis (HDGL v36 style)
// ============================================================================
__device__ __forceinline__ float wavelet_spectral_eval(
    float dphi, float local_amp, float neigh_amp,
    const float* __restrict__ wc, const float* __restrict__ ws, const float* __restrict__ wsig)
{
    float spec = 0.0f;
    float phi2 = dphi * dphi;

    #pragma unroll
    for (int k = 0; k < SPECTRAL_N; ++k) {
        float freq = (float)(1 << k);
        float sigma = wsig[k];
        float sigma2 = sigma * sigma + 1e-6f;
        float gauss = expf(-phi2 / (2.0f * sigma2));
        float kd = freq * dphi;
        spec += gauss * (wc[k] * cosf(kd) + ws[k] * sinf(kd));
    }
    spec += c_params.wavelet_self * local_amp;
    spec += c_params.wavelet_neigh * neigh_amp;
    return spec;
}

// ============================================================================
// BEAST KERNEL — Fused Multi-Stack + Residuals + Wavelet
// ============================================================================
__global__ void nullweaver_beast_full_kernel(
    cuFloatComplex* __restrict__ field,     // [B * L * D]  — flattened
    float* __restrict__ one_eff_out,        // optional [B * L]
    float* __restrict__ null_score_out,     // optional [B * L]
    float* __restrict__ coherence_out,      // optional [B * L]
    int B, int L, int D,                    // batch, seq, dim
    int num_stacks,
    int steps_per_stack)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int tid = threadIdx.x;
    if (idx >= B * L * D) return;

    // Decode 3D indices
    int d = idx % D;
    int l = (idx / D) % L;
    int b = idx / (D * L);

    const int base = (b * L + l) * D;           // start of this token's vector
    const int N_tokens = B * L;

    __shared__ float sh_u_mid[FIELD_BLOCK / 32];

    cuFloatComplex z = field[base + d];
    float amp = sqrtf(z.x*z.x + z.y*z.y + 1e-12f);
    float phase = atan2f(z.y, z.x);

    // Feistel partner (global across all tokens)
    int l_partner = (l + STRIDE_A) % L;
    int partner_idx = (b * L + l_partner) * D + d;
    cuFloatComplex zj = field[partner_idx];
    float ph_j = atan2f(zj.y, zj.x);
    float phj_n = fmodf(ph_j / 6.283185307179586f + 1.0f, 1.0f);

    float r_h = amp + 1e-8f;

    cuFloatComplex current = z;

    // === Multi-Stack with Residuals (fused) ===
    for (int stack = 0; stack < num_stacks; ++stack) {
        // Inner U-field evolution
        float inner_exp = 0.5f * phj_n + c_params.kappa_scale * KAPPA_BASE * logf(r_h + 1.0f);
        float u = expf(inner_exp * LN_PHI_F);

        for (int s = 0; s < steps_per_stack; ++s) {
            // Warp collective mean
            float u_sum = u;
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1)
                u_sum += __shfl_down_sync(0xffffffffu, u_sum, off);

            float u_mean = __shfl_sync(0xffffffffu, u_sum, 0) / 32.0f;

            float diff = fabsf(u - u_mean) + 1e-8f;
            float log_diff = logf(diff);

            // Beast phi-pi oscillatory kernel
            float osc = cosf(2.0f * M_PI_F * log_diff * PHI_F) *
                        sinf(M_PI_F * log_diff / PHI_F) *
                        cosf(0.618f * log_diff);   // golden harmonic

            float gauss = expf(-diff*diff / (2.0f * PHI_F*PHI_F));
            float interaction = osc * gauss * c_params.coupling_strength;

            float anchor = c_params.anchor_base / sqrtf((float)s + PHI_F);

            u = expf(u * LN_PHI_F) + interaction * anchor;
            u = fmaxf(1e-6f, fminf(800.0f, u));
        }

        // Block reduce → M_U
        if ((tid & 31) == 0)
            sh_u_mid[tid >> 5] = u;
        __syncthreads();

        if (tid == 0) {
            float mu = 0.0f;
            for (int w = 0; w < FIELD_BLOCK/32; ++w) mu += sh_u_mid[w];
            sh_u_mid[0] = mu / (FIELD_BLOCK / 32.0f);
        }
        __syncthreads();

        float M_U = fmaxf(sh_u_mid[0], 1e-6f);
        float lambda_phi = logf(M_U) / LN_PHI_F - INV_2PHI_F;

        // Phi-Pi final alignment
        float beta = lambda_phi - floorf(lambda_phi);
        float delta = fabsf(cosf(M_PI_F * beta * PHI_F)) *
                      (logf(ALPHA_INV_F) / powf(PHI_F, 2.0f + 0.618f * beta)) + 0.15f;

        float one_eff = fmaxf(0.6f, fminf(4.5f, 1.0f + delta));

        float phase_raw = M_PI_F * lambda_phi;
        float correction = M_PI_F - fmodf(phase_raw + 6.283185307179586f, 6.283185307179586f);

        cuFloatComplex mod = make_cuFloatComplex(one_eff * cosf(correction),
                                                 one_eff * sinf(correction));

        // Apply + residual skip connection
        cuFloatComplex modulated = c_mul(current, mod);
        current = make_cuFloatComplex(
            modulated.x * c_params.mix_weights[stack] + current.x * (1.0f - c_params.mix_weights[stack]),
            modulated.y * c_params.mix_weights[stack] + current.y * (1.0f - c_params.mix_weights[stack])
        );

        // Optional diagnostics (only on first dimension)
        if (d == 0) {
            if (one_eff_out)     one_eff_out[b*L + l] = one_eff;
            float null_score = fabsf(cosf(phase_raw) + one_eff) + fabsf(sinf(phase_raw));
            if (null_score_out)  null_score_out[b*L + l] = null_score;
            if (coherence_out)   coherence_out[b*L + l] = expf(-null_score * c_params.null_sharpness);
        }
    }

    // Write final result
    field[base + d] = current;
}

// ============================================================================
// Host launcher
// ============================================================================
extern "C" {

void launch_beast_omega_full(
    cuFloatComplex* field,
    float* one_eff_out,
    float* null_score_out,
    float* coherence_out,
    int B, int L, int D,
    int num_stacks,
    int steps_per_stack,
    cudaStream_t stream = 0)
{
    int total_elements = B * L * D;
    int block = FIELD_BLOCK;
    int grid = (total_elements + block - 1) / block;

    nullweaver_beast_full_kernel<<<grid, block, 0, stream>>>(
        field, one_eff_out, null_score_out, coherence_out,
        B, L, D, num_stacks, steps_per_stack);
}

} // extern "C"
