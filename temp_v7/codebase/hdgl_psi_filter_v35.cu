// ============================================================================
// HDGL Psi-Filter v35 — implementation
// Ported and adapted from psi_scanner_cuda_v2.cu (conscious-128-bit-floor)
// ============================================================================

#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <math.h>
#include "hdgl_psi_filter_v35.h"

// ---------------------------------------------------------------------------
// Riemann zeta zeros — 80 exact values, rest via Gram-point Newton iteration
// ---------------------------------------------------------------------------

__constant__ double d_zeros_exact[80] = {
    14.134725142,  21.022039639,  25.010857580,  30.424876126,  32.935061588,
    37.586178159,  40.918719012,  43.327073281,  48.005150881,  49.773832478,
    52.970321478,  56.446247697,  59.347044003,  60.831778525,  65.112544048,
    67.079810529,  69.546401711,  72.067157674,  75.704690699,  77.144840069,
    79.337375021,  82.910380854,  84.735492981,  87.425274613,  88.809111208,
    92.491899271,  94.651344041,  95.870634228,  98.831194218, 101.317851006,
   103.725538040, 105.446623052, 107.168611184, 111.029535543, 111.874659177,
   114.320220915, 116.226680321, 118.790782866, 121.370125002, 122.946829294,
   124.256818554, 127.516683880, 129.578704200, 131.087688531, 133.497737203,
   134.756509753, 138.116042055, 139.736208952, 141.123707404, 143.111845808,
   146.000982487, 147.422765343, 150.053589856, 150.925257612, 153.024693811,
   156.112909294, 157.597591818, 158.849988171, 161.188964138, 163.030709687,
   165.537069188, 167.184439978, 169.094515416, 169.911976479, 173.411536520,
   174.754191523, 176.441434298, 178.377407776, 179.916484020, 182.207078484,
   184.874467848, 185.598783678, 187.228922584, 189.416158656, 192.026656361,
   193.079726604, 195.265396680, 196.876481841, 198.015309676, 201.264751944
};

__device__ __forceinline__ double gram_zero_k(int k)
{
    /* Gram-point approximation for the k-th Riemann zero (0-indexed, k>=80).
     * Uses Backlund counting: t_n ≈ 2*pi*n / (log(n/(2*pi*e))).
     * Newton-polish to 6 iterations. */
    double n   = (double)(k + 1);
    double pi  = 3.14159265358979323846;
    double t   = 2.0 * pi * n / log(n / (2.0 * pi * 2.71828182845904523536));
    /* Newton: arg(zeta(0.5+it)) iterations via Stirling ϑ approximation */
    for (int iter = 0; iter < 6; iter++) {
        double theta = t * 0.5 * log(t / (2.0 * pi)) - t * 0.5 - pi / 8.0;
        double dtheta = 0.5 * log(t / (2.0 * pi));
        t -= (theta - pi * n) / dtheta;
    }
    return t;
}

__device__ __forceinline__ double zeta_zero(int k)
{
    return (k < 80) ? d_zeros_exact[k] : gram_zero_k(k);
}

// ---------------------------------------------------------------------------
// Psi spike computation: delta_psi(x) = psi(x) - psi(x-1)
// Approximated by the first B Riemann zeros.
// For prime x: delta_psi ≈ ln(x).
// Normalised spike = delta_psi / ln(x).  Primes → ~1.0; composites → ~0.
// ---------------------------------------------------------------------------

#define PASS1_B      500
#define PASS2_B      3000
#define PASS3_B      8000
#define SPIKE_THRESH1  0.12
#define SPIKE_THRESH2  0.08
#define CONV_EPS       5e-4

/* Compute normalised delta_psi for candidate x using B zeros.
 *
 * Explicit formula:
 *   psi(x) = x - sum_{gamma>0} 2*Re[x^rho/rho] - log(2*pi) - ...
 *
 * Re[x^rho/rho] where rho = 1/2 + i*gamma:
 *   = sqrt(x) * (cos(g*lx)/2 + g*sin(g*lx)) / (1/4 + g^2)
 *
 * delta_psi(x) = psi(x) - psi(x-1)
 *   ≈ 1  -  sum_{gamma>0} 2*(Re[x^rho/rho] - Re[(x-1)^rho/rho])
 *
 * Normalised: spike = delta_psi(x) / ln(x)
 *   Prime x: spike ≈ 1.0
 *   Composite (non prime-power) x: spike ≈ 0.0
 */
__device__ double compute_spike(double x, int B)
{
    double lx   = log(x);
    double lx1  = log(x - 1.0);
    double sqx  = sqrt(x);
    double sqx1 = sqrt(x - 1.0);
    double sum  = 0.0;

    for (int k = 0; k < B; k++) {
        double g     = zeta_zero(k);
        double denom = 0.25 + g * g;
        /* Re[x^rho / rho] */
        double tx  = sqx  * (0.5 * cos(g * lx)  + g * sin(g * lx))  / denom;
        /* Re[(x-1)^rho / rho] */
        double tx1 = sqx1 * (0.5 * cos(g * lx1) + g * sin(g * lx1)) / denom;
        sum += 2.0 * (tx - tx1);
    }

    double delta_psi = 1.0 - sum;   /* main term=1, subtract oscillatory part */
    return delta_psi / lx;
}

// ---------------------------------------------------------------------------
// Pass 1: coarse kill using B=PASS1_B zeros
// ---------------------------------------------------------------------------

__global__ void psi_pass1_kernel(const double *cands, int n,
                                  uint8_t *flags, double *scores)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    double x     = cands[idx];
    double spike = compute_spike(x, PASS1_B);
    scores[idx]  = spike;
    flags[idx]   = (spike >= SPIKE_THRESH1) ? 1 : 0;
}

// ---------------------------------------------------------------------------
// Pass 2: finer check on Pass-1 survivors
// ---------------------------------------------------------------------------

__global__ void psi_pass2_kernel(const double *cands, int n_live,
                                  const int *live_idx,
                                  uint8_t *flags, double *scores)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_live) return;

    int idx      = live_idx[i];
    double x     = cands[idx];
    double spike = compute_spike(x, PASS2_B);
    scores[idx]  = spike;
    if (spike < SPIKE_THRESH2) flags[idx] = 0;
}

// ---------------------------------------------------------------------------
// Pass 3: convergence check on Pass-2 survivors (B=PASS3_B)
// ---------------------------------------------------------------------------

__global__ void psi_pass3_kernel(const double *cands, int n_live,
                                  const int *live_idx,
                                  uint8_t *flags, double *scores)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_live) return;

    int idx       = live_idx[i];
    double x      = cands[idx];
    double spike_a = compute_spike(x, PASS3_B);
    double spike_b = compute_spike(x, PASS3_B - 200);
    scores[idx]   = spike_a;
    /* If the two estimates disagree too much, candidate is borderline — keep
     * but only if larger estimate is still above threshold */
    double delta = fabs(spike_a - spike_b);
    if (delta > CONV_EPS && spike_a < SPIKE_THRESH2 * 1.5) flags[idx] = 0;
}

// ---------------------------------------------------------------------------
// alloc / free
// ---------------------------------------------------------------------------

extern "C"
void hdgl_psi_filter_alloc(PsiFilterState *st, int max_cands)
{
    if (max_cands > PSI_MAX_CANDS) max_cands = PSI_MAX_CANDS;
    st->max_cands = max_cands;

    cudaMalloc(&st->d_cands,  max_cands * sizeof(double));
    cudaMalloc(&st->d_flags,  max_cands * sizeof(uint8_t));
    cudaMalloc(&st->d_live,   max_cands * sizeof(int));
    cudaMalloc(&st->d_scores, max_cands * sizeof(double));

    cudaMallocHost(&st->h_cands,  max_cands * sizeof(double));
    cudaMallocHost(&st->h_flags,  max_cands * sizeof(uint8_t));
    cudaMallocHost(&st->h_scores, max_cands * sizeof(double));
}

extern "C"
void hdgl_psi_filter_free(PsiFilterState *st)
{
    cudaFree(st->d_cands);
    cudaFree(st->d_flags);
    cudaFree(st->d_live);
    cudaFree(st->d_scores);
    cudaFreeHost(st->h_cands);
    cudaFreeHost(st->h_flags);
    cudaFreeHost(st->h_scores);
}

// ---------------------------------------------------------------------------
// run — main filter entry point
// ---------------------------------------------------------------------------

extern "C"
int hdgl_psi_filter_run(PsiFilterState *st,
                         const uint32_t *h_in, int n_in,
                         uint32_t *h_out, float *h_scores_out,
                         cudaStream_t stream)
{
    if (n_in <= 0) return 0;
    if (n_in > st->max_cands) n_in = st->max_cands;

    /* Convert uint32_t exponents → double candidates on host */
    for (int i = 0; i < n_in; i++)
        st->h_cands[i] = (double)h_in[i];

    /* Upload to device */
    cudaMemcpyAsync(st->d_cands, st->h_cands,
                    n_in * sizeof(double),
                    cudaMemcpyHostToDevice, stream);

    /* Zero flags */
    cudaMemsetAsync(st->d_flags, 0, n_in * sizeof(uint8_t), stream);

    /* Pass 1 */
    const int BLK = 64;
    int grd = (n_in + BLK - 1) / BLK;
    psi_pass1_kernel<<<grd, BLK, 0, stream>>>(
        st->d_cands, n_in, st->d_flags, st->d_scores);

    /* Sync, read back flags, compact live list on host */
    cudaStreamSynchronize(stream);
    cudaMemcpyAsync(st->h_flags,  st->d_flags,  n_in * sizeof(uint8_t),
                    cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(st->h_scores, st->d_scores, n_in * sizeof(double),
                    cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    int live_buf[PSI_MAX_CANDS];
    int n_live1 = 0;
    for (int i = 0; i < n_in; i++)
        if (st->h_flags[i]) live_buf[n_live1++] = i;

    if (n_live1 == 0) return 0;

    /* Pass 2 */
    int *d_live_tmp;
    cudaMalloc(&d_live_tmp, n_live1 * sizeof(int));
    cudaMemcpy(d_live_tmp, live_buf, n_live1 * sizeof(int), cudaMemcpyHostToDevice);

    grd = (n_live1 + BLK - 1) / BLK;
    psi_pass2_kernel<<<grd, BLK, 0, stream>>>(
        st->d_cands, n_live1, d_live_tmp, st->d_flags, st->d_scores);
    cudaStreamSynchronize(stream);
    cudaMemcpy(st->h_flags,  st->d_flags,  n_in * sizeof(uint8_t),  cudaMemcpyDeviceToHost);
    cudaMemcpy(st->h_scores, st->d_scores, n_in * sizeof(double), cudaMemcpyDeviceToHost);

    int n_live2 = 0;
    for (int i = 0; i < n_live1; i++) {
        int idx = live_buf[i];
        if (st->h_flags[idx]) live_buf[n_live2++] = idx;
    }
    cudaFree(d_live_tmp);

    if (n_live2 == 0) return 0;

    /* Pass 3 */
    cudaMalloc(&d_live_tmp, n_live2 * sizeof(int));
    cudaMemcpy(d_live_tmp, live_buf, n_live2 * sizeof(int), cudaMemcpyHostToDevice);

    grd = (n_live2 + BLK - 1) / BLK;
    psi_pass3_kernel<<<grd, BLK, 0, stream>>>(
        st->d_cands, n_live2, d_live_tmp, st->d_flags, st->d_scores);
    cudaStreamSynchronize(stream);
    cudaMemcpy(st->h_flags,  st->d_flags,  n_in * sizeof(uint8_t),  cudaMemcpyDeviceToHost);
    cudaMemcpy(st->h_scores, st->d_scores, n_in * sizeof(double), cudaMemcpyDeviceToHost);
    cudaFree(d_live_tmp);

    /* Collect survivors */
    int n_out = 0;
    for (int i = 0; i < n_in; i++) {
        if (st->h_flags[i]) {
            h_out[n_out] = h_in[i];
            if (h_scores_out) h_scores_out[n_out] = (float)st->h_scores[i];
            n_out++;
        }
    }
    return n_out;
}
