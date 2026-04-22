// ============================================================================
// HDGL Psi-Filter v35 — Riemann zeta pre-filter for Mersenne exponents
// ============================================================================
//
// 3-pass Riemann psi spike detector.  Filters an array of uint32_t Mersenne
// exponents p: for each candidate the spike delta_psi(p) = psi(p) - psi(p-1)
// is estimated from the first B Riemann zeros.  Primes produce a spike of
// height ~ln(p); composites produce near-zero spikes.
//
// Pass 1 (B=500  zeros) : hard kill threshold
// Pass 2 (B=5000 zeros) : softer second-pass on Pass-1 survivors
// Pass 3 (B=10000 zeros): convergence check on Pass-2 survivors
//
// API:
//   hdgl_psi_filter_alloc() — allocate device scratch
//   hdgl_psi_filter_run()   — filter a batch of host exponents, returns count
//   hdgl_psi_filter_free()  — release device memory
// ============================================================================

#pragma once
#include <stdint.h>
#include <cuda_runtime.h>

#define PSI_MAX_CANDS 512

typedef struct {
    double  *d_cands;   /* [max_cands] working double buffer on device    */
    uint8_t *d_flags;   /* [max_cands] pass/fail flag per candidate        */
    int     *d_live;    /* [max_cands] compacted live indices              */
    double  *d_scores;  /* [max_cands] normalised spike height             */
    double  *h_cands;   /* [max_cands] pinned host mirror                  */
    uint8_t *h_flags;   /* [max_cands] pinned host mirror                  */
    double  *h_scores;  /* [max_cands] pinned host mirror                  */
    int      max_cands;
} PsiFilterState;

#ifdef __cplusplus
extern "C" {
#endif

/* Allocate GPU scratch buffers for up to max_cands candidates. */
void hdgl_psi_filter_alloc(PsiFilterState *st, int max_cands);

/* Release all device/pinned buffers. */
void hdgl_psi_filter_free(PsiFilterState *st);

/* Run the 3-pass psi spike filter.
 *   h_in     : host array of Mersenne exponents to test
 *   n_in     : number of input candidates (must be <= st->max_cands)
 *   h_out    : host array to receive surviving exponents (size >= n_in)
 *   h_scores : (optional) normalised spike heights for survivors (size >= n_in)
 *   stream   : CUDA stream (may be 0 for default stream)
 * Returns:  number of survivors written to h_out / h_scores.
 */
int hdgl_psi_filter_run(PsiFilterState *st,
                         const uint32_t *h_in, int n_in,
                         uint32_t *h_out, float *h_scores,
                         cudaStream_t stream);

#ifdef __cplusplus
} /* extern "C" */
#endif
