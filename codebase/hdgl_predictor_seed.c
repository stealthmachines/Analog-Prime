/* ============================================================================
 * HDGL Predictor Seed — phi-lattice candidate scoring for Mersenne exponents
 * Adapted from phi_mersenne_predictor.c (conscious-128-bit-floor)
 * ============================================================================
 *
 * The phi-lattice coordinate of a Mersenne exponent p is:
 *   n(p) = log(log(p) / ln(phi)) / ln(phi)  −  1/(2*phi)
 *
 * 67 % of known Mersenne prime exponents have frac(n(p)) < 0.5.
 * The D_n amplitude score captures the local resonance strength.
 *
 * This file provides:
 *   hdgl_n_coord()       — compute n(p)
 *   hdgl_phi_lower_half()— 1 if frac(n(p)) < 0.5
 *   hdgl_predictor_top20()— scan sieve band, return top-N by D_n score
 * ============================================================================ */

#include <math.h>
#include <string.h>
#include "hdgl_predictor_seed.h"
#include "hdgl_phi_lang.h"   /* DN_EMPIRICAL_BETA: calibrated beta from BigG/Fudge10 */

/* ---- constants ----------------------------------------------------------- */
#define PHI      1.6180339887498948482
#define LN_PHI   0.4812118250596034748
#define NUM_DN   8

/* Fibonacci table for D_n amplitude */
static const double FIB8[NUM_DN] = {1.0, 1.0, 2.0, 3.0, 5.0, 8.0, 13.0, 21.0};
/* First 8 primes for D_n amplitude */
static const double PRM8[NUM_DN] = {2.0, 3.0, 5.0, 7.0, 11.0, 13.0, 17.0, 19.0};

/* Known Mersenne prime exponents (51 values, for mean_delta computation) */
static const uint32_t MERSENNE_EXP[51] = {
    2, 3, 5, 7, 13, 17, 19, 31, 61, 89, 107, 127, 521, 607, 1279, 2203,
    2281, 3217, 4253, 4423, 9689, 9941, 11213, 19937, 21701, 23209, 44497,
    86243, 110503, 132049, 216091, 756839, 859433, 1257787, 1398269, 2976221,
    3021377, 6972593, 13466917, 20996011, 24036583, 25964951, 30402457,
    32582657, 37156667, 42643801, 43112609, 57885161, 74207281, 77232917,
    136279841
};

/* Sieve band start (same as SIEVE_BASE_P in hdgl_sieve_v34.cu) */
#define BASE_P 82589934u
/* Scan range: scan 400000 odd candidates (~200000 numbers) */
#define SCAN_RANGE 400000

/* ---- internal helpers ---------------------------------------------------- */

static double n_of_x(double x)
{
    double lx = log(x);
    return log(lx / LN_PHI) / LN_PHI - 0.5 / PHI;
}

static double frac(double v)
{
    double f = v - floor(v);
    return (f < 0.0) ? f + 1.0 : f;
}

/* D_n amplitude for coordinate nc.
 * Uses DN_EMPIRICAL_BETA from hdgl_phi_lang.h: fractional correction
 * calibrated by BigG/Fudge10 empirical_validation (beta=0.360942,
 * VALIDATION 1 chi^2_red~0.000, VALIDATION 2 100% CODATA pass rate). */
static double dn_amp(double nc)
{
    double nc_cal = nc + (double)DN_EMPIRICAL_BETA;  /* apply calibrated beta */
    int    dim = (int)fabs(nc_cal) % NUM_DN;
    double r   = frac(nc_cal);
    int    k   = (dim + 1) / NUM_DN;   /* exponent for r */
    double om  = 0.5 + 0.5 * sin(nc_cal);  /* Omega */
    double base = PHI * FIB8[dim] * PRM8[dim] * 2.0 * om;
    if (base <= 0.0) base = 1e-9;
    return sqrt(base) * pow(r + 1e-9, k + 1);
}

/* Score function: D_n amplitude + bonus for lower-half */
static double phi_score(uint32_t p)
{
    double nc = n_of_x((double)p);
    double amp = dn_amp(nc);
    double bonus = (frac(nc) < 0.5) ? 2.0 : 1.0;
    return amp * bonus;
}

/* ---- public API ---------------------------------------------------------- */

double hdgl_n_coord(uint32_t p)
{
    return n_of_x((double)p);
}

int hdgl_phi_lower_half(uint32_t p)
{
    double nc = n_of_x((double)p);
    return (frac(nc) < 0.5) ? 1 : 0;
}

int hdgl_predictor_top20(uint32_t *out, int max_n)
{
    if (max_n <= 0) return 0;
    if (max_n > 20) max_n = 20;

    /* Compute mean_delta from the 51 known Mersenne exponents so we can
     * also use the forward-step prediction to bias the initial scan window. */
    double n_vals[51];
    for (int i = 0; i < 51; i++)
        n_vals[i] = n_of_x((double)MERSENNE_EXP[i]);

    double mean_delta = 0.0;
    for (int i = 1; i < 51; i++)
        mean_delta += n_vals[i] - n_vals[i - 1];
    mean_delta /= 50.0;

    /* n-coordinate of the last known exponent */
    double n_last = n_vals[50];   /* n(136279841) */

    /* Predicted n-coordinate range for next candidates: [n_last, n_last+20*delta] */
    double n_pred_lo = n_last;
    double n_pred_hi = n_last + (double)max_n * mean_delta;

    /* Heap: keep top max_n by score using simple insertion into sorted buffer */
    uint32_t best_p[20];
    double   best_s[20];
    int      n_best = 0;

    /* Scan odd candidates in the sieve band */
    uint32_t start = BASE_P | 1u;   /* ensure odd */
    for (uint32_t p = start; p < start + (uint32_t)SCAN_RANGE; p += 2) {
        double sc = phi_score(p);

        /* Optionally weight by proximity to predicted n-range */
        double nc = n_of_x((double)p);
        if (nc >= n_pred_lo && nc <= n_pred_hi) sc *= 1.5;

        if (n_best < max_n) {
            best_p[n_best] = p;
            best_s[n_best] = sc;
            n_best++;
            /* insertion sort step */
            for (int j = n_best - 1; j > 0 && best_s[j] > best_s[j-1]; j--) {
                uint32_t tp = best_p[j]; best_p[j] = best_p[j-1]; best_p[j-1] = tp;
                double   ts = best_s[j]; best_s[j] = best_s[j-1]; best_s[j-1] = ts;
            }
        } else if (sc > best_s[n_best - 1]) {
            best_p[n_best - 1] = p;
            best_s[n_best - 1] = sc;
            /* re-sort last element */
            for (int j = n_best - 1; j > 0 && best_s[j] > best_s[j-1]; j--) {
                uint32_t tp = best_p[j]; best_p[j] = best_p[j-1]; best_p[j-1] = tp;
                double   ts = best_s[j]; best_s[j] = best_s[j-1]; best_s[j-1] = ts;
            }
        }
    }

    for (int i = 0; i < n_best; i++)
        out[i] = best_p[i];
    return n_best;
}
