/* ============================================================================
 * HDGL Prismatic Recursion Scorer v35
 * Phi-prismatic resonance scoring for Mersenne exponent candidates.
 * ============================================================================
 *
 * P(p, r) = sqrt(phi^(p%16) * F[p%128] * 2^(p%16) * Pr[p%128] * Omega)
 *           * r^((p%7)+1)
 *
 * where:
 *   F[i]    = i-th Fibonacci number (precomputed table, 128 entries)
 *   Pr[i]   = i-th prime number     (precomputed table, 128 entries)
 *   Omega   = 0.5 + 0.5 * sin(pi * frac(n(p)) * phi)
 *   n(p)    = log(log(p) / ln(phi)) / ln(phi) - 1/(2*phi)
 *   phi     = 1.618033988749...
 * ============================================================================ */

#include <math.h>
#include "hdgl_prismatic_v35.h"

/* ---- constants ----------------------------------------------------------- */
#define PHI_P    1.6180339887498948482
#define LN_PHI_P 0.4812118250596034748
#define PI_P     3.14159265358979323846

/* Fibonacci numbers F(0)..F(127) — computed via double, stored as float */
static const float FIB128[128] = {
      1,  1,  2,  3,  5,  8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987,
   1597, 2584, 4181, 6765, 10946, 17711, 28657, 46368, 75025, 121393,
   196418, 317811, 514229, 832040, 1346269, 2178309,
   3524578, 5702887, 9227465, 14930352, 24157817, 39088169,
   63245986, 102334155, 165580141, 267914296, 433494437, 701408733,
   1134903170, 1836311903,
   /* F(46) onward — capped at float max to avoid overflow in sqrt */
   2971215073.0f, 4807526976.0f, 7778742049.0f, 1.259e10f, 2.037e10f,
   3.295e10f, 5.332e10f, 8.628e10f, 1.396e11f, 2.259e11f, 3.654e11f,
   5.914e11f, 9.568e11f, 1.548e12f, 2.505e12f, 4.053e12f,
   6.557e12f, 1.061e13f, 1.717e13f, 2.778e13f, 4.495e13f, 7.273e13f,
   1.177e14f, 1.904e14f, 3.081e14f, 4.985e14f, 8.065e14f, 1.305e15f,
   2.111e15f, 3.417e15f, 5.527e15f, 8.944e15f, 1.447e16f, 2.342e16f,
   3.789e16f, 6.130e16f, 9.920e16f, 1.605e17f, 2.597e17f, 4.202e17f,
   6.799e17f, 1.100e18f, 1.780e18f, 2.880e18f, 4.660e18f, 7.540e18f,
   1.220e19f, 1.974e19f, 3.194e19f, 5.168e19f, 8.362e19f, 1.353e20f,
   2.189e20f, 3.542e20f, 5.731e20f, 9.273e20f, 1.500e21f, 2.428e21f,
   3.928e21f, 6.355e21f, 1.028e22f, 1.664e22f, 2.692e22f, 4.356e22f,
   7.048e22f, 1.140e23f, 1.845e23f, 2.985e23f, 4.830e23f, 7.815e23f,
1.265e24f, 2.046e24f, 3.311e24f, 5.357e24f, 8.668e24f, 1.402e25f, // 6 = 122
    2.269e25f, 3.671e25f, 5.940e25f, 9.611e25f, 1.555e26f, 2.516e26f  // 6 = 128
};

/* First 128 prime numbers */
static const uint32_t PRIME128[128] = {
      2,   3,   5,   7,  11,  13,  17,  19,  23,  29,  31,  37,  41,  43,  47,  53,
     59,  61,  67,  71,  73,  79,  83,  89,  97, 101, 103, 107, 109, 113, 127, 131,
    137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199, 211, 223,
    227, 229, 233, 239, 241, 251, 257, 263, 269, 271, 277, 281, 283, 293, 307, 311,
    313, 317, 331, 337, 347, 349, 353, 359, 367, 373, 379, 383, 389, 397, 401, 409,
    419, 421, 431, 433, 439, 443, 449, 457, 461, 463, 467, 479, 487, 491, 499, 503,
    509, 521, 523, 541, 547, 557, 563, 569, 571, 577, 587, 593, 599, 601, 607, 613,
    617, 619, 631, 641, 643, 647, 653, 659, 661, 673, 677, 683, 691, 701, 709, 719
};

/* phi^k for k = 0..15 */
static float phi_pow16[16];
/* 2^k for k = 0..15 */
static float pow2_16[16];
static int tables_init = 0;

static void init_tables(void)
{
    if (tables_init) return;
    double p = 1.0;
    double t = 1.0;
    for (int k = 0; k < 16; k++) {
        phi_pow16[k] = (float)p;
        pow2_16[k]   = (float)t;
        p *= PHI_P;
        t *= 2.0;
    }
    tables_init = 1;
}

/* ---- public API ---------------------------------------------------------- */

float hdgl_prismatic_score(uint32_t p, float r_h)
{
    init_tables();

    int k16   = (int)(p % 16u);
    int k128  = (int)(p % 128u);
    int kpow  = (int)(p % 7u) + 1;   /* r exponent: 1..7 */

    /* Phi-lattice coordinate */
    double lp  = log((double)p);
    double nc  = log(lp / LN_PHI_P) / LN_PHI_P - 0.5 / PHI_P;
    double frc = nc - floor(nc);
    if (frc < 0.0) frc += 1.0;
    double omega = 0.5 + 0.5 * sin(PI_P * frc * PHI_P);

    /* Base = phi^k16 * F128[k128] * 2^k16 * Pr128[k128] * Omega */
    double base = (double)phi_pow16[k16]
                * (double)FIB128[k128]
                * (double)pow2_16[k16]
                * (double)PRIME128[k128]
                * omega;
    if (base <= 0.0) base = 1e-30;

    double r_norm = (r_h > 0.0f) ? (double)r_h : 1e-9;
    double score  = sqrt(base) * pow(r_norm, (double)kpow);

    /* Normalise to a finite float range using log-scale: exp(-|log(score)/100|) */
    if (!isfinite(score) || score <= 0.0) score = 1e-30;
    return (float)(log(score) + 1e-9);
}

void hdgl_prismatic_sort(uint32_t *exps, float *r_hs, int n)
{
    /* Insertion sort descending by score (n <= 256, so O(n^2) is fine) */
    for (int i = 1; i < n; i++) {
        uint32_t kp = exps[i];
        float    kr = r_hs[i];
        float    ks = hdgl_prismatic_score(kp, kr);
        int j = i - 1;
        while (j >= 0 && hdgl_prismatic_score(exps[j], r_hs[j]) < ks) {
            exps[j+1] = exps[j];
            r_hs[j+1] = r_hs[j];
            j--;
        }
        exps[j+1] = kp;
        r_hs[j+1] = kr;
    }
}
