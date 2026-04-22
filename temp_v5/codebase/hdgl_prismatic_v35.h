#pragma once
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Compute the phi-prismatic resonance score P(p, r_h) for Mersenne exponent p
 * and GRA state r_harmonic.  Higher score = stronger phi-lattice resonance.
 *
 * Formula:
 *   n  = log(log(p)/ln(phi)) / ln(phi) - 1/(2*phi)     (phi-lattice coord)
 *   Omega = 0.5 + 0.5 * sin(pi * frac(n) * phi)
 *   P(p,r) = sqrt(phi^(p%16) * F[p%128] * 2^(p%16) * P_prime[p%128] * Omega)
 *            * r^((p%7)+1)
 * where F[i] is the i-th Fibonacci number and P_prime[i] is the i-th prime. */
float hdgl_prismatic_score(uint32_t p, float r_h);

/* Sort arrays (exps[], r_hs[]) of length n in-place by descending
 * hdgl_prismatic_score().  O(n^2) insertion sort — fine for n <= 256. */
void hdgl_prismatic_sort(uint32_t *exps, float *r_hs, int n);

#ifdef __cplusplus
}
#endif
