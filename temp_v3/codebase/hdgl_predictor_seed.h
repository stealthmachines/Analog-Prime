#pragma once
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Scan the sieve band starting at BASE_P and return the top n exponents
 * (up to max_n <= 20) scored by D_n phi-lattice amplitude.
 * Candidates with frac(n(p)) < 0.5 receive a score bonus (67% bias confirmed).
 * out  : array of size >= max_n to receive predicted exponents
 * Returns number of candidates written to out (always <= max_n). */
int hdgl_predictor_top20(uint32_t *out, int max_n);

/* Phi-lattice coordinate of exponent p:
 *   n(p) = log(log(p) / ln(phi)) / ln(phi) - 1/(2*phi)
 * Returns the raw n-coordinate as a double. */
double hdgl_n_coord(uint32_t p);

/* Returns 1 if frac(n(p)) < 0.5 (phi-lattice lower-half; 67% of known
 * Mersenne exponents satisfy this). */
int hdgl_phi_lower_half(uint32_t p);

#ifdef __cplusplus
}
#endif
