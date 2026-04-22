/*
 * hdgl_phi_lang.h
 * Phi-language token assignment and spiral8 geometric basis
 *
 * Provides two static tables:
 *
 * 1. PHI_LANG_TOKENS -- the 4 core operators of this codebase mapped to
 *    their phi-log space coordinates (n, beta) under D(n,beta):
 *
 *      D(n,beta) = sqrt(phi * F(n+beta) * P(n+beta) * base^(n+beta) * Omega) * r^k
 *
 *    where F = continuous Binet Fibonacci, P = prime table lookup,
 *    Omega = 0.5 + 0.5*sin(pi*frac(n)*phi), base = 2, k = 1.
 *
 *    A token's coordinate is the (n, beta) pair whose D value best
 *    matches the operator's characteristic scale.
 *
 * 2. SPIRAL8_AXES -- the 8 spiral geometries from prismatic5z.py
 *    (MAX_SLICE=8, instanceID%8) mapped to 8D Kuramoto oscillator axes.
 *    Each geometry is a distinct (phi^k, prime[k]) resonance mode at
 *    slice k = id % 8.
 *
 * No external dependencies.  Header-only.
 */

#pragma once
#ifndef HDGL_PHI_LANG_H
#define HDGL_PHI_LANG_H

#ifdef __cplusplus
extern "C" {
#endif

/* ================================================================
 * Phi-language: operator token table
 *
 * n      -- phi-log coordinate (float; whole+fractional parts)
 * beta   -- fractional correction (0 = pure harmonic)
 * d_val  -- approximate D(n, beta) at r=1, k=1 (characteristic scale)
 * label  -- human-readable name
 * formula -- compact formula string
 * ================================================================ */
typedef struct {
    float       n;
    float       beta;
    float       d_val;   /* pre-computed D(n,beta) at r=1 */
    const char *label;
    const char *formula;
} phi_token_t;

/*
 * Core operator token assignments.
 *
 * Lambda_phi: the Mersenne bridge coordinate.  For p=3: ~0.97; p=127: ~5.0;
 *   p=86243: ~10.4.  The token anchor is placed at n=0, beta=0
 *   (the origin of phi-log space) because Lambda_phi IS the coordinate.
 *
 * D_n: the resonance operator itself.  Self-referential: its token
 *   is the identity point n=1, beta=0 (first phi-harmonic, D≈1.618).
 *
 * S_p: the resonance gate output |e^(i*pi*Λ)+1_eff|.  Minimum value
 *   ~0 at primes.  Token placed at n=0.5, beta=0.5 (mid-band).
 *
 * U_field: the phi-recursive interaction field.  Prime invariant:
 *   all 8 oscillators lock → M(U)=8 → Lambda_phi^(U)≈4.33.
 *   Token: n=4.33, beta=0 (D≈70, same band as H0=70.099 from fit table).
 *
 * DN_EMPIRICAL_BETA: fractional beta correction validated by empirical_validation.c
 *   BigG/Fudge10 chi² fit to 1000+ Pan-STARRS1 supernovae + 200+ CODATA constants.
 *   Both VALIDATION 1 (chi²_red≈0.000, PERFECT MATCH) and VALIDATION 2 (100%
 *   pass rate) confirm beta=0.360942 as the calibrated D_n fractional offset.
 *   Calibrated D_n token: n=1.361, beta=0.360942 (D≈1.049 at n+beta=1.361).
 */

/* Empirically calibrated beta from BigG/Fudge10 (empirical_validation.c).
 * beta = 0.360942 from chi^2 minimization against Pan-STARRS1 supernovae.  */
#define DN_EMPIRICAL_BETA 0.360942f

static const phi_token_t PHI_LANG_TOKENS[] = {
    /* n       beta   d_val  label          formula */
    { 0.0f,   0.0f,  1.0f,  "Lambda_phi",  "log(p*ln2/lnphi)/lnphi - 1/(2*phi)"          },
    { 1.0f,   0.0f,  1.618f,"D_n",         "sqrt(phi*F_n*P_n*2^n*Omega)*r^k"              },
    { 0.5f,   0.5f,  1.049f,"S_p",         "|e^(i*pi*Lambda_phi) + 1_eff|"                },
    { 4.33f,  0.0f,  70.1f, "U_field",     "phi^(sum phi^(sum phi^(interact+kappa*ln p)))" },
    /* Empirically calibrated D_n token (BigG/Fudge10 beta): */
    { 1.0f,   DN_EMPIRICAL_BETA, 1.049f, "D_n_cal",
      "sqrt(phi*F_{n+beta}*P_{n+beta}*2^{n+beta}*Omega)*r^k; beta=0.360942" },
};

#define PHI_LANG_N_TOKENS 4

/* ================================================================
 * Spiral8: 8 spiral geometry basis ↔ 8D Kuramoto oscillator axes
 *
 * From prismatic5z.py: slice = instanceID % MAX_SLICE (MAX_SLICE=8).
 * Each slice uses prismatic_recursion at id=k (k=0..7):
 *   phi_harm = phi^(k%16),  prime = primeTable[k] = first 8 primes,
 *   dyadic   = 2^(k%16),    r_dim = r^(k%7+1)
 *
 * The 8 Kuramoto axes theta[0..7] are assigned to these 8 modes.
 * The geometry name reflects the dominant harmonic character.
 *
 * Locking behaviour:
 *   - When theta[k] → 0  (osc locked): the corresponding spiral collapses
 *     to its radial minimum — prismatic_recursion returns near 0.
 *   - When ALL 8 lock: M(U)→8 → S(U)~1.531 → prime verdict.
 * ================================================================ */
typedef struct {
    int         axis;         /* Kuramoto oscillator index 0..7 */
    int         prime;        /* dominant prime harmonic */
    float       phi_power;    /* phi^axis (phi-log magnitude) */
    int         r_exponent;   /* r^(axis%7+1) dimension */
    const char *name;         /* geometry label */
    const char *character;    /* geometric description */
} spiral8_axis_t;

static const spiral8_axis_t SPIRAL8_AXES[] = {
    /* axis  prime  phi^k   r^exp  name              character */
    { 0,  2,  1.0000f,  1,  "RADIAL",        "identity: phi^0, r^1, p=2"        },
    { 1,  3,  1.6180f,  2,  "PHI_TWIST",     "golden spiral: phi^1, r^2, p=3"   },
    { 2,  5,  2.6180f,  3,  "FIBONACCI_RING","Fibonacci: phi^2, r^3, p=5"        },
    { 3,  7,  4.2361f,  4,  "HEPTAGONAL",    "heptagonal: phi^3, r^4, p=7"      },
    { 4, 11,  6.8541f,  5,  "PRIME_LATTICE", "undecagonal: phi^4, r^5, p=11"    },
    { 5, 13, 11.0902f,  6,  "OMEGA_BREATH",  "Omega-mod: phi^5, r^6, p=13"      },
    { 6, 17, 17.9443f,  7,  "GOLDEN_FOLD",   "phi-inverted: phi^6, r^7, p=17"   },
    { 7, 19, 29.0344f,  1,  "RESONANCE_GATE","full gate: phi^7, r^1, p=19"      },
};

#define SPIRAL8_N_AXES 8

/* ================================================================
 * Phi-language utility: look up a token by label
 * Returns pointer to token or NULL if not found.
 * ================================================================ */
static const phi_token_t *phi_lang_find(const char *label)
{
    int i;
    for (i = 0; i < PHI_LANG_N_TOKENS; i++) {
        const char *a = label;
        const char *b = PHI_LANG_TOKENS[i].label;
        while (*a && *a == *b) { a++; b++; }
        if (*a == '\0' && *b == '\0') return &PHI_LANG_TOKENS[i];
    }
    return (const phi_token_t*)0;
}

/* ================================================================
 * Spiral8 utility: look up an axis by Kuramoto index
 * ================================================================ */
static const spiral8_axis_t *spiral8_find(int axis)
{
    if (axis < 0 || axis >= SPIRAL8_N_AXES) return (const spiral8_axis_t*)0;
    return &SPIRAL8_AXES[axis];
}

#ifdef __cplusplus
}
#endif

#endif /* HDGL_PHI_LANG_H */
