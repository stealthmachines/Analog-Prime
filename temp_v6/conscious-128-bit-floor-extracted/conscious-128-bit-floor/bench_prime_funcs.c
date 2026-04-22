/* bench_prime_funcs.c — Benchmark for all quantum-prime scalar functions
 *
 * Functions benchmarked:
 *   Math primitives (1M+ iterations):
 *     1. fibonacci_real(n)            — Binet's formula, real-index
 *     2. prime_product_index(n,beta)  — modular 50-entry prime table
 *     3. D_n(n,beta,r,k,Omega,base)   — universal resonance amplitude
 *     4. n_of_2p(p)                   — phi-lattice coord for Mersenne p
 *     5. phi_filter(p)                — frac(n_of_2p(p)) < 0.5 predicate
 *     6. gram_zero_k(k)               — Lambert W Gram-point approx (6-iter Newton)
 *     7. zeta_zero_cpu(k)             — dispatch: hardcoded (<80) or gram_zero_k
 *     8. psi_score_cpu(x, B)          — scalar explicit formula, B=500 zeros
 *
 *   Pipeline throughput:
 *     9. segmented_sieve([1e7,1e7+1e5])  — enumerate primes in 100K window
 *    10. phi_filter_batch(51 Mersenne)   — frac test on all known Mersenne exps
 *    11. D_n_batch(candidates, n=1000)   — rank 1000 exponents by Dn score
 *    12. miller_rabin_64(p, 12 witnesses)— deterministic primality test
 *
 *   Full pipeline:
 *    13. pipeline_full([1e7,1e7+1e5])    — sieve→phi-filter→Dn-rank
 *
 * Build modes (all produce standalone exe, no external libraries):
 *   clang:  clang -O3 -march=native -x c -D_CRT_SECURE_NO_WARNINGS bench_prime_funcs.c -o bench_prime_funcs.exe
 *   gcc:    gcc   -O3 -march=native -D_CRT_SECURE_NO_WARNINGS -D_USE_MATH_DEFINES bench_prime_funcs.c -o bench_prime_funcs_gcc.exe -lm
 *   msvc:   cl /O2 /TC /D_CRT_SECURE_NO_WARNINGS /D_USE_MATH_DEFINES bench_prime_funcs.c /link
 *
 * Licensed per https://zchg.org/t/legal-notice-copyright-applicable-ip-and-licensing-read-me/440
 */

#ifndef _USE_MATH_DEFINES
#  define _USE_MATH_DEFINES
#endif
#ifndef _CRT_SECURE_NO_WARNINGS
#  define _CRT_SECURE_NO_WARNINGS
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

/* ── Windows high-resolution timer ──────────────────────────────────────── */
#ifdef _WIN32
#  include <windows.h>
static double bench_now_s(void) {
    LARGE_INTEGER freq, cnt;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&cnt);
    return (double)cnt.QuadPart / (double)freq.QuadPart;
}
#else
#  include <time.h>
static double bench_now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}
#endif

/* ═══════════════════════════════════════════════════════════════════════════
 * SECTION A — Constants
 * ═══════════════════════════════════════════════════════════════════════════ */
#define PHI      1.6180339887498948482
#define LN_PHI   0.4812118250596034748
#define SQRT5    2.2360679774997896
#define PI       3.14159265358979323846
#define INV_E    0.36787944117144232159
#define TWO_PI   6.28318530717958647693
#define M_LN2_V  0.6931471805599453094

/* First 50 primes for D_n prime_product_index */
static const int PRIMES50[50] = {
     2,  3,  5,  7, 11, 13, 17, 19, 23, 29,
    31, 37, 41, 43, 47, 53, 59, 61, 67, 71,
    73, 79, 83, 89, 97,101,103,107,109,113,
   127,131,137,139,149,151,157,163,167,173,
   179,181,191,193,197,199,211,223,227,229
};

/* All 51 known Mersenne prime exponents (M1..M51, as of 2026-04) */
static const uint64_t MERSENNE_EXP[51] = {
    2ULL, 3ULL, 5ULL, 7ULL, 13ULL, 17ULL, 19ULL, 31ULL, 61ULL, 89ULL,
    107ULL, 127ULL, 521ULL, 607ULL, 1279ULL, 2203ULL, 2281ULL, 3217ULL,
    4253ULL, 4423ULL, 9689ULL, 9941ULL, 11213ULL, 19937ULL, 21701ULL,
    23209ULL, 44497ULL, 86243ULL, 110503ULL, 132049ULL, 216091ULL,
    756839ULL, 859433ULL, 1257787ULL, 1398269ULL, 2976221ULL, 3021377ULL,
    6972593ULL, 13466917ULL, 20996011ULL, 24036583ULL, 25964951ULL,
    30402457ULL, 32582657ULL, 37156667ULL, 42643801ULL, 43112609ULL,
    57885161ULL, 74207281ULL, 77232917ULL, 136279841ULL
};
#define N_MERSENNE 51

/* First 80 exact Riemann zeta zeros (imaginary parts) */
static const double ZETA_ZEROS_80[80] = {
    14.134725141734693,  21.022039638771555,  25.010857580145688,
    30.424876125859513,  32.935061587739189,  37.586178158825671,
    40.918719012147495,  43.327073280914999,  48.005150881167159,
    49.773832477672302,  52.970321477714460,  56.446247697063246,
    59.347044002602352,  60.831778524609809,  65.112544048081560,
    67.079810529494173,  69.546401711173979,  72.067157674481907,
    75.704690699083933,  77.144840068874805,  79.337375020249367,
    82.910380854160462,  84.735492981074628,  87.425274613125229,
    88.809111207634465,  92.491899270593585,  94.651344040519681,
    95.870634228245332,  98.831194218193159, 101.317851006956152,
   103.725538040478419, 105.446623052947866, 107.168611184276793,
   111.029535543169970, 111.874659177229233, 114.320220915452460,
   116.226680321519019, 118.790782866217474, 121.370125002980428,
   122.946829294236573, 124.256818554513985, 127.516683879564406,
   129.578704200821853, 131.087688531430975, 133.497737202990660,
   134.756510050820649, 138.116042054533808, 139.736208952121808,
   141.123707404415728, 143.111845808910186, 146.000982487395827,
   147.422765343849989, 150.053520421293562, 150.925257612895526,
   153.024693791188948, 156.112909294982618, 157.597591818986345,
   158.849988365204885, 161.188964138954152, 163.030709687408168,
   165.537069188392498, 167.184439971994828, 169.094515416791259,
   169.911976498590630, 173.411536520135680, 174.754191523438771,
   176.441434188575954, 178.377407776468757, 179.916484018400656,
   182.207078484665730, 184.874467848130730, 185.598783678433914,
   187.228922291882030, 189.415759393773366, 192.026656325978780,
   193.079726604550355, 195.265396680495222, 196.876481841084053,
   198.015309585175508, 201.264751178782752
};

/* ═══════════════════════════════════════════════════════════════════════════
 * SECTION B — Quantum-prime math functions (exact copies from repo files)
 * ═══════════════════════════════════════════════════════════════════════════ */

/* ── 1. fibonacci_real(n) — from EMPIRICAL_VALIDATION_ASCII.c, prime_pipeline.c */
static double fibonacci_real(double n) {
    double term1 = pow(PHI, n) / SQRT5;
    double term2 = pow(1.0 / PHI, n) * cos(PI * n);
    return term1 - term2;
}

/* ── 2. prime_product_index(n, beta) — from prime_pipeline.c */
static double prime_product_index(double n, double beta) {
    int idx = ((int)floor(n + beta) + 50) % 50;
    return (double)PRIMES50[idx];
}

/* ── 3. D_n — from prime_pipeline.c / EMPIRICAL_VALIDATION_ASCII.c */
static double D_n(double n, double beta, double r, double k,
                  double Omega, double base) {
    double n_beta = n + beta;
    double Fn     = fibonacci_real(n_beta);
    double Pn     = prime_product_index(n, beta);
    double dyadic = pow(base, n_beta);
    double val    = PHI * fmax(Fn, 1e-15) * dyadic * Pn * Omega;
    return sqrt(fmax(val, 1e-15)) * pow(fabs(r), k);
}

/* ── 4. n_of_2p(p) — phi-lattice coordinate for Mersenne exponent p */
/*    From prime_pipeline.c.  n(2^p) = log(p·ln2 / ln(phi)) / ln(phi) - 1/(2phi) */
static double n_of_2p(uint64_t p) {
    double lx  = (double)p * M_LN2_V;
    double llx = log(lx / LN_PHI);
    if (llx <= 0.0) return -1.0;
    return llx / LN_PHI - 0.5 / PHI;
}

/* ── 5. phi_filter(p) — frac(n_of_2p(p)) < 0.5 */
static int phi_filter(uint64_t p) {
    double n = n_of_2p(p);
    if (n < 0.0) return 0;
    double frac = n - floor(n);
    return frac < 0.5;
}

/* ── 6. gram_zero_k(k) — Lambert W / Newton, from psi_scanner_cuda_v2.cu */
static double gram_zero_k(int k) {
    double z = (double)k * INV_E;
    if (z <= 0.0) return TWO_PI * (double)k; /* k=0 edge */
    double w = log(z + 1.0);
    double ew;
    ew=exp(w); w-=(w*ew-z)/(ew*(w+1.0));
    ew=exp(w); w-=(w*ew-z)/(ew*(w+1.0));
    ew=exp(w); w-=(w*ew-z)/(ew*(w+1.0));
    ew=exp(w); w-=(w*ew-z)/(ew*(w+1.0));
    ew=exp(w); w-=(w*ew-z)/(ew*(w+1.0));
    ew=exp(w); w-=(w*ew-z)/(ew*(w+1.0));
    return TWO_PI * (double)k / w;
}

/* ── 7. zeta_zero_cpu(k) — dispatch: hardcoded table (<80) or gram */
static double zeta_zero_cpu(int k) {
    return (k >= 0 && k < 80) ? ZETA_ZEROS_80[k] : gram_zero_k(k);
}

/* ── 8. psi_score_cpu(x, B) — scalar explicit formula, B zeros
 *    Computes pi_approx(x) - pi_approx(x-1) via prime counting function spike.
 *    From pass1_kernel in psi_scanner_cuda_v2.cu (CPU version). */
static double psi_score_cpu(double x, int B) {
    double lx  = log(x);
    double lxm = log(x - 1.0);
    double mx  = exp(0.5 * lx);
    double mm  = exp(0.5 * lxm);
    double px  = x, pm = x - 1.0;
    for (int k = 0; k < B; k++) {
        double t = zeta_zero_cpu(k);
        double d = 0.25 + t * t;
        px -= 2.0 * mx * (0.5 * cos(t * lx)  + t * sin(t * lx))  / d;
        pm -= 2.0 * mm * (0.5 * cos(t * lxm) + t * sin(t * lxm)) / d;
    }
    return px - pm;
}

/* ── 9. Miller-Rabin 64-bit — from prime_pipeline.c */
/* mulmod64: a*b mod m, correct for all 64-bit values.
 *  GCC (not Clang): __uint128_t (one hardware multiply).
 *  Clang on Windows uses lld and lacks __umodti3 (compiler-rt builtin),
 *  so fall through to portable binary shift-subtract for Clang and MSVC. */
#if defined(__GNUC__) && !defined(__clang__)
static uint64_t mulmod64(uint64_t a, uint64_t b, uint64_t m) {
    return (uint64_t)((__uint128_t)a * b % m);
}
#else
static uint64_t mulmod64(uint64_t a, uint64_t b, uint64_t m) {
    uint64_t r = 0;
    a %= m;
    while (b > 0) {
        if (b & 1) { r += a; if (r >= m) r -= m; }
        a = (a >= m - a) ? (a + a - m) : (a + a);
        b >>= 1;
    }
    return r;
}
#endif

static uint64_t powmod64(uint64_t base, uint64_t exp, uint64_t mod) {
    uint64_t result = 1;
    base %= mod;
    while (exp > 0) {
        if (exp & 1) result = mulmod64(result, base, mod);
        base = mulmod64(base, base, mod);
        exp >>= 1;
    }
    return result;
}

static int miller_rabin_witness(uint64_t n, uint64_t a) {
    if (n % a == 0) return (int)(n == a);
    uint64_t d = n - 1;
    int r = 0;
    while (!(d & 1)) { d >>= 1; r++; }
    uint64_t x = powmod64(a, d, n);
    if (x == 1 || x == n - 1) return 1;
    for (int i = 0; i < r - 1; i++) {
        x = mulmod64(x, x, n);
        if (x == n - 1) return 1;
    }
    return 0;
}

/* Deterministic for n < 3,317,044,064,679,887,385,961,981 */
static int is_prime_64(uint64_t n) {
    if (n < 2) return 0;
    if (n < 4) return 1;
    if (!(n & 1) || n % 3 == 0) return 0;
    static const uint64_t witnesses[] = {2,3,5,7,11,13,17,19,23,29,31,37};
    for (int i = 0; i < 12; i++)
        if (!miller_rabin_witness(n, witnesses[i])) return 0;
    return 1;
}

/* ── 10. Segmented sieve — primes in [lo, hi], returns malloc'd array */
static uint64_t *sieve_range(uint64_t lo, uint64_t hi, int *count) {
    *count = 0;
    if (hi < lo) return NULL;
    uint64_t cap   = 8 * ((hi - lo) / 50 + 64);
    uint64_t *out  = (uint64_t *)malloc(cap * sizeof(uint64_t));
    if (!out) return NULL;
    uint64_t span  = hi - lo + 1;
    uint8_t *sieve = (uint8_t *)calloc((size_t)span, 1);
    if (!sieve) { free(out); return NULL; }
    /* mark even composites */
    uint64_t s2 = (lo % 2 == 0) ? lo : lo + 1;
    for (uint64_t n = s2; n <= hi; n += 2)
        if (n > 2) sieve[(size_t)(n - lo)] = 1;
    /* sieve with small primes */
    uint64_t sq = (uint64_t)ceil(sqrt((double)hi)) + 1;
    for (uint64_t p = 3; p <= sq; p += 2) {
        int p_prime = 1;
        for (uint64_t d = 3; d * d <= p; d += 2)
            if (p % d == 0) { p_prime = 0; break; }
        if (!p_prime) continue;
        uint64_t first = ((lo + p - 1) / p) * p;
        if (first < p * p) first = p * p;
        if (first > hi) continue;
        if (first == p) first = p * p;
        for (uint64_t n = first; n <= hi; n += p)
            sieve[(size_t)(n - lo)] = 1;
    }
    for (uint64_t n = lo; n <= hi; n++) {
        if (sieve[(size_t)(n - lo)]) continue;
        if (n < 2) continue;
        if ((size_t)*count >= (size_t)(cap - 1)) {
            cap *= 2;
            uint64_t *tmp = (uint64_t *)realloc(out, cap * sizeof(uint64_t));
            if (!tmp) break;
            out = tmp;
        }
        out[(*count)++] = n;
    }
    free(sieve);
    return out;
}

/* ── 11. D_n batch rank — score and sort n candidates by Dn descending */
typedef struct { uint64_t p; double dn; double n_coord; double frac_n; } Candidate;

static int cand_cmp(const void *a, const void *b) {
    double da = ((const Candidate*)a)->dn;
    double db = ((const Candidate*)b)->dn;
    return (da < db) ? 1 : (da > db) ? -1 : 0;
}

static void dn_rank_batch(const uint64_t *primes, int n_primes, Candidate *out) {
    for (int i = 0; i < n_primes; i++) {
        double nc = n_of_2p(primes[i]);
        double frac = nc - floor(nc);
        /* Mersenne-specific Dn: base=2, beta=0, k=(n_int+1)/8, Omega=1, r=frac */
        int n_int = (int)floor(nc);
        double k_exp = (double)(n_int + 1) / 8.0;
        double dn = D_n(nc, 0.0, frac > 0.0 ? frac : 1e-6, k_exp, 1.0, 2.0);
        out[i].p       = primes[i];
        out[i].dn      = dn;
        out[i].n_coord = nc;
        out[i].frac_n  = frac;
    }
    qsort(out, n_primes, sizeof(Candidate), cand_cmp);
}

/* ═══════════════════════════════════════════════════════════════════════════
 * SECTION C — Benchmark infrastructure
 * ═══════════════════════════════════════════════════════════════════════════ */

#define BENCH_REPS 3

typedef struct {
    const char *name;
    double mean_us;
    double min_us;
    double max_us;
    int    iters;
    double throughput; /* ops/s */
} BenchResult;

/* Prevent dead-code elimination via volatile accumulator */
static volatile double g_sink = 0.0;
static volatile uint64_t g_sink_u = 0;

static FILE *g_tsv = NULL;

static void write_result(const BenchResult *r) {
    printf("  %-48s  mean=%8.2f µs  min=%7.2f  [%d iters]  %9.0f ops/s\n",
           r->name, r->mean_us, r->min_us, r->iters, r->throughput);
    if (g_tsv)
        fprintf(g_tsv, "%s\t%.4f\t%.4f\t%.4f\t%d\t%.0f\n",
                r->name, r->mean_us, r->min_us, r->max_us,
                r->iters, r->throughput);
}

/* ═══════════════════════════════════════════════════════════════════════════
 * SECTION D — Individual benchmarks
 * ═══════════════════════════════════════════════════════════════════════════ */

static void bench_fibonacci_real(void) {
    const int N = 500000;
    double acc = 0.0;
    double t0 = bench_now_s();
    for (int i = 0; i < N; i++)
        acc += fibonacci_real((double)(i % 40));
    double t1 = bench_now_s();
    g_sink = acc;

    BenchResult r;
    r.name       = "fibonacci_real(n) — Binet real-index";
    r.iters      = N;
    r.mean_us    = (t1 - t0) / N * 1e6;
    r.min_us     = r.mean_us;
    r.max_us     = r.mean_us;
    r.throughput = N / (t1 - t0);
    write_result(&r);
}

static void bench_prime_product_index(void) {
    const int N = 2000000;
    double acc = 0.0;
    double t0 = bench_now_s();
    for (int i = 0; i < N; i++)
        acc += prime_product_index((double)(i % 50), 0.0);
    double t1 = bench_now_s();
    g_sink = acc;

    BenchResult r;
    r.name       = "prime_product_index(n,beta) — table lookup";
    r.iters      = N;
    r.mean_us    = (t1 - t0) / N * 1e6;
    r.min_us     = r.mean_us;
    r.max_us     = r.mean_us;
    r.throughput = N / (t1 - t0);
    write_result(&r);
}

static void bench_D_n(void) {
    const int N = 200000;
    double acc = 0.0;
    double t0 = bench_now_s();
    for (int i = 0; i < N; i++) {
        double n  = 7.0 + (i % 40) * 0.25;
        double r  = 0.3 + (i % 7) * 0.1;
        double k  = (double)((i % 8) + 1) / 8.0;
        acc += D_n(n, 0.0, r, k, 1.0, 2.0);
    }
    double t1 = bench_now_s();
    g_sink = acc;

    BenchResult r;
    r.name       = "D_n(n,b,r,k,Omega,base) — resonance amplitude";
    r.iters      = N;
    r.mean_us    = (t1 - t0) / N * 1e6;
    r.min_us     = r.mean_us;
    r.max_us     = r.mean_us;
    r.throughput = N / (t1 - t0);
    write_result(&r);
}

static void bench_n_of_2p(void) {
    const int N = 1000000;
    double acc = 0.0;
    double t0 = bench_now_s();
    for (int i = 0; i < N; i++) {
        /* Cycle through all 51 Mersenne exponents */
        acc += n_of_2p(MERSENNE_EXP[i % N_MERSENNE]);
    }
    double t1 = bench_now_s();
    g_sink = acc;

    BenchResult r;
    r.name       = "n_of_2p(p) — phi-lattice coord for Mersenne p";
    r.iters      = N;
    r.mean_us    = (t1 - t0) / N * 1e6;
    r.min_us     = r.mean_us;
    r.max_us     = r.mean_us;
    r.throughput = N / (t1 - t0);
    write_result(&r);
}

static void bench_phi_filter(void) {
    const int N = 2000000;
    int acc = 0;
    double t0 = bench_now_s();
    for (int i = 0; i < N; i++)
        acc += phi_filter(MERSENNE_EXP[i % N_MERSENNE]);
    double t1 = bench_now_s();
    g_sink = (double)acc;

    BenchResult r;
    r.name       = "phi_filter(p) — frac(n_of_2p) < 0.5 predicate";
    r.iters      = N;
    r.mean_us    = (t1 - t0) / N * 1e6;
    r.min_us     = r.mean_us;
    r.max_us     = r.mean_us;
    r.throughput = N / (t1 - t0);
    write_result(&r);
}

static void bench_gram_zero_k(void) {
    const int N = 500000;
    double acc = 0.0;
    double t0 = bench_now_s();
    for (int i = 0; i < N; i++)
        acc += gram_zero_k((i % 10000) + 80); /* k>=80: no table, pure Newton */
    double t1 = bench_now_s();
    g_sink = acc;

    BenchResult r;
    r.name       = "gram_zero_k(k) — Lambert W, 6-iter Newton";
    r.iters      = N;
    r.mean_us    = (t1 - t0) / N * 1e6;
    r.min_us     = r.mean_us;
    r.max_us     = r.mean_us;
    r.throughput = N / (t1 - t0);
    write_result(&r);
}

static void bench_zeta_zero_cpu(void) {
    const int N = 2000000;
    double acc = 0.0;
    /* Half from table (<80), half from gram (>=80) */
    double t0 = bench_now_s();
    for (int i = 0; i < N; i++)
        acc += zeta_zero_cpu(i % 160);
    double t1 = bench_now_s();
    g_sink = acc;

    BenchResult r;
    r.name       = "zeta_zero_cpu(k) — table dispatch (<80) or gram";
    r.iters      = N;
    r.mean_us    = (t1 - t0) / N * 1e6;
    r.min_us     = r.mean_us;
    r.max_us     = r.mean_us;
    r.throughput = N / (t1 - t0);
    write_result(&r);
}

static void bench_psi_score_B500(void) {
    /* B=500 zeros: mimics psi_scanner pass1_kernel, per-candidate cost */
    const int N = 2000;
    double acc = 0.0;
    /* Use candidate range around M_127 = 2^127-1 */
    double base_x = 1.70141183e38; /* rough magnitude, use log-safe value */
    /* We'll use smaller but realistic x to keep log() well-behaved */
    double xs[8] = {1e12, 1e15, 1e18, 1e21, 1e24, 1e27, 1e30, 1e33};
    double t0 = bench_now_s();
    for (int i = 0; i < N; i++)
        acc += psi_score_cpu(xs[i % 8], 500);
    double t1 = bench_now_s();
    g_sink = acc;

    BenchResult r;
    r.name       = "psi_score_cpu(x, B=500) — scalar explicit formula";
    r.iters      = N;
    r.mean_us    = (t1 - t0) / N * 1e6;
    r.min_us     = r.mean_us;
    r.max_us     = r.mean_us;
    r.throughput = N / (t1 - t0);
    write_result(&r);
}

static void bench_miller_rabin(void) {
    /* Deterministic 12-witness test; use known primes & composites */
    const int N = 200000;
    /* Mix of known Mersenne exponents (prime) and nearby composites */
    static const uint64_t test_ns[8] = {
        127ULL, 521ULL, 1279ULL, 2203ULL,   /* prime exponents */
        128ULL, 522ULL, 1280ULL, 2204ULL    /* composite neighbours */
    };
    int acc = 0;
    double t0 = bench_now_s();
    for (int i = 0; i < N; i++)
        acc += is_prime_64(test_ns[i % 8]);
    double t1 = bench_now_s();
    g_sink_u = (uint64_t)acc;

    BenchResult r;
    r.name       = "is_prime_64(n) — Miller-Rabin, 12 witnesses";
    r.iters      = N;
    r.mean_us    = (t1 - t0) / N * 1e6;
    r.min_us     = r.mean_us;
    r.max_us     = r.mean_us;
    r.throughput = N / (t1 - t0);
    write_result(&r);
}

static void bench_segmented_sieve(void) {
    /* Sieve [1e7, 1e7+100000] — a typical prime_pipeline window */
    uint64_t lo = 10000000ULL, hi = 10100000ULL;
    double t0 = bench_now_s();
    int cnt = 0;
    uint64_t *primes = sieve_range(lo, hi, &cnt);
    double t1 = bench_now_s();
    free(primes);

    printf("    [sieve found %d primes in [%llu, %llu]]\n", cnt, (unsigned long long)lo, (unsigned long long)hi);

    BenchResult r;
    r.name       = "segmented_sieve([1e7, 1e7+1e5]) — prime enumeration";
    r.iters      = 1;
    r.mean_us    = (t1 - t0) * 1e6;
    r.min_us     = r.mean_us;
    r.max_us     = r.mean_us;
    r.throughput = (double)(hi - lo + 1) / (t1 - t0);
    write_result(&r);
}

static void bench_phi_filter_batch_mersenne(void) {
    /* Run phi_filter on all 51 known Mersenne exponents */
    int pass = 0;
    double t0 = bench_now_s();
    for (int rep = 0; rep < 10000; rep++) {
        int p2 = 0;
        for (int i = 0; i < N_MERSENNE; i++)
            p2 += phi_filter(MERSENNE_EXP[i]);
        pass = p2;
    }
    double t1 = bench_now_s();
    int total_calls = N_MERSENNE * 10000;
    g_sink = (double)pass;

    /* Report observed pass rate */
    int pass_single = 0;
    for (int i = 0; i < N_MERSENNE; i++)
        pass_single += phi_filter(MERSENNE_EXP[i]);
    printf("    [phi_filter: %d/%d Mersenne exps pass frac<0.5 = %.1f%%]\n",
           pass_single, N_MERSENNE, 100.0 * pass_single / N_MERSENNE);

    BenchResult r;
    r.name       = "phi_filter_batch(51 Mersenne exps × 10K reps)";
    r.iters      = total_calls;
    r.mean_us    = (t1 - t0) / total_calls * 1e6;
    r.min_us     = r.mean_us;
    r.max_us     = r.mean_us;
    r.throughput = total_calls / (t1 - t0);
    write_result(&r);
}

static void bench_dn_rank_batch(void) {
    /* Rank 1000 prime exponents from the sieve by Dn score */
    uint64_t lo = 10000000ULL, hi = 10100000ULL;
    int cnt = 0;
    uint64_t *primes = sieve_range(lo, hi, &cnt);
    if (!primes || cnt == 0) { printf("  [sieve empty]\n"); return; }

    /* Use up to 1000 */
    int n_use = cnt < 1000 ? cnt : 1000;
    Candidate *cands = (Candidate *)malloc(n_use * sizeof(Candidate));
    if (!cands) { free(primes); return; }

    double t0 = bench_now_s();
    dn_rank_batch(primes, n_use, cands);
    double t1 = bench_now_s();

    /* Print top 5 */
    printf("    [top 5 Dn-ranked candidates from [%llu, %llu]]:\n",
           (unsigned long long)lo, (unsigned long long)hi);
    for (int i = 0; i < 5 && i < n_use; i++)
        printf("      #%d  p=%-10llu  n=%.5f  frac=%.4f  Dn=%.4e\n",
               i+1, (unsigned long long)cands[i].p,
               cands[i].n_coord, cands[i].frac_n, cands[i].dn);

    BenchResult r;
    r.name       = "D_n_rank_batch(1000 primes) — score + sort";
    r.iters      = n_use;
    r.mean_us    = (t1 - t0) / n_use * 1e6;
    r.min_us     = r.mean_us;
    r.max_us     = r.mean_us;
    r.throughput = n_use / (t1 - t0);
    write_result(&r);

    free(cands);
    free(primes);
}

static void bench_full_pipeline(void) {
    /* sieve → phi-filter → Dn rank, as prime_pipeline.c does */
    uint64_t lo = 10000000ULL, hi = 10200000ULL;
    double t0 = bench_now_s();

    int n_sieve = 0;
    uint64_t *primes = sieve_range(lo, hi, &n_sieve);
    int n_pass = 0;
    uint64_t *filtered = (uint64_t*)malloc(n_sieve * sizeof(uint64_t));
    if (filtered && primes) {
        for (int i = 0; i < n_sieve; i++)
            if (phi_filter(primes[i])) filtered[n_pass++] = primes[i];
    }
    Candidate *cands = (Candidate*)malloc((n_pass > 0 ? n_pass : 1) * sizeof(Candidate));
    if (cands && n_pass > 0)
        dn_rank_batch(filtered, n_pass, cands);

    double t1 = bench_now_s();

    printf("    [pipeline: sieve=%d  phi-pass=%d (%.0f%%)  ranked=%d]\n",
           n_sieve, n_pass, n_sieve > 0 ? 100.0*n_pass/n_sieve : 0.0, n_pass);
    if (cands && n_pass > 0) {
        printf("    [top 3 pipeline candidates]:\n");
        for (int i = 0; i < 3 && i < n_pass; i++)
            printf("      #%d  p=%-10llu  frac=%.4f  Dn=%.4e\n",
                   i+1, (unsigned long long)cands[i].p,
                   cands[i].frac_n, cands[i].dn);
    }

    BenchResult r;
    r.name       = "full_pipeline([1e7,1e7+2e5]): sieve+filter+rank";
    r.iters      = 1;
    r.mean_us    = (t1 - t0) * 1e6;
    r.min_us     = r.mean_us;
    r.max_us     = r.mean_us;
    r.throughput = (double)(hi - lo) / (t1 - t0); /* candidates/s */
    write_result(&r);

    free(cands);
    free(filtered);
    free(primes);
}

/* ═══════════════════════════════════════════════════════════════════════════
 * SECTION E — main
 * ═══════════════════════════════════════════════════════════════════════════ */

int main(void) {
    /* Detect compiler */
#if defined(_MSC_VER)
    const char *compiler = "MSVC";
    int   cver   = _MSC_VER;
#elif defined(__clang__)
    const char *compiler = "Clang";
    int   cver   = __clang_major__ * 100 + __clang_minor__;
#elif defined(__GNUC__)
    const char *compiler = "GCC";
    int   cver   = __GNUC__ * 100 + __GNUC_MINOR__;
#else
    const char *compiler = "Unknown";
    int   cver   = 0;
#endif

    printf("\xC9");
    for (int i = 0; i < 66; i++) printf("\xCD");
    printf("\xBB\n");
    printf("\xBA  bench_prime_funcs \xC4 Quantum-Prime Math Function Benchmark  \xBA\n");
    printf("\xC8");
    for (int i = 0; i < 66; i++) printf("\xCD");
    printf("\xBC\n");
    printf("  Compiler: %s %d\n", compiler, cver);
    printf("  Functions: fibonacci_real, D_n, n_of_2p, phi_filter, gram_zero,\n");
    printf("             zeta_zero, psi_score, miller_rabin, sieve, pipeline\n\n");

    /* Open TSV output */
    g_tsv = fopen("bench_prime_results.tsv", "w");
    if (g_tsv)
        fprintf(g_tsv, "function\tmean_us\tmin_us\tmax_us\titers\tops_per_s\n");

    /* ── Section 1: Scalar math primitives ─────────────────────────────── */
    printf("\xC4\xC4 Scalar Math Primitives ");
    for (int i = 0; i < 46; i++) printf("\xC4");
    printf("\n");
    bench_fibonacci_real();
    bench_prime_product_index();
    bench_D_n();
    bench_n_of_2p();
    bench_phi_filter();
    bench_gram_zero_k();
    bench_zeta_zero_cpu();

    /* ── Section 2: Explicit formula (ψ-score) ──────────────────────────── */
    printf("\n\xC4\xC4 Psi-Score (Explicit Formula, CPU) ");
    for (int i = 0; i < 33; i++) printf("\xC4");
    printf("\n");
    bench_psi_score_B500();

    /* ── Section 3: Number-theory functions ─────────────────────────────── */
    printf("\n\xC4\xC4 Number Theory ");
    for (int i = 0; i < 53; i++) printf("\xC4");
    printf("\n");
    bench_miller_rabin();
    bench_segmented_sieve();

    /* ── Section 4: Pipeline ────────────────────────────────────────────── */
    printf("\n\xC4\xC4 Pipeline Benchmarks (φ-filter + D_n rank) ");
    for (int i = 0; i < 24; i++) printf("\xC4");
    printf("\n");
    bench_phi_filter_batch_mersenne();
    bench_dn_rank_batch();
    bench_full_pipeline();

    /* ── Phi-filter validation: empirical pass rate on known Mersenne exps */
    printf("\n\xC4\xC4 Phi-Filter Empirical Validation ");
    for (int i = 0; i < 35; i++) printf("\xC4");
    printf("\n");
    printf("  Known Mersenne exponents and their phi-lattice coordinates:\n");
    printf("  %10s  %10s  %8s  %6s  %12s\n", "p", "n_of_2p", "frac(n)", "pass?", "Dn");
    int n_pass_total = 0;
    for (int i = 0; i < N_MERSENNE; i++) {
        uint64_t p  = MERSENNE_EXP[i];
        double nc   = n_of_2p(p);
        double frac = nc - floor(nc);
        int    pass = frac < 0.5;
        int    n_int = (int)floor(nc);
        double k_exp = (double)(n_int + 1) / 8.0;
        double dn   = D_n(nc, 0.0, frac > 0.0 ? frac : 1e-6, k_exp, 1.0, 2.0);
        n_pass_total += pass;
        printf("  %10llu  %10.5f  %8.5f  %6s  %12.4e\n",
               (unsigned long long)p, nc, frac, pass ? "YES" : "no", dn);
    }
    printf("  Pass rate: %d/%d = %.1f%%  (uniform null = 50.0%%)\n\n",
           n_pass_total, N_MERSENNE, 100.0 * n_pass_total / N_MERSENNE);

    /* ── GPU note ──────────────────────────────────────────────────────── */
    printf("\xC4\xC4 GPU psi_scanner Note ");
    for (int i = 0; i < 47; i++) printf("\xC4");
    printf("\n");
    printf("  psi_scanner_cuda_v2.cu CUDA kernels (pass1/pass2/pass3) require:\n");
    printf("    nvcc -O3 -arch=sm_75 psi_scanner_cuda_v2.cu -o psi_scanner_v2.exe\n");
    printf("  GPU B=500 throughput on RTX 2060 SM7.5: ~1M candidates/s (estimated).\n");
    printf("  On Windows, cuQuantum GPU mode requires Docker or WSL2:\n");
    printf("    docker run --gpus all --rm -v .:/work \\\n");
    printf("      nvcr.io/nvidia/cuquantum-appliance:26.03 \\\n");
    printf("      bash -c \"nvcc -arch=sm_75 /work/psi_scanner_cuda_v2.cu -o /work/psi_scanner_v2\"\n\n");

    if (g_tsv) {
        fclose(g_tsv);
        printf("  Results saved to bench_prime_results.tsv\n\n");
    }
    printf("  Done.\n");
    return 0;
}
