// ============================================================================
// HDGL Analog Mainnet v3.1 — Closed-Form Accuracy Hemisphere
// ============================================================================
//
// HEMISPHERE ROLE  : ACCURACY / VERIFIER
// RECURSIVE FORM   : CLOSED — GRA r_harmonic fixed at init, NOT plastic
// INTEGRATION      : RK4, global dt computed once per lattice step
// PRECISION        : Full Slot4096 APA mantissa; in-place write-back (no malloc)
// LL PROBE         : 64-bit iterative squaring residue (anchored reward signal)
// SPECTRAL KERNEL  : 8 harmonics, global weights updated on consensus lock
// BRIDGE ROLE      : Consumes BridgeCandidate from v31b → emits BridgeReward
// GRA COUPLING     : BASE_GRA_MODULATION * r / (1 + r)  — bounded in [0, base]
//
// Fixes vs prior versions (per roadmap analysis):
//   [F1] state_flags uint32_t added to Slot4096 (was UB / memory corruption)
//   [F2] No heap allocation in RK4 hot loop (in-place mantissa write-back)
//   [F3] Global dt computed once per step, not mutated inside slot loop
//   [F4] GRA bounded: factor = base * r/(1+r)  ∈ [0, base_gra]
//   [F5] Bridge interface (v31 ↔ v31b ring buffer, zero-copy)
//   [F6] Spectral kernel replaces raw GRA as per-neighbor coupling modulator
//   [F7] Adaptive dt stored globally, clamped once, never inside per-slot path
// ============================================================================

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <unistd.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// --- System Constants ---
#define PHI                1.6180339887498948
#define MAX_INSTANCES      8388608
#define SLOTS_PER_INSTANCE 4
#define MAX_SLOTS          (MAX_INSTANCES * SLOTS_PER_INSTANCE)
#define CHUNK_SIZE         1048576
#define MSB_MASK           (1ULL << 63)

// --- Analog Constants ---
#define GAMMA              0.02
#define LAMBDA             0.05
#define SAT_LIMIT          1e6
#define NOISE_SIGMA        0.01
#define CONSENSUS_EPS      1e-6
#define CONSENSUS_N        100
#define ADAPT_THRESH       0.8
#define K_COUPLING         1.0

// --- GRA — Closed-Form (v31: canonical, non-plastic) ---
#define BASE_GRA_MODULATION 0.18   // coupling factor bounded in [0, BASE_GRA_MODULATION]
#define MAX_GRA_N           16

static const uint64_t FIB_TABLE[MAX_GRA_N + 1] = {
    0,1,1,2,3,5,8,13,21,34,55,89,144,233,377,610,987};
static const uint64_t PRIME_TABLE[MAX_GRA_N] = {
    2,3,5,7,11,13,17,19,23,29,31,37,41,43,47,53};

// --- Spectral Kernel (v31: 8 harmonics, global learned weights) ---
#define SPECTRAL_N 8
static double g_W_cos[SPECTRAL_N] = {0.10,0.05,0.03,0.02,0.01,0.01,0.005,0.005};
static double g_W_sin[SPECTRAL_N] = {0.00,0.00,0.00,0.00,0.00,0.00,0.000,0.000};
static double g_W_amp[2]          = {0.05, 0.05};  // [0]=self, [1]=neighbor

// --- LL Probe ---
#define LL_PROBE_ITERS 64

// --- Checkpoint ---
#define CHECKPOINT_INTERVAL 100
#define SNAPSHOT_MAX        10
#define SNAPSHOT_DECAY      0.95

// --- Bridge ---
#define BRIDGE_CAPACITY 256

// --- MPI Stub ---
#define MPI_REAL 0
#if MPI_REAL
#include <mpi.h>
#define MPI_BCAST(buf,cnt,type,root,comm)          MPI_Bcast(buf,cnt,type,root,MPI_COMM_WORLD)
#define MPI_REDUCE(buf,res,cnt,type,op,root,comm)  MPI_Reduce(buf,res,cnt,type,op,root,MPI_COMM_WORLD)
#else
#define MPI_BCAST(buf,cnt,type,root,comm)
#define MPI_REDUCE(buf,res,cnt,type,op,root,comm)
#define MPI_SUM 0
#endif

// --- Timing ---
#ifdef USE_DS3231
#include <i2c/smbus.h>
#define DS3231_ADDR 0x68
static int i2c_fd = -1;
#endif

static double get_normalized_rand(void) { return (double)rand() / (double)RAND_MAX; }
#define GET_RANDOM_UINT64() (((uint64_t)rand() << 32) | (uint64_t)rand())

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Timing Primitives
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

int64_t get_rtc_ns(void) {
#ifdef USE_DS3231
    if (i2c_fd >= 0) {
        uint8_t data[7];
        if (i2c_smbus_read_i2c_block_data(i2c_fd, DS3231_ADDR, 0x00, 7, data) == 7) {
            int sec = ((data[0] >> 4) * 10) + (data[0] & 0x0F);
            int min = ((data[1] >> 4) * 10) + (data[1] & 0x0F);
            int hr  = ((data[2] >> 4) * 10) + (data[2] & 0x0F);
            return (int64_t)(hr * 3600 + min * 60 + sec) * 1000000000LL;
        }
    }
#endif
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000000000LL + ts.tv_nsec;
}

void rtc_sleep_until(int64_t target_ns) {
    int64_t now = get_rtc_ns();
    if (target_ns <= now) return;
    struct timespec req = {
        .tv_sec  = (target_ns - now) / 1000000000LL,
        .tv_nsec = (target_ns - now) % 1000000000LL
    };
    nanosleep(&req, NULL);
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Golden Recursive Algebra — CLOSED FORM (v31 canonical: fixed at init)
// r_n = sqrt(phi * omega * F_n * 2^n * prod(p_k, k=0..n-1))
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

double gra_rn_closed(int n, double omega) {
    if (n < 1 || n > MAX_GRA_N) return 1.0;
    uint64_t F_n   = FIB_TABLE[n];
    double   two_n = pow(2.0, (double)n);
    uint64_t prod_p = 1;
    for (int k = 0; k < n; k++) prod_p *= PRIME_TABLE[k];
    double inside = PHI * omega * (double)F_n * two_n * (double)prod_p;
    return sqrt(inside > 0.0 ? inside : 1e-12);
}

// Recursive form (used for cache warm-up only; not used in dynamics)
double gra_rn_recursive(int n, double omega, double *r_cache) {
    if (n < 1) return 1.0;
    if (r_cache[n] > 0.0) return r_cache[n];
    if (n == 1) { r_cache[1] = sqrt(4.0 * PHI * omega); return r_cache[1]; }
    double r_prev = gra_rn_recursive(n - 1, omega, r_cache);
    double mult = sqrt(2.0 * (double)PRIME_TABLE[n-1]
                     * ((double)FIB_TABLE[n] / (double)FIB_TABLE[n-1]));
    r_cache[n] = r_prev * mult;
    return r_cache[n];
}

// GRA vector addition (orthogonal composition)
static inline double gra_add(double r_n, double r_m) {
    return sqrt(r_n * r_n + r_m * r_m);
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// MPI (Multi-word Integer)
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

typedef struct { uint64_t *words; size_t num_words; uint8_t sign; } MPI;

// APA state flags
#define APA_FLAG_SIGN_NEG  (1u << 0)
#define APA_FLAG_IS_NAN    (1u << 1)
#define APA_FLAG_GOI       (1u << 2)   // Grown-Over Infinity
#define APA_FLAG_GUZ       (1u << 3)   // Gone-Under Zero
#define APA_FLAG_CONSENSUS (1u << 4)

void mpi_init(MPI *m, size_t initial_words) {
    m->words = calloc(initial_words, sizeof(uint64_t));
    m->num_words = initial_words;
    m->sign = 0;
}
void mpi_free(MPI *m) {
    if (m->words) free(m->words);
    m->words = NULL; m->num_words = 0;
}
void mpi_copy(MPI *dest, const MPI *src) {
    mpi_free(dest);
    dest->num_words = src->num_words;
    dest->words = malloc(src->num_words * sizeof(uint64_t));
    if (src->words && dest->words)
        memcpy(dest->words, src->words, src->num_words * sizeof(uint64_t));
    dest->sign = src->sign;
}
void mpi_set_value(MPI *m, uint64_t value, uint8_t sign) {
    if (m->words) m->words[0] = value;
    m->sign = sign;
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Slot4096 — APA with Complex Coupling and GRA
// [F1] state_flags uint32_t explicitly present (fixes memory corruption)
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

typedef struct {
    uint64_t  *mantissa_words;
    MPI        num_words_mantissa;
    MPI        exponent_mpi;
    MPI        source_of_infinity;
    size_t     num_words;
    int64_t    exponent;
    float      base;
    int        bits_mant;
    int        bits_exp;
    uint32_t   state_flags;        // [F1] explicitly declared — no more UB
    uint16_t   exponent_base;

    // Complex oscillator state
    double     phase;
    double     phase_vel;
    double     freq;
    double     amp_im;

    // GRA — v31: CLOSED, fixed at init, non-plastic
    double     r_harmonic;
} Slot4096;

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// APA Helpers (no heap allocation in hot path — [F2])
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

void ap_free(Slot4096 *slot) {
    if (!slot) return;
    if (slot->mantissa_words) { free(slot->mantissa_words); slot->mantissa_words = NULL; }
    mpi_free(&slot->exponent_mpi);
    mpi_free(&slot->num_words_mantissa);
    mpi_free(&slot->source_of_infinity);
    slot->num_words = 0;
}

void ap_copy(Slot4096 *dest, const Slot4096 *src) {
    ap_free(dest);
    memcpy(dest, src, sizeof(Slot4096));
    dest->mantissa_words = malloc(src->num_words * sizeof(uint64_t));
    if (dest->mantissa_words && src->mantissa_words)
        memcpy(dest->mantissa_words, src->mantissa_words,
               src->num_words * sizeof(uint64_t));
    mpi_copy(&dest->exponent_mpi,       &src->exponent_mpi);
    mpi_copy(&dest->num_words_mantissa, &src->num_words_mantissa);
    mpi_copy(&dest->source_of_infinity, &src->source_of_infinity);
}

double ap_to_double(const Slot4096 *slot) {
    if (!slot || slot->num_words == 0 || !slot->mantissa_words) return 0.0;
    double mant = (double)slot->mantissa_words[0] / (double)UINT64_MAX;
    double val  = mant * pow(2.0, (double)slot->exponent);
    return (slot->state_flags & APA_FLAG_SIGN_NEG) ? -val : val;
}

// [F2] In-place write-back: no heap allocation, precision boundary acknowledged
static void ap_writeback_double(Slot4096 *slot, double A_re) {
    if (!slot->mantissa_words || slot->num_words == 0) return;
    if (A_re == 0.0) {
        memset(slot->mantissa_words, 0, slot->num_words * sizeof(uint64_t));
        slot->exponent = 0;
        slot->state_flags &= ~APA_FLAG_SIGN_NEG;
        mpi_set_value(&slot->exponent_mpi, 0, 0);
        return;
    }
    // Precision boundary: double → word[0], higher words zeroed (acknowledged)
    int exp_off;
    double mant = frexp(fabs(A_re), &exp_off);
    slot->mantissa_words[0] = (uint64_t)(mant * (double)UINT64_MAX);
    for (size_t i = 1; i < slot->num_words; i++) slot->mantissa_words[i] = 0;
    slot->exponent = (int64_t)exp_off;
    if (A_re < 0.0) slot->state_flags |=  APA_FLAG_SIGN_NEG;
    else            slot->state_flags &= ~APA_FLAG_SIGN_NEG;
    mpi_set_value(&slot->exponent_mpi,
                  (uint64_t)llabs(slot->exponent),
                  slot->exponent < 0 ? 1 : 0);
}

Slot4096* ap_from_double(double value, int bits_mant, int bits_exp) {
    Slot4096 *slot = malloc(sizeof(Slot4096));
    if (!slot) return NULL;
    memset(slot, 0, sizeof(Slot4096));
    slot->bits_mant  = bits_mant;
    slot->bits_exp   = bits_exp;
    slot->num_words  = 1;
    slot->mantissa_words = calloc(1, sizeof(uint64_t));
    if (!slot->mantissa_words) { free(slot); return NULL; }
    mpi_init(&slot->exponent_mpi,       1);
    mpi_init(&slot->num_words_mantissa, 1);
    mpi_init(&slot->source_of_infinity, 1);
    ap_writeback_double(slot, value);
    return slot;
}

void ap_shift_right_legacy(uint64_t *w, size_t nw, int64_t shift) {
    if (shift <= 0 || nw == 0) return;
    if (shift >= (int64_t)(nw * 64)) { memset(w, 0, nw * sizeof(uint64_t)); return; }
    int64_t ws = shift / 64;
    int     bs = (int)(shift % 64);
    if (ws > 0) {
        for (int64_t i = (int64_t)nw - 1; i >= ws; i--) w[i] = w[i - ws];
        memset(w, 0, (size_t)ws * sizeof(uint64_t));
    }
    if (bs > 0) {
        int rs = 64 - bs;
        for (size_t i = nw - 1; i > 0; i--)
            w[i] = (w[i] >> bs) | (w[i-1] << rs);
        w[0] >>= bs;
    }
}

void ap_normalize_legacy(Slot4096 *slot) {
    if (slot->num_words == 0 || !slot->mantissa_words) return;
    while (!(slot->mantissa_words[0] & MSB_MASK)) {
        if (slot->exponent <= -(1LL << (slot->bits_exp - 1))) {
            slot->state_flags |= APA_FLAG_GUZ;
            break;
        }
        uint64_t carry = 0;
        for (size_t i = slot->num_words - 1; i != (size_t)-1; i--) {
            uint64_t nc = (slot->mantissa_words[i] & MSB_MASK) ? 1 : 0;
            slot->mantissa_words[i] = (slot->mantissa_words[i] << 1) | carry;
            carry = nc;
        }
        slot->exponent--;
    }
    if (slot->mantissa_words[0] == 0) slot->exponent = 0;
}

void ap_add_legacy(Slot4096 *A, const Slot4096 *B) {
    if (A->num_words != B->num_words) return;
    Slot4096 Bal;
    memset(&Bal, 0, sizeof(Bal));
    ap_copy(&Bal, B);
    int64_t ed = A->exponent - Bal.exponent;
    if      (ed > 0) { ap_shift_right_legacy(Bal.mantissa_words, Bal.num_words,  ed); Bal.exponent = A->exponent; }
    else if (ed < 0) { ap_shift_right_legacy(A->mantissa_words,  A->num_words,  -ed); A->exponent  = Bal.exponent; }
    uint64_t carry = 0;
    for (size_t i = A->num_words - 1; i != (size_t)-1; i--) {
        uint64_t s = A->mantissa_words[i] + Bal.mantissa_words[i] + carry;
        carry = (s < A->mantissa_words[i] || (s == A->mantissa_words[i] && carry)) ? 1 : 0;
        A->mantissa_words[i] = s;
    }
    if (carry) {
        if (A->exponent >= (1LL << (A->bits_exp - 1))) A->state_flags |= APA_FLAG_GOI;
        else {
            A->exponent++;
            ap_shift_right_legacy(A->mantissa_words, A->num_words, 1);
            A->mantissa_words[0] |= MSB_MASK;
        }
    }
    ap_normalize_legacy(A);
    mpi_set_value(&A->exponent_mpi, (uint64_t)llabs(A->exponent), A->exponent < 0 ? 1 : 0);
    ap_free(&Bal);
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Slot Init — CLOSED-FORM GRA (r_harmonic fixed at init, non-plastic)
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Slot4096 slot_init_v31(int bits_mant, int bits_exp, int n_dim, double omega,
                        double *r_cache) {
    Slot4096 slot;
    memset(&slot, 0, sizeof(slot));
    slot.bits_mant = bits_mant;
    slot.bits_exp  = bits_exp;
    slot.num_words = (size_t)((bits_mant + 63) / 64);

    slot.mantissa_words = calloc(slot.num_words, sizeof(uint64_t));
    mpi_init(&slot.exponent_mpi,       1);
    mpi_init(&slot.num_words_mantissa, 1);
    mpi_init(&slot.source_of_infinity, 1);

    if (slot.num_words > 0 && slot.mantissa_words) {
        slot.mantissa_words[0] = GET_RANDOM_UINT64() | MSB_MASK;
    }

    int64_t exp_range = 1LL << bits_exp;
    slot.exponent      = (rand() % exp_range) - (1LL << (bits_exp - 1));
    slot.base          = (float)(PHI + get_normalized_rand() * 0.01);
    slot.exponent_base = 4096;

    slot.phase     = 2.0 * M_PI * get_normalized_rand();
    slot.phase_vel = 0.0;
    slot.freq      = 1.0 + 0.5 * get_normalized_rand();
    slot.amp_im    = 0.1  * get_normalized_rand();

    // CLOSED: compute once, never update during dynamics
    if (r_cache) {
        slot.r_harmonic = gra_rn_recursive(n_dim, omega, r_cache);
    } else {
        slot.r_harmonic = gra_rn_closed(n_dim, omega);
    }

    mpi_set_value(&slot.exponent_mpi,       (uint64_t)llabs(slot.exponent), slot.exponent < 0 ? 1 : 0);
    mpi_set_value(&slot.num_words_mantissa, (uint64_t)slot.num_words,       0);
    return slot;
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Spectral Kernel — 8 harmonics, global weights, learned from LL outcomes
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

static double spectral_eval(double dphi, double A_self, double A_neigh) {
    double acc = 0.0;
    for (int k = 1; k <= SPECTRAL_N; k++) {
        acc += g_W_cos[k-1] * cos((double)k * dphi);
        acc += g_W_sin[k-1] * sin((double)k * dphi);
    }
    acc += g_W_amp[0] * A_self + g_W_amp[1] * A_neigh;
    return acc;
}

static void spectral_learn(double dphi, double A_self, double A_neigh,
                            double reward) {
    const double lr   = 1e-4;
    const double clmp = 2.0;
    for (int k = 1; k <= SPECTRAL_N; k++) {
        g_W_cos[k-1] += lr * reward * cos((double)k * dphi);
        g_W_sin[k-1] += lr * reward * sin((double)k * dphi);
        if (g_W_cos[k-1] >  clmp) g_W_cos[k-1] =  clmp;
        if (g_W_cos[k-1] < -clmp) g_W_cos[k-1] = -clmp;
        if (g_W_sin[k-1] >  clmp) g_W_sin[k-1] =  clmp;
        if (g_W_sin[k-1] < -clmp) g_W_sin[k-1] = -clmp;
    }
    g_W_amp[0] += lr * reward * A_self;
    g_W_amp[1] += lr * reward * A_neigh;
    // clamp amplitude weights
    for (int i = 0; i < 2; i++) {
        if (g_W_amp[i] >  clmp) g_W_amp[i] =  clmp;
        if (g_W_amp[i] < -clmp) g_W_amp[i] = -clmp;
    }
}

// Normalize spectral weight vector (called on consensus lock)
static void spectral_normalize(void) {
    double norm = 0.0;
    for (int k = 0; k < SPECTRAL_N; k++)
        norm += g_W_cos[k]*g_W_cos[k] + g_W_sin[k]*g_W_sin[k];
    norm = sqrt(norm) + 1e-12;
    for (int k = 0; k < SPECTRAL_N; k++) {
        g_W_cos[k] /= norm;
        g_W_sin[k] /= norm;
    }
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// LL Residue Probe — 64-bit anchored reward signal
// s_{n+1} = s_n^2 - 2  (mod 2^64)
// Preserves squaring dynamics; lower residue ≈ more LL-like structure
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

static inline uint64_t ll_step64(uint64_t s) {
    // __uint128_t available on GCC/Clang x86-64
    unsigned __int128 sq = (unsigned __int128)s * s;
    return (uint64_t)(sq) - 2ULL;
}

double ll_residue_probe(uint64_t seed, int iters) {
    uint64_t s = seed | 1ULL;   // ensure non-zero seed
    for (int i = 0; i < iters; i++) s = ll_step64(s);
    return (double)s / (double)UINT64_MAX;
}

// Derive a seed from slot state (deterministic, reversible)
static uint64_t slot_to_ll_seed(const Slot4096 *slot) {
    uint64_t m = slot->mantissa_words ? slot->mantissa_words[0] : 0x9e3779b97f4a7c15ULL;
    uint64_t p = (uint64_t)(slot->phase * 1e9);
    return m ^ p ^ (uint64_t)(slot->r_harmonic * 1e6);
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Bridge v31 ↔ v31b  (ring-buffer, zero heap allocation per message)
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

typedef struct {
    int      slot_idx;
    double   phase;
    double   amplitude;
    double   r_harmonic;
    double   score;
    uint64_t ll_seed;
} BridgeCandidate;

typedef struct {
    int    slot_idx;
    double reward;
    double ll_residue;
    int    verified;    // +1 = confirmed, -1 = rejected, 0 = pending
} BridgeReward;

typedef struct {
    BridgeCandidate cands[BRIDGE_CAPACITY];
    BridgeReward    rewards[BRIDGE_CAPACITY];
    int cand_head, cand_tail;
    int rew_head,  rew_tail;
} BridgeBuffer;

static BridgeBuffer g_bridge;

static void bridge_init(void) { memset(&g_bridge, 0, sizeof(g_bridge)); }

int bridge_emit_candidate(int idx, double phase, double amp,
                           double r_h, double score) {
    int next = (g_bridge.cand_tail + 1) % BRIDGE_CAPACITY;
    if (next == g_bridge.cand_head) return 0;   // buffer full
    g_bridge.cands[g_bridge.cand_tail] = (BridgeCandidate){
        .slot_idx   = idx,
        .phase      = phase,
        .amplitude  = amp,
        .r_harmonic = r_h,
        .score      = score,
        .ll_seed    = ((uint64_t)(amp * (double)UINT64_MAX))
                     ^ ((uint64_t)(phase * 1e9))
                     ^ ((uint64_t)(r_h * 1e6))
    };
    g_bridge.cand_tail = next;
    return 1;
}

int bridge_pop_candidate(BridgeCandidate *out) {
    if (g_bridge.cand_head == g_bridge.cand_tail) return 0;
    *out = g_bridge.cands[g_bridge.cand_head];
    g_bridge.cand_head = (g_bridge.cand_head + 1) % BRIDGE_CAPACITY;
    return 1;
}

void bridge_inject_reward(int slot_idx, double reward, double residue,
                           int verified) {
    int next = (g_bridge.rew_tail + 1) % BRIDGE_CAPACITY;
    if (next == g_bridge.rew_head) return;
    g_bridge.rewards[g_bridge.rew_tail] = (BridgeReward){
        slot_idx, reward, residue, verified};
    g_bridge.rew_tail = next;
}

int bridge_pop_reward(BridgeReward *out) {
    if (g_bridge.rew_head == g_bridge.rew_tail) return 0;
    *out = g_bridge.rewards[g_bridge.rew_head];
    g_bridge.rew_head = (g_bridge.rew_head + 1) % BRIDGE_CAPACITY;
    return 1;
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// AnalogLink (GRA-extended)
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

typedef struct {
    double charge;
    double charge_im;
    double tension;
    double potential;
    double coupling;
    double r_harmonic;
} AnalogLink;

static void exchange_analog_links(AnalogLink *links, int rank, int size,
                                   int num_links) {
#if MPI_REAL
    MPI_BCAST(links, num_links * sizeof(AnalogLink), MPI_BYTE, rank, MPI_COMM_WORLD);
    AnalogLink *reduced = calloc(num_links, sizeof(AnalogLink));
    if (reduced) {
        MPI_REDUCE(links, reduced, num_links * sizeof(AnalogLink),
                   MPI_BYTE, MPI_SUM, 0, MPI_COMM_WORLD);
        for (int i = 0; i < num_links; i++) {
            links[i].charge    = reduced[i].charge / size;
            links[i].charge_im = reduced[i].charge_im / size;
            links[i].tension  *= 0.9;
        }
        free(reduced);
    }
#else
    (void)rank; (void)size;
    for (int i = 0; i < num_links; i++) {
        links[i].charge    *= 0.95;
        links[i].charge_im *= 0.95;
    }
#endif
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// RK4 — v31: global dt, in-place write-back, spectral coupling, bounded GRA
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

typedef struct { double A_re, A_im, phase, phase_vel; } ComplexState;

static ComplexState compute_deriv_v31(ComplexState st, double omega,
                                       const AnalogLink *nb, int num_nb,
                                       double r_h, double A_self) {
    ComplexState d = {0};
    double sum_sin = 0.0, gra_sum = 0.0;

    d.A_re = -GAMMA * st.A_re;
    d.A_im = -GAMMA * st.A_im;

    for (int k = 0; k < num_nb; k++) {
        double dphi    = nb[k].potential - st.phase;
        double A_neigh = fabs(nb[k].charge);

        // Spectral coupling (v31: 8-harmonic learned kernel)
        double spec = spectral_eval(dphi, A_self, A_neigh);

        // GRA coupling [F4]: bounded in [0, BASE_GRA_MODULATION]
        double combined = gra_add(r_h, nb[k].r_harmonic);
        double gra_fac  = BASE_GRA_MODULATION * combined / (1.0 + combined);

        double factor = spec + gra_fac;

        sum_sin += sin(dphi);
        d.A_re  += K_COUPLING * factor * cos(dphi);
        d.A_im  += K_COUPLING * factor * sin(dphi);
        gra_sum += gra_fac;
    }

    d.phase_vel = omega + K_COUPLING * sum_sin + 0.15 * gra_sum;
    d.phase     = st.phase_vel;
    return d;
}

// [F2] No heap allocation; [F3] dt is passed in (computed globally outside loop
void rk4_step_v31(Slot4096 *slot, double dt, const AnalogLink *nb,
                   int num_nb, double omega) {
    double A_self = sqrt(pow(ap_to_double(slot), 2.0) + pow(slot->amp_im, 2.0));

    ComplexState st = {
        .A_re     = ap_to_double(slot),
        .A_im     = slot->amp_im,
        .phase    = slot->phase,
        .phase_vel= slot->phase_vel
    };

    ComplexState k1 = compute_deriv_v31(st, omega, nb, num_nb, slot->r_harmonic, A_self);

    ComplexState t = st;
    t.A_re += dt * k1.A_re / 2.0; t.A_im += dt * k1.A_im / 2.0;
    t.phase += dt * k1.phase / 2.0; t.phase_vel += dt * k1.phase_vel / 2.0;
    ComplexState k2 = compute_deriv_v31(t, omega, nb, num_nb, slot->r_harmonic, A_self);

    t = st;
    t.A_re += dt * k2.A_re / 2.0; t.A_im += dt * k2.A_im / 2.0;
    t.phase += dt * k2.phase / 2.0; t.phase_vel += dt * k2.phase_vel / 2.0;
    ComplexState k3 = compute_deriv_v31(t, omega, nb, num_nb, slot->r_harmonic, A_self);

    t = st;
    t.A_re += dt * k3.A_re; t.A_im += dt * k3.A_im;
    t.phase += dt * k3.phase; t.phase_vel += dt * k3.phase_vel;
    ComplexState k4 = compute_deriv_v31(t, omega, nb, num_nb, slot->r_harmonic, A_self);

    st.A_re     += dt / 6.0 * (k1.A_re     + 2*k2.A_re     + 2*k3.A_re     + k4.A_re);
    st.A_im     += dt / 6.0 * (k1.A_im     + 2*k2.A_im     + 2*k3.A_im     + k4.A_im);
    st.phase    += dt / 6.0 * (k1.phase    + 2*k2.phase    + 2*k3.phase    + k4.phase);
    st.phase_vel+= dt / 6.0 * (k1.phase_vel+ 2*k2.phase_vel+ 2*k3.phase_vel+ k4.phase_vel);

    // Amplitude damping + saturation + noise
    double A = sqrt(st.A_re*st.A_re + st.A_im*st.A_im);
    A *= exp(-LAMBDA * dt);
    if (A > SAT_LIMIT) A = SAT_LIMIT;
    A += NOISE_SIGMA * (2.0 * get_normalized_rand() - 1.0);

    double norm = sqrt(st.A_re*st.A_re + st.A_im*st.A_im);
    if (norm > 1e-10) { st.A_re = (st.A_re / norm) * A; st.A_im = (st.A_im / norm) * A; }

    st.phase = fmod(st.phase, 2.0 * M_PI);
    if (st.phase < 0.0) st.phase += 2.0 * M_PI;

    // [F2] In-place write-back — no malloc
    ap_writeback_double(slot, st.A_re);
    slot->amp_im    = st.A_im;
    slot->phase     = st.phase;
    slot->phase_vel = st.phase_vel;
    // r_harmonic NOT updated — CLOSED hemisphere
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// HDGL Lattice (v31: closed-form GRA cache, global dt)
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

typedef struct { Slot4096 *slots; size_t allocated; } HDGLChunk;

typedef struct {
    HDGLChunk **chunks;
    int         num_chunks;
    int         num_instances;
    int         slots_per_instance;
    double      omega;
    double      time;
    int         consensus_steps;
    double      phase_var;
    int64_t     last_checkpoint_ns;
    double      r_cache[MAX_GRA_N + 1];
    double      dt_global;     // [F3] global dt for this hemisphere
} HDGLLattice;

HDGLLattice* lattice_init_v31(int num_instances, int slots_per_instance) {
    HDGLLattice *lat = malloc(sizeof(HDGLLattice));
    if (!lat) return NULL;
    memset(lat, 0, sizeof(HDGLLattice));
    lat->num_instances      = num_instances;
    lat->slots_per_instance = slots_per_instance;
    lat->omega              = 1.0;
    lat->time               = 0.0;
    lat->consensus_steps    = 0;
    lat->phase_var          = 1e6;
    lat->last_checkpoint_ns = get_rtc_ns();
    lat->dt_global          = 1.0 / 32768.0;

    // Warm closed-form GRA cache
    for (int n = 1; n <= MAX_GRA_N; n++)
        gra_rn_recursive(n, lat->omega, lat->r_cache);

    int total_slots  = num_instances * slots_per_instance;
    lat->num_chunks  = (total_slots + CHUNK_SIZE - 1) / CHUNK_SIZE;
    lat->chunks      = calloc(lat->num_chunks, sizeof(HDGLChunk*));
    if (!lat->chunks) { free(lat); return NULL; }
    return lat;
}

HDGLChunk* lattice_get_chunk(HDGLLattice *lat, int chunk_idx) {
    if (chunk_idx >= lat->num_chunks) return NULL;
    if (!lat->chunks[chunk_idx]) {
        HDGLChunk *chunk = malloc(sizeof(HDGLChunk));
        if (!chunk) return NULL;
        chunk->allocated = CHUNK_SIZE;
        chunk->slots     = malloc(CHUNK_SIZE * sizeof(Slot4096));
        if (!chunk->slots) { free(chunk); return NULL; }
        for (int i = 0; i < CHUNK_SIZE; i++) {
            int n_dim    = (i % MAX_GRA_N) + 1;
            int bits_m   = 4096 + (i % 8) * 64;
            int bits_e   = 16   + (i % 8) * 2;
            chunk->slots[i] = slot_init_v31(bits_m, bits_e, n_dim,
                                             lat->omega, lat->r_cache);
        }
        lat->chunks[chunk_idx] = chunk;
    }
    return lat->chunks[chunk_idx];
}

Slot4096* lattice_get_slot(HDGLLattice *lat, int idx) {
    int chunk_idx = idx / CHUNK_SIZE;
    int local_idx = idx % CHUNK_SIZE;
    HDGLChunk *chunk = lattice_get_chunk(lat, chunk_idx);
    return chunk ? &chunk->slots[local_idx] : NULL;
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Consensus Detection (triggers spectral kernel normalization on lock)
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

void detect_harmonic_consensus(HDGLLattice *lat) {
    int    total = lat->num_instances * lat->slots_per_instance;
    double mean  = 0.0, var = 0.0;
    int    count = 0;

    for (int i = 0; i < total; i++) {
        Slot4096 *s = lattice_get_slot(lat, i);
        if (s && !(s->state_flags & APA_FLAG_CONSENSUS)) {
            mean += s->phase; count++;
        }
    }
    if (count == 0) return;
    mean /= count;

    for (int i = 0; i < total; i++) {
        Slot4096 *s = lattice_get_slot(lat, i);
        if (s && !(s->state_flags & APA_FLAG_CONSENSUS)) {
            double d = s->phase - mean;
            if (d >  M_PI) d -= 2.0 * M_PI;
            if (d < -M_PI) d += 2.0 * M_PI;
            var += d * d;
        }
    }
    lat->phase_var = sqrt(var / count);

    if (lat->phase_var < CONSENSUS_EPS) {
        lat->consensus_steps++;
        if (lat->consensus_steps >= CONSENSUS_N) {
            printf("[v31 CONSENSUS] Domain locked t=%.4f var=%.8f\n",
                   lat->time, lat->phase_var);
            // Lock slots and normalize spectral weights (reward signal)
            for (int i = 0; i < total; i++) {
                Slot4096 *s = lattice_get_slot(lat, i);
                if (s && !(s->state_flags & APA_FLAG_CONSENSUS)) {
                    s->state_flags |= APA_FLAG_CONSENSUS;
                    s->phase_vel    = 0.0;
                }
            }
            spectral_normalize();
            lat->consensus_steps = 0;
        }
    } else {
        lat->consensus_steps = 0;
    }
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Lattice RK4 Integration — [F3] global dt, no per-slot mutation
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

void lattice_integrate_v31(HDGLLattice *lat) {
    int    total  = lat->num_instances * lat->slots_per_instance;
    double dt     = lat->dt_global;   // [F3] single value, not mutated per slot
    double amp_sq_sum = 0.0;
    int    active = 0;

    for (int i = 0; i < total; i++) {
        Slot4096 *slot = lattice_get_slot(lat, i);
        if (!slot || (slot->state_flags &
                      (APA_FLAG_GOI | APA_FLAG_IS_NAN | APA_FLAG_CONSENSUS)))
            continue;

        AnalogLink nb[8] = {0};
        int ni[8] = {
            (i - 1 + total) % total,
            (i + 1)         % total,
            (i - lat->slots_per_instance + total) % total,
            (i + lat->slots_per_instance)         % total,
            (i - lat->slots_per_instance - 1 + total) % total,
            (i - lat->slots_per_instance + 1 + total) % total,
            (i + lat->slots_per_instance - 1 + total) % total,
            (i + lat->slots_per_instance + 1)         % total
        };

        for (int j = 0; j < 8; j++) {
            Slot4096 *n = lattice_get_slot(lat, ni[j]);
            if (n) {
                double nA = ap_to_double(n);
                double sA = ap_to_double(slot);
                nb[j].charge      = nA;
                nb[j].charge_im   = n->amp_im;
                nb[j].tension     = (nA - sA) / dt;
                nb[j].potential   = n->phase - slot->phase;
                double amp_corr   = fabs(nA) / (fabs(sA) + 1e-10);
                nb[j].coupling    = K_COUPLING * exp(-fabs(1.0 - amp_corr));
                nb[j].r_harmonic  = n->r_harmonic;
            }
        }

        exchange_analog_links(nb, i % lat->num_instances, lat->num_instances, 8);
        rk4_step_v31(slot, dt, nb, 8, lat->omega);

        double amp = sqrt(pow(ap_to_double(slot), 2.0) + pow(slot->amp_im, 2.0));
        amp_sq_sum += amp * amp;
        active++;
    }

    // [F3] Adaptive dt updated ONCE globally after all slots
    if (active > 0) {
        double rms_amp = sqrt(amp_sq_sum / active);
        if      (rms_amp > ADAPT_THRESH)       lat->dt_global *= PHI;
        else if (rms_amp < ADAPT_THRESH / PHI) lat->dt_global /= PHI;
        if (lat->dt_global < 1e-6) lat->dt_global = 1e-6;
        if (lat->dt_global > 0.1)  lat->dt_global = 0.1;
    }

    detect_harmonic_consensus(lat);
    lat->omega += 0.01 * dt;
    lat->time  += dt;
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Bridge Processing Loop (v31 acts as verifier: pop candidate → LL probe → reward)
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

static int bridge_process_candidates(HDGLLattice *lat, int max_per_cycle) {
    BridgeCandidate cand;
    int processed = 0;

    while (processed < max_per_cycle && bridge_pop_candidate(&cand)) {
        double residue = ll_residue_probe(cand.ll_seed, LL_PROBE_ITERS);

        // Build reward: 1 = clean, 0 = chaotic
        double reward = 1.0 / (1.0 + residue * 10.0)
                      + 0.3 * (cand.score / 10.0)    // normalize score
                      + 0.2 * cand.amplitude;

        if (reward > 1.0) reward = 1.0;

        int verified = (residue < 0.05) ? 1 : -1;
        bridge_inject_reward(cand.slot_idx, reward, residue, verified);

        // Update global spectral weights based on LL outcome
        double dphi_proxy = cand.phase;
        spectral_learn(dphi_proxy, cand.amplitude, cand.r_harmonic, reward);

        // Also update any local slot that matches (if in this lattice)
        int total = lat->num_instances * lat->slots_per_instance;
        if (cand.slot_idx >= 0 && cand.slot_idx < total) {
            Slot4096 *s = lattice_get_slot(lat, cand.slot_idx);
            if (s && verified == 1) {
                // Reinforce coherence by nudging phase toward consensus
                s->phase_vel *= 0.95;
            }
        }
        processed++;
    }
    return processed;
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Checkpoint
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

typedef struct {
    int     evolution;
    int64_t timestamp_ns;
    double  phase_var;
    double  omega;
    double  weight;
} CheckpointMeta;

typedef struct {
    CheckpointMeta *snapshots;
    int count, capacity;
} CheckpointManager;

CheckpointManager* checkpoint_init(void) {
    CheckpointManager *m = malloc(sizeof(CheckpointManager));
    m->snapshots = malloc(SNAPSHOT_MAX * sizeof(CheckpointMeta));
    m->count     = 0;
    m->capacity  = SNAPSHOT_MAX;
    return m;
}

void checkpoint_add(CheckpointManager *m, int evo, HDGLLattice *lat) {
    if (m->count >= m->capacity) {
        int mi = 0; double mw = m->snapshots[0].weight;
        for (int i = 1; i < m->count; i++)
            if (m->snapshots[i].weight < mw) { mw = m->snapshots[i].weight; mi = i; }
        for (int i = mi; i < m->count - 1; i++) m->snapshots[i] = m->snapshots[i+1];
        m->count--;
    }
    m->snapshots[m->count++] = (CheckpointMeta){
        evo, get_rtc_ns(), lat->phase_var, lat->omega, 1.0};
    for (int i = 0; i < m->count - 1; i++) m->snapshots[i].weight *= SNAPSHOT_DECAY;
    printf("[v31 Checkpoint] evo=%d var=%.6f omega=%.4f\n",
           evo, lat->phase_var, lat->omega);
}

void checkpoint_free(CheckpointManager *m) {
    if (m) { free(m->snapshots); free(m); }
}

void lattice_free_v31(HDGLLattice *lat) {
    if (!lat) return;
    for (int i = 0; i < lat->num_chunks; i++) {
        if (lat->chunks[i]) {
            for (size_t j = 0; j < CHUNK_SIZE; j++)
                ap_free(&lat->chunks[i]->slots[j]);
            free(lat->chunks[i]->slots);
            free(lat->chunks[i]);
        }
    }
    free(lat->chunks);
    free(lat);
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Bootloader
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

void bootloader_v31(HDGLLattice *lat, int steps, CheckpointManager *ckpt) {
    printf("[v31 Bootloader] Closed-Form Accuracy Hemisphere init...\n");
    printf("[v31 Bootloader] %d instances × %d slots = %d total\n",
           lat->num_instances, lat->slots_per_instance,
           lat->num_instances * lat->slots_per_instance);

    int64_t step_ns      = 30517;
    int64_t next_step_ns = get_rtc_ns() + step_ns;

    for (int i = 0; i < steps; i++) {
        lattice_integrate_v31(lat);
        bridge_process_candidates(lat, 8);

        if (i % CHECKPOINT_INTERVAL == 0 && i > 0)
            checkpoint_add(ckpt, i, lat);

        rtc_sleep_until(next_step_ns);
        next_step_ns += step_ns;
    }

    printf("[v31 Bootloader] %d steps | dt=%.8f | omega=%.6f | var=%.6f\n",
           steps, lat->dt_global, lat->omega, lat->phase_var);
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Main
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

int main(int argc, char *argv[]) {
    (void)argc; (void)argv;
    srand((unsigned)time(NULL));
    bridge_init();

#ifdef USE_DS3231
    i2c_fd = i2c_open("/dev/i2c-1");
    if (i2c_fd >= 0) {
        i2c_smbus_write_byte_data(i2c_fd, DS3231_ADDR, 0x0E, 0x00);
        printf("[RTC] DS3231 online\n");
    } else {
        printf("[RTC] CLOCK_MONOTONIC fallback\n");
    }
#else
    printf("[RTC] CLOCK_MONOTONIC fallback\n");
#endif

    printf("=== HDGL Analog Mainnet v3.1 — Closed-Form Accuracy Hemisphere ===\n\n");

    HDGLLattice *lat = lattice_init_v31(4096, 4);
    if (!lat) { fprintf(stderr, "Fatal: lattice init failed\n"); return 1; }

    CheckpointManager *ckpt = checkpoint_init();
    bootloader_v31(lat, 500, ckpt);

    printf("\nGRA closed-form r_n (omega=%.4f):\n", lat->omega);
    for (int n = 1; n <= 8; n++)
        printf("  r_%d = %.8f\n", n, gra_rn_closed(n, lat->omega));

    printf("\nSpectral kernel weights after boot:\n");
    for (int k = 0; k < SPECTRAL_N; k++)
        printf("  W_cos[%d]=%.6f  W_sin[%d]=%.6f\n",
               k, g_W_cos[k], k, g_W_sin[k]);

    printf("\nFirst 8 slots:\n");
    for (int i = 0; i < 8; i++) {
        Slot4096 *s = lattice_get_slot(lat, i);
        if (s) {
            double amp = sqrt(pow(ap_to_double(s), 2.0) + pow(s->amp_im, 2.0));
            double res = ll_residue_probe(slot_to_ll_seed(s), LL_PROBE_ITERS);
            printf("  Slot %d: |A|=%.6e  φ=%.3f  r_h=%.6f  LL_res=%.6f  flags=0x%02x\n",
                   i, amp, s->phase, s->r_harmonic, res, s->state_flags);
        }
    }

    checkpoint_free(ckpt);
    lattice_free_v31(lat);

#ifdef USE_DS3231
    if (i2c_fd >= 0) i2c_close(i2c_fd);
#endif

    printf("\n=== v3.1 CLOSED-FORM ACCURACY HEMISPHERE OPERATIONAL ===\n");
    return 0;
}
