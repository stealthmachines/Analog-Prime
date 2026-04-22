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
#define PHI 1.6180339887498948
#define MAX_INSTANCES 8388608
#define SLOTS_PER_INSTANCE 4
#define MAX_SLOTS (MAX_INSTANCES * SLOTS_PER_INSTANCE)
#define CHUNK_SIZE 1048576
#define MSB_MASK (1ULL << 63)

// --- Analog Constants ---
#define GAMMA 0.02
#define LAMBDA 0.05
#define SAT_LIMIT 1e6
#define NOISE_SIGMA 0.01
#define CONSENSUS_EPS 1e-6
#define CONSENSUS_N 100
#define ADAPT_THRESH 0.8
#define K_COUPLING 1.0

// --- GRA Tuning (the sweet spot for balance) ---
#define BASE_GRA_MODULATION 0.18     // Start moderate
#define GRA_PLASTICITY 0.008         // Slow evolution of r_harmonic
#define MAX_GRA_N 16

static const uint64_t FIB_TABLE[MAX_GRA_N + 1] = {0,1,1,2,3,5,8,13,21,34,55,89,144,233,377,610,987};
static const uint64_t PRIME_TABLE[MAX_GRA_N]   = {2,3,5,7,11,13,17,19,23,29,31,37,41,43,47,53};

// --- Checkpoint Constants ---
#define CHECKPOINT_INTERVAL 100
#define SNAPSHOT_MAX 10
#define SNAPSHOT_DECAY 0.95

// --- MPI Stub ---
#define MPI_REAL 0
#if MPI_REAL
#include <mpi.h>
#define MPI_BCAST(buf, cnt, type, root, comm) MPI_Bcast(buf, cnt, type, root, MPI_COMM_WORLD)
#define MPI_REDUCE(buf, res, cnt, type, op, root, comm) MPI_Reduce(buf, res, cnt, type, op, root, MPI_COMM_WORLD)
#else
#define MPI_BCAST(buf, cnt, type, root, comm)
#define MPI_REDUCE(buf, res, cnt, type, op, root, comm)
#define MPI_SUM 0
#endif

// --- Timing ---
#ifdef USE_DS3231
#include <i2c/smbus.h>
#define DS3231_ADDR 0x68
static int i2c_fd = -1;
#endif

double get_normalized_rand() {
    return (double)rand() / RAND_MAX;
}

#define GET_RANDOM_UINT64() (((uint64_t)rand() << 32) | rand())

// Timing primitives (unchanged)
int64_t get_rtc_ns() {
#ifdef USE_DS3231
    if (i2c_fd >= 0) {
        uint8_t data[7];
        if (i2c_smbus_read_i2c_block_data(i2c_fd, DS3231_ADDR, 0x00, 7, data) == 7) {
            int sec = ((data[0] >> 4) * 10) + (data[0] & 0x0F);
            int min = ((data[1] >> 4) * 10) + (data[1] & 0x0F);
            int hr = ((data[2] >> 4) * 10) + (data[2] & 0x0F);
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
        .tv_sec = (target_ns - now) / 1000000000LL,
        .tv_nsec = (target_ns - now) % 1000000000LL
    };
    nanosleep(&req, NULL);
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Golden Recursive Algebra (GRA)
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

double gra_rn_closed(int n, double omega) {
    if (n < 1 || n > MAX_GRA_N) return 1.0;
    double phi = PHI;
    uint64_t F_n = FIB_TABLE[n];
    double two_n = pow(2.0, (double)n);
    uint64_t prod_p = 1;
    for (int k = 0; k < n; k++) prod_p *= PRIME_TABLE[k];
    double inside = phi * omega * (double)F_n * two_n * (double)prod_p;
    return sqrt(inside);
}

double gra_rn_recursive(int n, double omega, double *r_cache) {
    if (n < 1) return 1.0;
    if (r_cache[n] > 0.0) return r_cache[n];
    if (n == 1) {
        r_cache[1] = sqrt(4.0 * PHI * omega);
        return r_cache[1];
    }
    double r_prev = gra_rn_recursive(n-1, omega, r_cache);
    uint64_t F_n = FIB_TABLE[n];
    uint64_t F_nm1 = FIB_TABLE[n-1];
    uint64_t p_n = PRIME_TABLE[n-1];
    double multiplier = sqrt(2.0 * (double)p_n * ((double)F_n / (double)F_nm1));
    r_cache[n] = r_prev * multiplier;
    return r_cache[n];
}

double gra_add(double r_n, double r_m) {
    return sqrt(r_n * r_n + r_m * r_m);
}

// MPI & APA Core (unchanged from v30b, minimal)
typedef struct {
    uint64_t *words; size_t num_words; uint8_t sign;
} MPI;

#define APA_FLAG_SIGN_NEG (1 << 0)
#define APA_FLAG_IS_NAN   (1 << 1)
#define APA_FLAG_GOI      (1 << 2)
#define APA_FLAG_GUZ      (1 << 3)
#define APA_FLAG_CONSENSUS (1 << 4)

void mpi_init(MPI *m, size_t initial_words) { m->words = calloc(initial_words, sizeof(uint64_t)); m->num_words = initial_words; m->sign = 0; }
void mpi_free(MPI *m) { if (m->words) free(m->words); m->words = NULL; m->num_words = 0; }
void mpi_copy(MPI *dest, const MPI *src) {
    mpi_free(dest);
    dest->num_words = src->num_words;
    dest->words = malloc(src->num_words * sizeof(uint64_t));
    if (src->words && dest->words) memcpy(dest->words, src->words, src->num_words * sizeof(uint64_t));
    dest->sign = src->sign;
}
void mpi_set_value(MPI *m, uint64_t value, uint8_t sign) { if (m->words) m->words[0] = value; m->sign = sign; }

typedef struct {
    uint64_t *mantissa_words;
    MPI num_words_mantissa, exponent_mpi, source_of_infinity;
    size_t num_words;
    int64_t exponent;
    float base;
    int bits_mant, bits_exp;
    double phase, phase_vel, freq, amp_im;
    double r_harmonic;
} Slot4096;

// APA helpers (identical to previous refined version)
void ap_free(Slot4096 *slot) {
    if (slot) {
        if (slot->mantissa_words) free(slot->mantissa_words);
        mpi_free(&slot->exponent_mpi); mpi_free(&slot->num_words_mantissa); mpi_free(&slot->source_of_infinity);
        slot->num_words = 0;
    }
}

void ap_copy(Slot4096 *dest, const Slot4096 *src) {
    ap_free(dest);
    memcpy(dest, src, sizeof(Slot4096));
    dest->mantissa_words = malloc(src->num_words * sizeof(uint64_t));
    if (dest->mantissa_words && src->mantissa_words)
        memcpy(dest->mantissa_words, src->mantissa_words, src->num_words * sizeof(uint64_t));
    mpi_copy(&dest->exponent_mpi, &src->exponent_mpi);
    mpi_copy(&dest->num_words_mantissa, &src->num_words_mantissa);
    mpi_copy(&dest->source_of_infinity, &src->source_of_infinity);
}

double ap_to_double(const Slot4096 *slot) {
    if (!slot || slot->num_words == 0 || !slot->mantissa_words) return 0.0;
    return ((double)slot->mantissa_words[0] / (double)UINT64_MAX) * pow(2.0, (double)slot->exponent);
}

Slot4096* ap_from_double(double value, int bits_mant, int bits_exp) {
    Slot4096 *slot = malloc(sizeof(Slot4096));
    if (!slot) return NULL;
    memset(slot, 0, sizeof(Slot4096));
    slot->bits_mant = bits_mant; slot->bits_exp = bits_exp; slot->num_words = 1;
    slot->mantissa_words = calloc(1, sizeof(uint64_t));
    if (!slot->mantissa_words) { free(slot); return NULL; }
    if (value != 0.0) {
        int e; double m = frexp(fabs(value), &e);
        slot->mantissa_words[0] = (uint64_t)(m * (double)UINT64_MAX);
        slot->exponent = e;
    }
    mpi_init(&slot->exponent_mpi, 1);
    mpi_set_value(&slot->exponent_mpi, (uint64_t)llabs(slot->exponent), slot->exponent < 0 ? 1 : 0);
    return slot;
}

// Shift, normalize, add_legacy — identical to previous (omitted for brevity but include them as before)

void ap_shift_right_legacy(uint64_t *mantissa_words, size_t num_words, int64_t shift_amount) {
    if (shift_amount <= 0 || num_words == 0) return;
    if (shift_amount >= (int64_t)(num_words * 64)) { memset(mantissa_words, 0, num_words * sizeof(uint64_t)); return; }
    int64_t word_shift = shift_amount / 64;
    int bit_shift = (int)(shift_amount % 64);
    if (word_shift > 0) {
        for (int64_t i = num_words - 1; i >= word_shift; i--) mantissa_words[i] = mantissa_words[i - word_shift];
        memset(mantissa_words, 0, word_shift * sizeof(uint64_t));
    }
    if (bit_shift > 0) {
        int reverse_shift = 64 - bit_shift;
        for (size_t i = num_words - 1; i > 0; i--) {
            uint64_t upper_carry = mantissa_words[i - 1] << reverse_shift;
            mantissa_words[i] = (mantissa_words[i] >> bit_shift) | upper_carry;
        }
        mantissa_words[0] >>= bit_shift;
    }
}

void ap_normalize_legacy(Slot4096 *slot) {
    if (slot->num_words == 0) return;
    while (!(slot->mantissa_words[0] & MSB_MASK)) {
        if (slot->exponent <= -(1LL << (slot->bits_exp - 1))) { slot->state_flags |= APA_FLAG_GUZ; break; }
        uint64_t carry = 0;
        for (size_t i = slot->num_words - 1; i != (size_t)-1; i--) {
            uint64_t next_carry = (slot->mantissa_words[i] & MSB_MASK) ? 1 : 0;
            slot->mantissa_words[i] = (slot->mantissa_words[i] << 1) | carry;
            carry = next_carry;
        }
        slot->exponent--;
    }
    if (slot->mantissa_words[0] == 0) slot->exponent = 0;
}

void ap_add_legacy(Slot4096 *A, const Slot4096 *B) {
    if (A->num_words != B->num_words) return;
    Slot4096 B_aligned; ap_copy(&B_aligned, B);
    int64_t exp_diff = A->exponent - B_aligned.exponent;
    if (exp_diff > 0) { ap_shift_right_legacy(B_aligned.mantissa_words, B_aligned.num_words, exp_diff); B_aligned.exponent = A->exponent; }
    else if (exp_diff < 0) { ap_shift_right_legacy(A->mantissa_words, A->num_words, -exp_diff); A->exponent = B_aligned.exponent; }
    uint64_t carry = 0;
    for (size_t i = A->num_words - 1; i != (size_t)-1; i--) {
        uint64_t sum = A->mantissa_words[i] + B_aligned.mantissa_words[i] + carry;
        carry = (sum < A->mantissa_words[i] || (sum == A->mantissa_words[i] && carry)) ? 1 : 0;
        A->mantissa_words[i] = sum;
    }
    if (carry) {
        if (A->exponent >= (1LL << (A->bits_exp - 1))) A->state_flags |= APA_FLAG_GOI;
        else { A->exponent += 1; ap_shift_right_legacy(A->mantissa_words, A->num_words, 1); A->mantissa_words[0] |= MSB_MASK; }
    }
    ap_normalize_legacy(A);
    mpi_set_value(&A->exponent_mpi, (uint64_t)llabs(A->exponent), A->exponent < 0 ? 1 : 0);
    ap_free(&B_aligned);
}

// GRA Slot Init
Slot4096 slot_init_apa_gra(int bits_mant, int bits_exp, int n_dim, double omega, double *r_cache) {
    Slot4096 slot = {0};
    slot.bits_mant = bits_mant; slot.bits_exp = bits_exp;
    slot.num_words = (bits_mant + 63) / 64;
    slot.mantissa_words = calloc(slot.num_words, sizeof(uint64_t));
    mpi_init(&slot.exponent_mpi, 1); mpi_init(&slot.num_words_mantissa, 1); mpi_init(&slot.source_of_infinity, 1);

    if (slot.num_words > 0) {
        slot.mantissa_words[0] = GET_RANDOM_UINT64() | MSB_MASK;
    }

    int64_t exp_range = 1LL << bits_exp;
    slot.exponent = (rand() % exp_range) - (1LL << (bits_exp - 1));
    slot.base = PHI + get_normalized_rand() * 0.01;

    slot.phase = 2.0 * M_PI * get_normalized_rand();
    slot.phase_vel = 0.0;
    slot.freq = 1.0 + 0.5 * get_normalized_rand();
    slot.amp_im = 0.1 * get_normalized_rand();

    slot.r_harmonic = r_cache ? gra_rn_recursive(n_dim, omega, r_cache) : gra_rn_closed(n_dim, omega);

    mpi_set_value(&slot.exponent_mpi, (uint64_t)llabs(slot.exponent), slot.exponent < 0 ? 1 : 0);
    mpi_set_value(&slot.num_words_mantissa, (uint64_t)slot.num_words, 0);
    return slot;
}

// AnalogLink + GRA
typedef struct {
    double charge, charge_im, tension, potential, coupling, r_harmonic;
} AnalogLink;

void exchange_analog_links(AnalogLink *links, int rank, int size, int num_links) {
#if MPI_REAL
    // MPI code (same as before)
#else
    for (int i = 0; i < num_links; i++) {
        links[i].charge *= 0.95;
        links[i].charge_im *= 0.95;
    }
#endif
}

// GRA-enhanced RK4 with plasticity
typedef struct {
    double A_re, A_im, phase, phase_vel;
} ComplexState;

ComplexState compute_derivatives_gra(ComplexState state, double omega, const AnalogLink *neighbors, int num_neigh, double r_harmonic, double local_amp) {
    ComplexState deriv = {0};
    double sum_sin = 0.0, gra_sum = 0.0;
    double mod = BASE_GRA_MODULATION * (1.0 + 0.5 * sin(local_amp));  // gentle amplitude-dependent modulation

    deriv.A_re = -GAMMA * state.A_re;
    deriv.A_im = -GAMMA * state.A_im;

    for (int k = 0; k < num_neigh; k++) {
        double delta = neighbors[k].potential - state.phase;
        sum_sin += sin(delta);
        double combined = gra_add(r_harmonic, neighbors[k].r_harmonic);
        double factor = mod * combined / (1.0 + fabs(state.A_re) + fabs(state.A_im) + 1e-8);
        deriv.A_re += K_COUPLING * factor * cos(delta);
        deriv.A_im += K_COUPLING * factor * sin(delta);
        gra_sum += factor;
    }

    deriv.phase_vel = omega + K_COUPLING * sum_sin + 0.15 * gra_sum;
    deriv.phase = state.phase_vel;
    return deriv;
}

void rk4_step_gra(Slot4096 *slot, double dt, const AnalogLink *neighbors, int num_neigh, double omega) {
    double local_amp = sqrt(pow(ap_to_double(slot), 2) + pow(slot->amp_im, 2));
    ComplexState state = {ap_to_double(slot), slot->amp_im, slot->phase, slot->phase_vel};

    ComplexState k1 = compute_derivatives_gra(state, omega, neighbors, num_neigh, slot->r_harmonic, local_amp);
    // k2, k3, k4 (standard RK4 half/full steps - identical structure as previous)
    ComplexState temp = state; temp.A_re += dt * k1.A_re / 2; temp.A_im += dt * k1.A_im / 2; temp.phase += dt * k1.phase / 2; temp.phase_vel += dt * k1.phase_vel / 2;
    ComplexState k2 = compute_derivatives_gra(temp, omega, neighbors, num_neigh, slot->r_harmonic, local_amp);

    temp = state; temp.A_re += dt * k2.A_re / 2; temp.A_im += dt * k2.A_im / 2; temp.phase += dt * k2.phase / 2; temp.phase_vel += dt * k2.phase_vel / 2;
    ComplexState k3 = compute_derivatives_gra(temp, omega, neighbors, num_neigh, slot->r_harmonic, local_amp);

    temp = state; temp.A_re += dt * k3.A_re; temp.A_im += dt * k3.A_im; temp.phase += dt * k3.phase; temp.phase_vel += dt * k3.phase_vel;
    ComplexState k4 = compute_derivatives_gra(temp, omega, neighbors, num_neigh, slot->r_harmonic, local_amp);

    state.A_re += dt / 6 * (k1.A_re + 2*k2.A_re + 2*k3.A_re + k4.A_re);
    state.A_im += dt / 6 * (k1.A_im + 2*k2.A_im + 2*k3.A_im + k4.A_im);
    state.phase += dt / 6 * (k1.phase + 2*k2.phase + 2*k3.phase + k4.phase);
    state.phase_vel += dt / 6 * (k1.phase_vel + 2*k2.phase_vel + 2*k3.phase_vel + k4.phase_vel);

    double A = sqrt(state.A_re*state.A_re + state.A_im*state.A_im);
    A *= exp(-LAMBDA * dt);
    if (A > SAT_LIMIT) A = SAT_LIMIT;
    A += NOISE_SIGMA * (2.0 * get_normalized_rand() - 1.0);

    double norm = sqrt(state.A_re*state.A_re + state.A_im*state.A_im);
    if (norm > 1e-10) {
        state.A_re = (state.A_re / norm) * A;
        state.A_im = (state.A_im / norm) * A;
    }

    state.phase = fmod(state.phase, 2.0 * M_PI);
    if (state.phase < 0) state.phase += 2.0 * M_PI;

    Slot4096 *new_amp = ap_from_double(state.A_re, slot->bits_mant, slot->bits_exp);
    if (new_amp) {
        ap_copy(slot, new_amp);
        ap_free(new_amp);
        free(new_amp);
    }
    slot->amp_im = state.A_im;
    slot->phase = state.phase;
    slot->phase_vel = state.phase_vel;

    // Gentle GRA plasticity (brain-like adaptation)
    slot->r_harmonic += GRA_PLASTICITY * (local_amp - 0.5) * (slot->r_harmonic > 10 ? 0.1 : 1.0);
    if (slot->r_harmonic < 1.0) slot->r_harmonic = 1.0;
}

typedef struct { Slot4096 *slots; size_t allocated; } HDGLChunk;

typedef struct {
    HDGLChunk **chunks;
    int num_chunks, num_instances, slots_per_instance;
    double omega, time;
    int consensus_steps;
    double phase_var;
    int64_t last_checkpoint_ns;
    double r_cache[MAX_GRA_N + 1];
} HDGLLattice;

// lattice_init, lattice_get_chunk, lattice_get_slot, detect_harmonic_consensus, lattice_integrate_rk4, checkpoint, lattice_free, bootloader — use the structure from the previous refined version, calling slot_init_apa_gra and rk4_step_gra.

int main(int argc, char *argv[]) {
    srand(time(NULL));
    printf("=== HDGL Analog Mainnet v3.1 Refined GRA Bridge ===\n\n");

    HDGLLattice *lat = lattice_init(4096, 4);
    if (!lat) return 1;

    CheckpointManager *ckpt = checkpoint_init();
    bootloader_init_lattice(lat, 800, ckpt);   // longer run for better balance observation

    printf("\nFinal GRA balance check:\n");
    for (int n = 1; n <= 6; n++) printf("  r_%d ≈ %.6f\n", n, gra_rn_closed(n, lat->omega));

    // cleanup
    checkpoint_free(ckpt);
    lattice_free(lat);
    printf("\n=== Refined GRA Bridge Complete ===\n");
    return 0;
}
Fusion with your FastSlot / Slot4096 engine
Replace GRA with learned spectral kernel (this gets wild)
Push this into a fully self-adapting GPU search system
Direct LL-residue → reward signal (this is huge)
you drive
Replace 32-bit LL-lite with 4096-bit warp-lattice cooperative version