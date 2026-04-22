// ============================================================================
// HDGL Analog Mainnet v3.1b — Open-Recursive Throughput Hemisphere
// ============================================================================
//
// HEMISPHERE ROLE  : THROUGHPUT / EXPLORER
// RECURSIVE FORM   : OPEN — GRA r_harmonic is PLASTIC (evolves per slot)
// INTEGRATION      : Euler, single-pass, Structure-of-Arrays (SoA) layout
// PRECISION        : double for dynamics; APA not used in inner loop
// LL PROBE         : 32-bit LL-lite proxy (4 squaring iters per step)
// SPECTRAL KERNEL  : 4 harmonics, PER-SLOT learned weights (Hebbian)
// BRIDGE ROLE      : Emits BridgeCandidate to v31; consumes BridgeReward
// GRA COUPLING     : Open-recursive plasticity — r_harmonic grows/shrinks
//                    bounded: r in [1, R_MAX_PLASTIC]
//
// Design contract (this file):
//   — SoA layout eliminates struct pointer chasing in hot loop
//   — Global dt computed once per step, not per slot
//   — LL-lite 32-bit (4 iters) as cheap continuous reward signal
//   — Per-slot spectral weights with Hebbian updates + clamping
//   — Candidate emission uses accumulator threshold + residue gate
//   — Bridge reward injection adjusts local r_harmonic (open feedback)
//   — Throughput > Accuracy: explore, emit, let v31 verify
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

// --- GRA — Open-Recursive (v31b: plastic r_harmonic per slot) ---
#define BASE_GRA_MODULATION 0.18
#define GRA_PLASTICITY      0.008     // per-step evolution rate
#define R_MAX_PLASTIC       1000.0    // hard ceiling on r_harmonic
#define MAX_GRA_N           16

static const uint64_t FIB_TABLE_B[MAX_GRA_N + 1] = {
    0,1,1,2,3,5,8,13,21,34,55,89,144,233,377,610,987};
static const uint64_t PRIME_TABLE_B[MAX_GRA_N] = {
    2,3,5,7,11,13,17,19,23,29,31,37,41,43,47,53};

// --- Spectral Kernel (v31b: 4 harmonics, per-slot local weights) ---
#define SPECTRAL_N_B       4

// --- LL-lite ---
#define LL_LITE_ITERS      4      // 32-bit proxy iterations per step

// --- Candidate emission thresholds ---
#define CAND_ACCUM_THRESH  5.0
#define CAND_RESIDUE_MAX   0.02f
#define CAND_AMP_MIN       0.6

// --- Checkpoint ---
#define CHECKPOINT_INTERVAL_B 100
#define SNAPSHOT_MAX_B        10
#define SNAPSHOT_DECAY_B      0.95

// --- Bridge ---
#define BRIDGE_CAPACITY    256

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

static double get_normalized_rand_b(void) { return (double)rand() / (double)RAND_MAX; }

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Timing Primitives
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

int64_t get_rtc_ns_b(void) {
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

void rtc_sleep_until_b(int64_t target_ns) {
    int64_t now = get_rtc_ns_b();
    if (target_ns <= now) return;
    struct timespec req = {
        .tv_sec  = (target_ns - now) / 1000000000LL,
        .tv_nsec = (target_ns - now) % 1000000000LL
    };
    nanosleep(&req, NULL);
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// GRA — Closed-form seed (for initialisation only; runtime uses plastic r_h)
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

static double gra_rn_seed(int n, double omega) {
    if (n < 1 || n > MAX_GRA_N) return 1.0;
    uint64_t F_n   = FIB_TABLE_B[n];
    double   two_n = pow(2.0, (double)n);
    uint64_t prod_p = 1;
    for (int k = 0; k < n; k++) prod_p *= PRIME_TABLE_B[k];
    double inside = PHI * omega * (double)F_n * two_n * (double)prod_p;
    return sqrt(inside > 0.0 ? inside : 1e-12);
}

static inline double gra_add_b(double r_n, double r_m) {
    return sqrt(r_n * r_n + r_m * r_m);
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Bridge — same interface as v31 (compatible ring buffer)
// Both hemispheres share this struct definition; in a header-based build
// this would live in hdgl_bridge_v31.h
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

typedef struct {
    int      slot_idx;
    double   phase;
    double   amplitude;
    double   r_harmonic;
    double   score;
    uint64_t ll_seed;
} BridgeCandidate_b;

typedef struct {
    int    slot_idx;
    double reward;
    double ll_residue;
    int    verified;    // +1 confirmed, -1 rejected, 0 pending
} BridgeReward_b;

typedef struct {
    BridgeCandidate_b cands[BRIDGE_CAPACITY];
    BridgeReward_b    rewards[BRIDGE_CAPACITY];
    int cand_head, cand_tail;
    int rew_head,  rew_tail;
} BridgeBuffer_b;

static BridgeBuffer_b g_bridge_b;

static void bridge_b_init(void) { memset(&g_bridge_b, 0, sizeof(g_bridge_b)); }

static int bridge_b_emit_candidate(int idx, double phase, double amp,
                                    double r_h, double score) {
    int next = (g_bridge_b.cand_tail + 1) % BRIDGE_CAPACITY;
    if (next == g_bridge_b.cand_head) return 0;
    g_bridge_b.cands[g_bridge_b.cand_tail] = (BridgeCandidate_b){
        .slot_idx   = idx,
        .phase      = phase,
        .amplitude  = amp,
        .r_harmonic = r_h,
        .score      = score,
        .ll_seed    = ((uint64_t)(amp  * (double)UINT64_MAX))
                     ^ ((uint64_t)(phase * 1e9))
                     ^ ((uint64_t)(r_h   * 1e6))
    };
    g_bridge_b.cand_tail = next;
    return 1;
}

static int bridge_b_pop_reward(BridgeReward_b *out) {
    if (g_bridge_b.rew_head == g_bridge_b.rew_tail) return 0;
    *out = g_bridge_b.rewards[g_bridge_b.rew_head];
    g_bridge_b.rew_head = (g_bridge_b.rew_head + 1) % BRIDGE_CAPACITY;
    return 1;
}

// v31 calls this to inject its verdict back
void bridge_b_inject_reward(int slot_idx, double reward, double residue,
                              int verified) {
    int next = (g_bridge_b.rew_tail + 1) % BRIDGE_CAPACITY;
    if (next == g_bridge_b.rew_head) return;
    g_bridge_b.rewards[g_bridge_b.rew_tail] = (BridgeReward_b){
        slot_idx, reward, residue, verified};
    g_bridge_b.rew_tail = next;
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// FastLattice — Structure-of-Arrays (SoA)
// All per-slot fields are flat arrays; no struct pointer chasing in hot loop
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

typedef struct {
    // --- dynamics (SoA) ---
    double   *A_re;          // real amplitude
    double   *A_im;          // imaginary amplitude
    double   *phase;         // oscillator phase φ
    double   *phase_vel;     // dφ/dt
    double   *r_harmonic;    // OPEN: plastic GRA radius per slot

    // --- LL-lite proxy ---
    uint32_t *ll_state;      // 32-bit LL squaring state

    // --- reward accumulator ---
    double   *reward_accum;  // temporal reward with 0.9 decay

    // --- per-slot spectral weights (4 harmonics each) ---
    // layout: w_cos[i*SPECTRAL_N_B + k]  for slot i, harmonic k
    float    *w_cos;         // cosine weights [num_slots * SPECTRAL_N_B]
    float    *w_sin;         // sine   weights [num_slots * SPECTRAL_N_B]

    // --- global spectral amplitude weights (shared, momentum-updated) ---
    float     w_amp_self;
    float     w_amp_neigh;

    // --- lattice metadata ---
    int       num_slots;
    int       slots_per_instance;
    double    omega;
    double    time;
    int       consensus_steps;
    double    phase_var;
    double    dt_global;
    int64_t   last_checkpoint_ns;
} FastLattice;

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// FastLattice Init
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

FastLattice* fast_lattice_init(int num_instances, int slots_per_instance) {
    FastLattice *lat = malloc(sizeof(FastLattice));
    if (!lat) return NULL;
    memset(lat, 0, sizeof(FastLattice));

    int N = num_instances * slots_per_instance;
    lat->num_slots          = N;
    lat->slots_per_instance = slots_per_instance;
    lat->omega              = 1.0;
    lat->time               = 0.0;
    lat->consensus_steps    = 0;
    lat->phase_var          = 1e6;
    lat->dt_global          = 1.0 / 32768.0;
    lat->last_checkpoint_ns = get_rtc_ns_b();
    lat->w_amp_self         = 0.05f;
    lat->w_amp_neigh        = 0.05f;

    // Allocate all SoA arrays
    lat->A_re        = malloc(N * sizeof(double));
    lat->A_im        = malloc(N * sizeof(double));
    lat->phase       = malloc(N * sizeof(double));
    lat->phase_vel   = malloc(N * sizeof(double));
    lat->r_harmonic  = malloc(N * sizeof(double));
    lat->ll_state    = malloc(N * sizeof(uint32_t));
    lat->reward_accum= malloc(N * sizeof(double));
    lat->w_cos       = malloc(N * SPECTRAL_N_B * sizeof(float));
    lat->w_sin       = malloc(N * SPECTRAL_N_B * sizeof(float));

    if (!lat->A_re || !lat->A_im || !lat->phase || !lat->phase_vel ||
        !lat->r_harmonic || !lat->ll_state || !lat->reward_accum ||
        !lat->w_cos || !lat->w_sin) {
        fprintf(stderr, "[v31b] FastLattice allocation failed\n");
        free(lat->A_re); free(lat->A_im); free(lat->phase);
        free(lat->phase_vel); free(lat->r_harmonic); free(lat->ll_state);
        free(lat->reward_accum); free(lat->w_cos); free(lat->w_sin);
        free(lat);
        return NULL;
    }

    // Seed per-slot state
    for (int i = 0; i < N; i++) {
        int n_dim           = (i % MAX_GRA_N) + 1;
        lat->A_re[i]        = get_normalized_rand_b() * 0.5;
        lat->A_im[i]        = get_normalized_rand_b() * 0.1;
        lat->phase[i]       = 2.0 * M_PI * get_normalized_rand_b();
        lat->phase_vel[i]   = 0.0;
        // OPEN: seed from closed-form, then let plasticity run
        lat->r_harmonic[i]  = gra_rn_seed(n_dim, lat->omega);
        lat->ll_state[i]    = (uint32_t)rand() | 1u;
        lat->reward_accum[i]= 0.0;

        // Init spectral weights to small uniform values
        for (int k = 0; k < SPECTRAL_N_B; k++) {
            lat->w_cos[i * SPECTRAL_N_B + k] =  0.05f;
            lat->w_sin[i * SPECTRAL_N_B + k] =  0.00f;
        }
    }

    return lat;
}

void fast_lattice_free(FastLattice *lat) {
    if (!lat) return;
    free(lat->A_re);   free(lat->A_im);    free(lat->phase);
    free(lat->phase_vel); free(lat->r_harmonic); free(lat->ll_state);
    free(lat->reward_accum); free(lat->w_cos); free(lat->w_sin);
    free(lat);
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// LL-Lite 32-bit proxy  (4 squaring iterations per slot per step)
// s_{n+1} = (s_n^2 - 2) mod 2^32
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

static inline uint32_t ll_step32(uint32_t s) {
    uint64_t sq = (uint64_t)s * (uint64_t)s;
    return (uint32_t)(sq) - 2u;
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Euler Step — single-pass, all fixes applied
//   • Global dt (not mutated per slot)
//   • SoA reads (cache-friendly)
//   • Bounded GRA coupling
//   • Per-slot Hebbian spectral update
//   • Open GRA plasticity with ceiling
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

void fast_lattice_step(FastLattice *lat) {
    int    N  = lat->num_slots;
    int    S  = lat->slots_per_instance;
    double dt = lat->dt_global;   // global dt: set once, read-only inside loop

    double amp_sq_sum = 0.0;
    int    active     = 0;

    for (int i = 0; i < N; i++) {
        // Load state (SoA reads — sequential, cache-friendly)
        double A_re    = lat->A_re[i];
        double A_im    = lat->A_im[i];
        double phase   = lat->phase[i];
        double phvel   = lat->phase_vel[i];
        double r_h     = lat->r_harmonic[i];

        double local_amp = sqrt(A_re*A_re + A_im*A_im);

        // LL-lite: 4 squaring iterations, cheap continuous reward proxy
        uint32_t s = lat->ll_state[i];
        for (int k = 0; k < LL_LITE_ITERS; k++) s = ll_step32(s);
        lat->ll_state[i] = s;
        float residue = (float)s / 4294967295.0f;  // normalise to [0,1]

        // Neighbours: 4-connected (von Neumann; fast path)
        int ni[4] = {
            (i - 1 + N) % N,
            (i + 1)     % N,
            (i - S + N) % N,
            (i + S)     % N
        };

        double dA_re   = -GAMMA * A_re;
        double dA_im   = -GAMMA * A_im;
        double sum_sin = 0.0;
        double gra_sum = 0.0;

        float *wc = &lat->w_cos[i * SPECTRAL_N_B];
        float *ws = &lat->w_sin[i * SPECTRAL_N_B];

        // Primary coupling direction for Hebbian update (neighbour 1 = right)
        double dphi_primary = lat->phase[ni[1]] - phase;

        for (int j = 0; j < 4; j++) {
            int    n       = ni[j];
            double dphi    = lat->phase[n] - phase;
            double n_A_re  = lat->A_re[n];
            double n_A_im  = lat->A_im[n];
            double n_amp   = sqrt(n_A_re*n_A_re + n_A_im*n_A_im);

            // Per-slot spectral kernel (4 harmonics)
            double spec = 0.0;
            for (int k = 0; k < SPECTRAL_N_B; k++) {
                double kd = (double)(k + 1) * dphi;
                spec += (double)wc[k] * cos(kd);
                spec += (double)ws[k] * sin(kd);
            }
            spec += (double)lat->w_amp_self  * local_amp;
            spec += (double)lat->w_amp_neigh * n_amp;

            // GRA coupling — OPEN: r_h is plastic, but coupling bounded
            double combined = gra_add_b(r_h, lat->r_harmonic[n]);
            double gra_fac  = BASE_GRA_MODULATION * combined / (1.0 + combined);

            double factor = spec + gra_fac;

            sum_sin += sin(dphi);
            dA_re   += factor * cos(dphi);
            dA_im   += factor * sin(dphi);
            gra_sum += gra_fac;
        }

        double dphvel = lat->omega + K_COUPLING * sum_sin + 0.15 * gra_sum;

        // Euler update
        A_re  += dt * dA_re;
        A_im  += dt * dA_im;
        phase += dt * phvel;
        phvel += dt * dphvel;

        // Amplitude damping + saturation + noise
        double A = sqrt(A_re*A_re + A_im*A_im);
        A *= exp(-LAMBDA * dt);
        if (A > SAT_LIMIT) A = SAT_LIMIT;
        A += NOISE_SIGMA * (2.0 * get_normalized_rand_b() - 1.0);

        double norm = sqrt(A_re*A_re + A_im*A_im);
        if (norm > 1e-10) { A_re = (A_re / norm) * A; A_im = (A_im / norm) * A; }

        // Phase wrap
        phase = fmod(phase, 2.0 * M_PI);
        if (phase < 0.0) phase += 2.0 * M_PI;

        // GRA PLASTICITY — OPEN RECURSIVE: r_harmonic evolves
        // Slope: positive when amplitude is high (growing branch)
        //        negative when amplitude is low (pruning branch)
        double plastic_step = GRA_PLASTICITY * (local_amp - 0.5)
                            * (r_h > 50.0 ? 0.05 : 1.0);  // slowdown near ceiling
        r_h += plastic_step;
        if (r_h < 1.0)         r_h = 1.0;
        if (r_h > R_MAX_PLASTIC) r_h = R_MAX_PLASTIC;

        // Reward signal: LL-lite residue + phase coherence + amplitude
        float reward = 1.0f / (1.0f + residue)
                     + 0.25f * (float)fabs(sum_sin)
                     + 0.15f * (float)local_amp;

        // Per-slot Hebbian spectral weight update (clamped)
        const float lr   = 1e-3f;
        const float clmp = 2.0f;
        for (int k = 0; k < SPECTRAL_N_B; k++) {
            double kd = (double)(k + 1) * dphi_primary;
            wc[k] += lr * reward * (float)cos(kd);
            ws[k] += lr * reward * (float)sin(kd);
            if (wc[k] >  clmp) wc[k] =  clmp;
            if (wc[k] < -clmp) wc[k] = -clmp;
            if (ws[k] >  clmp) ws[k] =  clmp;
            if (ws[k] < -clmp) ws[k] = -clmp;
        }

        // Global amplitude weights (momentum update, once via representative slot)
        if (i == 0) {
            lat->w_amp_self  = 0.95f * lat->w_amp_self
                             + 0.05f * lr * reward * (float)local_amp;
            lat->w_amp_neigh = 0.95f * lat->w_amp_neigh
                             + 0.05f * lr * reward * (float)(A / (local_amp + 1e-8));
        }

        // Temporal reward accumulator (0.9 decay — trajectory, not noise)
        lat->reward_accum[i] = 0.9 * lat->reward_accum[i] + (double)reward;

        // Write back (SoA writes)
        lat->A_re[i]       = A_re;
        lat->A_im[i]       = A_im;
        lat->phase[i]      = phase;
        lat->phase_vel[i]  = phvel;
        lat->r_harmonic[i] = r_h;

        amp_sq_sum += local_amp * local_amp;
        active++;
    }

    // Global dt update — once, after all slots
    if (active > 0) {
        double rms = sqrt(amp_sq_sum / active);
        if      (rms > ADAPT_THRESH)       lat->dt_global *= PHI;
        else if (rms < ADAPT_THRESH / PHI) lat->dt_global /= PHI;
        if (lat->dt_global < 1e-6) lat->dt_global = 1e-6;
        if (lat->dt_global > 0.1)  lat->dt_global = 0.1;
    }

    lat->omega += 0.01 * dt;
    lat->time  += dt;
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Consensus Detection (lightweight: phase variance only)
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

void fast_lattice_detect_consensus(FastLattice *lat) {
    int    N     = lat->num_slots;
    double mean  = 0.0;
    int    count = N;

    for (int i = 0; i < N; i++) mean += lat->phase[i];
    mean /= count;

    double var = 0.0;
    for (int i = 0; i < N; i++) {
        double d = lat->phase[i] - mean;
        if (d >  M_PI) d -= 2.0 * M_PI;
        if (d < -M_PI) d += 2.0 * M_PI;
        var += d * d;
    }
    lat->phase_var = sqrt(var / count);

    if (lat->phase_var < CONSENSUS_EPS) {
        lat->consensus_steps++;
        if (lat->consensus_steps >= CONSENSUS_N) {
            printf("[v31b CONSENSUS] Domain locked t=%.4f var=%.8f\n",
                   lat->time, lat->phase_var);
            lat->consensus_steps = 0;
        }
    } else {
        lat->consensus_steps = 0;
    }
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Candidate Emission — emit high-confidence slots to bridge for v31 verification
// Gate: accumulated reward > threshold AND LL residue < gate AND amplitude > min
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

int fast_lattice_emit_candidates(FastLattice *lat) {
    int N = lat->num_slots;
    int emitted = 0;

    for (int i = 0; i < N; i++) {
        float residue = (float)lat->ll_state[i] / 4294967295.0f;
        double amp    = sqrt(lat->A_re[i]*lat->A_re[i]
                           + lat->A_im[i]*lat->A_im[i]);

        if (lat->reward_accum[i] > CAND_ACCUM_THRESH &&
            residue              < CAND_RESIDUE_MAX   &&
            amp                  > CAND_AMP_MIN) {

            int ok = bridge_b_emit_candidate(
                i, lat->phase[i], amp,
                lat->r_harmonic[i], lat->reward_accum[i]);

            if (ok) {
                // Decay accumulator after emission (prevents spam)
                lat->reward_accum[i] *= 0.5;
                emitted++;
            }
        }
    }
    return emitted;
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Bridge Reward Processing — apply v31's verdicts back to local state
// Verified (+1): reinforce r_harmonic (open branch grows)
// Rejected (-1): dampen r_harmonic   (open branch prunes)
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

int fast_lattice_apply_rewards(FastLattice *lat, int max_per_cycle) {
    BridgeReward_b rw;
    int applied = 0;

    while (applied < max_per_cycle && bridge_b_pop_reward(&rw)) {
        int idx = rw.slot_idx;
        if (idx < 0 || idx >= lat->num_slots) { applied++; continue; }

        if (rw.verified == 1) {
            // Reinforce: grow r_harmonic slightly (open recursive reward)
            lat->r_harmonic[idx] *= (1.0 + 0.05 * rw.reward);
            if (lat->r_harmonic[idx] > R_MAX_PLASTIC)
                lat->r_harmonic[idx] = R_MAX_PLASTIC;
            // Boost reward accumulator
            lat->reward_accum[idx] += rw.reward;
        } else if (rw.verified == -1) {
            // Prune: shrink r_harmonic (false positive penalty)
            lat->r_harmonic[idx] *= (1.0 - 0.02 * (1.0 - rw.reward));
            if (lat->r_harmonic[idx] < 1.0) lat->r_harmonic[idx] = 1.0;
            // Penalise accumulator
            lat->reward_accum[idx] *= 0.7;
        }
        applied++;
    }
    return applied;
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
} CheckpointMeta_b;

typedef struct {
    CheckpointMeta_b *snapshots;
    int count, capacity;
} CheckpointManager_b;

CheckpointManager_b* checkpoint_b_init(void) {
    CheckpointManager_b *m = malloc(sizeof(CheckpointManager_b));
    m->snapshots = malloc(SNAPSHOT_MAX_B * sizeof(CheckpointMeta_b));
    m->count     = 0;
    m->capacity  = SNAPSHOT_MAX_B;
    return m;
}

void checkpoint_b_add(CheckpointManager_b *m, int evo, FastLattice *lat) {
    if (m->count >= m->capacity) {
        int mi = 0; double mw = m->snapshots[0].weight;
        for (int i = 1; i < m->count; i++)
            if (m->snapshots[i].weight < mw) { mw = m->snapshots[i].weight; mi = i; }
        for (int i = mi; i < m->count - 1; i++) m->snapshots[i] = m->snapshots[i+1];
        m->count--;
    }
    m->snapshots[m->count++] = (CheckpointMeta_b){
        evo, get_rtc_ns_b(), lat->phase_var, lat->omega, 1.0};
    for (int i = 0; i < m->count - 1; i++) m->snapshots[i].weight *= SNAPSHOT_DECAY_B;
    printf("[v31b Checkpoint] evo=%d var=%.6f omega=%.4f\n",
           evo, lat->phase_var, lat->omega);
}

void checkpoint_b_free(CheckpointManager_b *m) {
    if (m) { free(m->snapshots); free(m); }
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Bootloader
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

void bootloader_v31b(FastLattice *lat, int steps, CheckpointManager_b *ckpt) {
    printf("[v31b Bootloader] Open-Recursive Throughput Hemisphere init...\n");
    printf("[v31b Bootloader] %d slots (Euler, SoA, LL-lite)\n", lat->num_slots);

    int64_t step_ns      = 30517;
    int64_t next_step_ns = get_rtc_ns_b() + step_ns;

    for (int i = 0; i < steps; i++) {
        fast_lattice_step(lat);

        // Emit candidates every 16 steps
        if (i % 16 == 0) fast_lattice_emit_candidates(lat);

        // Apply bridge rewards
        fast_lattice_apply_rewards(lat, 4);

        // Consensus detection every 32 steps (cheap)
        if (i % 32 == 0) fast_lattice_detect_consensus(lat);

        if (i % CHECKPOINT_INTERVAL_B == 0 && i > 0)
            checkpoint_b_add(ckpt, i, lat);

        rtc_sleep_until_b(next_step_ns);
        next_step_ns += step_ns;
    }

    printf("[v31b Bootloader] %d steps | dt=%.8f | omega=%.6f | var=%.6f\n",
           steps, lat->dt_global, lat->omega, lat->phase_var);
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Main
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

int main(int argc, char *argv[]) {
    (void)argc; (void)argv;
    srand((unsigned)time(NULL));
    bridge_b_init();

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

    printf("=== HDGL Analog Mainnet v3.1b — Open-Recursive Throughput Hemisphere ===\n\n");

    // 4096 instances × 4 slots = 16384 total (manageable for CPU demo)
    FastLattice *lat = fast_lattice_init(4096, 4);
    if (!lat) { fprintf(stderr, "Fatal: FastLattice init failed\n"); return 1; }

    CheckpointManager_b *ckpt = checkpoint_b_init();
    bootloader_v31b(lat, 800, ckpt);

    printf("\nFirst 8 slots (post-evolution):\n");
    for (int i = 0; i < 8; i++) {
        double amp     = sqrt(lat->A_re[i]*lat->A_re[i] + lat->A_im[i]*lat->A_im[i]);
        float  residue = (float)lat->ll_state[i] / 4294967295.0f;
        printf("  Slot %2d: |A|=%.6e  φ=%.3f  r_h=%.4f  "
               "accum=%.3f  res=%.4f\n",
               i, amp, lat->phase[i], lat->r_harmonic[i],
               lat->reward_accum[i], residue);
    }

    printf("\nSpectral weights slot 0 (4 harmonics):\n");
    for (int k = 0; k < SPECTRAL_N_B; k++)
        printf("  w_cos[%d]=%.4f  w_sin[%d]=%.4f\n",
               k, lat->w_cos[k], k, lat->w_sin[k]);

    printf("\nBridge candidates pending: %d\n",
           (g_bridge_b.cand_tail - g_bridge_b.cand_head + BRIDGE_CAPACITY)
           % BRIDGE_CAPACITY);

    checkpoint_b_free(ckpt);
    fast_lattice_free(lat);

#ifdef USE_DS3231
    if (i2c_fd >= 0) i2c_close(i2c_fd);
#endif

    printf("\n=== v3.1b OPEN-RECURSIVE THROUGHPUT HEMISPHERE OPERATIONAL ===\n");
    return 0;
}
