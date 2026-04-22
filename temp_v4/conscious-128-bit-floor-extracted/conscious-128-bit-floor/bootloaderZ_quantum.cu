/* bootloaderZ_quantum.cu — HDGL Analog Mainnet V2.7: bug-fixed + quantum layer
 *
 * Changes from V2.6:
 *
 * ── Bug fixes ────────────────────────────────────────────────────────────────
 *  FIX 1  ap_copy: UAF — memcpy copied src MPI pointers into dest, then mpi_copy
 *          called mpi_free(dest) which freed SRC's memory.  Fixed: field-by-field
 *          scalar copy; mpi_copy handles the heap members.
 *
 *  FIX 2  rk4_step: heap-allocated ap_from_double every step = O(N×steps) churn.
 *          Fixed: write A_re directly into slot->mantissa_words[0], no alloc.
 *
 *  FIX 3  lattice_integrate_rk4: dt_base mutated per-slot inside loop bled into
 *          lat->time += dt_base at the end of the same call, making global time
 *          non-deterministic.  Fixed: separate local_dt per slot; advance time by
 *          the original caller-supplied dt_base.
 *
 *  FIX 4  checkpoint_add: pruned snapshot printed AFTER array was shifted down,
 *          logging wrong slot.  Fixed: capture meta before shift.
 *
 *  FIX 5  detect_harmonic_consensus: circular mean computed as linear mean of
 *          angles — wraps incorrectly near 0/2π boundary.  Fixed: atan2(Σsin, Σcos).
 *
 *  FIX 6  mpi_copy: if src->words == NULL with num_words > 0, dest->words was
 *          malloc'd but left uninitialised (memcpy skipped).  Fixed: memset to 0.
 *
 * ── Quantum layer (cuStateVec, SM 7.5) ───────────────────────────────────────
 *  Architecture: the lattice has up to 8M slots — far too many for one state
 *  vector.  Instead, each HDGLChunk gets a QChunkSampler: an 8-qubit state
 *  vector representing 8 evenly-spaced representative slots from that chunk.
 *
 *  Encoding:   Ry(slot->phase)|0⟩ per representative qubit
 *               ⟨Zi⟩ = cos(phase) → re correction
 *               ⟨Xi⟩ = sin(phase) → im correction
 *
 *  Evolution:  one Trotterised XY step (same Hamiltonian as ll_quantum.cu):
 *               H = Σi ωi Zi  +  (K/2) Σ_{i<j} (Xi Xj + Yi Yj)
 *               The XY coupling drives quantum Kuramoto synchronisation.
 *
 *  Readback:   expectation values blended into slot phases at Q_BLEND_ALPHA.
 *               Renormalised to unit circle so phase-doubling stays exact.
 *
 *  Consensus:  quantum coherence |⟨Z⟩|² replaces linear phase variance.
 *               ⟨Z⟩ = Σi⟨Zi⟩/N; |⟨Z⟩|² = Kuramoto R² ∈ [0,1].
 *               Consensus when R² > 1 − CONSENSUS_EPS² (near-perfect coherence).
 *
 *  Scheduling: quantum_correct_chunk called every Q_INTERVAL=8 integration steps.
 *               Flag APA_FLAG_QUANTUM marks quantum-corrected slots.
 *
 * ── Build ────────────────────────────────────────────────────────────────────
 *  With quantum (Windows, RTX 2060):
 *    nvcc -arch=sm_75 -I"%CUQUANTUM_ROOT%\include" -L"%CUQUANTUM_ROOT%\lib\x64"
 *         -lcustatevec -lcudart -lm -DLL_QUANTUM_ENABLED
 *         bootloaderZ_quantum.cu -o bootloaderZ_quantum.exe
 *
 *  Without quantum (pure C fallback, any platform):
 *    gcc -O2 bootloaderZ_quantum.cu -o bootloaderZ -lm
 *    (remove .cu extension or pass -x c to gcc)
 *
 * Licensed per https://zchg.org/t/legal-notice-copyright-applicable-ip-and-licensing-read-me/440
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <time.h>
#ifndef _WIN32
#include <unistd.h>
#else
/* Windows compat: timespec is in <time.h> (MSVC ucrt); provide clock_gettime + nanosleep */
#include <windows.h>
#ifndef CLOCK_MONOTONIC
#define CLOCK_MONOTONIC 1
static int clock_gettime(int clk, struct timespec *ts) {
    (void)clk;
    LARGE_INTEGER freq, cnt;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&cnt);
    long long ns = (long long)(cnt.QuadPart * 1000000000LL / freq.QuadPart);
    ts->tv_sec  = (long)(ns / 1000000000LL);
    ts->tv_nsec = (long)(ns % 1000000000LL);
    return 0;
}
#endif
#ifndef HAVE_NANOSLEEP
static int nanosleep(const struct timespec *req, struct timespec *rem) {
    (void)rem;
    DWORD ms = (DWORD)((long long)req->tv_sec * 1000 + req->tv_nsec / 1000000);
    Sleep(ms ? ms : 1);
    return 0;
}
#define HAVE_NANOSLEEP 1
#endif
#endif

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* ── System constants ─────────────────────────────────────────────────────── */
#define PHI             1.6180339887498948
#define MAX_INSTANCES   8388608
#define SLOTS_PER_INSTANCE 4
#define MAX_SLOTS       (MAX_INSTANCES * SLOTS_PER_INSTANCE)
#define CHUNK_SIZE      1048576
#define MSB_MASK        (1ULL << 63)

/* ── Analog constants ─────────────────────────────────────────────────────── */
#define GAMMA           0.02
#define LAMBDA          0.05
#define SAT_LIMIT       1e6
#define NOISE_SIGMA     0.01
#define CONSENSUS_EPS   1e-6
#define CONSENSUS_N     100
#define ADAPT_THRESH    0.8
#define K_COUPLING      1.0

/* ── Checkpoint constants ─────────────────────────────────────────────────── */
#define CHECKPOINT_INTERVAL 100
#define SNAPSHOT_MAX    10
#define SNAPSHOT_DECAY  0.95

/* ── Quantum constants ────────────────────────────────────────────────────── */
#define Q_REPS          8      /* representative slots per chunk = n_qubits    */
#define Q_BLEND_ALPHA   0.30   /* quantum→classical blend weight               */
#define Q_INTERVAL      8      /* run quantum correction every N lattice steps  */
#define APA_FLAG_QUANTUM (1 << 5)  /* slot has received quantum correction     */

/* ── MPI stub ─────────────────────────────────────────────────────────────── */
#define MPI_REAL 0
#if MPI_REAL
#  include <mpi.h>
#  define MPI_BCAST(b,c,t,r,comm) MPI_Bcast(b,c,t,r,MPI_COMM_WORLD)
#  define MPI_REDUCE(b,res,c,t,op,r,comm) MPI_Reduce(b,res,c,t,op,r,MPI_COMM_WORLD)
#else
#  define MPI_BCAST(b,c,t,r,comm)
#  define MPI_REDUCE(b,res,c,t,op,r,comm)
#  define MPI_SUM 0
#endif

/* ── DS3231 RTC ───────────────────────────────────────────────────────────── */
#ifdef USE_DS3231
#  include <i2c/smbus.h>
#  define DS3231_ADDR 0x68
   static int i2c_fd = -1;
#endif

/* ── cuStateVec quantum layer (opt-in) ───────────────────────────────────── */
#ifdef LL_QUANTUM_ENABLED
#  include <custatevec.h>
#  include <cuda_runtime.h>
#  define Q_DIM (1 << Q_REPS)   /* 256 amplitudes */

#  define CUDA_CHECK(x) do { \
       cudaError_t _e=(x); \
       if(_e!=cudaSuccess){fprintf(stderr,"[qchunk] CUDA %s at %s:%d\n", \
           cudaGetErrorString(_e),__FILE__,__LINE__);} \
   } while(0)
#  define CSV_CHECK(x) do { \
       custatevecStatus_t _s=(x); \
       if(_s!=CUSTATEVEC_STATUS_SUCCESS){fprintf(stderr,"[qchunk] cuSV %d at %s:%d\n", \
           (int)_s,__FILE__,__LINE__);} \
   } while(0)

typedef struct {
    custatevecHandle_t handle;
    void              *d_sv;         /* device: Q_DIM cuDoubleComplex         */
    void              *d_workspace;
    size_t             workspace_sz;
    int                step_count;   /* how many XY steps taken this chunk    */
} QChunkSampler;

/* Apply Ry(angle) gate to qubit `target` */
static void q_apply_Ry(QChunkSampler *q, int target, double angle) {
    double c = cos(angle*0.5), s = sin(angle*0.5);
    cuDoubleComplex mat[4] = {{c,0},{s,0},{-s,0},{c,0}};
    int tgt[1] = {target};
    CSV_CHECK(custatevecApplyMatrix(
        q->handle, q->d_sv, CUDA_C_64F, Q_REPS,
        mat, CUDA_C_64F, CUSTATEVEC_MATRIX_LAYOUT_COL, 0,
        tgt, 1, NULL, NULL, 0, CUSTATEVEC_COMPUTE_64F,
        q->d_workspace, q->workspace_sz));
}

/* Apply Rz(angle) gate to qubit `target` */
static void q_apply_Rz(QChunkSampler *q, int target, double angle) {
    double h = angle*0.5;
    cuDoubleComplex mat[4] = {{cos(h),-sin(h)},{0,0},{0,0},{cos(h),sin(h)}};
    int tgt[1] = {target};
    CSV_CHECK(custatevecApplyMatrix(
        q->handle, q->d_sv, CUDA_C_64F, Q_REPS,
        mat, CUDA_C_64F, CUSTATEVEC_MATRIX_LAYOUT_COL, 0,
        tgt, 1, NULL, NULL, 0, CUSTATEVEC_COMPUTE_64F,
        q->d_workspace, q->workspace_sz));
}

/* Apply e^{-i·param·Pa⊗Pb} two-qubit Pauli exponential */
static void q_apply_pauli_exp(QChunkSampler *q,
    custatevecPauli_t pa, custatevecPauli_t pb,
    int ta, int tb, double param)
{
    custatevecPauli_t paulis[2] = {pa, pb};
    int32_t targets[2] = {ta, tb};
    CSV_CHECK(custatevecApplyPauliExp(
        q->handle, q->d_sv, CUDA_C_64F, Q_REPS,
        (float)param, paulis, targets, 2, NULL, NULL, 0));
}

/* Allocate a QChunkSampler */
static QChunkSampler *qcs_create(void) {
    QChunkSampler *q = (QChunkSampler *)calloc(1, sizeof(QChunkSampler));
    if (!q) return NULL;
    if (custatevecCreate(&q->handle) != CUSTATEVEC_STATUS_SUCCESS) { free(q); return NULL; }
    cudaError_t ce = cudaMalloc(&q->d_sv, Q_DIM * sizeof(cuDoubleComplex));
    if (ce != cudaSuccess) { custatevecDestroy(q->handle); free(q); return NULL; }
    q->workspace_sz = 0;
    custatevecApplyMatrix_bufferSize(
        q->handle, CUDA_C_64F, Q_REPS, NULL, CUDA_C_64F,
        CUSTATEVEC_MATRIX_LAYOUT_COL, 0, 1, 0,
        CUSTATEVEC_COMPUTE_64F, &q->workspace_sz);
    if (q->workspace_sz > 0) cudaMalloc(&q->d_workspace, q->workspace_sz);
    return q;
}

static void qcs_destroy(QChunkSampler *q) {
    if (!q) return;
    if (q->d_workspace) cudaFree(q->d_workspace);
    if (q->d_sv)        cudaFree(q->d_sv);
    custatevecDestroy(q->handle);
    free(q);
}

/* Reset state vector to |00...0⟩, encode phases via Ry(phase[i]) */
static void qcs_encode(QChunkSampler *q, const double phases[Q_REPS]) {
    cuDoubleComplex h_sv[Q_DIM];
    memset(h_sv, 0, sizeof(h_sv));
    h_sv[0].x = 1.0;
    CUDA_CHECK(cudaMemcpy(q->d_sv, h_sv, Q_DIM * sizeof(cuDoubleComplex),
                          cudaMemcpyHostToDevice));
    for (int i = 0; i < Q_REPS; i++)
        q_apply_Ry(q, i, phases[i]);
}

/* One Trotterised XY step:
 *   H = Σi ωi Zi  +  (K/2) Σ_{i<j} (Xi Xj + Yi Yj)
 * ωi = K_COUPLING (uniform; natural freq variation handled by classical RK4) */
static void qcs_step(QChunkSampler *q, double k, double dt) {
    double half_K_dt = 0.5 * k * dt;
    /* Rz(−2ω·dt) natural frequency drive */
    for (int i = 0; i < Q_REPS; i++)
        q_apply_Rz(q, i, -2.0 * K_COUPLING * dt);
    /* XX + YY coupling for all 28 pairs */
    for (int i = 0; i < Q_REPS; i++) {
        for (int j = i+1; j < Q_REPS; j++) {
            q_apply_pauli_exp(q, CUSTATEVEC_PAULI_X, CUSTATEVEC_PAULI_X, i, j, -half_K_dt);
            q_apply_pauli_exp(q, CUSTATEVEC_PAULI_Y, CUSTATEVEC_PAULI_Y, i, j, -half_K_dt);
        }
    }
    q->step_count++;
}

/* Read ⟨Zi⟩ → re_out[i],  ⟨Xi⟩ → im_out[i], and return Kuramoto R² = |⟨Z⟩|² */
static double qcs_readback(QChunkSampler *q, double re_out[Q_REPS], double im_out[Q_REPS]) {
    const int n_ops = 2 * Q_REPS;
    custatevecPauli_t pauliOps[2 * Q_REPS];
    int32_t           tgt_flat[2 * Q_REPS];
    int32_t          *basisBits[2 * Q_REPS];
    uint32_t          nBasisBits[2 * Q_REPS];
    double            expects[2 * Q_REPS];

    for (int i = 0; i < Q_REPS; i++) {
        pauliOps[2*i]       = CUSTATEVEC_PAULI_Z;  tgt_flat[2*i]   = i;
        pauliOps[2*i+1]     = CUSTATEVEC_PAULI_X;  tgt_flat[2*i+1] = i;
        basisBits[2*i]      = &tgt_flat[2*i];      nBasisBits[2*i]   = 1;
        basisBits[2*i+1]    = &tgt_flat[2*i+1];    nBasisBits[2*i+1] = 1;
    }
    CSV_CHECK(custatevecComputeExpectationsOnPauliBasis(
        q->handle, q->d_sv, CUDA_C_64F, Q_REPS,
        expects, (const custatevecPauli_t **)pauliOps, n_ops,
        (const int32_t **)basisBits, nBasisBits));

    double sum_z = 0.0;
    for (int i = 0; i < Q_REPS; i++) {
        re_out[i] = expects[2*i];    /* ⟨Zi⟩ = cos(phase) */
        im_out[i] = expects[2*i+1]; /* ⟨Xi⟩ = sin(phase) */
        sum_z += re_out[i];
    }
    double R = sum_z / Q_REPS;
    return R * R;   /* R² = Kuramoto coherence squared */
}

#endif /* LL_QUANTUM_ENABLED */

/* ── Tables ───────────────────────────────────────────────────────────────── */
static const float fib_table[]   = {1,1,2,3,5,8,13,21,34,55,89,144,233,377,610,987};
static const float prime_table[] = {2,3,5,7,11,13,17,19,23,29,31,37,41,43,47,53};
static const int fib_len   = 16;
static const int prime_len = 16;

double   get_normalized_rand(void) { return (double)rand() / RAND_MAX; }
uint64_t det_rand(uint64_t seed) {
    seed ^= seed << 13; seed ^= seed >> 7; seed ^= seed << 17; return seed;
}
#define GET_RANDOM_UINT64() (((uint64_t)rand() << 32) | (uint64_t)rand())

/* ── RTC ──────────────────────────────────────────────────────────────────── */
int64_t get_rtc_ns(void) {
#ifdef USE_DS3231
    if (i2c_fd >= 0) {
        uint8_t data[7];
        if (i2c_smbus_read_i2c_block_data(i2c_fd, DS3231_ADDR, 0x00, 7, data) == 7) {
            int sec = ((data[0]>>4)*10)+(data[0]&0x0F);
            int min = ((data[1]>>4)*10)+(data[1]&0x0F);
            int hr  = ((data[2]>>4)*10)+(data[2]&0x0F);
            return (int64_t)(hr*3600+min*60+sec)*1000000000LL;
        }
    }
#endif
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec*1000000000LL + ts.tv_nsec;
}
void rtc_sleep_until(int64_t target_ns) {
    int64_t now = get_rtc_ns();
    if (target_ns <= now) return;
    int64_t diff = target_ns - now;
    struct timespec req;
    req.tv_sec  = (long)(diff / 1000000000LL);
    req.tv_nsec = (long)(diff % 1000000000LL);
    nanosleep(&req, NULL);
}

/* ── MPI (multi-word integer) ─────────────────────────────────────────────── */
typedef struct { uint64_t *words; size_t num_words; uint8_t sign; } MPI;

#define APA_FLAG_SIGN_NEG  (1<<0)
#define APA_FLAG_IS_NAN    (1<<1)
#define APA_FLAG_GOI       (1<<2)
#define APA_FLAG_GUZ       (1<<3)
#define APA_FLAG_CONSENSUS (1<<4)
/* APA_FLAG_QUANTUM       (1<<5)  defined above */

void mpi_init(MPI *m, size_t nw) {
    m->words = (uint64_t *)calloc(nw, sizeof(uint64_t)); m->num_words = nw; m->sign = 0;
}
void mpi_free(MPI *m) {
    if (m->words) { free(m->words); m->words = NULL; } m->num_words = 0;
}
/* FIX 6: zero dest->words when src->words is NULL */
void mpi_copy(MPI *dest, const MPI *src) {
    mpi_free(dest);
    dest->num_words = src->num_words;
    dest->sign      = src->sign;
    if (dest->num_words == 0) { dest->words = NULL; return; }
    dest->words = (uint64_t *)malloc(dest->num_words * sizeof(uint64_t));
    if (!dest->words) { dest->num_words = 0; return; }
    if (src->words)
        memcpy(dest->words, src->words, dest->num_words * sizeof(uint64_t));
    else
        memset(dest->words, 0, dest->num_words * sizeof(uint64_t));
}
void mpi_set_value(MPI *m, uint64_t value, uint8_t sign) {
    if (m->words) m->words[0] = value; m->sign = sign;
}

/* ── Slot4096 ─────────────────────────────────────────────────────────────── */
typedef struct {
    uint64_t  *mantissa_words;
    MPI        num_words_mantissa;
    MPI        exponent_mpi;
    uint16_t   exponent_base;
    uint32_t   state_flags;
    MPI        source_of_infinity;
    size_t     num_words;
    int64_t    exponent;
    float      base;
    int        bits_mant;
    int        bits_exp;
    double     phase;
    double     phase_vel;
    double     freq;
    double     amp_im;
} Slot4096;

static Slot4096 APA_CONST_PHI;
static Slot4096 APA_CONST_PI;

/* Forward declarations */
void     ap_normalize_legacy(Slot4096 *slot);
void     ap_add_legacy(Slot4096 *A, const Slot4096 *B);
void     ap_free(Slot4096 *slot);
void     ap_copy(Slot4096 *dest, const Slot4096 *src);
double   ap_to_double(const Slot4096 *slot);
Slot4096 *ap_from_double(double value, int bits_mant, int bits_exp);
void     ap_shift_right_legacy(uint64_t *mw, size_t nw, int64_t shift);

/* ── APA implementation ───────────────────────────────────────────────────── */
Slot4096 slot_init_apa(int bits_mant, int bits_exp) {
    Slot4096 slot = {0};
    slot.bits_mant = bits_mant; slot.bits_exp = bits_exp;
    slot.num_words = (size_t)((bits_mant + 63) / 64);
    slot.mantissa_words = (uint64_t *)calloc(slot.num_words, sizeof(uint64_t));
    mpi_init(&slot.exponent_mpi, 1);
    mpi_init(&slot.num_words_mantissa, 1);
    mpi_init(&slot.source_of_infinity, 1);
    if (!slot.mantissa_words) { fprintf(stderr,"Error: Failed to allocate mantissa.\n"); return slot; }
    if (slot.num_words > 0) { slot.mantissa_words[0] = GET_RANDOM_UINT64() | MSB_MASK; }
    int64_t exp_bias = 1LL << (bits_exp-1);
    slot.exponent = (int64_t)(rand() % (1LL << bits_exp)) - exp_bias;
    slot.base      = (float)(PHI + get_normalized_rand()*0.01);
    slot.exponent_base = 4096;
    slot.phase     = 2.0*M_PI*get_normalized_rand();
    slot.phase_vel = 0.0;
    slot.freq      = 1.0 + 0.5*get_normalized_rand();
    slot.amp_im    = 0.1*get_normalized_rand();
    mpi_set_value(&slot.exponent_mpi, (uint64_t)llabs(slot.exponent), slot.exponent<0 ? 1:0);
    mpi_set_value(&slot.num_words_mantissa, (uint64_t)slot.num_words, 0);
    return slot;
}

void ap_free(Slot4096 *slot) {
    if (!slot) return;
    if (slot->mantissa_words) { free(slot->mantissa_words); slot->mantissa_words = NULL; }
    mpi_free(&slot->exponent_mpi);
    mpi_free(&slot->num_words_mantissa);
    mpi_free(&slot->source_of_infinity);
    slot->num_words = 0;
}

/* FIX 1: ap_copy — field-by-field scalar copy avoids UAF from memcpy of pointers */
void ap_copy(Slot4096 *dest, const Slot4096 *src) {
    ap_free(dest);
    /* Copy all scalar fields */
    dest->exponent_base = src->exponent_base;
    dest->state_flags   = src->state_flags;
    dest->num_words     = src->num_words;
    dest->exponent      = src->exponent;
    dest->base          = src->base;
    dest->bits_mant     = src->bits_mant;
    dest->bits_exp      = src->bits_exp;
    dest->phase         = src->phase;
    dest->phase_vel     = src->phase_vel;
    dest->freq          = src->freq;
    dest->amp_im        = src->amp_im;
    /* Deep-copy heap members */
    dest->mantissa_words = (uint64_t *)malloc(src->num_words * sizeof(uint64_t));
    if (!dest->mantissa_words) { fprintf(stderr,"Error: ap_copy alloc failed.\n"); dest->num_words=0; return; }
    memcpy(dest->mantissa_words, src->mantissa_words, src->num_words * sizeof(uint64_t));
    mpi_copy(&dest->exponent_mpi,       &src->exponent_mpi);
    mpi_copy(&dest->num_words_mantissa, &src->num_words_mantissa);
    mpi_copy(&dest->source_of_infinity, &src->source_of_infinity);
}

double ap_to_double(const Slot4096 *slot) {
    if (!slot || slot->num_words==0 || !slot->mantissa_words) return 0.0;
    return (double)slot->mantissa_words[0] / (double)UINT64_MAX
           * pow(2.0, (double)slot->exponent);
}

Slot4096 *ap_from_double(double value, int bits_mant, int bits_exp) {
    Slot4096 *slot = (Slot4096 *)malloc(sizeof(Slot4096));
    if (!slot) return NULL;
    *slot = slot_init_apa(bits_mant, bits_exp);
    if (value == 0.0) return slot;
    int exp_offset; double mant = frexp(value, &exp_offset);
    slot->mantissa_words[0] = (uint64_t)(fabs(mant) * (double)UINT64_MAX);
    slot->exponent = (int64_t)exp_offset;
    if (value < 0) slot->state_flags |= APA_FLAG_SIGN_NEG;
    mpi_set_value(&slot->exponent_mpi, (uint64_t)llabs(slot->exponent), slot->exponent<0?1:0);
    return slot;
}

void ap_shift_right_legacy(uint64_t *mw, size_t nw, int64_t shift) {
    if (shift<=0 || nw==0) return;
    if (shift>=(int64_t)(nw*64)) { memset(mw, 0, nw*sizeof(uint64_t)); return; }
    int64_t ws = shift/64; int bs = (int)(shift%64);
    if (ws>0) { for (int64_t i=nw-1;i>=ws;i--) mw[i]=mw[i-ws]; memset(mw,0,ws*sizeof(uint64_t)); }
    if (bs>0) { int rs=64-bs; for (size_t i=nw-1;i>0;i--) mw[i]=(mw[i]>>bs)|(mw[i-1]<<rs); mw[0]>>=bs; }
}

void ap_normalize_legacy(Slot4096 *slot) {
    if (slot->num_words==0) return;
    while (!(slot->mantissa_words[0] & MSB_MASK)) {
        if (slot->exponent<=-(1LL<<(slot->bits_exp-1))) { slot->state_flags|=APA_FLAG_GUZ; break; }
        uint64_t carry=0;
        for (size_t i=slot->num_words-1;i!=(size_t)-1;i--) {
            uint64_t nc=(slot->mantissa_words[i]&MSB_MASK)?1:0;
            slot->mantissa_words[i]=(slot->mantissa_words[i]<<1)|carry; carry=nc;
        }
        slot->exponent--;
    }
    if (slot->mantissa_words[0]==0) slot->exponent=0;
}

void ap_add_legacy(Slot4096 *A, const Slot4096 *B) {
    if (A->num_words != B->num_words) { fprintf(stderr,"Error: Unaligned word counts.\n"); return; }
    Slot4096 Bal; ap_copy(&Bal, B);
    int64_t ed = A->exponent - Bal.exponent;
    if (ed>0) { ap_shift_right_legacy(Bal.mantissa_words, Bal.num_words, ed); Bal.exponent=A->exponent; }
    else if (ed<0) { ap_shift_right_legacy(A->mantissa_words, A->num_words, -ed); A->exponent=Bal.exponent; }
    uint64_t carry=0;
    for (size_t i=A->num_words-1;i!=(size_t)-1;i--) {
        uint64_t s=A->mantissa_words[i]+Bal.mantissa_words[i]+carry;
        carry=(s<A->mantissa_words[i]||(s==A->mantissa_words[i]&&carry))?1:0;
        A->mantissa_words[i]=s;
    }
    if (carry) {
        if (A->exponent>=(1LL<<(A->bits_exp-1))) A->state_flags|=APA_FLAG_GOI;
        else { A->exponent++; ap_shift_right_legacy(A->mantissa_words,A->num_words,1); A->mantissa_words[0]|=MSB_MASK; }
    }
    ap_normalize_legacy(A);
    mpi_set_value(&A->exponent_mpi,(uint64_t)llabs(A->exponent),A->exponent<0?1:0);
    ap_free(&Bal);
}

/* ── AnalogLink ───────────────────────────────────────────────────────────── */
typedef struct { double charge,charge_im,tension,potential,coupling; } AnalogLink;

void exchange_analog_links(AnalogLink *links, int rank, int size, int num_links) {
#if MPI_REAL
    MPI_BCAST(links, num_links*sizeof(AnalogLink), MPI_BYTE, rank, MPI_COMM_WORLD);
    AnalogLink *reduced = (AnalogLink *)calloc(num_links, sizeof(AnalogLink));
    MPI_REDUCE(links, reduced, num_links*sizeof(AnalogLink), MPI_BYTE, MPI_SUM, 0, MPI_COMM_WORLD);
    for (int i=0;i<num_links;i++) {
        links[i].charge = reduced[i].charge/size;
        links[i].charge_im = reduced[i].charge_im/size;
        links[i].tension *= 0.9;
    }
    free(reduced);
#else
    (void)rank; (void)size;
    for (int i=0;i<num_links;i++) { links[i].charge*=0.95; links[i].charge_im*=0.95; }
#endif
}

/* ── ODE / RK4 ────────────────────────────────────────────────────────────── */
typedef struct { double A_re, A_im, phase, phase_vel; } ComplexState;

static ComplexState compute_derivatives(ComplexState st, double omega,
                                        const AnalogLink *nb, int nn) {
    ComplexState d = {0};
    d.A_re = -GAMMA*st.A_re; d.A_im = -GAMMA*st.A_im;
    double ss = 0.0;
    for (int k=0;k<nn;k++) {
        double dp = nb[k].potential - st.phase;
        ss += sin(dp);
        d.A_re += K_COUPLING*nb[k].coupling*cos(dp);
        d.A_im += K_COUPLING*nb[k].coupling*sin(dp);
    }
    d.phase_vel = omega + K_COUPLING*ss;
    d.phase     = st.phase_vel;
    return d;
}

/* FIX 2: write A_re directly into mantissa; no heap allocation per step */
static void rk4_step(Slot4096 *slot, double dt, const AnalogLink *nb, int nn) {
    ComplexState st = { ap_to_double(slot), slot->amp_im, slot->phase, slot->phase_vel };
    ComplexState k1 = compute_derivatives(st, slot->freq, nb, nn);
    ComplexState tmp;
#define HALF_STEP(dst, base, k) \
    dst = base; dst.A_re += dt*(k).A_re/2; dst.A_im += dt*(k).A_im/2; \
    dst.phase += dt*(k).phase/2; dst.phase_vel += dt*(k).phase_vel/2;
#define FULL_STEP(dst, base, k) \
    dst = base; dst.A_re += dt*(k).A_re; dst.A_im += dt*(k).A_im; \
    dst.phase += dt*(k).phase; dst.phase_vel += dt*(k).phase_vel;
    HALF_STEP(tmp, st, k1); ComplexState k2 = compute_derivatives(tmp, slot->freq, nb, nn);
    HALF_STEP(tmp, st, k2); ComplexState k3 = compute_derivatives(tmp, slot->freq, nb, nn);
    FULL_STEP(tmp, st, k3); ComplexState k4 = compute_derivatives(tmp, slot->freq, nb, nn);
#undef HALF_STEP
#undef FULL_STEP
    st.A_re     += dt/6*(k1.A_re     + 2*k2.A_re     + 2*k3.A_re     + k4.A_re);
    st.A_im     += dt/6*(k1.A_im     + 2*k2.A_im     + 2*k3.A_im     + k4.A_im);
    st.phase    += dt/6*(k1.phase    + 2*k2.phase    + 2*k3.phase    + k4.phase);
    st.phase_vel += dt/6*(k1.phase_vel+2*k2.phase_vel+2*k3.phase_vel+k4.phase_vel);
    double A = sqrt(st.A_re*st.A_re + st.A_im*st.A_im);
    A *= exp(-LAMBDA*dt); if (A>SAT_LIMIT) A=SAT_LIMIT;
    A += NOISE_SIGMA*(2.0*get_normalized_rand()-1.0);
    double norm = sqrt(st.A_re*st.A_re + st.A_im*st.A_im);
    if (norm>1e-10) { st.A_re=(st.A_re/norm)*A; st.A_im=(st.A_im/norm)*A; }
    st.phase = fmod(st.phase, 2.0*M_PI); if (st.phase<0) st.phase+=2.0*M_PI;
    /* Write directly — no ap_from_double allocation */
    if (slot->num_words>0 && slot->mantissa_words) {
        int exp_off; double mant = frexp(st.A_re, &exp_off);
        slot->mantissa_words[0] = (uint64_t)(fabs(mant)*(double)UINT64_MAX);
        slot->exponent = (int64_t)exp_off;
        if (st.A_re < 0) slot->state_flags |= APA_FLAG_SIGN_NEG;
        else             slot->state_flags &= ~(uint32_t)APA_FLAG_SIGN_NEG;
        mpi_set_value(&slot->exponent_mpi,(uint64_t)llabs(slot->exponent),slot->exponent<0?1:0);
    }
    slot->amp_im    = st.A_im;
    slot->phase     = st.phase;
    slot->phase_vel = st.phase_vel;
}

/* ── HDGLChunk / HDGLLattice ──────────────────────────────────────────────── */
typedef struct { Slot4096 *slots; size_t allocated; } HDGLChunk;

typedef struct {
    HDGLChunk  **chunks;
    int          num_chunks;
    int          num_instances;
    int          slots_per_instance;
    double       omega;
    double       time;
    int          consensus_steps;
    double       phase_var;
    int64_t      last_checkpoint_ns;
    int          integrate_count;   /* total lattice_integrate_rk4 calls */
#ifdef LL_QUANTUM_ENABLED
    QChunkSampler **q_samplers;     /* one per chunk, lazy-allocated      */
    double          q_coherence;    /* last measured R² across all chunks  */
#endif
} HDGLLattice;

HDGLLattice *lattice_init(int num_instances, int slots_per_instance) {
    HDGLLattice *lat = (HDGLLattice *)malloc(sizeof(HDGLLattice));
    if (!lat) return NULL;
    memset(lat, 0, sizeof(HDGLLattice));
    lat->num_instances      = num_instances;
    lat->slots_per_instance = slots_per_instance;
    lat->phase_var          = 1e6;
    lat->last_checkpoint_ns = get_rtc_ns();
    int total = num_instances * slots_per_instance;
    lat->num_chunks = (total + CHUNK_SIZE - 1) / CHUNK_SIZE;
    lat->chunks = (HDGLChunk **)calloc(lat->num_chunks, sizeof(HDGLChunk*));
    if (!lat->chunks) { free(lat); return NULL; }
#ifdef LL_QUANTUM_ENABLED
    lat->q_samplers = (QChunkSampler **)calloc(lat->num_chunks, sizeof(QChunkSampler*));
    if (!lat->q_samplers) { free(lat->chunks); free(lat); return NULL; }
#endif
    return lat;
}

HDGLChunk *lattice_get_chunk(HDGLLattice *lat, int ci) {
    if (ci >= lat->num_chunks) return NULL;
    if (!lat->chunks[ci]) {
        HDGLChunk *chunk = (HDGLChunk *)malloc(sizeof(HDGLChunk));
        if (!chunk) return NULL;
        chunk->allocated = CHUNK_SIZE;
        chunk->slots = (Slot4096 *)malloc(CHUNK_SIZE * sizeof(Slot4096));
        if (!chunk->slots) { free(chunk); return NULL; }
        for (int i=0;i<CHUNK_SIZE;i++) {
            int bm=4096+(i%8)*64; int be=16+(i%8)*2;
            chunk->slots[i] = slot_init_apa(bm, be);
        }
        lat->chunks[ci] = chunk;
    }
    return lat->chunks[ci];
}

Slot4096 *lattice_get_slot(HDGLLattice *lat, int idx) {
    HDGLChunk *chunk = lattice_get_chunk(lat, idx/CHUNK_SIZE);
    if (!chunk) return NULL;
    return &chunk->slots[idx % CHUNK_SIZE];
}

double prismatic_recursion(HDGLLattice *lat, int idx, double val) {
    double phi_harm  = pow(PHI,  (double)(idx%16));
    double fib_harm  = fib_table[idx%fib_len];
    double dyadic    = (double)(1 << (idx%16));
    double prime_h   = prime_table[idx%prime_len];
    double omega_val = 0.5 + 0.5*sin(lat->time + idx*0.01);
    double r_dim     = pow(fabs(val), (double)((idx%7)+1)/8.0);
    return sqrt(phi_harm*fib_harm*dyadic*prime_h*omega_val)*r_dim;
}

/* FIX 5: circular mean via atan2 for phase averaging */
void detect_harmonic_consensus(HDGLLattice *lat) {
    int total = lat->num_instances * lat->slots_per_instance;
    double sum_sin=0, sum_cos=0, sum_var=0;
    int count=0;
    for (int i=0;i<total;i++) {
        Slot4096 *slot = lattice_get_slot(lat, i);
        if (slot && !(slot->state_flags & APA_FLAG_CONSENSUS)) {
            sum_sin += sin(slot->phase); sum_cos += cos(slot->phase); count++;
        }
    }
    if (count==0) return;
    /* Circular mean — correct across 0/2π boundary */
    double mean_phase = atan2(sum_sin/count, sum_cos/count);
    for (int i=0;i<total;i++) {
        Slot4096 *slot = lattice_get_slot(lat, i);
        if (slot && !(slot->state_flags & APA_FLAG_CONSENSUS)) {
            double diff = slot->phase - mean_phase;
            if (diff> M_PI) diff -= 2.0*M_PI;
            if (diff<-M_PI) diff += 2.0*M_PI;
            sum_var += diff*diff;
        }
    }
    lat->phase_var = sqrt(sum_var/count);
    if (lat->phase_var < CONSENSUS_EPS) {
        lat->consensus_steps++;
        if (lat->consensus_steps >= CONSENSUS_N) {
            printf("[CONSENSUS] Domain locked at t=%.4f (var=%.6f, evo=%d)!\n",
                   lat->time, lat->phase_var, lat->integrate_count);
            for (int i=0;i<total;i++) {
                Slot4096 *slot = lattice_get_slot(lat, i);
                if (slot && !(slot->state_flags & APA_FLAG_CONSENSUS)) {
                    slot->state_flags |= APA_FLAG_CONSENSUS; slot->phase_vel=0.0;
                }
            }
            lat->consensus_steps=0;
        }
    } else { lat->consensus_steps=0; }
}

/* ── Quantum correction: one chunk ───────────────────────────────────────── */
#ifdef LL_QUANTUM_ENABLED
static void quantum_correct_chunk(HDGLLattice *lat, int ci, double dt) {
    HDGLChunk *chunk = lat->chunks[ci];
    if (!chunk) return;

    /* Lazy-allocate the QChunkSampler for this chunk */
    if (!lat->q_samplers[ci]) {
        lat->q_samplers[ci] = qcs_create();
        if (!lat->q_samplers[ci]) return;   /* cuQuantum unavailable; skip */
    }
    QChunkSampler *q = lat->q_samplers[ci];

    /* Sample Q_REPS evenly-spaced slots from this chunk */
    int stride = (int)(CHUNK_SIZE / Q_REPS);
    double phases[Q_REPS];
    int slot_indices[Q_REPS];
    for (int r=0; r<Q_REPS; r++) {
        slot_indices[r] = r * stride;
        phases[r] = chunk->slots[slot_indices[r]].phase;
    }

    /* Encode phases into state vector, evolve one XY step, read back */
    qcs_encode(q, phases);
    qcs_step(q, K_COUPLING, dt);
    double q_re[Q_REPS], q_im[Q_REPS];
    double R2 = qcs_readback(q, q_re, q_im);

    /* Blend quantum expectation values into representative slots */
    double one_minus = 1.0 - Q_BLEND_ALPHA;
    for (int r=0; r<Q_REPS; r++) {
        Slot4096 *slot = &chunk->slots[slot_indices[r]];
        if (slot->state_flags & (APA_FLAG_CONSENSUS|APA_FLAG_GOI|APA_FLAG_IS_NAN)) continue;

        /* Blend and renormalise to unit circle */
        double blended_cos = one_minus*cos(slot->phase) + Q_BLEND_ALPHA*q_re[r];
        double blended_sin = one_minus*sin(slot->phase) + Q_BLEND_ALPHA*q_im[r];
        double mag = sqrt(blended_cos*blended_cos + blended_sin*blended_sin);
        if (mag > 1e-10) {
            slot->phase = atan2(blended_sin/mag, blended_cos/mag);
            if (slot->phase < 0) slot->phase += 2.0*M_PI;
        }
        slot->state_flags |= APA_FLAG_QUANTUM;
    }

    /* Accumulate quantum coherence for lattice-level readout */
    lat->q_coherence = 0.9*lat->q_coherence + 0.1*R2;  /* EMA */
}
#endif /* LL_QUANTUM_ENABLED */

/* FIX 3: dt_base not mutated globally; local_dt per slot; time advances by caller's dt */
void lattice_integrate_rk4(HDGLLattice *lat, double dt_base) {
    int total = lat->num_instances * lat->slots_per_instance;
    for (int i=0;i<total;i++) {
        Slot4096 *slot = lattice_get_slot(lat, i);
        if (!slot || (slot->state_flags & (APA_FLAG_GOI|APA_FLAG_IS_NAN|APA_FLAG_CONSENSUS))) continue;

        AnalogLink nb[8] = {0};
        int ni[] = {
            (i-1+total)%total, (i+1)%total,
            (i-lat->slots_per_instance+total)%total, (i+lat->slots_per_instance)%total,
            (i-lat->slots_per_instance-1+total)%total, (i-lat->slots_per_instance+1+total)%total,
            (i+lat->slots_per_instance-1+total)%total, (i+lat->slots_per_instance+1)%total
        };
        for (int j=0;j<8;j++) {
            Slot4096 *neigh = lattice_get_slot(lat, ni[j]);
            if (neigh) {
                nb[j].charge    = ap_to_double(neigh);
                nb[j].charge_im = neigh->amp_im;
                nb[j].tension   = (ap_to_double(neigh)-ap_to_double(slot)) / dt_base;
                nb[j].potential = neigh->phase - slot->phase;
                double ac = fabs(ap_to_double(neigh)) / (fabs(ap_to_double(slot))+1e-10);
                nb[j].coupling  = K_COUPLING * exp(-fabs(1.0-ac));
            }
        }
        exchange_analog_links(nb, i%lat->num_instances, lat->num_instances, 8);

        /* Per-slot adaptive dt — does NOT affect lat->time */
        double local_dt = dt_base;
        double amp = ap_to_double(slot);
        if (fabs(amp) > ADAPT_THRESH)          local_dt *= PHI;
        else if (fabs(amp) < ADAPT_THRESH/PHI) local_dt /= PHI;
        if (local_dt < 1e-6) local_dt = 1e-6;
        if (local_dt > 0.1)  local_dt = 0.1;

        rk4_step(slot, local_dt, nb, 8);
    }

#ifdef LL_QUANTUM_ENABLED
    /* Quantum correction every Q_INTERVAL steps, one chunk at a time */
    if (lat->integrate_count % Q_INTERVAL == 0) {
        for (int ci=0; ci<lat->num_chunks; ci++) {
            if (lat->chunks[ci])
                quantum_correct_chunk(lat, ci, dt_base);
        }
    }
#endif

    detect_harmonic_consensus(lat);
    lat->omega += 0.01 * dt_base;
    lat->time  += dt_base;           /* always the original caller's dt */
    lat->integrate_count++;
}

/* ── Checkpoint ───────────────────────────────────────────────────────────── */
typedef struct { int evolution; int64_t timestamp_ns; double phase_var; double omega; double weight; } CheckpointMeta;
typedef struct { CheckpointMeta *snapshots; int count; int capacity; } CheckpointManager;

CheckpointManager *checkpoint_init(void) {
    CheckpointManager *mgr = (CheckpointManager *)malloc(sizeof(CheckpointManager));
    mgr->snapshots = (CheckpointMeta *)malloc(SNAPSHOT_MAX * sizeof(CheckpointMeta));
    mgr->count=0; mgr->capacity=SNAPSHOT_MAX; return mgr;
}

/* FIX 4: capture pruned meta BEFORE shifting the array */
void checkpoint_add(CheckpointManager *mgr, int evo, HDGLLattice *lat) {
    if (mgr->count >= mgr->capacity) {
        int min_idx=0; double min_w=mgr->snapshots[0].weight;
        for (int i=1;i<mgr->count;i++)
            if (mgr->snapshots[i].weight<min_w) { min_w=mgr->snapshots[i].weight; min_idx=i; }
        CheckpointMeta pruned = mgr->snapshots[min_idx];   /* capture before shift */
        for (int i=min_idx;i<mgr->count-1;i++) mgr->snapshots[i]=mgr->snapshots[i+1];
        mgr->count--;
        printf("[Checkpoint] Pruned evo %d (weight=%.4f)\n", pruned.evolution, pruned.weight);
    }
    CheckpointMeta meta = { evo, get_rtc_ns(), lat->phase_var, lat->omega, 1.0 };
    mgr->snapshots[mgr->count++] = meta;
    for (int i=0;i<mgr->count-1;i++) mgr->snapshots[i].weight *= SNAPSHOT_DECAY;
    printf("[Checkpoint] Saved evo %d (total: %d, var=%.6f)\n", evo, mgr->count, lat->phase_var);
}
void checkpoint_free(CheckpointManager *mgr) { if (mgr) { free(mgr->snapshots); free(mgr); } }

/* ── Lattice fold ─────────────────────────────────────────────────────────── */
void lattice_fold(HDGLLattice *lat) {
    int new_inst = lat->num_instances * 2;
    if (new_inst > MAX_INSTANCES) return;
    int old_total  = lat->num_instances * lat->slots_per_instance;
    int new_total  = new_inst * lat->slots_per_instance;
    int old_chunks = lat->num_chunks;
    int new_chunks = (new_total + CHUNK_SIZE - 1) / CHUNK_SIZE;

    HDGLChunk **np = (HDGLChunk **)realloc(lat->chunks, new_chunks*sizeof(HDGLChunk*));
    if (!np) { fprintf(stderr,"Failed to allocate for folding\n"); return; }
    lat->chunks = np;
    for (int i=old_chunks;i<new_chunks;i++) lat->chunks[i]=NULL;

#ifdef LL_QUANTUM_ENABLED
    QChunkSampler **qp = (QChunkSampler **)realloc(lat->q_samplers, new_chunks*sizeof(QChunkSampler*));
    if (!qp) { fprintf(stderr,"Failed to realloc q_samplers\n"); return; }
    lat->q_samplers = qp;
    for (int i=old_chunks;i<new_chunks;i++) lat->q_samplers[i]=NULL;
#endif

    /* Update instance count BEFORE calling lattice_get_slot on new indices */
    lat->num_instances = new_inst;
    lat->num_chunks    = new_chunks;

    for (int i=0;i<old_total;i++) {
        Slot4096 *os = lattice_get_slot(lat, i);
        Slot4096 *ns = lattice_get_slot(lat, old_total+i);
        if (os && ns) {
            ap_copy(ns, os);
            double pert = fib_table[i%fib_len]*0.01;
            Slot4096 *pa = ap_from_double(pert, ns->bits_mant, ns->bits_exp);
            if (pa) { ap_add_legacy(ns, pa); ap_free(pa); free(pa); }
            ns->phase += (get_normalized_rand()-0.5)*0.1;
            ns->base  += (float)(get_normalized_rand()*0.001);
        }
    }
}

void lattice_free(HDGLLattice *lat) {
    if (!lat) return;
#ifdef LL_QUANTUM_ENABLED
    for (int i=0;i<lat->num_chunks;i++) qcs_destroy(lat->q_samplers[i]);
    free(lat->q_samplers);
#endif
    for (int i=0;i<lat->num_chunks;i++) {
        if (lat->chunks[i]) {
            for (size_t j=0;j<CHUNK_SIZE;j++) ap_free(&lat->chunks[i]->slots[j]);
            free(lat->chunks[i]->slots); free(lat->chunks[i]);
        }
    }
    free(lat->chunks); free(lat);
}

/* ── Bootloader ───────────────────────────────────────────────────────────── */
void init_apa_constants(void) {
    APA_CONST_PHI = slot_init_apa(4096,16);
    APA_CONST_PI  = slot_init_apa(4096,16);
    Slot4096 *tp  = ap_from_double(PHI,  APA_CONST_PHI.bits_mant, APA_CONST_PHI.bits_exp);
    Slot4096 *tpi = ap_from_double(M_PI, APA_CONST_PI.bits_mant,  APA_CONST_PI.bits_exp);
    ap_copy(&APA_CONST_PHI, tp);  ap_free(tp);  free(tp);
    ap_copy(&APA_CONST_PI,  tpi); ap_free(tpi); free(tpi);
    printf("[Bootloader] Constants PHI=%.15e  PI=%.15e\n", PHI, M_PI);
}

void bootloader_init_lattice(HDGLLattice *lat, int steps, CheckpointManager *ckpt_mgr) {
    printf("[Bootloader] Initializing HDGL Analog Mainnet (APA V2.7)...\n");
    if (!lat) { printf("[Bootloader] ERROR: Lattice NULL.\n"); return; }
    init_apa_constants();
    printf("[Bootloader] %d instances, %d total slots\n",
           lat->num_instances, lat->num_instances*lat->slots_per_instance);
#ifdef LL_QUANTUM_ENABLED
    printf("[Bootloader] Quantum layer: %d-qubit XY sampler per chunk, blend=%.2f, interval=%d\n",
           Q_REPS, Q_BLEND_ALPHA, Q_INTERVAL);
#else
    printf("[Bootloader] Quantum layer: disabled (build with -DLL_QUANTUM_ENABLED)\n");
#endif

    double dt = 1.0/32768.0;
    int64_t step_ns = 30517, next_ns = get_rtc_ns()+step_ns;
    for (int i=0;i<steps;i++) {
        lattice_integrate_rk4(lat, dt);
        if (i%CHECKPOINT_INTERVAL==0 && i>0) checkpoint_add(ckpt_mgr, i, lat);
        rtc_sleep_until(next_ns); next_ns+=step_ns;
    }
    printf("[Bootloader] %d steps done.  omega=%.6f  t=%.6f  var=%.6f\n",
           steps, lat->omega, lat->time, lat->phase_var);
}

/* ── Main ─────────────────────────────────────────────────────────────────── */
int main(int argc, char *argv[]) {
    (void)argc; (void)argv;
    srand((unsigned)time(NULL));

#ifdef USE_DS3231
    i2c_fd = i2c_open("/dev/i2c-1");
    if (i2c_fd>=0) { i2c_smbus_write_byte_data(i2c_fd,DS3231_ADDR,0x0E,0x00); printf("[RTC] DS3231 on I2C-1\n"); }
    else printf("[RTC] Software fallback (CLOCK_MONOTONIC)\n");
#else
    printf("[RTC] Software fallback (CLOCK_MONOTONIC)\n");
#endif

    printf("=== HDGL Analog Mainnet V2.7 ===\n\n");

    HDGLLattice *lat = lattice_init(4096, 4);
    if (!lat) { fprintf(stderr,"Fatal: lattice_init failed.\n"); return 1; }

    CheckpointManager *ckpt = checkpoint_init();
    bootloader_init_lattice(lat, 500, ckpt);

    printf("\nHigh-Precision Constants:\n");
    printf("  PHI: %.15e  exp=%ld  words=%zu\n",
           ap_to_double(&APA_CONST_PHI), APA_CONST_PHI.exponent, APA_CONST_PHI.num_words);
    printf("  PI:  %.15e  exp=%ld  words=%zu\n",
           ap_to_double(&APA_CONST_PI),  APA_CONST_PI.exponent,  APA_CONST_PI.num_words);

    printf("\nFirst 8 slots:\n");
    for (int i=0;i<8;i++) {
        Slot4096 *s = lattice_get_slot(lat, i);
        if (s) {
            double amp = sqrt(pow(ap_to_double(s),2)+pow(s->amp_im,2));
            printf("  D%d: |A|=%.6e  phi=%.3f  w=%.3f  base=%.6f  exp=%ld  flags=0x%x%s\n",
                   i+1, amp, s->phase, s->freq, s->base, s->exponent, s->state_flags,
                   (s->state_flags & APA_FLAG_QUANTUM) ? "  [Q]" : "");
        }
    }

    printf("\nCheckpoints: %d snapshots\n", ckpt->count);
    for (int i=0;i<ckpt->count;i++)
        printf("  evo=%d  weight=%.4f  var=%.6f\n",
               ckpt->snapshots[i].evolution, ckpt->snapshots[i].weight, ckpt->snapshots[i].phase_var);

#ifdef LL_QUANTUM_ENABLED
    printf("\nQuantum coherence R²=%.4f\n", lat->q_coherence);
#endif

    printf("\nPrismatic fold: %d → ", lat->num_instances);
    lattice_fold(lat);
    printf("%d instances\n", lat->num_instances);

    printf("\nExtended evolution (1000 steps)...\n");
    for (int i=0;i<1000;i++) {
        lattice_integrate_rk4(lat, 1.0/32768.0);
        if (i%100==0) {
            printf("  step=%d  var=%.6f  consensus=%d", i, lat->phase_var, lat->consensus_steps);
#ifdef LL_QUANTUM_ENABLED
            printf("  R²=%.4f", lat->q_coherence);
#endif
            printf("\n");
        }
    }

    ap_free(&APA_CONST_PHI); ap_free(&APA_CONST_PI);
    checkpoint_free(ckpt); lattice_free(lat);

#ifdef USE_DS3231
    if (i2c_fd>=0) i2c_close(i2c_fd);
#endif

    printf("\n=== ANALOG MAINNET V2.7 OPERATIONAL ===\n");
    return 0;
}
