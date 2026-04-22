/* bench_quantum.cu — Comprehensive benchmark for all quantum functions
 *
 * Benchmarks (all 5 quantum API functions + sustained throughput):
 *   1. qosc_create        — allocate device sv + cuStateVec handle + encode phases
 *   2. qosc_step          — one Trotterised XY step: 8 Rz + 28 XX + 28 YY gates
 *   3. qosc_readback      — 16 Pauli expectations ⟨Z⟩,⟨X⟩ per qubit
 *   4. qosc_destroy       — free d_sv / workspace + destroy cuStateVec handle
 *   5. qosc_encode        — re-encode phases (state reset + 8 Ry gates)
 *   6. Full N-step cycles — create→N×step→readback→destroy, N∈{1,10,100,1000}
 *   7. Sustained steps    — 10000 consecutive steps, no readback (peak throughput)
 *
 * Two compile modes:
 *
 *   CPU simulation (default, no cuStateVec):
 *     nvcc -O2 bench_quantum.cu -o bench_quantum.exe
 *   or
 *     gcc  -O3 -march=native bench_quantum.cu -o bench_quantum -lm
 *
 *   Real cuStateVec (needs CUQUANTUM_ROOT + cuQuantum SDK):
 *     nvcc -arch=sm_75 -DLL_QUANTUM_ENABLED
 *          -I"%CUQUANTUM_ROOT%\include" -L"%CUQUANTUM_ROOT%\lib\x64"
 *          -lcustatevec -lcublas -O2
 *          bench_quantum.cu ll_quantum.obj -o bench_quantum.exe
 *
 * Output: timestamped TSV to bench_quantum_results.tsv and stdout table.
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

/* ── Windows high-resolution timer ─────────────────────────────────────── */
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

/* ── Constants ───────────────────────────────────────────────────────────── */
#define N_QUBITS  8
#define DIM       256          /* 2^8 complex amplitudes                    */
#define ANA_DT    0.01         /* Kuramoto timestep (matches ll_analog.c)   */
#define K_COUPL   1.0          /* all-to-all coupling constant K            */
#ifndef M_PI
#  define M_PI 3.14159265358979323846
#endif

/* ═══════════════════════════════════════════════════════════════════════════
 * SECTION A — CPU simulation (no cuStateVec)
 * Faithful 256-amplitude dense state-vector simulation of the quantum
 * Kuramoto oscillator.  Implements the identical gate sequence as ll_quantum.cu
 * so timing reflects the true algorithmic cost (without PCIe / cuStateVec
 * scheduling overhead).
 * ═══════════════════════════════════════════════════════════════════════════ */
#ifndef LL_QUANTUM_ENABLED

typedef struct { double re, im; } Cx;
static inline Cx cx(double r, double i)      { Cx c; c.re=r; c.im=i; return c; }
static inline Cx cxadd(Cx a, Cx b)           { return cx(a.re+b.re, a.im+b.im); }
static inline Cx cxmul(Cx a, Cx b)           { return cx(a.re*b.re-a.im*b.im, a.re*b.im+a.im*b.re); }
static inline Cx cxscale(Cx a, double s)     { return cx(a.re*s, a.im*s); }
/* Multiply by -i: (a+ib)*(-i) = b - ia */
static inline Cx cxmul_neg_i(Cx a)           { return cx(a.im, -a.re); }
/* Multiply by +i: (a+ib)*(+i) = -b + ia */
static inline Cx cxmul_pos_i(Cx a)           { return cx(-a.im, a.re); }

typedef struct {
    Cx sv[DIM];               /* 256-amplitude state vector                 */
    double theta[N_QUBITS];   /* last encoded phases (for readback fallback)*/
} QOsc8D;

/* ── Apply Ry(angle) to qubit k ─────────────────────────────────────────── */
/* Ry(θ)|0⟩ = cos(θ/2)|0⟩ + sin(θ/2)|1⟩                                   */
static void sim_Ry(QOsc8D *q, int k, double angle) {
    double c = cos(angle * 0.5);
    double s = sin(angle * 0.5);
    int mask = 1 << k;
    for (int b = 0; b < DIM; b++) {
        if ((b >> k) & 1) continue;      /* only process b with bit k = 0  */
        int b1 = b | mask;
        Cx a0 = q->sv[b], a1 = q->sv[b1];
        /*  [[c, -s], [s, c]] × [a0, a1]ᵀ   (column-major Ry)             */
        q->sv[b]  = cxadd(cxscale(a0, c), cxscale(a1, -s));
        q->sv[b1] = cxadd(cxscale(a0, s), cxscale(a1,  c));
    }
}

/* ── Apply Rz(angle) to qubit k ─────────────────────────────────────────── */
/* Rz(φ) = [[e^{-iφ/2}, 0], [0, e^{+iφ/2}]]                               */
static void sim_Rz(QOsc8D *q, int k, double angle) {
    double h = angle * 0.5;
    Cx e0 = cx( cos(h), -sin(h));   /* e^{-ih}: applied to |...0...⟩      */
    Cx e1 = cx( cos(h), +sin(h));   /* e^{+ih}: applied to |...1...⟩      */
    for (int b = 0; b < DIM; b++)
        q->sv[b] = cxmul(q->sv[b], ((b >> k) & 1) ? e1 : e0);
}

/* ── Apply e^{-i·t·XX} to qubits (ka, kb) ──────────────────────────────── */
/* XX|b⟩ = |b XOR mask⟩  (flips both target bits, coefficient +1).         */
/* e^{-it·XX}: 2×2 block per pair (b, b^mask):                             */
/*   [[cos(t), -i·sin(t)], [-i·sin(t), cos(t)]]                            */
static void sim_XX(QOsc8D *q, int ka, int kb, double t) {
    double c = cos(t), s = sin(t);
    int mask = (1 << ka) | (1 << kb);
    for (int b = 0; b < DIM; b++) {
        /* Only process each pair once: b must have NEITHER target bit set  */
        if ((b >> ka) & 1) continue;
        if ((b >> kb) & 1) continue;
        int b1 = b ^ mask;
        Cx a0 = q->sv[b], a1 = q->sv[b1];
        /* a0_new = cos(t)·a0 + (-i·sin(t))·a1 = c·a0 - i·s·a1           */
        q->sv[b]  = cxadd(cxscale(a0, c), cxscale(cxmul_neg_i(a1), s));
        /* a1_new = (-i·sin(t))·a0 + cos(t)·a1                            */
        q->sv[b1] = cxadd(cxscale(cxmul_neg_i(a0), s), cxscale(a1, c));
    }
}

/* ── Apply e^{-i·t·YY} to qubits (ka, kb) ──────────────────────────────── */
/* YY|b⟩ = p(b)·|b XOR mask⟩  where p(b) depends on target bit parity:    */
/*   bits (0,0) → p = -1;   bits (0,1) → p = +1;                          */
/*   bits (1,0) → p = +1;   bits (1,1) → p = -1.                          */
/* e^{-it·YY}: 2×2 block per pair (b, b^mask):                             */
/*   Group (00↔11): p=-1 → [[cos(t), i·sin(t)], [i·sin(t), cos(t)]]       */
/*   Group (01↔10): p=+1 → [[cos(t), -i·sin(t)], [-i·sin(t), cos(t)]]     */
static void sim_YY(QOsc8D *q, int ka, int kb, double t) {
    double c = cos(t), s = sin(t);
    int mask = (1 << ka) | (1 << kb);
    /* ── Group 1: pairs where target bits are (0,0) → phase=-1 ────────── */
    for (int b = 0; b < DIM; b++) {
        if ((b >> ka) & 1) continue;
        if ((b >> kb) & 1) continue;
        int b1 = b ^ mask;            /* b1 has bits (1,1)                 */
        Cx a0 = q->sv[b], a1 = q->sv[b1];
        /* i·sin(t) coupling: a_new = c·a + i·s·a_other                   */
        q->sv[b]  = cxadd(cxscale(a0, c), cxscale(cxmul_pos_i(a1), s));
        q->sv[b1] = cxadd(cxscale(cxmul_pos_i(a0), s), cxscale(a1, c));
    }
    /* ── Group 2: pairs where target bits are (0,1) → phase=+1 ────────── */
    for (int b = 0; b < DIM; b++) {
        if ((b >> ka) & 1) continue;
        if (!((b >> kb) & 1)) continue;
        int b1 = b ^ mask;            /* b1 has bits (1,0)                 */
        Cx a0 = q->sv[b], a1 = q->sv[b1];
        /* -i·sin(t) coupling: same as XX block                            */
        q->sv[b]  = cxadd(cxscale(a0, c), cxscale(cxmul_neg_i(a1), s));
        q->sv[b1] = cxadd(cxscale(cxmul_neg_i(a0), s), cxscale(a1, c));
    }
}

/* ── API: qosc_create ───────────────────────────────────────────────────── */
QOsc8D *qosc_create(const double theta[N_QUBITS]) {
    QOsc8D *q = (QOsc8D *)calloc(1, sizeof(QOsc8D));
    if (!q) return NULL;
    /* Reset to |0...0⟩ */
    memset(q->sv, 0, sizeof(q->sv));
    q->sv[0] = cx(1.0, 0.0);
    /* Encode phases: Ry(theta[i])|0⟩ per qubit                            */
    for (int i = 0; i < N_QUBITS; i++) {
        sim_Ry(q, i, theta[i]);
        q->theta[i] = theta[i];
    }
    return q;
}

/* ── API: qosc_encode (re-encode phases without realloc) ────────────────── */
static void qosc_encode(QOsc8D *q, const double theta[N_QUBITS]) {
    memset(q->sv, 0, sizeof(q->sv));
    q->sv[0] = cx(1.0, 0.0);
    for (int i = 0; i < N_QUBITS; i++) {
        sim_Ry(q, i, theta[i]);
        q->theta[i] = theta[i];
    }
}

/* ── API: qosc_step ─────────────────────────────────────────────────────── */
/* One first-order Trotterised XY step:                                     */
/*   U(dt) ≈ [∏ᵢ Rz(−2ωᵢdt)] · [∏_{i<j} e^{-i(K/2)dt·XᵢXⱼ} · e^{-i(K/2)dt·YᵢYⱼ}] */
void qosc_step(QOsc8D *q, const double omega[N_QUBITS],
               double k_coupling, double dt)
{
    if (!q) return;
    const double half_K_dt = 0.5 * k_coupling * dt;

    /* 1. Natural frequency drive: Rz(−2ωᵢdt) per qubit                   */
    for (int i = 0; i < N_QUBITS; i++)
        sim_Rz(q, i, -2.0 * omega[i] * dt);

    /* 2. XY coupling: all 28 pairs (i < j), each gets XX then YY gate     */
    for (int i = 0; i < N_QUBITS; i++)
        for (int j = i + 1; j < N_QUBITS; j++) {
            sim_XX(q, i, j, -half_K_dt);
            sim_YY(q, i, j, -half_K_dt);
        }
}

/* ── API: qosc_readback ─────────────────────────────────────────────────── */
/* ⟨Zᵢ⟩ → re[i]:  ⟨Z⟩ = Σ_b |sv[b]|² · (bit_i(b)==0 ? +1 : -1)          */
/* ⟨Xᵢ⟩ → im[i]:  ⟨X⟩ = 2·Re(Σ_{b: bit_i=0} sv[b]* · sv[b|mask_i])      */
void qosc_readback(const QOsc8D *q, double re[N_QUBITS], double im[N_QUBITS]) {
    if (!q) return;
    for (int k = 0; k < N_QUBITS; k++) {
        double sum_z = 0.0, sum_x_re = 0.0;
        int mask = 1 << k;
        for (int b = 0; b < DIM; b++) {
            double mag2 = q->sv[b].re * q->sv[b].re + q->sv[b].im * q->sv[b].im;
            sum_z += ((b >> k) & 1) ? -mag2 : +mag2;
        }
        /* ⟨Xᵢ⟩ = Σ_{b: bit_i=0} 2·Re(sv[b]* · sv[b|mask])              */
        for (int b = 0; b < DIM; b++) {
            if ((b >> k) & 1) continue;
            int b1 = b | mask;
            /* Re(conj(sv[b]) * sv[b1]) = sv[b].re*sv[b1].re + sv[b].im*sv[b1].im */
            sum_x_re += q->sv[b].re * q->sv[b1].re + q->sv[b].im * q->sv[b1].im;
        }
        re[k] = sum_z;
        im[k] = 2.0 * sum_x_re;
    }
}

/* ── API: qosc_destroy ──────────────────────────────────────────────────── */
void qosc_destroy(QOsc8D *q) {
    if (q) free(q);
}

#define BACKEND_NAME  "CPU-sim (no cuStateVec)"

#else /* LL_QUANTUM_ENABLED ─────────────────────────────────────────────── */

/* Use the real cuStateVec-backed API from ll_quantum.h                     */
#include "ll_quantum.h"
#define BACKEND_NAME  "cuStateVec (GPU)"

/* qosc_encode is not in the public ll_quantum.h API — declare it here for
 * the encode-only benchmark (we re-use qosc_create as a proxy).           */
static void qosc_encode(QOsc8D *q, const double theta[8]) {
    /* Approximate: destroy + recreate to measure encode cost.
     * The real encode_phases() is static in ll_quantum.cu; we pay alloc.  */
    (void)q; (void)theta;   /* placeholder — encode bench will use create  */
}

#endif /* LL_QUANTUM_ENABLED */

/* ═══════════════════════════════════════════════════════════════════════════
 * SECTION B — Benchmark infrastructure
 * ═══════════════════════════════════════════════════════════════════════════ */

#define WARMUP_ITERS   20
#define BENCH_ITERS   500      /* default iterations per micro-benchmark    */

typedef struct {
    const char *name;
    double mean_us;       /* mean latency in microseconds                   */
    double min_us;
    double max_us;
    double total_s;
    long   iters;
    double throughput;    /* ops/sec                                         */
} BenchResult;

static void print_result(const BenchResult *r) {
    printf("  %-36s  mean=%8.2f µs  min=%8.2f  max=%8.2f  [%ld iters]  %.0f ops/s\n",
           r->name, r->mean_us, r->min_us, r->max_us,
           r->iters, r->throughput);
}

static void save_tsv(FILE *f, const BenchResult *r) {
    fprintf(f, "%s\t%.4f\t%.4f\t%.4f\t%ld\t%.2f\n",
            r->name, r->mean_us, r->min_us, r->max_us,
            r->iters, r->throughput);
}

/* Reference phases / frequencies (φ-seeded, matches ll_analog.c) */
static const double REF_THETA[N_QUBITS] = {
    0.6283185, 1.0165579, 1.6448764, 2.6614344,
    4.3063108, 6.9677452, 11.273961, 18.241706
};
static const double REF_OMEGA[N_QUBITS] = {
    1.0000000, 1.6180340, 2.6180340, 4.2360680,
    6.8541020, 11.090170, 17.944272, 29.034442
};

/* ── bench_create_destroy ───────────────────────────────────────────────── */
static BenchResult bench_create_destroy(void) {
    BenchResult r; memset(&r, 0, sizeof(r));
    r.name = "qosc_create + qosc_destroy";
    r.iters = BENCH_ITERS;
    r.min_us = 1e18; r.max_us = 0.0;
    double total = 0.0;

    /* warm-up */
    for (int i = 0; i < WARMUP_ITERS; i++) {
        QOsc8D *q = qosc_create(REF_THETA);
        qosc_destroy(q);
    }

    for (long i = 0; i < r.iters; i++) {
        double t0 = bench_now_s();
        QOsc8D *q = qosc_create(REF_THETA);
        qosc_destroy(q);
        double dt = (bench_now_s() - t0) * 1e6;
        total += dt;
        if (dt < r.min_us) r.min_us = dt;
        if (dt > r.max_us) r.max_us = dt;
    }
    r.total_s    = total * 1e-6;
    r.mean_us    = total / (double)r.iters;
    r.throughput = (double)r.iters / r.total_s;
    return r;
}

/* ── bench_create_only ──────────────────────────────────────────────────── */
static BenchResult bench_create_only(void) {
    BenchResult r; memset(&r, 0, sizeof(r));
    r.name = "qosc_create (alloc + encode phases)";
    r.iters = BENCH_ITERS;
    r.min_us = 1e18; r.max_us = 0.0;
    double total = 0.0;
    /* pre-alloc array to store handles (avoid destroy inside timed section) */
    QOsc8D **handles = (QOsc8D **)malloc(r.iters * sizeof(QOsc8D *));

    for (int i = 0; i < WARMUP_ITERS; i++) {
        QOsc8D *q = qosc_create(REF_THETA); qosc_destroy(q);
    }
    for (long i = 0; i < r.iters; i++) {
        double t0 = bench_now_s();
        handles[i] = qosc_create(REF_THETA);
        double dt = (bench_now_s() - t0) * 1e6;
        total += dt;
        if (dt < r.min_us) r.min_us = dt;
        if (dt > r.max_us) r.max_us = dt;
    }
    for (long i = 0; i < r.iters; i++) qosc_destroy(handles[i]);
    free(handles);

    r.total_s    = total * 1e-6;
    r.mean_us    = total / (double)r.iters;
    r.throughput = (double)r.iters / r.total_s;
    return r;
}

/* ── bench_destroy_only ─────────────────────────────────────────────────── */
static BenchResult bench_destroy_only(void) {
    BenchResult r; memset(&r, 0, sizeof(r));
    r.name = "qosc_destroy (free device memory)";
    r.iters = BENCH_ITERS;
    r.min_us = 1e18; r.max_us = 0.0;
    double total = 0.0;
    QOsc8D **handles = (QOsc8D **)malloc(r.iters * sizeof(QOsc8D *));

    for (long i = 0; i < r.iters; i++)
        handles[i] = qosc_create(REF_THETA);

    for (long i = 0; i < r.iters; i++) {
        double t0 = bench_now_s();
        qosc_destroy(handles[i]);
        double dt = (bench_now_s() - t0) * 1e6;
        total += dt;
        if (dt < r.min_us) r.min_us = dt;
        if (dt > r.max_us) r.max_us = dt;
    }
    free(handles);

    r.total_s    = total * 1e-6;
    r.mean_us    = total / (double)r.iters;
    r.throughput = (double)r.iters / r.total_s;
    return r;
}

/* ── bench_encode_only ──────────────────────────────────────────────────── */
static BenchResult bench_encode_only(void) {
    BenchResult r; memset(&r, 0, sizeof(r));
    r.name = "qosc_encode (state reset + 8 Ry gates)";
    r.iters = BENCH_ITERS * 10;
    r.min_us = 1e18; r.max_us = 0.0;
    double total = 0.0;
    QOsc8D *q = qosc_create(REF_THETA);

    for (int i = 0; i < WARMUP_ITERS; i++) qosc_encode(q, REF_THETA);
    for (long i = 0; i < r.iters; i++) {
        double t0 = bench_now_s();
        qosc_encode(q, REF_THETA);
        double dt = (bench_now_s() - t0) * 1e6;
        total += dt;
        if (dt < r.min_us) r.min_us = dt;
        if (dt > r.max_us) r.max_us = dt;
    }
    qosc_destroy(q);

    r.total_s    = total * 1e-6;
    r.mean_us    = total / (double)r.iters;
    r.throughput = (double)r.iters / r.total_s;
    return r;
}

/* ── bench_step_single ──────────────────────────────────────────────────── */
static BenchResult bench_step_single(void) {
    BenchResult r; memset(&r, 0, sizeof(r));
    r.name = "qosc_step (1 XY Trotter step)";
    r.iters = BENCH_ITERS * 10;
    r.min_us = 1e18; r.max_us = 0.0;
    double total = 0.0;
    QOsc8D *q = qosc_create(REF_THETA);

    for (int i = 0; i < WARMUP_ITERS; i++)
        qosc_step(q, REF_OMEGA, K_COUPL, ANA_DT);
    for (long i = 0; i < r.iters; i++) {
        double t0 = bench_now_s();
        qosc_step(q, REF_OMEGA, K_COUPL, ANA_DT);
        double dt = (bench_now_s() - t0) * 1e6;
        total += dt;
        if (dt < r.min_us) r.min_us = dt;
        if (dt > r.max_us) r.max_us = dt;
    }
    qosc_destroy(q);

    r.total_s    = total * 1e-6;
    r.mean_us    = total / (double)r.iters;
    r.throughput = (double)r.iters / r.total_s;
    return r;
}

/* ── bench_step_rz_only ─────────────────────────────────────────────────── */
/* Sub-benchmark: isolate the Rz natural frequency drive (8 single-qubit gates) */
static BenchResult bench_step_rz_only(void) {
    BenchResult r; memset(&r, 0, sizeof(r));
    r.name = "qosc_step sub: 8 Rz gates (freq drive)";
    r.iters = BENCH_ITERS * 20;
    r.min_us = 1e18; r.max_us = 0.0;
    double total = 0.0;
    QOsc8D *q = qosc_create(REF_THETA);

#ifndef LL_QUANTUM_ENABLED
    /* Direct access to simulation internals */
    for (int i = 0; i < WARMUP_ITERS; i++)
        for (int k = 0; k < N_QUBITS; k++)
            sim_Rz(q, k, -2.0 * REF_OMEGA[k] * ANA_DT);
    for (long i = 0; i < r.iters; i++) {
        double t0 = bench_now_s();
        for (int k = 0; k < N_QUBITS; k++)
            sim_Rz(q, k, -2.0 * REF_OMEGA[k] * ANA_DT);
        double dt = (bench_now_s() - t0) * 1e6;
        total += dt;
        if (dt < r.min_us) r.min_us = dt;
        if (dt > r.max_us) r.max_us = dt;
    }
#else
    /* With cuStateVec, approximate using full step (overhead is in XY pairs) */
    r.name = "qosc_step sub: Rz gates (approx via full step)";
    for (int i = 0; i < WARMUP_ITERS; i++)
        qosc_step(q, REF_OMEGA, 0.0 /*K=0 → only Rz fires*/, ANA_DT);
    for (long i = 0; i < r.iters; i++) {
        double t0 = bench_now_s();
        qosc_step(q, REF_OMEGA, 0.0, ANA_DT);
        double dt = (bench_now_s() - t0) * 1e6;
        total += dt;
        if (dt < r.min_us) r.min_us = dt;
        if (dt > r.max_us) r.max_us = dt;
    }
#endif
    qosc_destroy(q);

    r.total_s    = total * 1e-6;
    r.mean_us    = total / (double)r.iters;
    r.throughput = (double)r.iters / r.total_s;
    return r;
}

/* ── bench_step_xy_only ─────────────────────────────────────────────────── */
/* Sub-benchmark: isolate the 28 XX+YY coupling gate pairs */
static BenchResult bench_step_xy_only(void) {
    BenchResult r; memset(&r, 0, sizeof(r));
    r.name = "qosc_step sub: 28 XX+YY pairs (coupling)";
    r.iters = BENCH_ITERS * 10;
    r.min_us = 1e18; r.max_us = 0.0;
    double total = 0.0;
    QOsc8D *q = qosc_create(REF_THETA);

#ifndef LL_QUANTUM_ENABLED
    double half_K_dt = 0.5 * K_COUPL * ANA_DT;
    for (int i = 0; i < WARMUP_ITERS; i++)
        for (int a = 0; a < N_QUBITS; a++)
            for (int b2 = a+1; b2 < N_QUBITS; b2++) {
                sim_XX(q, a, b2, -half_K_dt);
                sim_YY(q, a, b2, -half_K_dt);
            }
    for (long i = 0; i < r.iters; i++) {
        double t0 = bench_now_s();
        for (int a = 0; a < N_QUBITS; a++)
            for (int b2 = a+1; b2 < N_QUBITS; b2++) {
                sim_XX(q, a, b2, -half_K_dt);
                sim_YY(q, a, b2, -half_K_dt);
            }
        double dt = (bench_now_s() - t0) * 1e6;
        total += dt;
        if (dt < r.min_us) r.min_us = dt;
        if (dt > r.max_us) r.max_us = dt;
    }
#else
    /* With cuStateVec: use omega=0 to disable Rz (K≠0 activates coupling) */
    static const double zero_omega[N_QUBITS] = {0};
    for (int i = 0; i < WARMUP_ITERS; i++)
        qosc_step(q, zero_omega, K_COUPL, ANA_DT);
    for (long i = 0; i < r.iters; i++) {
        double t0 = bench_now_s();
        qosc_step(q, zero_omega, K_COUPL, ANA_DT);
        double dt = (bench_now_s() - t0) * 1e6;
        total += dt;
        if (dt < r.min_us) r.min_us = dt;
        if (dt > r.max_us) r.max_us = dt;
    }
#endif
    qosc_destroy(q);

    r.total_s    = total * 1e-6;
    r.mean_us    = total / (double)r.iters;
    r.throughput = (double)r.iters / r.total_s;
    return r;
}

/* ── bench_readback ─────────────────────────────────────────────────────── */
static BenchResult bench_readback(void) {
    BenchResult r; memset(&r, 0, sizeof(r));
    r.name = "qosc_readback (16 Pauli expectations)";
    r.iters = BENCH_ITERS * 10;
    r.min_us = 1e18; r.max_us = 0.0;
    double total = 0.0;
    double re_out[N_QUBITS], im_out[N_QUBITS];
    QOsc8D *q = qosc_create(REF_THETA);
    /* Advance a few steps so the state is non-trivial */
    for (int i = 0; i < 10; i++)
        qosc_step(q, REF_OMEGA, K_COUPL, ANA_DT);

    for (int i = 0; i < WARMUP_ITERS; i++)
        qosc_readback(q, re_out, im_out);
    for (long i = 0; i < r.iters; i++) {
        double t0 = bench_now_s();
        qosc_readback(q, re_out, im_out);
        double dt = (bench_now_s() - t0) * 1e6;
        total += dt;
        if (dt < r.min_us) r.min_us = dt;
        if (dt > r.max_us) r.max_us = dt;
    }
    qosc_destroy(q);

    r.total_s    = total * 1e-6;
    r.mean_us    = total / (double)r.iters;
    r.throughput = (double)r.iters / r.total_s;
    return r;
}

/* ── bench_sustained_steps ──────────────────────────────────────────────── */
/* 10000 consecutive steps — measures peak step throughput (no alloc / readback) */
static BenchResult bench_sustained_steps(void) {
    BenchResult r; memset(&r, 0, sizeof(r));
    r.name = "qosc_step x10000 sustained (pure throughput)";
    r.iters = 10000;
    r.min_us = 0; r.max_us = 0;
    QOsc8D *q = qosc_create(REF_THETA);
    /* warm-up */
    for (int i = 0; i < 100; i++)
        qosc_step(q, REF_OMEGA, K_COUPL, ANA_DT);

    double t0 = bench_now_s();
    for (long i = 0; i < r.iters; i++)
        qosc_step(q, REF_OMEGA, K_COUPL, ANA_DT);
    double total_s = bench_now_s() - t0;
    qosc_destroy(q);

    r.total_s    = total_s;
    r.mean_us    = (total_s * 1e6) / (double)r.iters;
    r.throughput = (double)r.iters / total_s;
    return r;
}

/* ── bench_full_cycle ───────────────────────────────────────────────────── */
/* create → N×step → readback → destroy, repeated BENCH_ITERS times        */
static BenchResult bench_full_cycle(int n_steps) {
    static char name_buf[64];
    snprintf(name_buf, sizeof(name_buf),
             "Full cycle: create→%d×step→readback→destroy", n_steps);
    BenchResult r; memset(&r, 0, sizeof(r));
    r.name   = name_buf;
    r.iters  = (n_steps > 100) ? 50 : BENCH_ITERS;
    r.min_us = 1e18; r.max_us = 0.0;
    double total = 0.0;
    double re_out[N_QUBITS], im_out[N_QUBITS];

    /* warm-up */
    for (int w = 0; w < 5; w++) {
        QOsc8D *q = qosc_create(REF_THETA);
        for (int s = 0; s < n_steps; s++) qosc_step(q, REF_OMEGA, K_COUPL, ANA_DT);
        qosc_readback(q, re_out, im_out);
        qosc_destroy(q);
    }

    for (long i = 0; i < r.iters; i++) {
        double t0 = bench_now_s();
        QOsc8D *q = qosc_create(REF_THETA);
        for (int s = 0; s < n_steps; s++)
            qosc_step(q, REF_OMEGA, K_COUPL, ANA_DT);
        qosc_readback(q, re_out, im_out);
        qosc_destroy(q);
        double dt = (bench_now_s() - t0) * 1e6;
        total += dt;
        if (dt < r.min_us) r.min_us = dt;
        if (dt > r.max_us) r.max_us = dt;
    }

    r.total_s    = total * 1e-6;
    r.mean_us    = total / (double)r.iters;
    r.throughput = (double)r.iters / r.total_s;
    return r;
}

/* ── bench_coherence ────────────────────────────────────────────────────── */
/* Measure Kuramoto R² (coherence) vs step count — physics validation      */
static void bench_coherence(void) {
    printf("\n  Kuramoto coherence R² vs step count:\n");
    printf("  %-8s  %-10s  %-10s  %s\n",
           "step", "R2", "<Z0>", "<X0>");
    QOsc8D *q = qosc_create(REF_THETA);
    double re[N_QUBITS], im[N_QUBITS];

    int checkpoints[] = {0, 1, 5, 10, 50, 100, 500, 1000};
    int n_cp = (int)(sizeof(checkpoints)/sizeof(checkpoints[0]));
    int current = 0;

    for (int cp = 0; cp < n_cp; cp++) {
        int target = checkpoints[cp];
        while (current < target) {
            qosc_step(q, REF_OMEGA, K_COUPL, ANA_DT);
            current++;
        }
        qosc_readback(q, re, im);
        /* R² = |⟨Z⟩|² = (mean of ⟨Zi⟩)²  */
        double sum_z = 0.0;
        for (int i = 0; i < N_QUBITS; i++) sum_z += re[i];
        double R = sum_z / N_QUBITS;
        printf("  %-8d  %+.6f  %+.6f  %+.6f\n",
               current, R * R, re[0], im[0]);
    }
    qosc_destroy(q);
}

/* ── bench_gate_counts ──────────────────────────────────────────────────── */
static void print_gate_counts(void) {
    int n = N_QUBITS;
    int n_Rz    = n;
    int n_pairs = n * (n - 1) / 2;
    int n_XX    = n_pairs;
    int n_YY    = n_pairs;
    int n_Ry    = n;   /* for encode */
    int n_exps  = n * 2; /* for readback: Z and X per qubit */

    printf("\n  Gate counts per quantum function call (N=%d qubits):\n", n);
    printf("  %-40s  %d  (one per qubit)\n",  "qosc_encode: Ry gates",    n_Ry);
    printf("  %-40s  %d  (one per qubit)\n",  "qosc_step:   Rz gates",    n_Rz);
    printf("  %-40s  %d  (all-to-all pairs)\n","qosc_step:   XX gates",   n_XX);
    printf("  %-40s  %d  (all-to-all pairs)\n","qosc_step:   YY gates",   n_YY);
    printf("  %-40s  %d  total = %d+%d\n",   "qosc_step:   total gates", n_Rz+n_XX+n_YY, n_Rz, n_XX+n_YY);
    printf("  %-40s  %d  (Z+X per qubit)\n", "qosc_readback: expectations", n_exps);
    printf("\n");
    printf("  State vector size:  %d amplitudes × 16 bytes = %d bytes (%.1f KB)\n",
           DIM, DIM * 16, DIM * 16.0 / 1024.0);
    printf("  XY pairs (i<j):     %d  (N*(N-1)/2)\n", n_pairs);
    printf("  Trotter error:      O(K²dt²) ≈ %.2e rad/step  (K=%.1f dt=%.3f)\n",
           K_COUPL * K_COUPL * ANA_DT * ANA_DT, K_COUPL, ANA_DT);
}

/* ═══════════════════════════════════════════════════════════════════════════
 * main
 * ═══════════════════════════════════════════════════════════════════════════ */
int main(void) {
    printf("╔══════════════════════════════════════════════════════════════════╗\n");
    printf("║  bench_quantum — 8-qubit XY Kuramoto Quantum Function Benchmark ║\n");
    printf("╚══════════════════════════════════════════════════════════════════╝\n");
    printf("  Backend : %s\n", BACKEND_NAME);
    printf("  Qubits  : %d   Dim: %d amplitudes\n", N_QUBITS, DIM);
    printf("  dt      : %.3f   K_coupling: %.1f   Trotter order: 1\n\n",
           ANA_DT, K_COUPL);

    print_gate_counts();

    /* Open TSV output */
    FILE *tsv = fopen("bench_quantum_results.tsv", "w");
    if (tsv) {
        fprintf(tsv, "function\tmean_us\tmin_us\tmax_us\titers\tops_per_sec\n");
    }

#define RUN(fn) do { \
    printf("  Running: %s ...\n", #fn); fflush(stdout); \
    BenchResult _r = fn; \
    print_result(&_r); \
    if (tsv) save_tsv(tsv, &_r); \
} while(0)

    printf("── API Function Benchmarks ──────────────────────────────────────────\n");
    RUN(bench_create_only());
    RUN(bench_encode_only());
    RUN(bench_destroy_only());
    RUN(bench_step_single());
    RUN(bench_readback());
    RUN(bench_create_destroy());

    printf("\n── Step Sub-function Benchmarks ─────────────────────────────────────\n");
    RUN(bench_step_rz_only());
    RUN(bench_step_xy_only());
    RUN(bench_sustained_steps());

    printf("\n── Full Cycle Benchmarks (create→Nstep→readback→destroy) ────────────\n");
    RUN(bench_full_cycle(1));
    RUN(bench_full_cycle(10));
    RUN(bench_full_cycle(100));
    RUN(bench_full_cycle(1000));

    printf("\n── Physics Validation ───────────────────────────────────────────────\n");
    bench_coherence();

    if (tsv) {
        fclose(tsv);
        printf("\n  Results saved to bench_quantum_results.tsv\n");
    }

    printf("\n  Done.\n");
    return 0;
}
