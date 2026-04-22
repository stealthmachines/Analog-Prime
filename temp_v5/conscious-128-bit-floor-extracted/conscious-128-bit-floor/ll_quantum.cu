/* ll_quantum.cu — 8-qubit cuStateVec quantum Kuramoto oscillator
 *
 * Implements qosc_create / qosc_step / qosc_readback / qosc_destroy.
 * See ll_quantum.h for full design rationale and API contract.
 *
 * Compile:
 *   nvcc -arch=sm_75 -I"%CUQUANTUM_ROOT%\include" -L"%CUQUANTUM_ROOT%\lib\x64"
 *        -lcustatevec -lcublas -O2 -c ll_quantum.cu -o ll_quantum.obj
 *
 * Licensed per https://zchg.org/t/legal-notice-copyright-applicable-ip-and-licensing-read-me/440
 */

#include "ll_quantum.h"
#include <custatevec.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ── Internal constants ───────────────────────────────────────────────────── */
#define QOSC_N_QUBITS   8
#define QOSC_DIM        (1 << QOSC_N_QUBITS)   /* 256 amplitudes */

/* ── Error helpers ────────────────────────────────────────────────────────── */
#define CUDA_CHECK(x) do {                                              \
    cudaError_t _e = (x);                                              \
    if (_e != cudaSuccess) {                                           \
        fprintf(stderr, "[qosc] CUDA error %s at %s:%d\n",            \
                cudaGetErrorString(_e), __FILE__, __LINE__);           \
        return;                                                        \
    }                                                                  \
} while(0)

#define CSV_CHECK(x) do {                                              \
    custatevecStatus_t _s = (x);                                       \
    if (_s != CUSTATEVEC_STATUS_SUCCESS) {                             \
        fprintf(stderr, "[qosc] cuStateVec error %d at %s:%d\n",      \
                (int)_s, __FILE__, __LINE__);                          \
        return;                                                        \
    }                                                                  \
} while(0)

#define CSV_CHECK_NULL(x) do {                                         \
    custatevecStatus_t _s = (x);                                       \
    if (_s != CUSTATEVEC_STATUS_SUCCESS) {                             \
        fprintf(stderr, "[qosc] cuStateVec error %d at %s:%d\n",      \
                (int)_s, __FILE__, __LINE__);                          \
        return NULL;                                                   \
    }                                                                  \
} while(0)

/* ── Opaque state struct ──────────────────────────────────────────────────── */
struct QOsc8D_s {
    custatevecHandle_t  handle;
    void               *d_sv;        /* device state vector: QOSC_DIM cuDoubleComplex */
    void               *d_workspace; /* cuStateVec scratch                             */
    size_t              workspace_sz;
    double              theta[QOSC_N_QUBITS];   /* last known phases (for re-encode) */
};

/* ── Gate helpers ─────────────────────────────────────────────────────────── */

/* Apply Ry(angle) to qubit `target` of the state vector.
 * Ry(θ) = [[cos(θ/2), -sin(θ/2)], [sin(θ/2), cos(θ/2)]]             */
static custatevecStatus_t apply_Ry(QOsc8D *q, int target, double angle) {
    double c = cos(angle * 0.5);
    double s = sin(angle * 0.5);
    /* cuStateVec expects column-major 2×2 complex matrix as 4 cuDoubleComplex */
    cuDoubleComplex mat[4] = {
        {c, 0}, {s, 0},   /* col 0: [cos, sin] */
        {-s, 0}, {c, 0}   /* col 1: [-sin, cos] */
    };
    int targets[1] = {target};
    return custatevecApplyMatrix(
        q->handle, q->d_sv, CUDA_C_64F, QOSC_N_QUBITS,
        mat, CUDA_C_64F, CUSTATEVEC_MATRIX_LAYOUT_COL,
        0,              /* adjoint=0 */
        targets, 1,
        NULL, NULL, 0,  /* no controls */
        CUSTATEVEC_COMPUTE_64F,
        q->d_workspace, q->workspace_sz);
}

/* Apply Rz(angle) to qubit `target`.
 * Rz(φ) = [[e^{-iφ/2}, 0], [0, e^{iφ/2}]]                           */
static custatevecStatus_t apply_Rz(QOsc8D *q, int target, double angle) {
    double h = angle * 0.5;
    cuDoubleComplex mat[4] = {
        {cos(h), -sin(h)}, {0, 0},
        {0, 0},            {cos(h),  sin(h)}
    };
    int targets[1] = {target};
    return custatevecApplyMatrix(
        q->handle, q->d_sv, CUDA_C_64F, QOSC_N_QUBITS,
        mat, CUDA_C_64F, CUSTATEVEC_MATRIX_LAYOUT_COL,
        0,
        targets, 1,
        NULL, NULL, 0,
        CUSTATEVEC_COMPUTE_64F,
        q->d_workspace, q->workspace_sz);
}

/* Apply e^{-i·param·P_a⊗P_b} to qubits (ta, tb) via custatevecApplyPauliExp.
 * Used for XX and YY terms of the XY coupling.                        */
static custatevecStatus_t apply_pauli_exp(QOsc8D *q,
    custatevecPauli_t pa, custatevecPauli_t pb,
    int ta, int tb, double param)
{
    custatevecPauli_t paulis[2]  = {pa, pb};
    int32_t           targets[2] = {ta, tb};
    return custatevecApplyPauliExp(
        q->handle, q->d_sv, CUDA_C_64F, QOSC_N_QUBITS,
        (float)param,       /* cuStateVec takes float param for PauliExp */
        paulis, targets, 2,
        NULL, NULL, 0);     /* no controls */
}

/* ── Encode classical phases into the state vector ────────────────────────── */
static void encode_phases(QOsc8D *q, const double theta[QOSC_N_QUBITS]) {
    /* Reset to |00...0⟩ */
    cuDoubleComplex h_sv[QOSC_DIM];
    memset(h_sv, 0, sizeof(h_sv));
    h_sv[0].x = 1.0;  h_sv[0].y = 0.0;
    cudaMemcpy(q->d_sv, h_sv, QOSC_DIM * sizeof(cuDoubleComplex),
               cudaMemcpyHostToDevice);

    /* Apply Ry(theta[i]) to each qubit i — encodes phase as Bloch-sphere latitude.
     * Ry(θ)|0⟩ = cos(θ/2)|0⟩ + sin(θ/2)|1⟩
     *   ⟨Z⟩ = cos²(θ/2) − sin²(θ/2) = cos(θ)   ← re[i]
     *   ⟨X⟩ = 2cos(θ/2)sin(θ/2)     = sin(θ)   ← im[i]  */
    for (int i = 0; i < QOSC_N_QUBITS; i++) {
        apply_Ry(q, i, theta[i]);
        q->theta[i] = theta[i];
    }
}

/* ── qosc_create ──────────────────────────────────────────────────────────── */
QOsc8D *qosc_create(const double theta[QOSC_N_QUBITS]) {
    QOsc8D *q = (QOsc8D *)calloc(1, sizeof(QOsc8D));
    if (!q) return NULL;

    /* cuStateVec handle */
    CSV_CHECK_NULL(custatevecCreate(&q->handle));

    /* Device state vector: QOSC_DIM = 256 cuDoubleComplex = 4096 bytes */
    cudaError_t ce = cudaMalloc(&q->d_sv, QOSC_DIM * sizeof(cuDoubleComplex));
    if (ce != cudaSuccess) {
        fprintf(stderr, "[qosc] cudaMalloc sv failed: %s\n",
                cudaGetErrorString(ce));
        custatevecDestroy(q->handle);
        free(q);
        return NULL;
    }

    /* Query and allocate workspace */
    q->workspace_sz = 0;
    custatevecApplyMatrix_bufferSize(
        q->handle, CUDA_C_64F, QOSC_N_QUBITS,
        NULL, CUDA_C_64F, CUSTATEVEC_MATRIX_LAYOUT_COL,
        0, 1, 0, CUSTATEVEC_COMPUTE_64F, &q->workspace_sz);
    if (q->workspace_sz > 0) {
        ce = cudaMalloc(&q->d_workspace, q->workspace_sz);
        if (ce != cudaSuccess) {
            fprintf(stderr, "[qosc] cudaMalloc workspace failed\n");
            cudaFree(q->d_sv);
            custatevecDestroy(q->handle);
            free(q);
            return NULL;
        }
    }

    encode_phases(q, theta);

    fprintf(stderr, "[qosc] init: %d qubits, sv=%zu bytes, workspace=%zu bytes\n",
            QOSC_N_QUBITS,
            (size_t)QOSC_DIM * sizeof(cuDoubleComplex),
            q->workspace_sz);
    return q;
}

/* ── qosc_step ────────────────────────────────────────────────────────────── */
/* One first-order Trotterised XY step:
 *
 *   U(dt) ≈ [∏ᵢ Rz(−2ωᵢdt)] · [∏_{i<j} e^{-i(K/2)dt·XᵢXⱼ} · e^{-i(K/2)dt·YᵢYⱼ}]
 *
 * The XX+YY pair gate is the quantum Kuramoto coupling:
 *   e^{-i·t·(XᵢXⱼ+YᵢYⱼ)} conserves total Sz and drives phase synchronisation
 *   exactly as sin(θⱼ−θᵢ) does in the classical Kuramoto model.
 *
 * Trotter error: O(K²dt²) per step.  At K≤5, dt=0.01: error ≈ 2.5×10⁻³ rad.
 * Acceptable for a correction signal; the exact LL arithmetic is unaffected.  */
void qosc_step(QOsc8D *q, const double omega[QOSC_N_QUBITS],
               double k_coupling, double dt)
{
    if (!q) return;
    const double half_K_dt = 0.5 * k_coupling * dt;   /* param for XX, YY gates */

    /* 1. Natural frequency drive: Rz(−2ωᵢdt) per qubit */
    for (int i = 0; i < QOSC_N_QUBITS; i++) {
        custatevecStatus_t s = apply_Rz(q, i, -2.0 * omega[i] * dt);
        if (s != CUSTATEVEC_STATUS_SUCCESS) {
            fprintf(stderr, "[qosc] Rz(%d) failed: %d\n", i, (int)s);
            return;
        }
    }

    /* 2. XY coupling: all 28 pairs (i < j), each gets XX then YY gate */
    for (int i = 0; i < QOSC_N_QUBITS; i++) {
        for (int j = i + 1; j < QOSC_N_QUBITS; j++) {
            custatevecStatus_t sx = apply_pauli_exp(
                q, CUSTATEVEC_PAULI_X, CUSTATEVEC_PAULI_X,
                i, j, -half_K_dt);
            custatevecStatus_t sy = apply_pauli_exp(
                q, CUSTATEVEC_PAULI_Y, CUSTATEVEC_PAULI_Y,
                i, j, -half_K_dt);
            if (sx != CUSTATEVEC_STATUS_SUCCESS ||
                sy != CUSTATEVEC_STATUS_SUCCESS) {
                fprintf(stderr, "[qosc] XY gate (%d,%d) failed\n", i, j);
                return;
            }
        }
    }
}

/* ── qosc_readback ────────────────────────────────────────────────────────── */
/* Compute ⟨Zᵢ⟩ → re[i] and ⟨Xᵢ⟩ → im[i] for each qubit.
 *
 * custatevecComputeExpectationsOnPauliBasis evaluates a batch of Pauli
 * expectation values in one call, which is more efficient than individual
 * matrix expectations.  Each qubit gets two entries: one Z, one X.
 *
 * Memory layout:
 *   pauliOps[2*N]    — alternating Z, X operators per qubit
 *   targets[2*N][1]  — qubit index for each operator
 *   expects[2*N]     — output expectation values (real, since Z and X are Hermitian)  */
void qosc_readback(const QOsc8D *q, double re[QOSC_N_QUBITS],
                   double im[QOSC_N_QUBITS])
{
    if (!q) return;

    const int n_ops = 2 * QOSC_N_QUBITS;   /* Z and X per qubit = 16 */

    custatevecPauli_t  pauliOps[2 * QOSC_N_QUBITS];
    int32_t            targets_flat[2 * QOSC_N_QUBITS];
    int32_t           *basisBits[2 * QOSC_N_QUBITS];
    uint32_t           nBasisBits[2 * QOSC_N_QUBITS];
    double             expects[2 * QOSC_N_QUBITS];

    for (int i = 0; i < QOSC_N_QUBITS; i++) {
        /* Even index: Z expectation → re[i] */
        pauliOps[2*i]       = CUSTATEVEC_PAULI_Z;
        targets_flat[2*i]   = i;
        basisBits[2*i]      = &targets_flat[2*i];
        nBasisBits[2*i]     = 1;
        /* Odd index: X expectation → im[i] */
        pauliOps[2*i+1]     = CUSTATEVEC_PAULI_X;
        targets_flat[2*i+1] = i;
        basisBits[2*i+1]    = &targets_flat[2*i+1];
        nBasisBits[2*i+1]   = 1;
    }

    custatevecStatus_t s = custatevecComputeExpectationsOnPauliBasis(
        q->handle, q->d_sv, CUDA_C_64F, QOSC_N_QUBITS,
        expects,
        (const custatevecPauli_t **)&pauliOps,  /* one Pauli per operator */
        n_ops,
        (const int32_t **)basisBits,
        nBasisBits);

    if (s != CUSTATEVEC_STATUS_SUCCESS) {
        fprintf(stderr, "[qosc] readback expectation failed: %d\n", (int)s);
        /* On failure return the classical-consistent values */
        for (int i = 0; i < QOSC_N_QUBITS; i++) {
            re[i] = cos(q->theta[i]);
            im[i] = sin(q->theta[i]);
        }
        return;
    }

    for (int i = 0; i < QOSC_N_QUBITS; i++) {
        re[i] = expects[2*i];     /* ⟨Zᵢ⟩ = cos(θᵢ) */
        im[i] = expects[2*i+1];   /* ⟨Xᵢ⟩ = sin(θᵢ) */
    }
}

/* ── qosc_destroy ─────────────────────────────────────────────────────────── */
void qosc_destroy(QOsc8D *q) {
    if (!q) return;
    if (q->d_workspace) cudaFree(q->d_workspace);
    if (q->d_sv)        cudaFree(q->d_sv);
    custatevecDestroy(q->handle);
    free(q);
}
