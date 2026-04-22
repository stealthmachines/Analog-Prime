/* ll_quantum.h — 8-qubit cuStateVec layer for the Kuramoto oscillator
 *
 * Implements the quantum simulation of the Kuramoto model via the XY Hamiltonian:
 *
 *   H = Σᵢ ωᵢ Zᵢ  +  (K/2) Σ_{i<j} (XᵢXⱼ + YᵢYⱼ)
 *
 * The XY term is the quantum analog of classical Kuramoto coupling:
 *   <XᵢXⱼ + YᵢYⱼ> ∝ cos(θᵢ−θⱼ) + i·sin(θᵢ−θⱼ)
 * Its imaginary part recovers exactly the sin(θⱼ−θᵢ) term in ana_deriv.
 *
 * State encoding:  Ry(θᵢ)|0⟩ = cos(θᵢ/2)|0⟩ + sin(θᵢ/2)|1⟩
 *   → ⟨Zᵢ⟩ = cos(θᵢ)  = re[i]  in AnaOsc8D
 *   → ⟨Xᵢ⟩ = sin(θᵢ)  = im[i]  in AnaOsc8D (on equator; no Rz offset)
 *
 * Time evolution (Trotterised, first-order, one step = ANA_DT):
 *   1. Single-qubit Rz(−2ωᵢ·dt):  natural frequency drive
 *   2. XX+YY pair gates e^{−i(K/2)dt(XᵢXⱼ+YᵢYⱼ)}:  Kuramoto coupling
 *      Applied via two custatevecApplyPauliExp calls per pair (XX then YY).
 *      28 pairs for N=8 qubits (all-to-all, matching ana_deriv mean-field).
 *   3. Readback: expectation values ⟨Zᵢ⟩ → re[i],  ⟨Xᵢ⟩ → im[i]
 *
 * Blending with classical RK4 (in ll_analog.c):
 *   re[i] ← (1−α)·re_classical[i]  +  α·re_quantum[i]
 *   im[i] ← (1−α)·im_classical[i]  +  α·im_quantum[i]
 *   α is phase-adaptive: Pluck=0.1, Sustain=0.2, FineTune=0.35, Lock=0.5
 *   Blend is renormalised to unit circle after mixing.
 *
 * Hardware: RTX 2060 (SM 7.5).  cuStateVec requires SM ≥ 7.0 — compatible.
 * State vector: 2^8 = 256 complex<double> amplitudes = 4 KB on device.
 * Cost per step: 28×2 PauliExp calls + 8 Rz + 8 expectation reads.
 *   At 256 amplitudes this is negligible vs the CPU schoolbook squaring.
 *
 * Build:
 *   nvcc -arch=sm_75 -lcustatevec -lcublas ll_quantum.cu -c -o ll_quantum.obj
 *   (link ll_quantum.obj alongside ll_analog.obj into ll_mpi.exe)
 *
 * Licensed per https://zchg.org/t/legal-notice-copyright-applicable-ip-and-licensing-read-me/440
 */
#pragma once
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Opaque quantum oscillator handle (hides cuStateVec internals from C). */
typedef struct QOsc8D_s QOsc8D;

/* Allocate and initialise an 8-qubit state vector.
 * theta[8]: initial Kuramoto phases — encoded as Ry(theta[i])|0⟩ per qubit.
 * Returns NULL on cuStateVec init failure (no GPU / no cuQuantum library). */
QOsc8D *qosc_create(const double theta[8]);

/* Evolve the state vector by one Trotterised XY step of duration dt.
 * omega[8]: natural frequencies (rad/step).
 * k_coupling: all-to-all coupling constant K (same as AnaOsc8D.k_coupling).
 * dt: timestep (ANA_DT = 0.01).  */
void qosc_step(QOsc8D *q, const double omega[8], double k_coupling, double dt);

/* Read back ⟨Zᵢ⟩ → re[8] and ⟨Xᵢ⟩ → im[8].
 * These are the same semantic quantities as AnaOsc8D.re / .im. */
void qosc_readback(const QOsc8D *q, double re[8], double im[8]);

/* Free all device memory and destroy the cuStateVec handle. */
void qosc_destroy(QOsc8D *q);

/* Phase-adaptive blend weight: Pluck→0.1, Sustain→0.2, FineTune→0.35, Lock→0.5.
 * Matches APhase enum values 0..3. */
static inline double qosc_blend_alpha(int aphase) {
    static const double A[4] = {0.10, 0.20, 0.35, 0.50};
    return (aphase >= 0 && aphase <= 3) ? A[aphase] : 0.10;
}

#ifdef __cplusplus
}
#endif
