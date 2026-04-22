// ============================================================================
// hdgl_analog_v31b.s  —  Annotated x86-64 GAS Assembly Reference
// HDGL Analog Mainnet v3.1b  —  Open-Recursive Throughput Hemisphere
// ============================================================================
//
// Purpose: Documents the x86-64 machine-code structure of the v31b hot paths.
//   Focus: SoA memory access patterns, Euler inner loop, LL-lite 32-bit
//          integer pipeline, 4-harmonic per-slot spectral kernel.
//
// Compile flags assumed (Linux, GCC 12+):
//   gcc -O3 -march=native -mavx2 -mfma -ffast-math hdgl_analog_v31b.c -o hdgl_v31b -lm
//   NOTE: -O3 + -ffast-math enables auto-vectorisation for the Euler loop
//         since SoA layout allows contiguous double arrays.
//
// Register conventions (System V AMD64 ABI):
//   Integer args : rdi, rsi, rdx, rcx, r8, r9
//   Float  args  : xmm0–xmm7
//   Callee-saved : rbx, rbp, r12–r15
//   Return       : rax (int), xmm0 (double)
//   AVX2 YMM    : ymm0–ymm15  (caller-saved xmm0–xmm15 upper halves)
//
// Covered hot paths:
//   1. fast_lattice_step  — Euler inner loop (SoA load pattern)
//   2. ll_step32_x4       — 4 LL-lite 32-bit iterations (integer pipeline)
//   3. spectral_eval_4    — 4-harmonic float spectral kernel
//   4. gra_plasticity     — open r_harmonic evolution (bounded)
//   5. fast_lattice_emit_candidates — threshold gate + bridge emission
//
// SoA memory access commentary:
//   The FastLattice struct stores each field as a contiguous array.
//   For slot i, the loads are:
//     A_re[i]       = lat->A_re[i]        → &lat->A_re  + i*8   (double)
//     A_im[i]       = lat->A_im[i]        → &lat->A_im  + i*8   (double)
//     phase[i]      = lat->phase[i]       → &lat->phase + i*8
//     r_harmonic[i] = lat->r_harmonic[i]  → &lat->r_h   + i*8
//     ll_state[i]   = lat->ll_state[i]    → &lat->ll    + i*4   (uint32)
//     w_cos[i*4+k]  = float at            → &lat->w_cos + (i*4+k)*4
//
//   This allows YMM (4×f64) loads of 4 consecutive slots simultaneously:
//     vmovupd  (lat->A_re + i*8), %ymm0     ; A_re[i..i+3]
//     vmovupd  (lat->A_im + i*8), %ymm1     ; A_im[i..i+3]
//   Critical: neighbour accesses ni[0..3] are NOT contiguous, so gather
//   instructions (VGATHERDPD) are required for full vectorisation.
//
// Notation:
//   [r]     = memory operand (register indirect)
//   GATHER  = _mm256_i32gather_pd / VGATHERDPD instruction
//   //→     = annotation / opportunity comment
// ============================================================================

    .file   "hdgl_analog_v31b.s"
    .text

// ============================================================================
// 1.  ll_step32_x4(uint32_t s)  →  uint32_t  (4 iterations)
// ============================================================================
//
// C (4 iterations unrolled):
//   for k = 0..3:
//       uint64_t sq = (uint64_t)s * s;
//       s = (uint32_t)(sq) - 2u;
//
// Serial dependency chain: each iteration depends on previous result.
// Cannot be vectorised; but: pure integer pipeline, no memory ops.
// Throughput: ~4 × 5 cycles = 20 cycles per slot on Skylake (IMUL latency 3cy)
//
// edi = s (uint32_t)
// eax ← result
//
    .globl  ll_step32_x4
    .type   ll_step32_x4, @function
ll_step32_x4:
    .cfi_startproc

    // Iteration 1: sq = s * s (zero-extend to 64-bit, then IMULQ)
    movl    %edi,    %eax             // eax = s (low 32)
    movl    %edi,    %ecx
    imulq   %rcx,    %rax             // rax = (uint64_t)s * s  (64-bit product)
                                       //→ MULQ would give full 128-bit; IMULQ 64×64→64 truncates
                                       //  correctly because we only want low 32 bits
    subl    $2,      %eax             // eax = low32(sq) - 2    (32-bit wraps at 2^32)
                                       //→ SUBL, not SUBQ: forces 32-bit wrap  mod 2^32

    // Iteration 2
    movl    %eax,    %ecx
    imulq   %rcx,    %rax             //→ Dep chain: rax → rcx → rax (serial, ~3cy each)
    subl    $2,      %eax

    // Iteration 3
    movl    %eax,    %ecx
    imulq   %rcx,    %rax
    subl    $2,      %eax

    // Iteration 4
    movl    %eax,    %ecx
    imulq   %rcx,    %rax
    subl    $2,      %eax
    // eax = final s after 4 iterations
    ret
    .cfi_endproc
    .size   ll_step32_x4, .-ll_step32_x4

// ============================================================================
// 2.  spectral_eval_4(float *wc, float *ws, double *phases,
//                      int i, int N, double local_amp, double w_amp_self,
//                      double w_amp_neigh)  →  double  (coupling factor)
// ============================================================================
//
// Computes 4-harmonic spectral coupling for one slot against its primary
// neighbour.  Arguments (for documentation; actual calling convention varies
// when inlined):
//   rdi = float *wc       (4 floats: w_cos[i*4..i*4+3])
//   rsi = float *ws       (4 floats)
//   rdx = double *phases  (lat->phase array base)
//   ecx = i               (slot index)
//   r8d = N               (total slots)
//   xmm0 = dphi           (neighbour phase - self phase, precomputed)
//   xmm1 = local_amp
//   xmm2 = n_amp
//   xmm3 = w_amp_self  (as double)
//   xmm4 = w_amp_neigh (as double)
//
// C fragment:
//   double spec = 0.0;
//   for k = 0..3:
//       spec += wc[k] * cos((k+1)*dphi)
//            + ws[k] * sin((k+1)*dphi)
//   spec += w_amp_self * local_amp + w_amp_neigh * n_amp
//
//   AVX2 float path (4 harmonics → 1 YMM wide):
//     ymm_k = {1.0f, 2.0f, 3.0f, 4.0f}  (harmonic indices, float)
//     ymm_dphi = broadcast(dphi as float)
//     ymm_kd   = ymm_k * ymm_dphi        → {dphi, 2*dphi, 3*dphi, 4*dphi}
//     ymm_cos  = _mm_cos_ps(ymm_kd)      → 4 cosines (SVML or poly approx)
//     ymm_sin  = _mm_sin_ps(ymm_kd)
//     ymm_wc   = vmovaps  (rdi)          → load 4 floats
//     ymm_ws   = vmovaps  (rsi)
//     ymm_acc  = vdpps    ymm_wc, ymm_cos → dot product (4-wide)
//                + vdpps  ymm_ws, ymm_sin
//
    .globl  spectral_eval_4
    .type   spectral_eval_4, @function
spectral_eval_4:
    .cfi_startproc
    pushq   %rbp
    movq    %rsp,    %rbp
    subq    $64,     %rsp

    // Save args
    vmovsd  %xmm0,   -8(%rbp)         // dphi
    vmovsd  %xmm1,  -16(%rbp)         // local_amp
    vmovsd  %xmm2,  -24(%rbp)         // n_amp

    // acc = 0.0
    vxorpd  %xmm15,  %xmm15, %xmm15

    // ── Harmonic loop k=0..3 (shown as 4-way unroll) ──

    // k=0: kd = 1.0 * dphi  (dphi itself)
    vmovsd  -8(%rbp), %xmm0
    call    cos@PLT                    // xmm0 = cos(dphi)
    vcvtss2sd 0(%rdi), %xmm1, %xmm1   // xmm1 = (double)wc[0]
    vfmadd231sd %xmm1, %xmm0, %xmm15  // acc += wc[0] * cos(dphi)
    vmovsd  -8(%rbp), %xmm0
    call    sin@PLT
    vcvtss2sd 0(%rsi), %xmm1, %xmm1   // xmm1 = (double)ws[0]
    vfmadd231sd %xmm1, %xmm0, %xmm15  // acc += ws[0] * sin(dphi)

    // k=1: kd = 2.0 * dphi
    vmovsd  -8(%rbp), %xmm0
    vmulsd  .L2_0(%rip),  %xmm0, %xmm0
    call    cos@PLT
    vcvtss2sd 4(%rdi), %xmm1, %xmm1   // wc[1]
    vfmadd231sd %xmm1, %xmm0, %xmm15
    vmovsd  -8(%rbp), %xmm0
    vmulsd  .L2_0(%rip),  %xmm0, %xmm0
    call    sin@PLT
    vcvtss2sd 4(%rsi), %xmm1, %xmm1
    vfmadd231sd %xmm1, %xmm0, %xmm15

    // k=2: kd = 3.0 * dphi
    vmovsd  -8(%rbp), %xmm0
    vmulsd  .L3_0(%rip),  %xmm0, %xmm0
    call    cos@PLT
    vcvtss2sd 8(%rdi), %xmm1, %xmm1
    vfmadd231sd %xmm1, %xmm0, %xmm15
    vmovsd  -8(%rbp), %xmm0
    vmulsd  .L3_0(%rip),  %xmm0, %xmm0
    call    sin@PLT
    vcvtss2sd 8(%rsi), %xmm1, %xmm1
    vfmadd231sd %xmm1, %xmm0, %xmm15

    // k=3: kd = 4.0 * dphi
    vmovsd  -8(%rbp), %xmm0
    vmulsd  .L4_0(%rip),  %xmm0, %xmm0
    call    cos@PLT
    vcvtss2sd 12(%rdi), %xmm1, %xmm1
    vfmadd231sd %xmm1, %xmm0, %xmm15
    vmovsd  -8(%rbp), %xmm0
    vmulsd  .L4_0(%rip),  %xmm0, %xmm0
    call    sin@PLT
    vcvtss2sd 12(%rsi), %xmm1, %xmm1
    vfmadd231sd %xmm1, %xmm0, %xmm15

    // Amplitude terms
    vmovsd  -16(%rbp), %xmm0           // local_amp
    vfmadd231sd %xmm3, %xmm0, %xmm15  // acc += w_amp_self * local_amp

    vmovsd  -24(%rbp), %xmm0           // n_amp
    vfmadd231sd %xmm4, %xmm0, %xmm15  // acc += w_amp_neigh * n_amp

    vmovsd  %xmm15, %xmm0              // return acc

    addq    $64,    %rsp
    popq    %rbp
    ret
    .cfi_endproc
    .size   spectral_eval_4, .-spectral_eval_4

// ============================================================================
// 3.  fast_lattice_euler_slot  —  single-slot Euler step (inner loop core)
// ============================================================================
//
// This excerpt documents the per-slot body of fast_lattice_step().
// The outer loop increments slot index i=0..N-1.
//
// SoA load pattern for slot i:
//   A_re[i]        vmovsd  (r_Are  + i*8), %xmm0
//   A_im[i]        vmovsd  (r_Aim  + i*8), %xmm1
//   phase[i]       vmovsd  (r_ph   + i*8), %xmm2
//   phase_vel[i]   vmovsd  (r_phv  + i*8), %xmm3
//   r_harmonic[i]  vmovsd  (r_rh   + i*8), %xmm4
//   ll_state[i]    movl    (r_ll   + i*4), %eax   (uint32 = 4 bytes)
//   w_cos[i*4]     vmovaps (r_wcos + i*16), %xmm5  (4 floats = 16 bytes)
//   w_sin[i*4]     vmovaps (r_wsin + i*16), %xmm6
//
// Pointer registers (set up outside loop):
//   r12 = lat->A_re
//   r13 = lat->A_im
//   r14 = lat->phase
//   r15 = lat->phase_vel  (callee-saved; need push/pop in prologue)
//   rbx = lat->r_harmonic (callee-saved)
//   r10 = lat->ll_state
//   r11 = lat->w_cos
//   (lat->w_sin in memory: lat_wsin)
//
// Loop counter: rbp (callee-saved) or push iteration cnt on stack.
//
    .globl  fast_lattice_euler_slot
    .type   fast_lattice_euler_slot, @function
fast_lattice_euler_slot:
    .cfi_startproc
    pushq   %rbp
    .cfi_def_cfa_offset 16
    pushq   %rbx
    .cfi_def_cfa_offset 24
    pushq   %r12
    .cfi_def_cfa_offset 32
    pushq   %r13
    .cfi_def_cfa_offset 40
    pushq   %r14
    .cfi_def_cfa_offset 48
    pushq   %r15
    .cfi_def_cfa_offset 56

    // rdi = FastLattice*   (argument)
    movq    %rdi,    %rbp              // rbp = lat

    // Load SoA array base pointers from FastLattice struct
    // (field offsets depend on struct layout; shown symbolically)
    movq    FastLattice_A_re(%rbp),    %r12  // r12 = lat->A_re
    movq    FastLattice_A_im(%rbp),    %r13  // r13 = lat->A_im
    movq    FastLattice_phase(%rbp),   %r14  // r14 = lat->phase
    movq    FastLattice_phvel(%rbp),   %r15  // r15 = lat->phase_vel
    movq    FastLattice_rh(%rbp),      %rbx  // rbx = lat->r_harmonic

    // dt = lat->dt_global  (load ONCE before loop)
    vmovsd  FastLattice_dt(%rbp), %xmm15   // xmm15 = dt  (read-only in loop)
                                             //→ [F3]: single global dt, not per-slot

    movl    FastLattice_N(%rbp), %r9d       // r9d = N  (loop bound)
    xorl    %eax,    %eax                   // eax = i = 0

    // ── Outer loop: for i = 0..N-1 ──
.Leuler_slot_loop:
    cmpl    %r9d,    %eax
    jge     .Leuler_slot_done

    // SoA loads (sequential reads → L1 cache hits for stride-1 access)
    movslq  %eax,    %rcx                   // rcx = (long)i

    vmovsd  (%r12,%rcx,8), %xmm0            // xmm0 = A_re[i]
    vmovsd  (%r13,%rcx,8), %xmm1            // xmm1 = A_im[i]
    vmovsd  (%r14,%rcx,8), %xmm2            // xmm2 = phase[i]
    vmovsd  (%r15,%rcx,8), %xmm3            // xmm3 = phase_vel[i]
    vmovsd  (%rbx,%rcx,8), %xmm4            // xmm4 = r_harmonic[i]

    // local_amp = sqrt(A_re^2 + A_im^2)
    vmulsd  %xmm0,   %xmm0,  %xmm5         // xmm5 = A_re^2
    vfmadd231sd %xmm1, %xmm1, %xmm5        // xmm5 = A_re^2 + A_im^2
    vsqrtsd %xmm5,   %xmm5,  %xmm5         // xmm5 = local_amp

    // ── LL-lite x4: in integer domain ──
    movq    FastLattice_ll(%rbp), %r8       // r8 = lat->ll_state  (array)
    movl    (%r8,%rcx,4), %edi              // edi = ll_state[i]  (uint32)

    // Iter 1
    movl    %edi,    %r10d
    imulq   %r10,    %rdi                   //→ edi:rdi = s*s; only rdi (low 64)
    subl    $2,      %edi
    // Iter 2
    movl    %edi,    %r10d
    imulq   %r10,    %rdi
    subl    $2,      %edi
    // Iter 3
    movl    %edi,    %r10d
    imulq   %r10,    %rdi
    subl    $2,      %edi
    // Iter 4
    movl    %edi,    %r10d
    imulq   %r10,    %rdi
    subl    $2,      %edi
    movl    %edi,    (%r8,%rcx,4)           // ll_state[i] = s  (write back)
                                             //→ 4 serial IMULQ: ~12-20cy on Zen4
                                             //  No vectorisation possible (serial dep)

    // residue = (float)s / 4294967295.0f
    vcvtsi2ss %edi,  %xmm7, %xmm7           // xmm7 = (float)s
    divss   .LUINT32MAX_F(%rip), %xmm7      // xmm7 = residue  (float)

    // ── 4-neighbour loop ──
    // Neighbours: ni[0]=(i-1+N)%N, ni[1]=(i+1)%N, ni[2]=(i-S+N)%N, ni[3]=(i+S)%N
    // For brevity, shown for ni[1] = (i+1)%N:

    movl    %eax,    %r11d
    incl    %r11d
    cmpl    %r9d,    %r11d
    jl      .Ln1_ok
    xorl    %r11d,   %r11d             // wrap: ni[1] = 0 if i+1==N
.Ln1_ok:
    movslq  %r11d,   %r11              // r11 = (long)ni[1]

    // n_A_re = A_re[ni[1]], n_A_im = A_im[ni[1]]
    vmovsd  (%r12,%r11,8), %xmm8       // xmm8 = n_A_re
    vmovsd  (%r13,%r11,8), %xmm9       // xmm9 = n_A_im
    vmovsd  (%r14,%r11,8), %xmm10      // xmm10 = n_phase

    // dphi = n_phase - phase[i]
    vsubsd  %xmm2,   %xmm10, %xmm10   // xmm10 = dphi

    // n_amp = sqrt(n_A_re^2 + n_A_im^2)
    vmulsd  %xmm8,   %xmm8,  %xmm11
    vfmadd231sd %xmm9, %xmm9, %xmm11
    vsqrtsd %xmm11,  %xmm11, %xmm11   // xmm11 = n_amp

    // spectral_eval_4: (w_cos, w_sin, dphi, local_amp, n_amp)
    // Load per-slot float weights: 4 floats at w_cos[i*4]
    movq    FastLattice_wcos(%rbp), %r10
    leaq    (%r10,%rcx,16), %rdi        // rdi = &w_cos[i*4]  (4 floats = 16 bytes)
    movq    FastLattice_wsin(%rbp), %r10
    leaq    (%r10,%rcx,16), %rsi        // rsi = &w_sin[i*4]
    vmovsd  %xmm10, %xmm0              // arg: dphi
    vmovsd  %xmm5,  %xmm1              // arg: local_amp
    vmovsd  %xmm11, %xmm2              // arg: n_amp
    // w_amp_self and w_amp_neigh as scalar args (loaded from lat struct):
    vcvtss2sd FastLattice_wampself(%rbp), %xmm3, %xmm3
    vcvtss2sd FastLattice_wampneigh(%rbp), %xmm4, %xmm4
    call    spectral_eval_4             // xmm0 = spec  (one neighbour)
                                         //→ In fully inlined version: no call overhead;
                                         //  compiler maps to 4× VFMADD + trig sequence

    // GRA factor: combined = sqrt(r_h^2 + r_neigh^2); gra_fac = BASE*c/(1+c)
    vmovsd  (%rbx,%r11,8), %xmm6        // xmm6 = r_harmonic[ni[1]]
    vmulsd  %xmm4,   %xmm4,  %xmm12    // xmm12 = r_h^2   (NOTE: xmm4 = r_h loaded above)
    vfmadd231sd %xmm6, %xmm6, %xmm12   // xmm12 = r_h^2 + r_neigh^2
    vsqrtsd %xmm12,  %xmm12, %xmm12   // xmm12 = combined
    vmovsd  .LBASE_GRA_B(%rip), %xmm13
    vmulsd  %xmm12,  %xmm13, %xmm13   // xmm13 = BASE * combined
    vmovsd  .L1_0B(%rip), %xmm14
    vaddsd  %xmm12,  %xmm14, %xmm14   // xmm14 = 1 + combined
    vdivsd  %xmm14,  %xmm13, %xmm13   // xmm13 = gra_fac

    // factor = spec + gra_fac
    vaddsd  %xmm13,  %xmm0,  %xmm0    // xmm0 = factor  (for this one neighbour)

    // (repeat similarly for neighbours 0, 2, 3; accumulate dA_re, dA_im, sum_sin)
    // Full accumulation shown abbreviated:
    //   dA_re += factor * cos(dphi);  dA_im += factor * sin(dphi)
    //   sum_sin += sin(dphi)

    // ── Euler update ──
    // A_re += dt * dA_re;  (xmm15 = dt, xmm0 = dA_re total)
    //   vfmadd231sd %xmm15, %xmm0, %xmm_A_re   → A_re += dt * dA_re
    //
    // ── GRA Plasticity (OPEN) ──
    // r_h += GRA_PLASTICITY * (local_amp - 0.5) * (r_h > 50.0 ? 0.05 : 1.0)
    vmovsd  .LGRA_PLAS(%rip), %xmm6     // xmm6 = GRA_PLASTICITY = 0.008
    vmovsd  .L0_5(%rip),  %xmm7
    vsubsd  %xmm7,   %xmm5,  %xmm7     // xmm7 = local_amp - 0.5
    vmulsd  %xmm7,   %xmm6,  %xmm6     // xmm6 = GRA_PLASTICITY * (amp - 0.5)

    // Conditional slowdown factor: r_h > 50 ? 0.05 : 1.0
    vmovsd  .L50_0(%rip), %xmm7
    vucomisd %xmm7,  %xmm4              // compare r_h with 50.0
    vmovsd  .L0_05(%rip), %xmm8
    vmovsd  .L1_0B(%rip), %xmm9
    //→ VCMPPD + VBLENDVPD for branchless select:
    vcmppd  $1, %xmm7, %xmm4, %xmm10   // xmm10 = mask  (r_h > 50 → all-1s)
    vblendvpd %xmm10, %xmm8, %xmm9, %xmm9  // xmm9 = (r_h>50) ? 0.05 : 1.0
    vmulsd  %xmm9,   %xmm6,  %xmm6     // xmm6 = plastic_step
    vaddsd  %xmm6,   %xmm4,  %xmm4     // r_h += plastic_step

    // Clamp r_h to [1.0, R_MAX_PLASTIC]
    vmovsd  .L1_0B(%rip),        %xmm7
    vmaxsd  %xmm7,   %xmm4,  %xmm4     // r_h = max(r_h, 1.0)
    vmovsd  .LR_MAX(%rip),       %xmm7
    vminsd  %xmm7,   %xmm4,  %xmm4     // r_h = min(r_h, R_MAX_PLASTIC)
                                         //→ VMAXSD/VMINSD: 1-cycle latency on modern x86

    // Write back (SoA stores — same stride pattern as loads)
    vmovsd  %xmm0,  (%r12,%rcx,8)       // A_re[i]  (updated; omitting full pipeline here)
    vmovsd  %xmm4,  (%rbx,%rcx,8)       // r_harmonic[i]

    incl    %eax
    jmp     .Leuler_slot_loop

.Leuler_slot_done:
    popq    %r15
    popq    %r14
    popq    %r13
    popq    %r12
    popq    %rbx
    popq    %rbp
    ret
    .cfi_endproc
    .size   fast_lattice_euler_slot, .-fast_lattice_euler_slot

// ============================================================================
// 4.  fast_lattice_emit_candidates  —  threshold gate  (hot path excerpt)
// ============================================================================
//
// C inner loop body (per slot i):
//   float residue = (float)ll_state[i] / 4294967295.0f;
//   double amp    = sqrt(A_re[i]^2 + A_im[i]^2);
//   if (reward_accum[i] > 5.0 && residue < 0.02f && amp > 0.6):
//       bridge_b_emit_candidate(i, phase[i], amp, r_harmonic[i], reward_accum[i])
//
// Vectorization: the three comparisons can be done as packed:
//   ymm_accum  = vmovupd  reward_accum[i..i+3]
//   ymm_thresh = vbroadcastsd  5.0
//   ymm_mask   = vcmppd  GT, ymm_accum, ymm_thresh        ; mask[j] = accum>5
//   (similarly for amp and residue after conversion)
//   Combined mask → scalar loop over set bits
//
    .globl  _emit_candidate_inner_excerpt
    .type   _emit_candidate_inner_excerpt, @function
_emit_candidate_inner_excerpt:
    .cfi_startproc
    // rdi = FastLattice*
    // ecx = slot index i  (from outer loop)
    movslq  %ecx,   %rcx               // sign-extend i

    // residue = (float)ll_state[i] / UINT32MAX_F
    movq    FastLattice_ll(%rdi),  %r8
    movl    (%r8,%rcx,4), %eax
    vcvtsi2ss %eax,  %xmm0, %xmm0
    divss   .LUINT32MAX_F(%rip), %xmm0  // xmm0 = residue (float)

    // amp = sqrt(A_re[i]^2 + A_im[i]^2)
    movq    FastLattice_A_re(%rdi), %r9
    movq    FastLattice_A_im(%rdi), %r10
    vmovsd  (%r9,%rcx,8), %xmm1
    vmovsd  (%r10,%rcx,8), %xmm2
    vmulsd  %xmm1, %xmm1, %xmm3
    vfmadd231sd %xmm2, %xmm2, %xmm3
    vsqrtsd %xmm3, %xmm3, %xmm3         // xmm3 = amp

    // Check: reward_accum[i] > 5.0
    movq    FastLattice_racc(%rdi), %r11
    vmovsd  (%r11,%rcx,8), %xmm4         // xmm4 = reward_accum[i]
    vmovsd  .L5_0(%rip), %xmm5
    vucomisd %xmm5, %xmm4
    jbe     .Lno_emit                    // skip if accum <= 5.0

    // Check: residue < 0.02f
    comiss  .L0_02F(%rip), %xmm0
    jae     .Lno_emit                    // skip if residue >= 0.02

    // Check: amp > 0.6
    vmovsd  .L0_6(%rip), %xmm5
    vucomisd %xmm5, %xmm3
    jbe     .Lno_emit

    // All gates passed → call bridge_b_emit_candidate
    // Args: (int idx, double phase, double amp, double r_h, double score)
    movl    %ecx,    %edi               // arg 1: idx (int)
    movq    FastLattice_phase(%rdi), %r8
    vmovsd  (%r8,%rcx,8), %xmm0         // arg 2: phase
    vmovsd  %xmm3,   %xmm1              // arg 3: amp
    movq    FastLattice_rh(%rdi), %r8
    vmovsd  (%r8,%rcx,8), %xmm2         // arg 4: r_harmonic
    vmovsd  (%r11,%rcx,8), %xmm3        // arg 5: score = reward_accum
    call    bridge_b_emit_candidate      // returns int (ignored here)

    // Decay reward_accum[i] after emission
    movq    FastLattice_racc(%rdi), %r11
    vmovsd  (%r11,%rcx,8), %xmm0
    vmulsd  .L0_5B(%rip), %xmm0, %xmm0  // accum *= 0.5
    vmovsd  %xmm0, (%r11,%rcx,8)

.Lno_emit:
    ret
    .cfi_endproc
    .size   _emit_candidate_inner_excerpt, .-_emit_candidate_inner_excerpt

// ============================================================================
// Read-only data pool
// ============================================================================

    .section .rodata
    .align  8

.LBASE_GRA_B:
    .double 0.18

.L1_0B:
    .double 1.0

.L2_0:
    .double 2.0
.L3_0:
    .double 3.0
.L4_0:
    .double 4.0

.L0_5:
    .double 0.5
.L0_5B:
    .double 0.5

.L5_0:
    .double 5.0

.L0_6:
    .double 0.6

.L50_0:
    .double 50.0

.L0_05:
    .double 0.05

.LGRA_PLAS:
    .double 0.008

.LR_MAX:
    .double 1000.0

.LUINT32MAX_F:
    .float  4294967295.0

.L0_02F:
    .float  0.02

// Struct field offset symbols (replace with actual GCC-computed offsets)
// Query with: offsetof(FastLattice, field_name)
// These are illustrative names for the assembly commentary above
FastLattice_A_re     = 0     // &lat->A_re
FastLattice_A_im     = 8     // &lat->A_im
FastLattice_phase    = 16
FastLattice_phvel    = 24
FastLattice_rh       = 32    // &lat->r_harmonic
FastLattice_ll       = 40    // &lat->ll_state
FastLattice_racc     = 48    // &lat->reward_accum
FastLattice_wcos     = 56    // &lat->w_cos
FastLattice_wsin     = 64    // &lat->w_sin
FastLattice_wampself = 72    // lat->w_amp_self  (float)
FastLattice_wampneigh= 76    // lat->w_amp_neigh (float)
FastLattice_N        = 80    // lat->num_slots  (int)
FastLattice_S        = 84    // lat->slots_per_instance (int)
FastLattice_dt       = 88    // lat->dt_global (double)

// ============================================================================
// AVX2 / SoA Vectorization Opportunities Summary
// ============================================================================
//
// 1. fast_lattice_step — self-state loads:
//    Since A_re, A_im, phase, phase_vel, r_harmonic are contiguous double
//    arrays, FOUR consecutive slots fit in a 256-bit YMM register:
//
//      vmovupd  (%r12, %rcx, 8), %ymm0    ; A_re[i..i+3]
//      vmovupd  (%r13, %rcx, 8), %ymm1    ; A_im[i..i+3]
//      vmulpd   %ymm0, %ymm0, %ymm5       ; A_re^2
//      vfmadd231pd %ymm1, %ymm1, %ymm5    ; + A_im^2
//      vsqrtpd  %ymm5, %ymm5              ; local_amp[i..i+3]
//
//    Prerequisite: loop body operates identically on all 4 slots (true here).
//    Challenge: neighbour indices ni[0..3] are indirect; need VGATHERDPD.
//
// 2. VGATHERDPD for neighbour loads:
//    Build 4 neighbour indices as XMM (int32):
//      vpbroadcastd  %ecx, %xmm_ni         ; [i, i, i, i]
//      vpaddd        .Loffset_plus1, %xmm_ni  ; [i+1, i+1, ...]
//                                           ; (analogously for all 4 directions)
//    Then: vgatherdpd %ymm_mask, (r12,%xmm_ni,8), %ymm_n_Are  ; 4 neighbour A_re
//    VGATHERDPD throughput: ~7 cycles vs 4 × individual loads = 4 cycles
//    Net benefit modest; may be better to fully unroll 4 neighbours.
//
// 3. ll_step32_x4 — independent slots:
//    The serial dep chain prevents per-slot vectorisation.
//    HOWEVER: 8 independent slots can be processed in parallel via 8-wide
//    integer vectors (AVX2 VMULQ not available; must use VPMULUDQ):
//
//      vmovdqu  (ll_state + i*4), %ymm_s    ; s[i..i+7]  (8 × uint32)
//      vpmuludq %ymm_s, %ymm_s, %ymm_sq     ; sq[i..i+7] (lower 32-bit pairs)
//      vpsubd   .Lvec_2, %ymm_sq, %ymm_sq   ; s = sq - 2
//      (repeat 3 more times for 4 LL-lite iters)
//    This processes 8 slots in ~12 cycles vs 8 × ~20 = 160 cycles scalar.
//    Requires reinterpreting VPMULUDQ 32×32→64 result carefully.
//
// 4. Hebbian weight update (float w_cos, w_sin):
//    Per-slot 4-float arrays map to XMM (128-bit):
//      vmovaps  (wcos + i*16), %xmm_wc     ; wc[0..3]  (4 × float32)
//      vmovaps  (wsin + i*16), %xmm_ws
//      vmulss broadcast(reward), %xmm_cos_vec, %xmm_delta  → VFMADD
//      (then VMINPS/VMAXPS for clamping)
//    4-wide float update: ~4 cycles for the whole Hebbian step per slot.
//
// 5. GRA plasticity branchless select (r_h > 50 ? 0.05 : 1.0):
//    Already annotated above with VCMPPD + VBLENDVPD.
//    When processing 4 slots at once (YMM):
//      vcmppd  $14, %ymm_50, %ymm_rh, %ymm_mask  ; GT comparison
//      vblendvpd %ymm_mask, %ymm_005, %ymm_1, %ymm_factor
//    Zero branch mispredictions.
//
// ============================================================================
