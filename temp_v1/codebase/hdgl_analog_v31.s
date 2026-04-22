// ============================================================================
// hdgl_analog_v31.s  —  Annotated x86-64 GAS Assembly Reference
// HDGL Analog Mainnet v3.1  —  Closed-Form Accuracy Hemisphere
// ============================================================================
//
// Purpose: Documents the x86-64 machine-code structure of the v31 hot paths.
//   Serves as: (a) micro-architecture commentary, (b) AVX2/FMA opportunity
//   annotations, (c) reference for manual tuning and verification.
//
// Compile flags assumed (Linux, GCC 12+):
//   gcc -O2 -march=native -mavx2 -mfma -ffast-math hdgl_analog_v31.c -o hdgl_v31 -lm
//
// Register conventions (System V AMD64 ABI):
//   Integer args : rdi, rsi, rdx, rcx, r8, r9
//   Float  args  : xmm0–xmm7
//   Callee-saved : rbx, rbp, r12–r15
//   Return       : rax (int), xmm0 (double)
//   Scratch      : r10, r11, xmm8–xmm15
//
// Covered hot paths:
//   1. gra_rn_closed       — closed-form GRA, scalar double precision
//   2. ll_step64           — 128-bit squaring, residue probe
//   3. compute_deriv_v31   — RK4 derivative inner loop (8 neighbours, spectral)
//   4. ap_writeback_double — in-place mantissa pack (precision boundary)
//   5. spectral_eval       — 8-harmonic global kernel inner loop
//   6. rk4_step_v31        — full RK4 stage composition
//
// Notation:
//   [r]     = memory operand (register indirect)
//   {k}     = AVX512 mask register (future target)
//   //→     = annotation / opportunity comment
// ============================================================================

    .file   "hdgl_analog_v31.s"
    .text

// ============================================================================
// 1.  gra_rn_closed(int n, double omega)  →  double
// ============================================================================
//
// C signature: double gra_rn_closed(int n, double omega)
//   edi  = n            (int)
//   xmm0 = omega        (double)
//   xmm0 ← result       (double)
//
// Algorithm:
//   prod_p = prod(PRIME_TABLE[0..n-1])   (integer, 64-bit)
//   result = sqrt(PHI * omega * FIB[n] * 2^n * prod_p)
//
// Register allocation:
//   eax / rax   = loop counter k
//   rcx         = &PRIME_TABLE[0]  (rip-relative)
//   rdx         = prod_p accumulator  (uint64_t)
//   xmm0        = omega (in) / result (out)
//   xmm1        = PHI constant
//   xmm2        = FIB_TABLE[n]  (converted to double)
//   xmm3        = 2^n            (via VCVTSI2SD + VADDSD pattern)
//   xmm4        = prod_p         (uint64 → double)
//   xmm5        = accumulator for product under sqrt
//
    .globl  gra_rn_closed
    .type   gra_rn_closed, @function
gra_rn_closed:
    .cfi_startproc
    // Prologue
    pushq   %rbp
    .cfi_def_cfa_offset 16
    movq    %rsp,    %rbp
    .cfi_offset 6, -16
    .cfi_def_cfa_register 6

    // Guard: n < 1 || n > MAX_GRA_N → return 1.0
    cmpl    $1,      %edi
    jl      .Lgra_one
    cmpl    $16,     %edi
    jg      .Lgra_one

    // rdx = prod_p = 1
    movl    $1,      %edx             // prod_p = 1

    // Compute prod(PRIME_TABLE[0..n-1]) in rdx
    // rax = loop counter k = 0; loop while k < n
    xorl    %eax,    %eax
    leaq    PRIME_TABLE(%rip), %rcx   //→ rip-relative load of static array

.Lgra_prime_loop:
    cmpl    %edi,    %eax
    jge     .Lgra_prime_done
    movq    (%rcx,%rax,8), %r8        // r8 = PRIME_TABLE[k]  (8-byte load)
    imulq   %r8,     %rdx             // prod_p *= PRIME_TABLE[k]
                                       //→ MULQ would give 128-bit; IMULQ fine for n≤16
    incl    %eax
    jmp     .Lgra_prime_loop

.Lgra_prime_done:
    // xmm2 = (double)FIB_TABLE[n]
    leaq    FIB_TABLE(%rip), %r9
    movq    (%r9,%rdi,8), %r10        // r10 = FIB_TABLE[n]
    vcvtsi2sdq %r10, %xmm2, %xmm2    //→ VCVTSI2SDQ avoids partial stall

    // xmm3 = pow(2.0, n)  = (double)(1 << n)  for n <= 62
    movl    $1,      %r11d
    movl    %edi,    %ecx
    shlq    %cl,     %r11             // r11 = 2^n  (uint64_t)
    vcvtsi2sdq %r11, %xmm3, %xmm3

    // xmm4 = (double)prod_p
    vcvtsi2sdq %rdx, %xmm4, %xmm4

    // PHI constant: xmm1 = 1.6180339887498948
    movsd   .LPHI(%rip), %xmm1        //→ load from .rodata literal pool

    // Accumulate inside = PHI * omega * FIB[n] * 2^n * prod_p
    // xmm5 = PHI * omega
    vmulsd  %xmm1,   %xmm0,  %xmm5   // xmm5 = PHI * omega
    vmulsd  %xmm2,   %xmm5,  %xmm5   // xmm5 *= FIB[n]
    vmulsd  %xmm3,   %xmm5,  %xmm5   // xmm5 *= 2^n
    vmulsd  %xmm4,   %xmm5,  %xmm5   // xmm5 *= prod_p
                                       //→ VFMADD213SD could fuse PHI*omega*FIB

    // result = sqrt(max(inside, 1e-12))
    // guard against negative / zero via VMAXSD then VSQRTSD
    movsd   .L1em12(%rip), %xmm6
    vmaxsd  %xmm6,   %xmm5,  %xmm5
    vsqrtsd %xmm5,   %xmm0,  %xmm0   //→ VSQRTSD latency ~20cy; bottleneck here
    jmp     .Lgra_ret

.Lgra_one:
    movsd   .L1_0(%rip), %xmm0        // return 1.0

.Lgra_ret:
    popq    %rbp
    .cfi_def_cfa 7, 8
    ret
    .cfi_endproc
    .size   gra_rn_closed, .-gra_rn_closed

// ============================================================================
// 2.  ll_step64(uint64_t s)  →  uint64_t
// ============================================================================
//
// C: static inline uint64_t ll_step64(uint64_t s)
//   rdi = s
//   rax ← result
//
// Implementation: (s * s) mod 2^128 → low 64 bits, subtract 2
//   MULQ computes RDX:RAX = RAX * RDI (unsigned 64×64 → 128)
//   We need only the low 64 bits (RAX) then subtract 2.
//
//   Critically: (s^2 - 2) mod 2^64 wraps naturally via integer overflow.
//   The __uint128_t in C compiles to exactly this sequence.
//
    .globl  ll_step64
    .type   ll_step64, @function
ll_step64:
    .cfi_startproc
    movq    %rdi,    %rax             // rax = s
    mulq    %rdi                      // rdx:rax = s * s   (unsigned)
                                       // NOTE: rdx (high 64) discarded
                                       //→ Only RAX (low 64) needed
    subq    $2,      %rax             // rax = (s*s mod 2^64) - 2
                                       //→ wraparound at 2^64 is intentional
    ret
    .cfi_endproc
    .size   ll_step64, .-ll_step64

// ============================================================================
// 2b.  ll_residue_probe(uint64_t seed, int iters)  →  double
// ============================================================================
//
// rdi = seed
// esi = iters   (default LL_PROBE_ITERS = 64)
// xmm0 ← residue in [0, 1]
//
// Inner loop: 64 × ll_step64; then convert to double / UINT64_MAX
// Throughput: each MULQ is ~3cy throughput on modern x86-64
// 64 iterations ≈ 200 cycles worst-case
//
    .globl  ll_residue_probe
    .type   ll_residue_probe, @function
ll_residue_probe:
    .cfi_startproc
    testl   %esi,    %esi
    jle     .Lprobe_done

    movq    %rdi,    %r8              // r8 = s (working state)
    orl     $1,      %r8d            // ensure non-zero seed (OR with 1)
    movl    %esi,    %ecx            // ecx = iteration counter

.Lprobe_loop:
    movq    %r8,     %rax
    mulq    %r8                       // rdx:rax = s^2
    subq    $2,      %rax
    movq    %rax,    %r8             // s = low64(s^2) - 2
    decl    %ecx
    jnz     .Lprobe_loop
                                       //→ Pipelining: MULQ serialises; can unroll 2×
                                       //  but dependency chain limits ILP

    // result = (double)r8 / (double)UINT64_MAX
    vcvtsi2sdq %r8,  %xmm0, %xmm0    // xmm0 = (double)s  (signed conversion OK: MSB clear statistically)
    movsd   .LUINT64MAX_D(%rip), %xmm1
    vdivsd  %xmm1,   %xmm0,  %xmm0  // result = s / UINT64_MAX
    ret

.Lprobe_done:
    vxorpd  %xmm0,   %xmm0,  %xmm0  // return 0.0
    ret
    .cfi_endproc
    .size   ll_residue_probe, .-ll_residue_probe

// ============================================================================
// 3.  spectral_eval(double dphi, double A_self, double A_neigh)  →  double
// ============================================================================
//
// xmm0 = dphi
// xmm1 = A_self
// xmm2 = A_neigh
// xmm0 ← evaluated sum
//
// C inner loop (8 harmonics):
//   for k = 1..8:
//       acc += g_W_cos[k-1] * cos(k * dphi)
//            + g_W_sin[k-1] * sin(k * dphi)
//   acc += g_W_amp[0] * A_self + g_W_amp[1] * A_neigh
//
// Vectorization opportunity: 4 harmonics fit in YMM (4 × f64 = 256-bit)
// Below shows the scalar version as emitted with -O2 (no -ffast-math autovec)
// AVX2 opportunity is annotated with //→ AVX2:
//
    .globl  spectral_eval
    .type   spectral_eval, @function
spectral_eval:
    .cfi_startproc
    pushq   %rbp
    movq    %rsp,    %rbp
    subq    $48,     %rsp            // local scratch for trig intermediates

    vmovsd  %xmm0,   -8(%rbp)       // spill dphi
    vmovsd  %xmm1,  -16(%rbp)       // spill A_self
    vmovsd  %xmm2,  -24(%rbp)       // spill A_neigh

    vxorpd  %xmm8,   %xmm8,  %xmm8  // xmm8 = acc = 0.0

    // Loop k = 1..8 (scalar; GCC will emit 8 unrolled COS/SIN calls via libm)
    // For illustration, k=1 shown; then k=2..8 are identical with incremented
    // addresses into g_W_cos / g_W_sin

    // k=1: kd = 1.0 * dphi = dphi
    vmovsd  -8(%rbp),  %xmm0         // xmm0 = dphi
    call    cos@PLT                   // xmm0 = cos(dphi)
                                       //→ AVX2: precompute k*dphi for k=1..4 as YMM
                                       //  then VCOS via SVML (Intel): ymm = _mm256_cos_pd(ymm)
                                       //  saves 3 of 8 serial libm calls
    vmovsd  g_W_cos(%rip), %xmm1     // xmm1 = g_W_cos[0]
    vfmadd231sd %xmm1, %xmm0, %xmm8 // acc += g_W_cos[0] * cos(1*dphi)
                                       //→ VFMADD231SD: fused multiply-add, no rounding err

    vmovsd  -8(%rbp),  %xmm0
    call    sin@PLT                   // xmm0 = sin(dphi)
    vmovsd  g_W_sin(%rip), %xmm1
    vfmadd231sd %xmm1, %xmm0, %xmm8 // acc += g_W_sin[0] * sin(1*dphi)

    // k=2..8: multiply dphi by 2..8 first
    // Pattern repeated; shown abstractly:
    //   vmulsd   k_const(%rip), -8(%rbp), xmm0  ; kd = k * dphi
    //   call cos@PLT
    //   vfmadd231sd g_W_cos+8*(k-1)(%rip), xmm0, xmm8
    //   ... (sin analogously)
    //   k_const = .L2_0, .L3_0, ..., .L8_0  in rodata
    //→ Fully unrolled by GCC -O3; 8 cos + 8 sin calls dominates runtime
    //→ AVX2 SVML: halves call count; meaningful on kernel-intensive runs

    // Amplitude terms
    vmovsd  -16(%rbp), %xmm0         // A_self
    vmovsd  g_W_amp(%rip),   %xmm1   // g_W_amp[0]
    vfmadd231sd %xmm1, %xmm0, %xmm8

    vmovsd  -24(%rbp), %xmm0         // A_neigh
    vmovsd  g_W_amp+8(%rip), %xmm1   // g_W_amp[1]
    vfmadd231sd %xmm1, %xmm0, %xmm8

    vmovsd  %xmm8,   %xmm0           // return acc
    addq    $48,     %rsp
    popq    %rbp
    ret
    .cfi_endproc
    .size   spectral_eval, .-spectral_eval

// ============================================================================
// 4.  compute_deriv_v31 inner neighbour loop  (annotated excerpt)
// ============================================================================
//
// This excerpt shows the hot path inside compute_deriv_v31 for ONE neighbour.
// Full loop iterates 8 times (8-connected lattice topology).
//
// C fragment:
//   double dphi   = nb[k].potential - st.phase;    // AnalogLink.potential
//   double A_neigh = fabs(nb[k].charge);
//   double spec   = spectral_eval(dphi, A_self, A_neigh);
//   double combined = sqrt(r_h*r_h + nb[k].r_harmonic*nb[k].r_harmonic);
//   double gra_fac  = BASE_GRA_MODULATION * combined / (1.0 + combined);
//   double factor   = spec + gra_fac;
//   sum_sin += sin(dphi);
//   d.A_re  += factor * cos(dphi);
//   d.A_im  += factor * sin(dphi);
//   gra_sum += gra_fac;
//
// Register allocation for inner body (r12=loop k, r13=&nb[k], r14=phase,
//   r15=r_h; xmm12=d.A_re, xmm13=d.A_im, xmm14=sum_sin, xmm15=gra_sum):
//
    .globl  _compute_deriv_v31_inner_excerpt
    .type   _compute_deriv_v31_inner_excerpt, @function
_compute_deriv_v31_inner_excerpt:
    .cfi_startproc
    // Assumes: r13 = &nb[k]  (AnalogLink layout: charge,charge_im,tension,potential,coupling,r_harmonic)
    //          xmm9 = st.phase    (caller loads once before loop)
    //          xmm10 = A_self     (caller computes once before loop)
    //          xmm11 = r_h        (slot->r_harmonic, immutable in v31)

    // dphi = nb[k].potential - st.phase
    vmovsd  24(%r13),  %xmm0          // nb[k].potential  (offset 24: 3rd double)
    vsubsd  %xmm9,    %xmm0, %xmm0   // xmm0 = dphi

    // A_neigh = fabs(nb[k].charge)
    vmovsd  0(%r13),   %xmm1          // nb[k].charge  (offset 0)
    vandpd  .Labs_mask(%rip), %xmm1, %xmm1 // fabs via mask clear sign bit
                                        //→ VANDPD on XMM is 1-cycle lat, better than SSE2 AND

    // spec = spectral_eval(dphi, A_self, A_neigh)
    // xmm0=dphi, xmm1=A_neigh already set; xmm10 → xmm1 (arg 2), shift args
    vmovsd  %xmm10,   %xmm2           // xmm2 = A_neigh (arg3)
    vmovsd  %xmm1,    %xmm1           // xmm1 = A_self  (arg2) already OK... actually:
    // Correction: ABI is xmm0=dphi, xmm1=A_self, xmm2=A_neigh
    vmovsd  %xmm10,   %xmm1           // arg2 = A_self
    // xmm2 was set above... re-set:
    vmovsd  0(%r13),   %xmm2
    vandpd  .Labs_mask(%rip), %xmm2, %xmm2  // fabs → A_neigh
    // (xmm0=dphi, xmm1=A_self, xmm2=A_neigh) → call spectral_eval
    call    spectral_eval             // xmm0 = spec

    // Save spec; need dphi again → reload from stack spill or recompute
    vmovsd  %xmm0,    -32(%rbp)      // spill spec
    vmovsd  24(%r13),  %xmm0
    vsubsd  %xmm9,    %xmm0, %xmm0  // dphi (recomputed; compiler may CSE)

    // combined = sqrt(r_h^2 + nb[k].r_harmonic^2)
    vmovsd  40(%r13),  %xmm1          // nb[k].r_harmonic  (offset 40: 6th double)
    vmulsd  %xmm11,   %xmm11, %xmm4  // xmm4 = r_h * r_h
    vfmadd231sd %xmm1, %xmm1, %xmm4  // xmm4 = r_h^2 + r_neigh^2  (FMA: a*b+c)
                                        //→ VFMADD231SD: c += a*b  — single rounding
    vsqrtsd %xmm4,    %xmm4, %xmm4   // xmm4 = combined

    // gra_fac = BASE_GRA_MODULATION * combined / (1.0 + combined)
    // = BASE * combined * (1/(1+combined))  — avoid division via Newton if hot
    vmovsd  .LBASE_GRA(%rip), %xmm5
    vmulsd  %xmm4,    %xmm5,  %xmm5  // xmm5 = BASE * combined
    movsd   .L1_0(%rip), %xmm6
    vaddsd  %xmm4,    %xmm6,  %xmm6  // xmm6 = 1.0 + combined
    vdivsd  %xmm6,    %xmm5,  %xmm5  // xmm5 = gra_fac
                                        //→ VDIVSD latency ~15cy; Newton-Raphson VRCPSD
                                        //  + VFNMADD213SD is faster if throughput-critical

    // factor = spec + gra_fac
    vmovsd  -32(%rbp),  %xmm3         // xmm3 = spec
    vaddsd  %xmm5,    %xmm3,  %xmm3  // xmm3 = factor

    // sin(dphi), cos(dphi) — saved/reused if dphi unchanged
    vmovsd  %xmm0,    -40(%rbp)       // spill dphi
    call    sin@PLT
    vmovsd  %xmm0,    -48(%rbp)       // spill sin(dphi)
    vmovsd  -40(%rbp),  %xmm0
    call    cos@PLT                    // xmm0 = cos(dphi)
    vmovsd  -48(%rbp),  %xmm1         // xmm1 = sin(dphi)

    // sum_sin += sin(dphi)
    vaddsd  %xmm1,    %xmm14, %xmm14

    // d.A_re += factor * cos(dphi)
    vfmadd231sd %xmm3, %xmm0, %xmm12 //→ VFMADD231SD: xmm12 += xmm3 * xmm0

    // d.A_im += factor * sin(dphi)
    vfmadd231sd %xmm3, %xmm1, %xmm13

    // gra_sum += gra_fac
    vaddsd  %xmm5,    %xmm15, %xmm15

    // r13 += sizeof(AnalogLink)  = 6 * 8 = 48 bytes; loop continues
    addq    $48,      %r13
    ret
    .cfi_endproc
    .size   _compute_deriv_v31_inner_excerpt, .-_compute_deriv_v31_inner_excerpt

// ============================================================================
// 5.  ap_writeback_double excerpt  —  precision boundary pack
// ============================================================================
//
// C fragment:
//   int exp_off;
//   double mant = frexp(fabs(A_re), &exp_off);
//   slot->mantissa_words[0] = (uint64_t)(mant * (double)UINT64_MAX);
//   slot->exponent = (int64_t)exp_off;
//   if (A_re < 0) slot->state_flags |= APA_FLAG_SIGN_NEG;
//
// Key: mant ∈ [0.5, 1.0); mant * UINT64_MAX fills word[0] with full range.
// Higher words zeroed → precision boundary acknowledged.
//
// rdi = Slot4096*  (struct pointer)
// xmm0 = A_re
//
    .globl  ap_writeback_double
    .type   ap_writeback_double, @function
ap_writeback_double:
    .cfi_startproc
    pushq   %rbp
    movq    %rsp,   %rbp
    subq    $16,    %rsp

    // Early-out: A_re == 0.0
    vxorpd  %xmm1,  %xmm1, %xmm1
    vucomisd %xmm1,  %xmm0
    je      .Lwb_zero

    // fabs(A_re) → frexp
    vandpd  .Labs_mask(%rip), %xmm0, %xmm0   // xmm0 = |A_re|
    // frexp: returns mantissa in xmm0, exponent in *ptr (rsp)
    leaq    -8(%rbp), %rsi                    // &exp_off on stack
    call    frexp@PLT                          // xmm0 = mant ∈ [0.5,1); -8(%rbp) = exp
                                               //→ VGETEXPSD + VGETMANTSD (AVX-512): single instruction
                                               //  frexp otherwise needs full libm call

    // slot->mantissa_words[0] = (uint64_t)(mant * UINT64_MAX)
    movq    (%rdi), %r8                        // r8 = slot->mantissa_words  (ptr, first field)
    testq   %r8,   %r8
    jz      .Lwb_ret
    movsd   .LUINT64MAX_D(%rip), %xmm1
    vmulsd  %xmm1, %xmm0,  %xmm0             // xmm0 = mant * UINT64_MAX
    vcvttsd2si %xmm0, %rax                    // rax = (uint64_t)(truncate)
                                               //→ VCVTTSD2USI (AVX-512) for unsigned truncate
    movq    %rax,  (%r8)                       // word[0] = value

    // Zero higher words
    movq    32(%rdi), %rcx                     // rcx = slot->num_words  (field offset 32 example)
    decq    %rcx
    jle     .Lwb_exp
    leaq    8(%r8),   %r9
    movq    $0,       %rax
    // rep stosq: zero rcx words starting at r9
    // (compiler emits memset call or rep-stos for larger clears)
    //→ For N≤64 words: 64-byte AVX512 zmm stores are faster
    movq    %rcx, %rdx
.Lwb_zero_loop:
    movq    $0,   (%r9)
    addq    $8,   %r9
    decq    %rdx
    jnz     .Lwb_zero_loop

.Lwb_exp:
    // slot->exponent = (int64_t)exp_off
    movl    -8(%rbp),  %eax                   // eax = exp_off (int from frexp)
    movslq  %eax,      %rax                   // sign-extend to 64-bit
    // Slot4096 field layout: exponent is at known struct offset (here symbolic)
    movq    %rax,   exponent_offset(%rdi)      //→ replace exponent_offset with real GCC offset

    // Update sign flag
    // Re-check original A_re sign (before fabs) — caller must pass original xmm
    // In real code: compiler keeps original in xmm_save; illustrated here
    // testq MSB of original A_re...
    // slot->state_flags &= ~APA_FLAG_SIGN_NEG  (clear) or |= APA_FLAG_SIGN_NEG  (set)
    // Field access via struct offset arithmetic omitted (compiler knows offsets)

    jmp     .Lwb_ret

.Lwb_zero:
    // Zero entire mantissa and exponent
    movq    (%rdi),  %r8
    testq   %r8,     %r8
    jz      .Lwb_ret
    movq    32(%rdi), %rcx
    xorl    %eax,    %eax
.Lwb_clr:
    movq    $0,      (%r8)
    addq    $8,      %r8
    decq    %rcx
    jnz     .Lwb_clr

.Lwb_ret:
    addq    $16,    %rsp
    popq    %rbp
    ret
    .cfi_endproc
    .size   ap_writeback_double, .-ap_writeback_double

// ============================================================================
// 6.  rk4_step_v31  —  Stage composition overview (not full listing)
// ============================================================================
//
// The RK4 step calls compute_deriv_v31 four times with intermediate states.
// From a micro-architecture perspective:
//
//   Stage k1: derive(st)                       → k1 {A_re, A_im, phase, phase_vel}
//   Stage k2: derive(st + dt/2 * k1)           → k2
//   Stage k3: derive(st + dt/2 * k2)           → k3
//   Stage k4: derive(st + dt   * k3)           → k4
//   Update:   st += dt/6 * (k1 + 2k2 + 2k3 + k4)
//
// The update step maps to 4 VFMADD chains (one per state variable):
//
//   vmovsd   k1_A_re,  %xmm0
//   vaddsd   k4_A_re,  %xmm0, %xmm0           ; k1 + k4
//   vfmadd231sd k2_A_re, two, %xmm0            ; += 2*k2
//   vfmadd231sd k3_A_re, two, %xmm0            ; += 2*k3
//   vmulsd   dt_sixth,  %xmm0, %xmm0          ; * (dt/6)
//   vaddsd   st_A_re,   %xmm0, %xmm0          ; + st.A_re
//
// Register pressure: 4 stages × 4 state vars × 2 (k + intermediate) = 32 slots
// YMM registers (16 available on x86-64 without AVX-512) insufficient to hold all.
// GCC spills to stack; AVX-512 (32 ZMM) would avoid all spills.
//
// Memory layout for intermediate states (stack frame):
//   -160(%rbp): k1  {A_re, A_im, phase, phase_vel}  =  4 × 8 = 32 bytes
//   -128(%rbp): k2
//    -96(%rbp): k3
//    -64(%rbp): k4
//    -32(%rbp): temp state t
//
// [F3 note]: dt loaded ONCE before RK4 stages begin, not recomputed per stage.
//   movsd lat->dt_global(%r12), %xmm15    ; dt in xmm15, read-only during RK4
//
    .globl  rk4_step_v31_overview
    .type   rk4_step_v31_overview, @function
rk4_step_v31_overview:
    // Annotated overview only; see C source for full implementation.
    // Key instruction sequences documented above in inner excerpt.
    ret
    .size   rk4_step_v31_overview, .-rk4_step_v31_overview

// ============================================================================
// Read-only data pool
// ============================================================================

    .section .rodata
    .align  8

.LPHI:
    .double 1.6180339887498948

.L1_0:
    .double 1.0

.L1em12:
    .double 1.0e-12

.LBASE_GRA:
    .double 0.18

.LUINT64MAX_D:
    .double 18446744073709551615.0    // (double)UINT64_MAX

.Labs_mask:
    .quad  0x7FFFFFFFFFFFFFFF         // fabs mask (clear sign bit)
    .quad  0x7FFFFFFFFFFFFFFF         // upper lane (VANDPD XMM uses 128-bit)

// ============================================================================
// AVX2 Vectorization Opportunities Summary
// ============================================================================
//
// 1. gra_prime_loop: integer IMULQ chain is serial; n≤16 so negligible.
//
// 2. spectral_eval (8 harmonics):
//    — Pack 4 × kd values (k=1..4) into YMM:
//        vmovddup  dphi_xmm, %ymm0         ; broadcast dphi to all 4 lanes
//        vmulpd   .Lharmonic_k1234(%rip), %ymm0, %ymm0  ; kd[0..3] = k*dphi
//        vcos_pd   %ymm0, %ymm0            ; 4 cosines (SVML _mm256_cos_pd)
//        vmulpd    g_W_cos(%rip), %ymm0, %ymm0
//        vhaddpd / vpermilpd + vaddpd for horizontal sum
//    — Reduces 8 serial cos calls to 2 SVML _mm256_cos_pd calls.
//    — Speedup ≈ 3× on kernel-heavy workloads.
//
// 3. ll_residue_probe:
//    — Cannot vectorise due to serial dependency s_n+1 = f(s_n).
//    — Unroll 2 independent LL chains from different seeds → 2× throughput.
//
// 4. lattice_integrate_v31 slot loop:
//    — Phase and amplitude arrays (when refactored to SoA) allow:
//        vmovapd  lat->phase[i..i+3], %ymm0   ; 4 phases in YMM
//        vsubpd   st_phase_broadcast, %ymm0   ; 4 dphi values simultaneously
//    — v31 AoS layout (Slot4096) prevents this in current form.
//    — v31b SoA layout (FastLattice) is designed for this path.
//
// 5. ap_writeback_double:
//    — The word-zeroing loop maps to VMOVDQU 256-bit stores when num_words≥4.
//        vpxor   %ymm0, %ymm0, %ymm0
//        vmovdqu %ymm0, 8(%r8)              ; clear 4 words at once
//
// ============================================================================
