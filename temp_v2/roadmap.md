# Conscious 2.0 -- Roadmap
*Last updated: April 20, 2026*

**Prime directive:** Does this change evolve or devolve the codebase?
Only pull in external assets if they directly advance one of the three evolutions below.

---

## Current Codebase (codebase/)

Files in `codebase/` are the active project. Everything else is reference until it earns a place.

| File | Role | Status |
|------|------|--------|
| hdgl_analog_v33.cu | GPU field evolution + Markov trit verdict gate | OK, 17/17 |
| hdgl_analog_v34.cu | Feistel phase (golden-angle fixed-point destroyed) + KAPPA fix | OK, builds |
| hdgl_analog_v35.cu | Full U-field bridge: M(U) warp/block reduce → Λ_φ^(U) → S(U) | OK, builds |
| hdgl_warp_ll_v33.cu | gpucarry CUDA graph (p<400K) + NTT LL (M61) | OK |
| hdgl_critic_v33.c/.h | CPU 5->8->1 MLP critic, TD(0) training | OK |
| hdgl_sieve_v34.cu | Mersenne sieve, BASE_P=82589934 | OK |
| hdgl_multigpu_v34.c | Multi-GPU P2P skeleton | skeleton only |
| hdgl_host_v33.c | 3-stream async host, critic<-LL-residue loop | OK |
| hdgl_bench_v33.cu | Test harness | 17/17 pass |
| hdgl_bench.exe | Built binary | 0.42 GSlots/s |
| BigG + Fudge10.md | APA C code -- emergent G(z)/c(z) vs Pan-STARRS1 | reference |
| base4096-dict-builder.md | Rosetta Stone + compression stack plan | reference |

**Build command (PowerShell):**
```
$env:PATH = "C:\Program Files (x86)\Microsoft Visual Studio\2017\BuildTools\VC\Tools\MSVC\14.16.27023\bin\Hostx64\x64;$env:PATH"
nvcc -O3 -arch=sm_75 -allow-unsupported-compiler hdgl_analog_v33.cu hdgl_warp_ll_v33.cu hdgl_sieve_v34.cu hdgl_critic_v33.c hdgl_bench_v33.cu -o hdgl_bench.exe
```

---

## Evolution 1 -- Session Memory Compression

**Problem:** Every new session loses full project state. Re-grounding costs ~10KB of plain-text
summary per session. The codebase concepts (Lambda_phi, mulmod61, Slot4096, the gate math,
build commands, file lineage) are dense and verbose in plain English.

**Solution:** A self-describing compressed state token that encodes the entire session context
into a single Base4096 block, using this project's own vocabulary as the dictionary.

### The Compression Stack (bottom to top)

```
1. Concept layer     -- raw idea: e.g. "resonance gate uses Lambda_phi = log(p*ln2/lnphi)/lnphi - 1/(2phi)"
                        |
2. Token layer       -- phi-based language: operator encoded as position in phi-log space
                        vector/DNA language: geometric vector in 8D Kuramoto space
                        spiral8: 8-geometry basis (zchg.org/t/spiral8-8-geometries-dna/876)
                        |
3. Compression layer -- MEGC: TernaryNode parity tree + BreathingEntropyCoder (arithmetic)
                        GoldenContext: phi-scaled frequency model
                        Onion Shell Encoding: multi-layer wrapping for discourse corpus
                        |
4. Adaptive layer    -- fold26 wu-wei: 5 adaptive strategies, lattice-selected
                        fold26_bridge: lk_ AEAD gateway for sealed state blocks
                        |
5. Wire layer        -- Base4096 v2.0.1: 12-bit/char, 3 bytes -> 2 Unicode glyphs
                        Rosetta Stone prefix: 0x1200x = mode (dict/raw/encrypted/grammar)
```

### Rosetta Stone Dictionary -- This Project

The dictionary is NOT built from general corpora first. It is built from this codebase.
Source files for phrase extraction (phrase_extractor_to_json.py from base4096-dict-builder.md
gives the method -- apply it to project docs, not just literary corpora):

- README.md
- roadmap.md (this file)
- Source file headers and comments from codebase/
- The 51 Mersenne exponents table
- Math constants: phi, ln_phi, M61, NTT_SIZE, LIMBS, KAPPA_PHI

Expected dictionary size: ~500-1000 project-specific tokens covering 95% of re-grounding cost.
General English corpus (Bible + Moby Dick + Dao) provides the base; project vocab overlays it.

### Short term
- [x] Run phrase_extractor_to_json.py against README.md + roadmap.md + source headers
- [x] Build project-specific Rosetta Stone JSON (map top phrases -> Base4096 tokens)  -- `codebase/rosetta_stone.json`
- [x] Prototype encode/decode of a session-state block in Python
- [x] Verify round-trip fidelity (encode session summary -> decode -> matches original)

### Medium term
- [ ] Port MEGC (TernaryNode + BreathingEntropyCoder) from Python to C
- [ ] Integrate fold26 wu-wei adaptive strategy selection
- [ ] Onion Shell: layer the state block (math layer / code layer / build layer)
- [ ] phi-language token assignment for the core operators (Lambda_phi, D_n, S(p), U-field)
- [ ] spiral8 geometric basis: map 8 spiral geometries to the 8D Kuramoto oscillator axes
- [ ] vector/DNA language: encode field evolution trajectories as geometric DNA strands

### Long term
- [ ] hdgl_corpus_seeder.c integration: codebase emits its own compressed state as output
- [ ] Self-provisioning: running hdgl_bench.exe produces a Base4096 state token automatically
- [ ] Session handoff: one Base4096 block restores full project context in any new session

### Key assets (in conscious-128-bit-floor-extracted/)
- base4096_hdgl_fully_selfcontained.py -- complete encode/decode
- fold26_wuwei.c / fold26_bridge.c -- wu-wei stream + lk_ gateway
- hdgl_corpus_seeder.c -- corpus ingestion
- frozen_base4096_alphabet.txt -- canonical 4096-char alphabet
- base4096_hdgl_selfprovisioning.json -- self-provisioning dictionary seed
- prismatic5z.py -- spiral8 / prismatic_recursion visualizer (GLSL, 32768 instances)
- phrase_extractor_to_json.py -- extract top N phrases to JSON (method in base4096-dict-builder.md)

---

## Evolution 2 -- LL Verifier (Competitive Scale)

**Problem:** hdgl_warp_ll_v33 has one path (NTT over M61, 4096-bit, p < ~4000).
Gap to GIMPS frontier (p=136,279,841): 33,000x. Gap to competitive range (p=110,503): 27x.

**The 128-bit floor has ll_mpi.cu with 7 paths already tested to p=110,503.**

### Benchmark comparison (RTX 2060, April 2026)

| Path | p=21,701 | p=44,497 | p=86,243 | p=110,503 |
|------|----------|----------|----------|-----------|
| gpucarry (ll_mpi) | 1.2 s | 3.4 s | 11.7 s | 19.2 s |
| schoolbook | 3.6 s | 7.3 s | 14.9 s | 22.1 s |
| NTT (ll_mpi) | 5.2 s | 10.9 s | 22.9 s | 32.0 s |
| warp_ll_v33 (current) | ~p<4000 only | -- | -- | -- |

gpucarry is 3.1x faster than NTT at small p. Auto-threshold: schoolbook wins below ~386K, NTT above.

### Short term
- [x] Port ll_gpu_gpucarry from ll_mpi.cu into hdgl_warp_ll_v33.cu — 11 primes, 5 composites verified
- [x] Add NTT_AUTO_THRESHOLD = 400000 dispatch
- [x] Verify at p=21701, p=44497 (match 1.2s, 3.4s benchmarks)

### Medium term
- [ ] Add psi_scanner_cuda as Stage 0 (Riemann zeta pre-filter, kills composites before squaring)
      Uses zeta_zeros_10k.json (10,000 Riemann zeros, already in 128-bit floor)
- [ ] Feed phi_mersenne_predictor.c top-20 candidates into sieve priority queue
      67% of known Mersenne exponents have frac(n) < 0.5 -- use this as pre-filter
- [ ] prismatic_recursion as refinement stage between resonance gate and LL call
      Formula: sqrt(phi^(id%16) * F[id%128] * 2^(id%16) * P[id%128] * Omega) * r^(id%7+1)

### Long term
- [ ] Extend to p~1M+ (requires multi-limb NTT beyond M61, or NTT over Z/(2^64-2^32+1))
- [ ] CommLayerState (v6.c) integration for correct async multi-GPU exponent sync
- [ ] 8D Kuramoto analog path (ll_analog.c) as CUDA-free fallback + double-confirmation
      Prime signal: osc LOCKED + residue=0 simultaneously

### Key assets (in conscious-128-bit-floor-extracted/)
- ll_mpi.cu (108KB) -- 7-path LL dispatcher, full gpucarry + NTT implementation
- ll_analog.c / ll_analog.h -- 8D Kuramoto + exact APA squaring
- phi_mersenne_predictor.c -- all 51 exponents, 67% frac<0.5 result, top-20 predictions
- prime_pipeline.c -- segmented sieve + D_n scoring -> pipe to ll_cuda
- psi_scanner_cuda.cu / psi_scanner_cuda_v2.cu -- Riemann psi pre-filter
- zeta_zeros_10k.json -- 10,000 Riemann zeta zeros
- v6.c -- Slot4096 with CommLayerState, async exponent decoupling

---

## Evolution 3 -- Resonance Gate (Principled Verdict)

**Problem:** Current gate is a bare threshold: resonance < 0.25 && acc > 5.0 && amp > 0.6.
This is a heuristic approximation. The 128-bit floor has a 6-stage Markov trit fused engine
with warp majority-vote, Slot4096 slow-sync, and a principled verdict derived from field
statistics -- not a fixed threshold.

Beyond that, v33 computes Lambda_phi directly from p. The correct derivation is from field
state M(U): p biases the field, field settles, field produces Lambda_phi^(U).

### Short term
- [x] Replace threshold gate with lambda_k - sigma Markov trit verdict rule:
        phi+ > 0.35  -->  ACCEPT  (prime lock signal)
        phi- > 0.45  -->  REJECT
        R = 1.2*phi- + 0.8*gamma - phi+ > 0.6  -->  REJECT
        else  -->  UNCERTAIN
- [x] Port verdict rule from conscious_fused_engine.cu into hdgl_analog_v33.cu

### Medium term
- [x] Add Slot4096 slow-sync correction (every 16 steps, SYNC_GAIN=0.08)
      Fast <-> slow error accumulation currently missing
- [x] Add warp majority-vote sigma correction (ballot + __popc)
      Warp consensus on trit state eliminates single-thread noise

### Long term
- [x] Full U-field bridge (COMPLETE — hdgl_analog_v35.cu):
        v34: Feistel phase map on T², KAPPA*log(p) wired, golden-angle trap destroyed
        v35: closes the full loop — S(U) now derived from field state, not r_harmonic directly
          u_inner = φ^(0.5·phj_n + κ·log(r_h+1))        per-thread inner φ-exponent
          M_inner = warp_mean(u_inner)                     warp reduce (32 lanes)
          u_mid   = φ^(M_inner)                            middle φ-layer
          M_U     = block_mean(u_mid)  [sh_u_mid[8]]       block reduce (8 warps)
          Λ_φ^(U) = log(M_U)/ln(φ) − 1/(2φ)              from field, not from p directly
          S(U) = phi_resonance_from_lambda(Λ_φ^(U))
        Prime invariant: coherent field → u_inner uniform → M_U stable → S(U) at prime minimum
        Flow: p → r_h (KAPPA) → phvel → Feistel → phj_n → u_inner → M_U → Λ_φ^(U) → S(U)
        U^(p) = φ^(Σ_i φ^(Σ_j φ^(interaction(U_i,U_j) + κ·log(p))))
        interaction(U_i,U_j) = 0.5·phj_n  (Feistel coupling, already computed)
- [ ] BigG/Fudge10 validation: if D(n,beta) fits 200+ CODATA constants, feed tuned
      beta back into gate as calibrated parameter (closes physics <-> codebase loop)

### Key assets (in conscious-128-bit-floor-extracted/)
- conscious_fused_engine.cu (24KB) -- full 6-stage lambda_k-sigma fused kernel
- analog_engine.c / analog_engine.h -- AnalogContainer1, Pluck->Sustain->FineTune->Lock
- EMPIRICAL_VALIDATION_ASCII.c -- Fudge10 / CODATA 200+ constant fits (compilable now)
- EMPIRICAL_VALIDATION_MICROTUNED.c -- precision-tuned version

---

## Origin Goals (from v30b -- the soul)

These five bullets were written at the end of the v30b source. Everything since is executing them.

| Goal | Status | Where |
|------|--------|-------|
| Fusion with FastSlot / Slot4096 engine | DONE | hdgl_analog_v33.cu + hdgl_warp_ll_v33.cu |
| Replace GRA with learned spectral kernel | DONE (stage 1) | wavelet_spectral_eval() in v33 is the spectral kernel; making it *learned* is Evolution 3 |
| Push into fully self-adapting GPU search system | IN PROGRESS | Evolution 2 + 3 combined |
| Direct LL-residue -> reward signal (this is huge) | NOT YET | see below -- this is the critical missing loop |
| Replace 32-bit LL-lite with 4096-bit warp-lattice cooperative version | DONE | hdgl_warp_ll_v33.cu |

### GRA Heritage

The old system's core was Golden Recursive Algebra:
```
r_n = sqrt(phi * F_n * 2^n * product(primes_1..n) * omega)   (closed form)
r_n = r_{n-1} * sqrt(2 * p_n * F_n/F_{n-1})                  (recursive)
```
r_harmonic is still alive in Slot4096 and fed into the gate via KAPPA_PHI=0.02.
GRA plasticity (v30b): r_harmonic += GRA_PLASTICITY * (local_amp - 0.5)
This brain-like adaptation was the first self-modifying mechanism. wavelet_spectral_eval()
is its successor -- a 4-scale Morlet kernel replacing the static GRA table lookup.
Making the wavelet scales *learned* (via critic backprop) is what "learned spectral kernel" means.

### LL-residue -> reward signal (the missing loop)

This is the highest-value unimplemented item. The full cycle is:
```
p candidate -> sieve -> resonance gate ACCEPT -> LL test -> residue
                                                              |
                        critic_reward() <-------- residue == 0 ? reward=+1 : reward=-1
                              |
                        update gate weights (TD(0))
                              |
                        next candidate filtered by updated gate
```
Currently the full loop IS wired: `run_sync_cycle()` in hdgl_host_v33.c harvests candidates,
runs `hdgl_warp_ll_v33_launch`, gets residue, computes `reward = (residue < LL_RESIDUE_EPS) ? +1.0f : -1.0f`,
calls `critic_observe(s, td_target)`, and uploads updated weights back to GPU each cycle.

- [x] Wire ll_residue_val from hdgl_warp_ll_v33 -> host -> critic_update() in hdgl_critic_v33.c
- [x] Replace proxy reward with: reward = (residue < RESIDUE_THRESH) ? +1.0f : -1.0f
- [ ] Verify gate converges: prime candidates should score higher over successive exponents
- [ ] This closes the full loop: GPU field -> gate -> LL verifier -> reward -> gate

---

## Mathematical Foundation (reference)

All three evolutions share this math. Do not change it without a proof.

```
phi = 1.6180339887498948482
ln_phi = 0.4812118250596035
M61 = 0x1FFFFFFFFFFFFFFF  (2^61 - 1)

-- X+1=0 / phi-resonance gate --
S(p) = |e^(i*pi*Lambda_phi) + 1_eff|
Lambda_phi = log(p*ln2/lnphi) / lnphi - 1/(2*phi)     (Mersenne bridge)
1_eff = 1 + delta(i)
delta(i) = |cos(pi*beta*phi)| * ln(P_n) / phi^(n+beta)  (lattice correction)
|Omega*C^2| = 1  ->  C^2 = 1/Omega  ->  cancels         (U(1) normalization)

-- phi-lattice coordinate --
n(x) = log(log(x)/lnphi) / lnphi - 1/(2*phi)
n(2^p) = log(p*ln2/lnphi) / lnphi - 1/(2*phi)
Empirical: 67% of 51 known Mersenne exponents have frac(n) < 0.5

-- D_n resonance operator --
D_n(r) = sqrt(phi * F_{n+beta} * P_{n+beta} * base^{n+beta} * Omega) * r^k
F = continuous Binet Fibonacci, P = prime table
Omega = 0.5 + 0.5*sin(pi*frac(n)*phi)

-- U-field bridge (target, not yet implemented) --
Lambda_phi^(U) = log(M(U)) / ln(phi) - 1/(2*phi)
Prime invariant: all 8 Kuramoto oscillators lock -> theta->0, M(U)->8, S(U)~1.531
```

---

## Hardware Reference

```
GPU:   RTX 2060  sm_75  1920 CUDA cores  6 GB GDDR6
CUDA:  13.2  nvcc
CC:    MSVC 2017 cl.exe  (must be on PATH before nvcc)
Flag:  -allow-unsupported-compiler
smem:  48 KB/block default, 96 KB configurable
Bandwidth: 336 GB/s theoretical, ~115 GB/s measured (33% util)
Occupancy bottleneck: sh_gc[4][256] = 12 KB/block -> max 4 active blocks at 48 KB
```

---

*Does this change evolve or devolve the codebase?*
