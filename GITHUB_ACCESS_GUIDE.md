# Analog-Prime on GitHub - Quick Start Guide

## Repository Location
**https://github.com/stealthmachines/Analog-Prime**

## Clone the Repository
```bash
git clone https://github.com/stealthmachines/Analog-Prime.git
cd Analog-Prime
```

## Available Versions

### Latest Version (v40)
**Branch:** `main`
**Tag:** `v40.0.0`
```bash
git checkout main
# or
git checkout v40.0.0
```
**Description:** Circular reward accumulator on S¹, 97k candidates/600 cycles, 34/34 bench pass

### Historical Versions
All versions are available as both branches and semantic version tags:

| Branch | Tag | Version | Description |
|--------|-----|---------|-------------|
| v1-initial | v1.0.0 | v1 | Initial codebase |
| v2-supplemental | v2.0.0 | v2 | Codebase + supplemental |
| v3-development | v3.0.0 | v3 | Development iteration |
| v4-features | v4.0.0 | v4 | Feature additions |
| v5-optimization | v5.0.0 | v5 | Optimization pass |
| v6-refinement | v6.0.0 | v6 | Refinement and testing |
| v7-stable | v7.0.0 | v7 | Stable baseline |
| main | v40.0.0 | v40 | Latest with circular reward |

## Checking Out Versions
```bash
# Via branch
git checkout v3-development

# Via tag
git checkout v5.0.0

# List all available branches
git branch -a

# List all available tags
git tag
```

## Repository Structure
```
Analog-Prime/
├── codebase/                          # Core source files (CUDA, C, Python)
│   ├── hdgl_analog_v35.cu             # GPU field kernel (v40 latest)
│   ├── hdgl_warp_ll_v33.cu            # Lucas-Lehmer verifier (gpucarry + NTT)
│   ├── hdgl_sieve_v34.cu              # Mersenne sieve
│   ├── hdgl_psi_filter_v35.cu         # Riemann psi pre-filter
│   ├── hdgl_predictor_seed.c          # Phi-lattice candidate predictor
│   ├── hdgl_prismatic_v35.c           # Prismatic recursion scorer
│   ├── hdgl_host_v33.c                # Host orchestration + stream pipeline
│   ├── hdgl_critic_v33.c              # TD(0) critic network
│   ├── ll_analog.c                    # 8D Kuramoto analog LL (CUDA-free)
│   ├── hdgl_corpus_seeder.c           # Self-provisioning state codec
│   ├── hdgl_fold26.c                  # Wu Wei adaptive compression
│   ├── hdgl_onion.c                   # Multi-layer state wrapper
│   ├── hdgl_megc.c                    # MEGC TernaryNode + DNA codec
│   ├── hdgl_bench_v33.cu              # Test harness
│   ├── empirical_validation.c         # BigG/Fudge10 CODATA validation
│   ├── phrase_extractor_to_json.py    # Rosetta Stone phrase extraction
│   └── ... (43 total files, 275 project files including docs)
├── README.md                          # Architecture guide + build instructions
├── LICENSE.md                         # ZCHG.org licensing
└── .gitignore                         # Build artifacts ignored
```

## Key Files

### Core CUDA Kernels
- **hdgl_analog_v35.cu**: GPU field evolution with circular reward accumulator, φ-resonance gate
- **hdgl_warp_ll_v33.cu**: Lucas-Lehmer verifier with gpucarry (p<400K) + NTT paths
- **hdgl_sieve_v34.cu**: Mersenne exponent sieve with priority seeding
- **hdgl_psi_filter_v35.cu**: Riemann zeta pre-filter (kills composites early)

### CPU/Host
- **hdgl_host_v33.c**: 3-stream async host managing sieve → psi filter → prismatic → LL pipeline
- **hdgl_critic_v33.c**: TD(0) critic network (5→8→1 MLP), learns resonance gate
- **ll_analog.c**: 8D Kuramoto analog simulator (CUDA-free) for cross-verification
- **hdgl_predictor_seed.c**: Phi-lattice predictor, top-20 candidate scoring

### Scoring & Refinement
- **hdgl_prismatic_v35.c**: Prismatic recursion scorer
- **hdgl_psi_filter_v35.cu**: Riemann psi spike filter (3-pass)

### Utilities & Codecs
- **hdgl_corpus_seeder.c**: Codebase self-emission (3-layer onion: MATH/CODE/BUILD)
- **hdgl_fold26.c**: Wu Wei adaptive compression (5 strategies)
- **hdgl_onion.c**: Multi-layer state block wrapper
- **hdgl_megc.c**: MEGC codec (TernaryNode + BreathingEntropyCoder)

### Testing & Validation
- **hdgl_bench_v33.cu**: Test harness (34/34 pass on v40)
- **empirical_validation.c**: BigG/Fudge10 CODATA validation (100% pass, χ²≈0)

### Data & Documentation
- **README.md**: Full architecture guide, build commands, evolution status
- **LICENSE.md**: ZCHG.org licensing (all branches)
- **rosetta_stone.json**: 500-phrase project Rosetta Stone (Base4096 tokens)
- **frozen_base4096_alphabet.txt**: Canonical 4096-character alphabet

## Build Environment
- **GPU**: RTX 2060 (sm_75, 1920 CUDA cores)
- **CUDA**: 13.2+
- **Compiler**: MSVC 2017 (must use `-allow-unsupported-compiler` flag)
- **Performance**: 0.42 GSlots/s (v35b), 34/34 bench pass (v40)

## Build Instructions
See [README.md](README.md) for complete build guide.

**Quick start (PowerShell):**
```powershell
$env:PATH = "C:\Program Files (x86)\Microsoft Visual Studio\2017\BuildTools\VC\Tools\MSVC\14.16.27023\bin\Hostx64\x64;$env:PATH"
nvcc -O2 -arch=sm_75 -allow-unsupported-compiler -o hdgl_bench_v40.exe hdgl_bench_v33.cu hdgl_warp_ll_v33.cu ...
```

## Architecture Highlights

### Three Evolutions
1. **Evolution 1**: Compression stack (fold26, MEGC, Rosetta Stone, self-provisioning)
2. **Evolution 2**: LL Verifier (gpucarry, psi pre-filter, phi-lattice predictor, prismatic scoring)
3. **Evolution 3**: Resonance Gate (Markov trit verdict, U-field bridge, empirical calibration)

### Core Algorithm
```
Slot i tracks candidate Mersenne exponent p_i
  ↓
8 Kuramoto oscillators couple via Feistel phase map
  ↓
GPU field computes M(U) (warp+block reduces)
  ↓
Lambda_φ^(U) derived from field state (not p directly)
  ↓
φ-resonance gate: S(U) = |e^(iπΛ_φ) + 1_eff|
  ↓
Candidates with S(U) < threshold → Lucas-Lehmer verification
  ↓
LL residue → reward signal → critic TD(0) update
  ↓
Updated gate weights uploaded to GPU next cycle
```

## Performance Notes
- **Circular reward accumulator**: Phase angle ∈ [0,2π], gates on cos(acc) > 0.5
- **Wu Wei exploration**: Bonus when cos(acc) < 0, encourages phase exploration
- **Stall detection**: Fires Pluck kernel (random phase injection) after 50 cycles without new candidates
- **97k candidates per 600 cycles** at default settings (v40)

## License
All software is the property of ZCHG.org. See [LICENSE.md](LICENSE.md) for full terms.

## Questions?
Consult README.md for detailed architecture or review source comments in codebase/ files.
