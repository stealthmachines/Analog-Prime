# docker_cuquantum.ps1
# Run cuQuantum GPU benchmarks in a Linux container via Docker Desktop (WSL2 backend).
#
# Prerequisites:
#   1. Docker Desktop for Windows — https://www.docker.com/products/docker-desktop/
#      Enable WSL2 backend (Settings → General → "Use WSL 2 based engine")
#   2. NVIDIA Container Toolkit for Docker Desktop / WSL2:
#      https://docs.nvidia.com/cuda/wsl-user-guide/
#      In a WSL2 terminal: sudo apt-get install -y nvidia-container-toolkit
#      Then: sudo nvidia-ctk runtime configure --runtime=docker && sudo service docker restart
#   3. GPU must be SM ≥ 7.0 — RTX 2060 is SM 7.5 ✓
#
# Usage:
#   .\docker_cuquantum.ps1              # build + run bench_quantum GPU mode
#   .\docker_cuquantum.ps1 --psi        # build + run psi_scanner_cuda_v2 GPU bench
#   .\docker_cuquantum.ps1 --shell      # interactive shell in cuQuantum container
#   .\docker_cuquantum.ps1 --check      # verify GPU is visible inside Docker
#
# Image: nvcr.io/nvidia/cuquantum-appliance (includes CUDA 12 + cuStateVec + cuTensorNet)
# For RTX 2060 (SM 7.5): cuquantum-appliance 24.11 or 25.x both work.
#
# Licensed per https://zchg.org/t/legal-notice-copyright-applicable-ip-and-licensing-read-me/440

param(
    [switch]$Psi,
    [switch]$Shell,
    [switch]$Check
)

$IMAGE   = "nvcr.io/nvidia/cuquantum-appliance:24.11-cuda12.x-ubuntu22.04"
$WORKDIR = $PSScriptRoot
$ARCH    = "sm_75"   # RTX 2060

# ── Sanity checks ─────────────────────────────────────────────────────────
if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    Write-Host "[docker_cuquantum] ERROR: 'docker' not found in PATH."
    Write-Host "  Install Docker Desktop: https://www.docker.com/products/docker-desktop/"
    exit 1
}

Write-Host "[docker_cuquantum] Checking Docker daemon..."
$dockerInfo = docker info 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "[docker_cuquantum] ERROR: Docker daemon not running. Start Docker Desktop."
    exit 1
}

# ── Mode: --check ─────────────────────────────────────────────────────────
if ($Check) {
    Write-Host "[docker_cuquantum] Verifying NVIDIA GPU visibility inside Docker..."
    docker run --gpus all --rm "$IMAGE" nvidia-smi
    if ($LASTEXITCODE -eq 0) {
        Write-Host "[docker_cuquantum] GPU visible. RTX 2060 SM7.5 is compatible with cuStateVec."
    } else {
        Write-Host "[docker_cuquantum] GPU not visible. Ensure NVIDIA Container Toolkit is installed:"
        Write-Host "  In WSL2: sudo apt-get install -y nvidia-container-toolkit"
        Write-Host "  Then:    sudo nvidia-ctk runtime configure --runtime=docker"
        Write-Host "  Docs:    https://docs.nvidia.com/cuda/wsl-user-guide/"
    }
    exit $LASTEXITCODE
}

# ── Mode: --shell ─────────────────────────────────────────────────────────
if ($Shell) {
    Write-Host "[docker_cuquantum] Opening interactive shell in cuQuantum container..."
    Write-Host "  Workspace mounted at /work"
    Write-Host "  Build with: nvcc -arch=$ARCH -DLL_QUANTUM_ENABLED \\"
    Write-Host "    -I`$CUQUANTUM_ROOT/include -L`$CUQUANTUM_ROOT/lib64 \"
    Write-Host "    -lcustatevec -lcublas -O2 bench_quantum.cu ll_quantum.cu -o bench_quantum"
    docker run --gpus all --rm -it `
        -v "${WORKDIR}:/work" `
        -w /work `
        "$IMAGE" bash
    exit $LASTEXITCODE
}

# ── Mode: --psi (psi_scanner_cuda_v2, CUDA only — no cuQuantum needed) ────
if ($Psi) {
    Write-Host "[docker_cuquantum] Building and running psi_scanner_cuda_v2 (CUDA only)..."
    Write-Host "  Note: psi_scanner uses CUDA runtime only — no cuQuantum SDK required."
    Write-Host "  This can also be built directly on Windows with nvcc."
    Write-Host ""

    $cmd = @"
set -e
cd /work
echo "=== Building psi_scanner_cuda_v2 ==="
nvcc -O3 -arch=$ARCH -o psi_scanner_v2 psi_scanner_cuda_v2.cu
echo "=== Build OK ==="
echo ""
echo "=== Benchmarking: lattice scan n=[1.0, 2.0], step=0.001 ==="
./psi_scanner_v2 --lattice 1.0 2.0 0.001
echo ""
echo "=== Benchmarking: x-range [1e9, 1e9+1000] (explicit formula) ==="
./psi_scanner_v2 1000000000 1000001000
"@

    docker run --gpus all --rm `
        -v "${WORKDIR}:/work" `
        -w /work `
        "$IMAGE" bash -c $cmd

    Write-Host ""
    Write-Host "[docker_cuquantum] psi_scanner benchmark complete."
    exit $LASTEXITCODE
}

# ── Default mode: build bench_quantum GPU mode (cuStateVec) ───────────────
Write-Host "[docker_cuquantum] Building and running bench_quantum with cuStateVec (LL_QUANTUM_ENABLED)..."
Write-Host "  Image : $IMAGE"
Write-Host "  GPU   : RTX 2060, $ARCH"
Write-Host "  Mount : $WORKDIR -> /work"
Write-Host ""

$cmd = @"
set -e
cd /work
echo "=== Checking cuQuantum ==="
ls \$CUQUANTUM_ROOT/include/custatevec.h 2>/dev/null && echo "custatevec.h found" || echo "WARNING: custatevec.h not at \$CUQUANTUM_ROOT/include"

echo ""
echo "=== Building ll_quantum.cu ==="
nvcc -arch=$ARCH \
  -I\$CUQUANTUM_ROOT/include \
  -L\$CUQUANTUM_ROOT/lib64 \
  -lcustatevec -lcublas \
  -O2 -c ll_quantum.cu -o ll_quantum.o

echo "=== Building bench_quantum (GPU mode) ==="
nvcc -arch=$ARCH \
  -DLL_QUANTUM_ENABLED \
  -I\$CUQUANTUM_ROOT/include \
  -L\$CUQUANTUM_ROOT/lib64 \
  -lcustatevec -lcublas \
  -O2 bench_quantum.cu ll_quantum.o -o bench_quantum_gpu

echo "=== Build OK ==="
echo ""
echo "=== Running GPU benchmark ==="
./bench_quantum_gpu
echo ""
echo "Results saved to bench_quantum_results.tsv"
"@

docker run --gpus all --rm `
    -v "${WORKDIR}:/work" `
    -w /work `
    "$IMAGE" bash -c $cmd

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "[docker_cuquantum] GPU benchmark complete. Results in bench_quantum_results.tsv"
} else {
    Write-Host ""
    Write-Host "[docker_cuquantum] Build or run failed."
    Write-Host "  If image not found, pull it first:"
    Write-Host "    docker pull $IMAGE"
    Write-Host "  Or browse available tags:"
    Write-Host "    https://catalog.ngc.nvidia.com/orgs/nvidia/containers/cuquantum-appliance/tags"
    Write-Host "  Alternative image (latest):"
    Write-Host "    nvcr.io/nvidia/cuquantum-appliance:25.03-cuda12.x-ubuntu22.04"
}

exit $LASTEXITCODE
