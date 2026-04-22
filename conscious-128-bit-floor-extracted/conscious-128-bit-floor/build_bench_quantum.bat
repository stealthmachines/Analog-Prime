@echo off
setlocal enabledelayedexpansion
REM build_bench_quantum.bat — Build bench_quantum.cu (oscillator) and bench_prime_funcs.c (primes)
REM
REM   Oscillator modes:
REM     (default)    clang -x c -O3         -> bench_quantum.exe
REM     --gcc        GCC 15.2 (MinGW64)     -> bench_quantum_gcc.exe
REM     --msvc       MSVC 19.x (VS2022/VS2017 Build Tools) -> bench_quantum_msvc.exe
REM     --all        Build all three compilers, run all, compare
REM     --quantum    cuStateVec GPU build   -> (Linux-only SDK, prints instructions)
REM
REM   Prime-function modes (reuses same compiler subroutines):
REM     --primes     clang only             -> bench_prime_funcs.exe
REM     --primes-all Build all three compilers, run all, compare
REM
REM   NOTE: NVIDIA cuQuantum SDK is Linux-only; --quantum mode documents
REM         the build command but cannot run on Windows without WSL2.

setlocal

set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2"
set "SRC=%~dp0bench_quantum.cu"
set "OUT=%~dp0bench_quantum.exe"
set "PRIME_SRC=%~dp0bench_prime_funcs.c"
set "PRIME_OUT=%~dp0bench_prime_funcs.exe"
set "PRIME_GCC_OUT=%~dp0bench_prime_funcs_gcc.exe"
set "PRIME_MSVC_OUT=%~dp0bench_prime_funcs_msvc.exe"
set "GCC_EXE=C:\msys64\mingw64\bin\gcc.exe"
REM Prefer VS2022 Build Tools; fall back to VS2017
set "VCVARS=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
if not exist "!VCVARS!" set "VCVARS=C:\Program Files (x86)\Microsoft Visual Studio\2017\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
set "MSVC_OUT=%~dp0bench_quantum_msvc.exe"
set "GCC_OUT=%~dp0bench_quantum_gcc.exe"

REM Parse optional flags
set QUANTUM=0
set USE_GCC=0
set USE_MSVC=0
set USE_ALL=0
set USE_PRIMES=0
set USE_PRIMES_ALL=0
for %%A in (%*) do (
    if "%%A"=="--quantum"    set QUANTUM=1
    if "%%A"=="--gcc"        set USE_GCC=1
    if "%%A"=="--msvc"       set USE_MSVC=1
    if "%%A"=="--all"        set USE_ALL=1
    if "%%A"=="--primes"     set USE_PRIMES=1
    if "%%A"=="--primes-all" set USE_PRIMES_ALL=1
)

if "%USE_PRIMES_ALL%"=="1" goto :build_primes_all
if "%USE_PRIMES%"=="1"     goto :build_primes
if "%USE_ALL%"=="1"        goto :build_all
if "%USE_GCC%"=="1"        goto :build_gcc
if "%USE_MSVC%"=="1"       goto :build_msvc
if "%QUANTUM%"=="1"        goto :build_quantum
goto :build_cpu

REM ─────────────────────────────────────────────────────────────────────────
:build_all
echo [bench_quantum] ===== BUILD ALL COMPILERS =====
echo.
call :do_clang
echo.
call :do_gcc
echo.
call :do_msvc
echo.
echo [bench_quantum] ===== RUN ALL BENCHMARKS =====
echo.
echo --- Clang (CPU-sim) ---
"%OUT%"
echo.
echo --- GCC 15.2 MinGW64 (CPU-sim) ---
"%GCC_OUT%"
echo.
echo --- MSVC (VS2022/VS2017 CPU-sim) ---
"%MSVC_OUT%"
endlocal
exit /b 0

REM ─────────────────────────────────────────────────────────────────────────
:build_cpu
echo [bench_quantum] Mode: CPU simulation (clang)
call :do_clang
if %ERRORLEVEL% neq 0 exit /b 1
goto :run

:do_clang
if exist "!VCVARS!" call "!VCVARS!" >nul 2>&1
echo [bench_quantum] Building with clang -x c -O3 -march=native...
clang ^
  -O3 ^
  -march=native ^
  -D_CRT_SECURE_NO_WARNINGS ^
  -x c ^
  "%SRC%" ^
  -o "%OUT%"
if %ERRORLEVEL% neq 0 (
    echo [bench_quantum] clang build FAILED.
    exit /b 1
)
echo [bench_quantum] Build OK (clang): %OUT%
exit /b 0

REM ─────────────────────────────────────────────────────────────────────────
:build_gcc
echo [bench_quantum] Mode: GCC 15.2 (MSYS2 MinGW64)
call :do_gcc
if %ERRORLEVEL% neq 0 exit /b 1
set OUT=%GCC_OUT%
goto :run

:do_gcc
if not exist "!GCC_EXE!" (
    echo [bench_quantum] GCC not found at C:\msys64\mingw64\bin\gcc.exe
    echo [bench_quantum] Install MSYS2 and run: pacman -S mingw-w64-x86_64-gcc
    exit /b 1
)
echo [bench_quantum] Building with GCC 15.2 -x c -O3 -march=native...
REM MinGW GCC requires its own bin dirs on PATH for cc1/ld DLL resolution
set "PATH=C:\msys64\mingw64\bin;C:\msys64\usr\bin;!PATH!"
"!GCC_EXE!" ^
  -O3 ^
  -march=native ^
  -D_CRT_SECURE_NO_WARNINGS ^
  -D_USE_MATH_DEFINES ^
  -x c ^
  "!SRC!" ^
  -o "!GCC_OUT!"
if !ERRORLEVEL! neq 0 (
    echo [bench_quantum] GCC build FAILED.
    exit /b 1
)
echo [bench_quantum] Build OK (GCC 15.2): !GCC_OUT!
exit /b 0

REM ─────────────────────────────────────────────────────────────────────────
:build_msvc
echo [bench_quantum] Mode: MSVC (VS2022/VS2017 Build Tools)
call :do_msvc
if %ERRORLEVEL% neq 0 exit /b 1
set OUT=%MSVC_OUT%
goto :run

:do_msvc
if not exist "!VCVARS!" (
    echo [bench_quantum] vcvars64.bat not found. Install VS2017/2022 Build Tools.
    exit /b 1
)
echo [bench_quantum] Setting up MSVC environment via vcvars64.bat...
call "!VCVARS!"
echo [bench_quantum] Building with cl.exe /O2 /TC...
cl.exe /O2 /W3 /TC /D_CRT_SECURE_NO_WARNINGS /D_USE_MATH_DEFINES "!SRC!" /Fe:"!MSVC_OUT!" /link
if !ERRORLEVEL! neq 0 (
    echo [bench_quantum] MSVC build FAILED.
    exit /b 1
)
echo [bench_quantum] Build OK (MSVC): !MSVC_OUT!
exit /b 0

REM ─────────────────────────────────────────────────────────────────────────
:build_quantum
echo [bench_quantum] Mode: cuStateVec GPU (LL_QUANTUM_ENABLED)
echo.
echo [bench_quantum] NOTICE: NVIDIA cuQuantum SDK is Linux-only.
echo [bench_quantum] There are no Windows binaries in NVIDIA's redistribution
echo [bench_quantum] (only linux-x86_64 and linux-sbsa tarballs are provided).
echo.
echo [bench_quantum] To use GPU mode on Windows, use WSL2 with NVIDIA CUDA support:
echo [bench_quantum]   https://docs.nvidia.com/cuda/wsl-user-guide/
echo [bench_quantum]   Then: apt-get install cuquantum-dev-cu12
echo.
echo [bench_quantum] The build commands for Linux/WSL2 are:
echo [bench_quantum]
echo [bench_quantum]   export CUQUANTUM_ROOT=/usr/local/cuquantum
echo [bench_quantum]   nvcc -arch=sm_75 -I$CUQUANTUM_ROOT/include \
echo [bench_quantum]        -L$CUQUANTUM_ROOT/lib64 -lcustatevec -lcublas \
echo [bench_quantum]        -O2 -c ll_quantum.cu -o ll_quantum.o
echo [bench_quantum]   nvcc -arch=sm_75 -DLL_QUANTUM_ENABLED \
echo [bench_quantum]        -I$CUQUANTUM_ROOT/include \
echo [bench_quantum]        -L$CUQUANTUM_ROOT/lib64 -lcustatevec -lcublas \
echo [bench_quantum]        -O2 bench_quantum.cu ll_quantum.o -o bench_quantum
exit /b 1

REM ─────────────────────────────────────────────────────────────────────────
REM Prime-function benchmark modes (reuse :do_clang / :do_gcc / :do_msvc
REM but swap SRC/OUT to bench_prime_funcs.c targets first)

:build_primes
echo [bench_prime] Mode: clang, bench_prime_funcs.c
set "SRC=!PRIME_SRC!"
set "OUT=!PRIME_OUT!"
call :do_clang
if !ERRORLEVEL! neq 0 exit /b 1
echo.
echo [bench_prime] Running clang prime benchmark...
echo ============================================================
"!OUT!"
echo ============================================================
echo [bench_prime] Results saved to bench_prime_results.tsv
endlocal
exit /b 0

:build_primes_all
echo [bench_prime] ===== BUILD ALL COMPILERS (prime functions) =====
echo.
REM Set up MSVC environment first — clang on Windows uses MSVC/SDK headers
if exist "!VCVARS!" call "!VCVARS!"
set "SRC=!PRIME_SRC!"
set "OUT=!PRIME_OUT!"
set "GCC_OUT=!PRIME_GCC_OUT!"
set "MSVC_OUT=!PRIME_MSVC_OUT!"
call :do_clang
echo.
call :do_gcc
echo.
call :do_msvc
echo.
echo [bench_prime] ===== RUN ALL (prime functions) =====
echo.
echo --- Clang -O3 ---
"!PRIME_OUT!"
echo.
echo --- GCC 15.2 MinGW64 -O3 ---
"!PRIME_GCC_OUT!"
echo.
echo --- MSVC 19.16 /O2 ---
"!PRIME_MSVC_OUT!"
endlocal
exit /b 0

REM ─────────────────────────────────────────────────────────────────────────
:run
echo.
echo [bench_quantum] Running benchmark...
echo ============================================================
"%OUT%"
echo ============================================================
echo [bench_quantum] Results saved to bench_quantum_results.tsv
endlocal
