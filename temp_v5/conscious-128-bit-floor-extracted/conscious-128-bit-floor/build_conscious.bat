@echo off
REM build_conscious.bat -- Build conscious.exe + bootloaderZ.exe - Dual-Slot Fused Engine and quantum-layer bootloader for GPU-accelerated random number generation
REM "conscious, by zchg.org"
REM
REM Usage:
REM   build_conscious.bat              (default: sm_75, N=8192, steps=1024)
REM   build_conscious.bat --debug      (add -G -lineinfo for nvcc-gdb)
REM   build_conscious.bat --selftest   (build then run selftest)
REM   build_conscious.bat --no-boot    (skip bootloader build)
REM
REM Requirements:
REM   nvcc in PATH (CUDA Toolkit 12+)
REM   curand library (-lcurand)
REM   sm_75+ GPU (RTX 2060/2070/2080, RTX 30xx, RTX 40xx)
REM   Optional: CUQUANTUM_ROOT set for quantum-layer bootloader

setlocal enabledelayedexpansion

REM Resolve host compiler for nvcc (-ccbin):
REM   1. cl.exe (MSVC VS2019+) if already in PATH
REM   2. cl.exe via vcvars64.bat (VS2017 OK with -allow-unsupported-compiler)
REM   3. clang-cl.exe from LLVM install (needs Windows SDK)
set CCBIN=
where cl.exe >nul 2>&1
if %ERRORLEVEL% equ 0 for /f "delims=" %%P in ('where cl.exe 2^>nul') do set CCBIN=%%P

if "%CCBIN%"=="" (
    set VCVARS=
    for %%V in (
        "C:\Program Files\Microsoft Visual Studio\2022\Enterprise\VC\Auxiliary\Build\vcvars64.bat"
        "C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvars64.bat"
        "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
        "C:\Program Files\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
        "C:\Program Files (x86)\Microsoft Visual Studio\2019\Enterprise\VC\Auxiliary\Build\vcvars64.bat"
        "C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvars64.bat"
        "C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
        "C:\Program Files (x86)\Microsoft Visual Studio\2017\Enterprise\VC\Auxiliary\Build\vcvars64.bat"
        "C:\Program Files (x86)\Microsoft Visual Studio\2017\Community\VC\Auxiliary\Build\vcvars64.bat"
        "C:\Program Files (x86)\Microsoft Visual Studio\2017\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
    ) do if not defined VCVARS if exist %%V set VCVARS=%%V
    if defined VCVARS (
        echo [conscious] Bootstrapping MSVC from !VCVARS!...
        call !VCVARS! >nul 2>&1
        for /f "delims=" %%P in ('where cl.exe 2^>nul') do set CCBIN=%%P
    )
)

if "%CCBIN%"=="" if exist "C:\Program Files\LLVM\bin\clang-cl.exe" set "CCBIN=C:\Program Files\LLVM\bin\clang-cl.exe"
if "%CCBIN%"=="" if exist "C:\msys64\clang64\bin\clang-cl.exe" set "CCBIN=C:\msys64\clang64\bin\clang-cl.exe"
if "%CCBIN%"=="" (
    echo [ERROR] No supported host compiler found ^(cl.exe or clang-cl.exe^).
    echo         Install VS Build Tools 2017+ or LLVM.
    exit /b 1
)
echo [conscious] Host compiler: %CCBIN%

set SRC=conscious_fused_engine.cu
set OUT=conscious.exe
set SM=sm_75
set ARCH=-arch=%SM%
set OPT=-O3
set LIBS=-lcurand
set FLAGS=-D_CRT_SECURE_NO_WARNINGS -allow-unsupported-compiler

REM Parse args
set SELFTEST=0
set NO_BOOT=0
set DEBUG_FLAGS=
for %%A in (%*) do (
    if /i "%%A"=="--debug"    set DEBUG_FLAGS=-G -lineinfo -O0
    if /i "%%A"=="--selftest" set SELFTEST=1
    if /i "%%A"=="--no-boot"  set NO_BOOT=1
)
if defined DEBUG_FLAGS set OPT=%DEBUG_FLAGS%

REM Locate nvcc
where nvcc >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo [ERROR] nvcc not found in PATH.
    echo         Install CUDA Toolkit and add to PATH:
    echo         C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.x\bin
    exit /b 1
)
echo H

if not exist "%SRC%" (
    echo [ERROR] %SRC% not found in current directory
    exit /b 1
)

REM Kill any running instance before link
taskkill /f /im "%OUT%" >nul 2>&1

echo [conscious] Building %SRC% ...
echo [conscious] Target: %SM%  Opt: %OPT%

nvcc %OPT% %ARCH% %FLAGS% %LIBS% -ccbin "%CCBIN%" "%SRC%" -o "%OUT%" 2>&1

if %ERRORLEVEL% neq 0 (
    echo.
    echo [FAILED] Build error -- check output above.
    exit /b 1
)

echo.
echo [OK] Built: %OUT%

REM Show exe size
for %%F in ("%OUT%") do echo      Size: %%~zF bytes

echo.
echo [conscious] Usage:
echo   conscious.exe                       ^(N=8192, steps=1024^)
echo   conscious.exe --N 16384 --steps 512
echo   conscious.exe --quiet
echo   conscious.exe --selftest
echo   conscious.exe --seed DEADBEEF00001234

REM ══════════════════════════════════════════════════════════════════════════
REM  BOOTLOADER BUILD
REM  Priority 1: bootloaderZ_quantum.cu with cuStateVec (if CUQUANTUM_ROOT set)
REM  Priority 2: bootloaderZ_quantum.cu without quantum  (nvcc -x cu, no cuSV)
REM  Priority 3: bootloaderZ.c via gcc/clang             (pure C fallback)
REM ══════════════════════════════════════════════════════════════════════════
if "%NO_BOOT%"=="1" goto :skip_boot

echo.
echo [bootloader] Building bootloaderZ ...

set BOOT_SRC_Q=bootloaderZ_quantum.cu
set BOOT_SRC_C=bootloaderZ.c
set BOOT_OUT=bootloaderZ.exe
set BOOT_OK=0

taskkill /f /im "%BOOT_OUT%" >nul 2>&1

REM -- Priority 1: quantum build (requires CUQUANTUM_ROOT) --
if not defined CUQUANTUM_ROOT goto :boot_no_quantum
if not exist "%CUQUANTUM_ROOT%\include\custatevec.h" goto :boot_no_quantum
echo [bootloader] cuQuantum found at %CUQUANTUM_ROOT% -- building with quantum layer
nvcc %OPT% %ARCH% %FLAGS% -DLL_QUANTUM_ENABLED ^
    -I"%CUQUANTUM_ROOT%\include" -L"%CUQUANTUM_ROOT%\lib\x64" ^
    -lcustatevec -lcudart ^
    -ccbin "%CCBIN%" "%BOOT_SRC_Q%" -o "%BOOT_OUT%" 2>&1
if %ERRORLEVEL% equ 0 (
    set BOOT_OK=1
    echo [bootloader] Quantum layer: ENABLED
    goto :boot_done
)
echo [bootloader] Quantum build failed -- falling through

:boot_no_quantum
REM -- Priority 2: bootloaderZ_quantum.cu without cuStateVec --
if not exist "%BOOT_SRC_Q%" goto :boot_c_fallback
echo [bootloader] Building %BOOT_SRC_Q% without quantum layer...
nvcc %OPT% %ARCH% %FLAGS% ^
    -ccbin "%CCBIN%" "%BOOT_SRC_Q%" -o "%BOOT_OUT%" 2>&1
if %ERRORLEVEL% equ 0 (
    set BOOT_OK=1
    echo [bootloader] Quantum layer: DISABLED ^(no cuQuantum^)
    goto :boot_done
)
echo [bootloader] nvcc build failed -- trying C fallback

:boot_c_fallback
REM -- Priority 3: bootloaderZ.c via gcc or clang --
if not exist "%BOOT_SRC_C%" (
    echo [bootloader] WARNING: %BOOT_SRC_C% not found, skipping bootloader
    goto :boot_done
)
set CC_TOOL=
where gcc >nul 2>&1
if %ERRORLEVEL% equ 0 for /f "delims=" %%P in ('where gcc 2^>nul') do set CC_TOOL=%%P
if "%CC_TOOL%"=="" (
    where clang >nul 2>&1
    if %ERRORLEVEL% equ 0 for /f "delims=" %%P in ('where clang 2^>nul') do set CC_TOOL=%%P
)
if "%CC_TOOL%"=="" if exist "C:\msys64\mingw64\bin\gcc.exe" set "CC_TOOL=C:\msys64\mingw64\bin\gcc.exe"
if "%CC_TOOL%"=="" (
    echo [bootloader] WARNING: no C compiler found for fallback, skipping bootloader
    goto :boot_done
)
echo [bootloader] Building %BOOT_SRC_C% with %CC_TOOL% ...
"%CC_TOOL%" -O2 -D_CRT_SECURE_NO_WARNINGS "%BOOT_SRC_C%" -o "%BOOT_OUT%" -lm 2>&1
if %ERRORLEVEL% equ 0 (
    set BOOT_OK=1
    echo [bootloader] Pure-C build ^(no CUDA^)
)

:boot_done
if "%BOOT_OK%"=="1" (
    for %%F in ("%BOOT_OUT%") do echo [OK] Built: %BOOT_OUT%  ^(%%~zF bytes^)
) else (
    echo [FAILED] bootloaderZ build failed -- conscious.exe will self-seed Slot4096
)

:skip_boot

if "%SELFTEST%"=="1" (
    echo.
    echo [conscious] Running selftest...
    "%OUT%" --selftest
)

endlocal
