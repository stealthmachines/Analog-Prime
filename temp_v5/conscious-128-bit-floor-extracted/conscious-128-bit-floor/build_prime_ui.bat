@echo off
REM build_prime_ui.bat — Build prime_ui.exe (Windows TUI for quantum-prime library)
REM
REM Usage:
REM   build_prime_ui.bat          (default: clang)
REM   build_prime_ui.bat --gcc    (MinGW GCC)
REM   build_prime_ui.bat --msvc   (MSVC cl.exe)
REM   build_prime_ui.bat --all    (all three compilers)
REM
REM Outputs: prime_ui.exe  prime_ui_gcc.exe  prime_ui_msvc.exe

setlocal enabledelayedexpansion

set SRC=prime_ui.c
set OUT=prime_ui.exe
set GCC_OUT=prime_ui_gcc.exe
set MSVC_OUT=prime_ui_msvc.exe

set CLANG=C:\Program Files\LLVM\bin\clang.exe
set GCC=C:\msys64\mingw64\bin\gcc.exe
set VCVARS=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat

set MODE=clang
if /i "%~1"=="--gcc"  set MODE=gcc
if /i "%~1"=="--msvc" set MODE=msvc
if /i "%~1"=="--all"  set MODE=all

if not exist "%SRC%" (
    echo [ERROR] %SRC% not found in current directory
    exit /b 1
)

if "%MODE%"=="clang" goto build_clang
if "%MODE%"=="gcc"   goto build_gcc
if "%MODE%"=="msvc"  goto build_msvc
if "%MODE%"=="all"   goto build_all

:build_clang
taskkill /f /im "%OUT%" >nul 2>&1
echo [prime_ui] Building with clang -O2...
"%CLANG%" ^
    -O2 ^
    -mavx2 -mfma -msse4.1 ^
    -D_CRT_SECURE_NO_WARNINGS ^
    -D_USE_MATH_DEFINES ^
    "%SRC%" -o "%OUT%"
if %ERRORLEVEL% neq 0 (
    echo [FAILED] clang build
    exit /b 1
)
echo [OK] Built: %OUT%
if "%MODE%"=="clang" (
    echo.
    echo [prime_ui] Run:  prime_ui.exe
    endlocal & exit /b 0
)

:build_gcc
if "%MODE%"=="gcc" taskkill /f /im "%GCC_OUT%" >nul 2>&1
echo [prime_ui] Building with GCC -O2...
if not exist "%GCC%" (
    echo [SKIP] GCC not found at %GCC%
    goto :eof
)
set "PATH=C:\msys64\mingw64\bin;%PATH%"
"%GCC%" ^
    -O2 ^
    -D_CRT_SECURE_NO_WARNINGS ^
    -D_USE_MATH_DEFINES ^
    "%SRC%" -o "%GCC_OUT%" -lm
if %ERRORLEVEL% neq 0 (
    echo [FAILED] GCC build
    if "%MODE%"=="gcc" exit /b 1
) else (
    echo [OK] Built: %GCC_OUT%
)
if "%MODE%"=="gcc" (
    echo.
    echo [prime_ui] Run:  %GCC_OUT%
    endlocal & exit /b 0
)

:build_msvc
if "%MODE%"=="msvc" taskkill /f /im "%MSVC_OUT%" >nul 2>&1
echo [prime_ui] Building with MSVC cl /O2...
if not exist "%VCVARS%" (
    echo [SKIP] MSVC vcvars64.bat not found
    goto :eof
)
call "%VCVARS%" >nul 2>&1
cl /nologo /O2 /TC ^
    /D_CRT_SECURE_NO_WARNINGS ^
    /D_USE_MATH_DEFINES ^
    "%SRC%" /Fe:"%MSVC_OUT%" /link >nul
if %ERRORLEVEL% neq 0 (
    echo [FAILED] MSVC build
    if "%MODE%"=="msvc" exit /b 1
) else (
    echo [OK] Built: %MSVC_OUT%
)
if "%MODE%"=="msvc" (
    echo.
    echo [prime_ui] Run:  %MSVC_OUT%
    endlocal & exit /b 0
)

:build_all
call :build_clang
call :build_gcc
call :build_msvc
echo.
echo [prime_ui] All builds complete.
echo   clang:  %OUT%
echo   gcc:    %GCC_OUT%
echo   msvc:   %MSVC_OUT%
endlocal
exit /b 0

:eof
