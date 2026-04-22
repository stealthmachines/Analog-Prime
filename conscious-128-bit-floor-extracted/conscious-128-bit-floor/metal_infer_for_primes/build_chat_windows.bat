@echo off
setlocal enabledelayedexpansion
REM build_chat_windows.bat — Build chat_win.c (Windows TUI chat client)
REM
REM   chat_win.exe is the Windows port of chat.m (macOS/POSIX).
REM   Uses Winsock2 for TCP, Windows Console API for ANSI/history.
REM
REM   Modes:
REM     (default)   clang -O2   -> chat_win.exe
REM     --gcc       GCC 15.2    -> chat_win_gcc.exe
REM     --msvc      MSVC        -> chat_win_msvc.exe
REM     --all       all three
REM
REM   Run:
REM     chat_win.exe [--port 8000] [--show-think] [--resume <id>]
REM                  [--sessions] [--hdgl] [--help]

set "SRC=%~dp0chat_win.c"
set "OUT=%~dp0chat_win.exe"
set "GCC_OUT=%~dp0chat_win_gcc.exe"
set "MSVC_OUT=%~dp0chat_win_msvc.exe"
set "GCC_EXE=C:\msys64\mingw64\bin\gcc.exe"

REM VS2022 first, fall back to VS2017
set "VCVARS=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
if not exist "!VCVARS!" set "VCVARS=C:\Program Files (x86)\Microsoft Visual Studio\2017\BuildTools\VC\Auxiliary\Build\vcvars64.bat"

set USE_GCC=0
set USE_MSVC=0
set USE_ALL=0
for %%A in (%*) do (
    if "%%A"=="--gcc"  set USE_GCC=1
    if "%%A"=="--msvc" set USE_MSVC=1
    if "%%A"=="--all"  set USE_ALL=1
)

if "%USE_ALL%"=="1"  goto :build_all
if "%USE_GCC%"=="1"  goto :build_gcc
if "%USE_MSVC%"=="1" goto :build_msvc
goto :build_clang

REM ─────────────────────────────────────────────────────────────────────────
:build_all
echo [chat_win] ===== BUILD ALL COMPILERS =====
echo.
call :do_clang
echo.
call :do_gcc
echo.
call :do_msvc
echo.
echo [chat_win] ===== All builds complete =====
echo   clang: %OUT%
echo   gcc:   %GCC_OUT%
echo   msvc:  %MSVC_OUT%
endlocal
exit /b 0

REM ─────────────────────────────────────────────────────────────────────────
:build_clang
call :do_clang
if %ERRORLEVEL% neq 0 exit /b 1
echo.
echo [chat_win] Run: "%OUT%" --help
endlocal
exit /b 0

:do_clang
if exist "!VCVARS!" call "!VCVARS!" >nul 2>&1
REM Kill any running chat_win.exe so linker can overwrite it
taskkill /f /im chat_win.exe >nul 2>&1
echo [chat_win] Building with clang -O2 (Winsock2)...
clang ^
    -O2 ^
    -D_CRT_SECURE_NO_WARNINGS ^
    -D_WIN32_WINNT=0x0601 ^
    "%SRC%" ^
    -o "%OUT%" ^
    -lws2_32
if %ERRORLEVEL% neq 0 (
    echo [chat_win] clang build FAILED.
    exit /b 1
)
echo [chat_win] Build OK (clang): %OUT%
exit /b 0

REM ─────────────────────────────────────────────────────────────────────────
:build_gcc
call :do_gcc
if %ERRORLEVEL% neq 0 exit /b 1
echo.
echo [chat_win] Run: "%GCC_OUT%" --help
endlocal
exit /b 0

:do_gcc
if not exist "!GCC_EXE!" (
    echo [chat_win] GCC not found at C:\msys64\mingw64\bin\gcc.exe
    echo [chat_win] Install MSYS2 and run: pacman -S mingw-w64-x86_64-gcc
    exit /b 1
)
echo [chat_win] Building with GCC 15.2 -O2 (Winsock2)...
set "PATH=C:\msys64\mingw64\bin;C:\msys64\usr\bin;!PATH!"
"!GCC_EXE!" ^
    -O2 ^
    -D_CRT_SECURE_NO_WARNINGS ^
    -D_WIN32_WINNT=0x0601 ^
    "!SRC!" ^
    -o "!GCC_OUT!" ^
    -lws2_32
if !ERRORLEVEL! neq 0 (
    echo [chat_win] GCC build FAILED.
    exit /b 1
)
echo [chat_win] Build OK (GCC): !GCC_OUT!
exit /b 0

REM ─────────────────────────────────────────────────────────────────────────
:build_msvc
echo [chat_win] Mode: MSVC (VS2022/VS2017 Build Tools)
if not exist "!VCVARS!" (
    echo [chat_win] vcvars64.bat not found. Install VS2022 or VS2017 Build Tools.
    exit /b 1
)
call "!VCVARS!"
call :do_msvc
if %ERRORLEVEL% neq 0 exit /b 1
echo.
echo [chat_win] Run: "%MSVC_OUT%" --help
endlocal
exit /b 0

:do_msvc
echo [chat_win] Building with MSVC /O2 (Winsock2)...
cl ^
    /O2 /TC ^
    /D_CRT_SECURE_NO_WARNINGS ^
    /D_WIN32_WINNT=0x0601 ^
    "%SRC%" ^
    ws2_32.lib ^
    /Fe:"%MSVC_OUT%"
if %ERRORLEVEL% neq 0 (
    echo [chat_win] MSVC build FAILED.
    exit /b 1
)
echo [chat_win] Build OK (MSVC): %MSVC_OUT%
exit /b 0
