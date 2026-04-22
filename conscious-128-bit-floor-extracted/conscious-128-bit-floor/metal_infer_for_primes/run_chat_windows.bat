@echo off
REM run_chat_windows.bat - Build + launch the analog-bot server + chat TUI
REM
REM Usage:
REM   run_chat_windows.bat                  - build both, serve on 8000, open chat
REM   run_chat_windows.bat --port 9000      - use a different port
REM   run_chat_windows.bat --corpus FILE    - pass a custom corpus to the server
REM   run_chat_windows.bat --no-build       - skip build, just launch

setlocal

set "PORT=8000"
set "CORPUS=..\pipeline\sft\train.jsonl"
set "NO_BUILD=0"

:parse
if "%~1"=="" goto done_parse
if /i "%~1"=="--port" (
    set "PORT=%~2"
    shift
    shift
    goto parse
)
if /i "%~1"=="--corpus" (
    set "CORPUS=%~2"
    shift
    shift
    goto parse
)
if /i "%~1"=="--no-build" (
    set "NO_BUILD=1"
    shift
    goto parse
)
shift
goto parse
:done_parse

if "%NO_BUILD%"=="0" (
    echo [run_chat] Building server bot.exe...
    call build_bot_windows.bat
    if errorlevel 1 (
        echo [ERROR] Server build failed - aborting.
        exit /b 1
    )
    echo [run_chat] Building chat client chat_win.exe...
    call build_chat_windows.bat
    if errorlevel 1 (
        echo [ERROR] Chat client build failed - aborting.
        exit /b 1
    )
)

if not exist "bot.exe" (
    echo [ERROR] bot.exe not found.  Run: build_bot_windows.bat
    exit /b 1
)
if not exist "chat_win.exe" (
    echo [ERROR] chat_win.exe not found.  Run: build_chat_windows.bat
    exit /b 1
)

REM ── Start server in a new window ──────────────────────────────────────────
echo [run_chat] Starting analog-bot server on port %PORT%...
if exist "%CORPUS%" (
    start "analog-bot-server" cmd /c "bot.exe --serve %PORT% --corpus %CORPUS%"
) else (
    echo [warn] Corpus not found: %CORPUS%
    echo [warn] Server will run without corpus.
    start "analog-bot-server" cmd /c "bot.exe --serve %PORT%"
)

REM ── Wait briefly for server to bind ───────────────────────────────────────
echo [run_chat] Waiting for server to start...
timeout /t 2 /nobreak >nul

REM ── Launch chat client ────────────────────────────────────────────────────
echo [run_chat] Launching chat client...
chat_win.exe --port %PORT%

endlocal
