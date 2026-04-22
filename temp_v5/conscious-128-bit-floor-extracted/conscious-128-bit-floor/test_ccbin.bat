@echo off
setlocal enabledelayedexpansion
echo step1
set CCBIN=
where cl.exe >nul 2>&1
echo step2
if %ERRORLEVEL% equ 0 for /f "delims=" %%P in ('where cl.exe 2^>nul') do set CCBIN=%%P
echo step3: CCBIN=%CCBIN%
if "%CCBIN%"=="" if exist "C:\Program Files\LLVM\bin\clang-cl.exe" set "CCBIN=C:\Program Files\LLVM\bin\clang-cl.exe"
echo step4: CCBIN=%CCBIN%
