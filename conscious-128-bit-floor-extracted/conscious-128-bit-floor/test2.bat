@echo off
setlocal enabledelayedexpansion
set SELFTEST=0
set DEBUG_FLAGS=
for %%A in (%*) do (
    if /i "%%A"=="--debug"    set DEBUG_FLAGS=-G -lineinfo -O0
    if /i "%%A"=="--selftest" set SELFTEST=1
)
echo SELFTEST=%SELFTEST%
echo         Install LLVM from https://releases.llvm.org/ ok
endlocal
