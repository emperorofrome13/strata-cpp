@echo off
REM strata-cpp build: MSVC -> CUDA include/lib on the environment -> ninja (cl.exe).
REM Outputs build\strata_cpp.dll (product ABI, src\capi.cpp), build\strata_cpp_dev.dll (test ABI),
REM build\sc_score.exe and build\sc_bench.exe.
REM
REM MSVC: vcvars64.bat is taken from the VCVARS environment variable when set; otherwise the newest
REM       Visual Studio install with the C++ tools is found with vswhere.exe.
REM CUDA: CUDA_PATH must point at a CUDA 12.x toolkit root (the CUDA installer sets it).
REM
REM Usage:  build.bat
REM         set SC_NO_PAUSE=1 before calling from a script; the window then does not wait on failure.
setlocal EnableExtensions

set "ROOT=%~dp0"

if defined VCVARS goto :have_vcvars
set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
if not exist "%VSWHERE%" set "VSWHERE=%ProgramFiles%\Microsoft Visual Studio\Installer\vswhere.exe"
if not exist "%VSWHERE%" goto :no_vcvars
for /f "usebackq delims=" %%i in (`"%VSWHERE%" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do set "VSINSTALL=%%i"
if not defined VSINSTALL goto :no_vcvars
set "VCVARS=%VSINSTALL%\VC\Auxiliary\Build\vcvars64.bat"

:have_vcvars
if not exist "%VCVARS%" goto :no_vcvars
REM vsdevcmd.bat (called by vcvars) runs vswhere itself; keep the installer directory on PATH so its
REM lookup does not fail on a machine where the installer is not otherwise on PATH.
if defined VSWHERE for %%d in ("%VSWHERE%") do set "PATH=%%~dpd;%PATH%"

if not defined CUDA_PATH goto :no_cuda
if not exist "%CUDA_PATH%\include\cuda.h" (
  echo [build] ERROR: cuda.h not found under "%CUDA_PATH%\include"
  echo [build] set CUDA_PATH to a CUDA 12.x toolkit root
  goto :fail
)
if not exist "%CUDA_PATH%\lib\x64\cuda.lib" (
  echo [build] ERROR: cuda.lib not found under "%CUDA_PATH%\lib\x64"
  echo [build] set CUDA_PATH to a CUDA 12.x toolkit root
  goto :fail
)

echo [build] MSVC: "%VCVARS%"
echo [build] CUDA: "%CUDA_PATH%"
call "%VCVARS%" >nul
if errorlevel 1 (
  echo [build] ERROR: vcvars64.bat failed
  goto :fail
)

set "INCLUDE=%CUDA_PATH%\include;%INCLUDE%"
set "LIB=%CUDA_PATH%\lib\x64;%LIB%"
set "PATH=%CUDA_PATH%\bin;%PATH%"

if not exist "%ROOT%build" mkdir "%ROOT%build"
if not exist "%ROOT%build\obj\main" mkdir "%ROOT%build\obj\main"
if not exist "%ROOT%build\obj\dev" mkdir "%ROOT%build\obj\dev"
if not exist "%ROOT%build\obj\tools" mkdir "%ROOT%build\obj\tools"

pushd "%ROOT%"
ninja -f build.ninja
set "RC=%ERRORLEVEL%"
popd

if not "%RC%"=="0" (
  echo [build] ERROR: ninja exited %RC%; the compiler output above shows the first error
  goto :fail
)
echo [build] ok: build\strata_cpp.dll, build\strata_cpp_dev.dll, build\sc_score.exe, build\sc_bench.exe
exit /b 0

:no_vcvars
echo [build] ERROR: vcvars64.bat not found.
echo [build] install Visual Studio 2022 Build Tools with the "Desktop development with C++" workload,
echo [build] or set VCVARS to the full path of your vcvars64.bat.
goto :fail

:no_cuda
echo [build] ERROR: CUDA_PATH is not set.
echo [build] install the CUDA 12.x toolkit (its installer sets CUDA_PATH), or set CUDA_PATH
echo [build] to the toolkit root that contains include\cuda.h and lib\x64\cuda.lib.
goto :fail

:fail
echo [build] FAILED
if not defined SC_NO_PAUSE pause
exit /b 1
