@echo off
REM strata-cpp MXFP4 build: the same toolchain setup as build.bat, but every output goes to build_mx\ so that a
REM running build\strata_cpp.dll (loaded by a benchmark) is never overwritten. Builds build_mx\strata_cpp.dll,
REM build_mx\strata_cpp_dev.dll, build_mx\sc_score.exe, build_mx\sc_bench.exe (build_mx.ninja) and
REM build_mx\strata_cpu_test.dll (build_cpu_mx.ninja).
REM
REM MSVC: VCVARS when set, otherwise vswhere.exe. CUDA: CUDA_PATH (set by the CUDA installer).
REM Usage:  build_mx.bat
REM         set SC_NO_PAUSE=1 before calling from a script; the window then does not wait on failure.
REM Tests find the directory through SC_BUILD_DIR=build_mx (see tests\*.py).
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
  echo [build_mx] ERROR: cuda.h not found under "%CUDA_PATH%\include"
  echo [build_mx] set CUDA_PATH to a CUDA 12.x toolkit root
  goto :fail
)
if not exist "%CUDA_PATH%\lib\x64\cuda.lib" (
  echo [build_mx] ERROR: cuda.lib not found under "%CUDA_PATH%\lib\x64"
  echo [build_mx] set CUDA_PATH to a CUDA 12.x toolkit root
  goto :fail
)

echo [build_mx] MSVC: "%VCVARS%"
echo [build_mx] CUDA: "%CUDA_PATH%"
call "%VCVARS%" >nul
if errorlevel 1 (
  echo [build_mx] ERROR: vcvars64.bat failed
  goto :fail
)

set "INCLUDE=%CUDA_PATH%\include;%INCLUDE%"
set "LIB=%CUDA_PATH%\lib\x64;%LIB%"
set "PATH=%CUDA_PATH%\bin;%PATH%"

if not exist "%ROOT%build_mx" mkdir "%ROOT%build_mx"
if not exist "%ROOT%build_mx\obj\main" mkdir "%ROOT%build_mx\obj\main"
if not exist "%ROOT%build_mx\obj\dev" mkdir "%ROOT%build_mx\obj\dev"
if not exist "%ROOT%build_mx\obj\tools" mkdir "%ROOT%build_mx\obj\tools"
if not exist "%ROOT%build_mx\cpu" mkdir "%ROOT%build_mx\cpu"

pushd "%ROOT%"
ninja -f build_mx.ninja
set "RC=%ERRORLEVEL%"
if not "%RC%"=="0" (
  popd
  echo [build_mx] ERROR: ninja build_mx.ninja exited %RC%; the compiler output above shows the first error
  goto :fail
)
ninja -f build_cpu_mx.ninja
set "RC=%ERRORLEVEL%"
popd

if not "%RC%"=="0" (
  echo [build_mx] ERROR: ninja build_cpu_mx.ninja exited %RC%; the compiler output above shows the first error
  goto :fail
)
echo [build_mx] ok: build_mx\strata_cpp.dll, build_mx\strata_cpp_dev.dll, build_mx\sc_score.exe, build_mx\sc_bench.exe, build_mx\strata_cpu_test.dll
exit /b 0

:no_vcvars
echo [build_mx] ERROR: vcvars64.bat not found.
echo [build_mx] install Visual Studio 2022 Build Tools with the "Desktop development with C++" workload,
echo [build_mx] or set VCVARS to the full path of your vcvars64.bat.
goto :fail

:no_cuda
echo [build_mx] ERROR: CUDA_PATH is not set.
echo [build_mx] install the CUDA 12.x toolkit (its installer sets CUDA_PATH), or set CUDA_PATH
echo [build_mx] to the toolkit root that contains include\cuda.h and lib\x64\cuda.lib.
goto :fail

:fail
echo [build_mx] FAILED
if not defined SC_NO_PAUSE pause
exit /b 1
