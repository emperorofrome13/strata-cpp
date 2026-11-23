@echo off
REM strata-cpu build: the CPU expert path (src\cpu_moe.cpp) + its test ABI -> build\strata_cpu_test.dll.
REM Same toolchain setup as build.bat: MSVC (VCVARS or vswhere) -> CUDA include/lib on the environment -> ninja.
REM
REM Usage:  build_cpu.bat
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
  echo [build_cpu] ERROR: cuda.h not found under "%CUDA_PATH%\include"
  echo [build_cpu] set CUDA_PATH to a CUDA 12.x toolkit root
  goto :fail
)
if not exist "%CUDA_PATH%\lib\x64\cuda.lib" (
  echo [build_cpu] ERROR: cuda.lib not found under "%CUDA_PATH%\lib\x64"
  echo [build_cpu] set CUDA_PATH to a CUDA 12.x toolkit root
  goto :fail
)

echo [build_cpu] MSVC: "%VCVARS%"
echo [build_cpu] CUDA: "%CUDA_PATH%"
call "%VCVARS%" >nul
if errorlevel 1 (
  echo [build_cpu] ERROR: vcvars64.bat failed
  goto :fail
)

set "INCLUDE=%CUDA_PATH%\include;%INCLUDE%"
set "LIB=%CUDA_PATH%\lib\x64;%LIB%"
set "PATH=%CUDA_PATH%\bin;%PATH%"

if not exist "%ROOT%build" mkdir "%ROOT%build"
if not exist "%ROOT%build\cpu" mkdir "%ROOT%build\cpu"

pushd "%ROOT%"
ninja -f build_cpu.ninja
set "RC=%ERRORLEVEL%"
popd

if not "%RC%"=="0" (
  echo [build_cpu] ERROR: ninja exited %RC%; the compiler output above shows the first error
  goto :fail
)
echo [build_cpu] ok: build\strata_cpu_test.dll
exit /b 0

:no_vcvars
echo [build_cpu] ERROR: vcvars64.bat not found.
echo [build_cpu] install Visual Studio 2022 Build Tools with the "Desktop development with C++" workload,
echo [build_cpu] or set VCVARS to the full path of your vcvars64.bat.
goto :fail

:no_cuda
echo [build_cpu] ERROR: CUDA_PATH is not set.
echo [build_cpu] install the CUDA 12.x toolkit (its installer sets CUDA_PATH), or set CUDA_PATH
echo [build_cpu] to the toolkit root that contains include\cuda.h and lib\x64\cuda.lib.
goto :fail

:fail
echo [build_cpu] FAILED
if not defined SC_NO_PAUSE pause
exit /b 1
