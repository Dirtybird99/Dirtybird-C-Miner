@echo off
REM DirtyBird Optimization Benchmark Runner
REM Usage: run_optimization.bat [mode] [options]
REM
REM Modes:
REM   sweep     - Run full optimization sweep (default)
REM   baseline  - Run baseline benchmark only
REM   compare   - Compare two result files
REM   analyze   - Analyze existing results

setlocal EnableDelayedExpansion

set SCRIPT_DIR=%~dp0
set PROJECT_DIR=%SCRIPT_DIR%..
set CONFIG_FILE=%PROJECT_DIR%\configs\optimization_matrix.json
set OUTPUT_DIR=%PROJECT_DIR%\benchmark_results

REM Check for Python
where python >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo ERROR: Python not found in PATH
    exit /b 1
)

REM Parse mode argument
set MODE=%1
if "%MODE%"=="" set MODE=sweep

REM Execute based on mode
if "%MODE%"=="sweep" (
    echo Running full optimization sweep...
    echo Configuration: %CONFIG_FILE%
    echo Output: %OUTPUT_DIR%
    echo.
    python "%SCRIPT_DIR%bench_optimize.py" --sweep --matrix "%CONFIG_FILE%" --output "%OUTPUT_DIR%" --include-sweeps
    goto :analyze
)

if "%MODE%"=="baseline" (
    echo Running baseline benchmark...
    python "%SCRIPT_DIR%bench_optimize.py" --config baseline_no_spsa --output "%OUTPUT_DIR%"
    goto :end
)

if "%MODE%"=="spsa" (
    echo Running SPSA benchmark...
    python "%SCRIPT_DIR%bench_optimize.py" --config default_spsa --output "%OUTPUT_DIR%"
    goto :end
)

if "%MODE%"=="cache" (
    echo Running cache batch benchmark...
    python "%SCRIPT_DIR%bench_optimize.py" --config cache_batch --output "%OUTPUT_DIR%"
    goto :end
)

if "%MODE%"=="analyze" (
    goto :analyze
)

if "%MODE%"=="compare" (
    if "%~2"=="" (
        echo Usage: run_optimization.bat compare file1.json file2.json
        exit /b 1
    )
    python "%SCRIPT_DIR%bench_optimize.py" --compare "%~2" "%~3"
    goto :end
)

if "%MODE%"=="quick" (
    echo Running quick baseline comparison (30s each)...
    python "%SCRIPT_DIR%bench_optimize.py" --config baseline_no_spsa --runs 1 --duration 30 --output "%OUTPUT_DIR%"
    python "%SCRIPT_DIR%bench_optimize.py" --config default_spsa --runs 1 --duration 30 --output "%OUTPUT_DIR%"
    goto :analyze
)

echo Unknown mode: %MODE%
echo.
echo Available modes:
echo   sweep    - Full optimization sweep
echo   baseline - Baseline (no SPSA) only
echo   spsa     - SPSA enabled only
echo   cache    - Cache batch only
echo   quick    - Quick 30s comparison
echo   analyze  - Analyze existing results
echo   compare  - Compare two result files
exit /b 1

:analyze
echo.
echo Analyzing results...
python "%SCRIPT_DIR%analyze_results.py" "%OUTPUT_DIR%" --report
goto :end

:end
echo.
echo Done.
pause
