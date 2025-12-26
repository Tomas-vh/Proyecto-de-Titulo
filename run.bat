@echo off
setlocal
REM Cambia a la carpeta donde está este .bat
cd /d %~dp0
if not exist "input" mkdir input
if not exist "output" mkdir output

echo ===== Ejecutando pipeline Python =====
python --version
python run_pipeline.py
echo.
echo Hecho. Revisa la carpeta "output".
pause
