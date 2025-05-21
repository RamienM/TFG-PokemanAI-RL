@echo off
setlocal enabledelayedexpansion

title Entrenamiento Pokémon Red - Reinicio automático con checkpoint dinámico
echo Iniciando entrenamiento en bucle...

:inicio
echo --------------------------
echo Verificando checkpoint previo...
echo --------------------------

set "CHECKPOINT_PATH="

if exist last_checkpoint_path.txt (
    set /p CHECKPOINT_PATH=<last_checkpoint_path.txt
    echo Checkpoint detectado: !CHECKPOINT_PATH!
) else (
    echo No se encontró checkpoint previo.
)

echo --------------------------
echo Ejecutando entrenamiento...
echo --------------------------

if defined CHECKPOINT_PATH (
    python train.py "!CHECKPOINT_PATH!"
) else (
    python train.py
)

echo.
echo Entrenamiento finalizado. Reiniciando en 5 segundos...
timeout /t 5 >nul
goto inicio

