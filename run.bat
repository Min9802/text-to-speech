@echo off
setlocal enabledelayedexpansion

set PYTHON_VERSION=3.10

REM Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Python is not installed or not in PATH. Please install Python %PYTHON_VERSION% or later.
    pause
    exit /b 1
)

echo Using Python version:
python --version

REM Check if virtual environment exists and is OK
if exist .env\ok (
    echo Activating existing virtual environment
    call .env\Scripts\activate.bat
) else (
    echo The environment is not ok. Running setup
    if exist .env (
        rmdir /s /q .env
    )
    
    echo Creating virtual environment
    python -m venv .env
    if %errorlevel% neq 0 (
        echo Failed to create virtual environment.
        pause
        exit /b 1
    )
    
    call .env\Scripts\activate.bat
    
    echo Upgrading pip setuptools and wheel
    python -m pip install --upgrade pip setuptools wheel
    
    echo Updating git submodules
    git submodule update --init --recursive
    if %errorlevel% neq 0 (
        echo Failed to update git submodules.
        pause
        exit /b 1
    )
    
    cd TTS
    echo Fetching tags and checking out version 0.1.1
    git fetch --tags
    git checkout 0.1.1
    if %errorlevel% neq 0 (
        echo Failed to checkout TTS version 0.1.1.
        cd ..
        pause
        exit /b 1
    )
    
    echo Installing TTS in development mode
    pip install -e . -q
    if %errorlevel% neq 0 (
        echo Failed to install TTS.
        cd ..
        pause
        exit /b 1
    )
    
    cd ..
    
    echo Installing project requirements (gradio, transformers, etc.)
    pip install -r requirements.txt -q
    
    echo Downloading Japanese/Chinese tokenizer
    @REM python -m unidic download
    if %errorlevel% neq 0 (
        echo Warning: Failed to download unidic tokenizer but continuing
    )
    
    echo Setup completed successfully.
    echo. > .env\ok
)

echo Starting app
python app.py
if %errorlevel% neq 0 (
    echo Failed to run app.py
    pause
    exit /b 1
)

pause