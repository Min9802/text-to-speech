@echo off
setlocal enabledelayedexpansion

set PYTHON_VERSION=3.10

REM Kiểm tra Python đã được cài đặt chưa
python --version >nul 2>&1
if errorlevel 1 (
    echo Python %PYTHON_VERSION% chua duoc cai dat. Vui long cai dat Python.
    exit /b 1
)

echo Su dung Python phien ban: %PYTHON_VERSION%

REM Kiểm tra môi trường ảo
if exist .env\ok (
    call .env\Scripts\activate.bat
) else (
    echo Moi truong chua san sang. Dang thiet lap...
    rmdir /s /q .env 2>nul
    
    REM Tạo môi trường ảo mới
    python -m venv .env
    if errorlevel 1 (
        echo Khong the tao moi truong ao.
        exit /b 1
    )
    REM Cập nhật submodules
    git submodule update --init --recursive
    
    cd TTS
    git fetch --tags
    @REM git checkout 0.1.1
    
    echo Dang cai dat TTS...
    pip install -e .[all]
    
    cd ..
    
    echo Dang cai dat cac yeu cau khac...
    pip install -r requirements.txt -q
    
    echo Dang tai xuong tokenizer tieng Nhat/Trung...
    python -m unidic download
    
    echo. > .env\ok
)

REM Chạy ứng dụng
python app.py

endlocal
