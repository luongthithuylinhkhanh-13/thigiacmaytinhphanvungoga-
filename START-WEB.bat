@echo off
setlocal enabledelayedexpansion

:: ============================================================================
:: PotholeSeg - Pothole Detection and Segmentation System
:: Script tu dong khoi dong Web Server Flask
:: ============================================================================

title PotholeSeg - Web Server

:: 1. Chuyen huong ve dung thu muc goc cua project
cd /d "%~dp0"

echo ===============================================================================
echo                POTHOLESEG - HE THONG PHAN VUNG O GA BANG AI
echo ===============================================================================
echo [1/4] Thu muc lam viec: %cd%

:: 2. Kiem tra va kich hoat moi truong ao Python neu co (venv hoac .venv)
set "VENV_DIR="
if exist "%~dp0.venv\Scripts\activate.bat" (
    set "VENV_DIR=%~dp0.venv"
    echo [2/4] Phat hien moi truong ao .venv tai project.
    echo [*] Dang kich hoat moi truong ao .venv...
    call "%~dp0.venv\Scripts\activate.bat"
    goto :PYTHON_DETECT
)

if exist "%~dp0venv\Scripts\activate.bat" (
    set "VENV_DIR=%~dp0venv"
    echo [2/4] Phat hien moi truong ao venv tai project.
    echo [*] Dang kich hoat moi truong ao venv...
    call "%~dp0venv\Scripts\activate.bat"
    goto :PYTHON_DETECT
)

echo [2/4] Khong co thu muc .venv hoac venv. Su dung Python he thong.

:PYTHON_DETECT
set "PYTHON_CMD="
if defined VENV_DIR (
    if exist "!VENV_DIR!\Scripts\python.exe" (
        set "PYTHON_CMD=!VENV_DIR!\Scripts\python.exe"
    )
)

if not defined PYTHON_CMD (
    python -c "exit(0)" >nul 2>&1
    if not errorlevel 1 (
        set "PYTHON_CMD=python"
    ) else (
        py -3 -c "exit(0)" >nul 2>&1
        if not errorlevel 1 (
            set "PYTHON_CMD=py -3"
        ) else (
            py -c "exit(0)" >nul 2>&1
            if not errorlevel 1 (
                set "PYTHON_CMD=py"
            ) else if exist "%LOCALAPPDATA%\Python\bin\python.exe" (
                set "PYTHON_CMD=%LOCALAPPDATA%\Python\bin\python.exe"
            ) else if exist "%LOCALAPPDATA%\Python\pythoncore-3.14-64\python.exe" (
                set "PYTHON_CMD=%LOCALAPPDATA%\Python\pythoncore-3.14-64\python.exe"
            )
        )
    )
)

if not defined PYTHON_CMD (
    echo.
    echo ===============================================================================
    echo [LOI] Khong tim thay trinh thong dich Python tren he thong!
    echo Vui long cai dat Python hoac tao moi truong ao cho project.
    echo ===============================================================================
    echo.
    pause
    exit /b 1
)

echo [*] Python executable: %PYTHON_CMD%

:: 3. Kiem tra file khoi dong web (app.py)
set "ENTRY_POINT=app.py"
if not exist "%~dp0%ENTRY_POINT%" (
    echo.
    echo ===============================================================================
    echo [LOI] Khong tim thay file khoi dong web: %ENTRY_POINT%
    echo Vui long kiem tra lai thu muc du an.
    echo ===============================================================================
    echo.
    pause
    exit /b 1
)

:: 4. Kiem tra port 5000 de tranh khoi dong trung lap
set "PORT=5000"
set "URL=http://127.0.0.1:%PORT%"

set "FOUND_PID="
for /f "tokens=5" %%a in ('netstat -ano ^| findstr /C:":%PORT% " ^| findstr LISTENING 2^>nul') do (
    set "FOUND_PID=%%a"
)

if not defined FOUND_PID goto :START_SERVER

:: Neu port 5000 da co tien trinh chay
set "PROC_NAME=Khong xac dinh"
for /f "tokens=1" %%b in ('tasklist /FI "PID eq %FOUND_PID%" /NH 2^>nul') do (
    set "PROC_NAME=%%b"
)

echo.
echo ===============================================================================
echo [THONG BAO] Port %PORT% hien dang hoat dong!
echo   - Tien trinh PID: %FOUND_PID% - %PROC_NAME%
echo   - Web server cua project da duoc bat truoc do.
echo   - He thong khong tao server trung lap.
echo.
echo [*] Dang tu dong mo trinh duyet toi: %URL%
echo ===============================================================================
start "" "%URL%"
echo.
echo Nhan phim bat ky de thoat cua so nay...
pause >nul
exit /b 0

:START_SERVER
:: 5. Bat tien trinh ngam theo doi server khoi dong va tu dong mo trinh duyet
echo [3/4] Khoi tao bo theo doi tu dong mo trinh duyet khi server san sang...
start "" /B powershell -ExecutionPolicy Bypass -NoProfile -WindowStyle Hidden -Command "$p=%PORT%; $u='%URL%'; for ($i=0; $i -lt 90; $i++) { Start-Sleep -Milliseconds 600; try { $c = New-Object System.Net.Sockets.TcpClient; $c.Connect('127.0.0.1', $p); if ($c.Connected) { $c.Close(); Start-Process $u; break; } } catch {} }"

:: 6. Khoi dong Web Server Flask (app.py)
echo [4/4] Dang khoi dong Web Server: %ENTRY_POINT%...
echo ===============================================================================
echo Server dang khoi chay tai: %URL%
echo Nhan to hop phim Ctrl + C de dung server khi can thiet.
echo ===============================================================================
echo.

%PYTHON_CMD% "%~dp0%ENTRY_POINT%"

:: 7. Xu ly khi tien trinh server ket thuc hoac co loi
set "EXIT_CODE=%errorlevel%"
echo.
echo ===============================================================================
if not "%EXIT_CODE%"=="0" (
    echo [LOI / CANH BAO] Server da dung lai hoac gap loi khi khoi dong!
    echo Ma loi Exit code: %EXIT_CODE%
    echo Neu thieu thu vien, vui long kiem tra thong bao loi o tren.
) else (
    echo [THONG BAO] Server da dong binh thuong.
)
echo ===============================================================================
echo.
echo Cua so CMD duoc giu lai de ban co the doc thong bao loi.
echo Nhan phim bat ky de dong cua so nay...
pause >nul
