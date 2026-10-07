@echo off
setlocal
cd /d "%~dp0"
if errorlevel 1 goto :failed
if not exist ".git" (
    echo [ERROR] No Git repository at "%~dp0".
    goto :failed
)
git --version >nul 2>&1
if errorlevel 1 goto :failed
echo Repository: %cd%
git remote get-url origin
if errorlevel 1 goto :failed
if /I "%~1"=="--check" goto :check
git pull
if errorlevel 1 goto :failed
echo [OK] Synchronization completed.
pause
exit /b 0

:failed
echo [ERROR] Synchronization stopped. See the error above.
if /I not "%~1"=="--check" pause
exit /b 1

:check
git status --short --branch
exit /b %errorlevel%
