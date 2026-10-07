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
git add .
if errorlevel 1 goto :failed
git diff --cached --quiet
if errorlevel 2 goto :failed
if not errorlevel 1 goto :push
git commit -m "Automatic synchronization updates by default"
if errorlevel 1 goto :failed
:push
git push origin main
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
