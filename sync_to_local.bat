@echo off
REM ==========================================
REM Sync current "Link" GitHub repository
REM Repository: https://github.com/Arash279/Link.git
REM Author: ChatGPT (GPT-5)
REM ==========================================

setlocal

REM Expected repo directory name
set REPO_DIR=Link
set REPO_URL=https://github.com/Arash279/Link.git

REM Get current directory name
for %%I in ("%cd%") do set CURRENT_DIR=%%~nxI

echo.
echo === GitHub Repository Sync Script ===
echo Repository URL: %REPO_URL%
echo Current Directory: %cd%
echo.

REM Check if current directory is "Link"
if /I not "%CURRENT_DIR%"=="%REPO_DIR%" (
    echo [ERROR] This script must be placed and run inside the "%REPO_DIR%" folder.
    echo Current folder: "%CURRENT_DIR%"
    echo Expected folder: "%REPO_DIR%"
    pause
    exit /b 1
)

REM Ensure Git is available
git --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Git is not installed or not in PATH.
    pause
    exit /b 1
)

REM Pull latest changes
echo Updating repository...
git pull
if %errorlevel% neq 0 (
    echo [ERROR] Update failed. Please check your network or local modifications.
    pause
    exit /b 1
)

echo.
echo === Repository successfully updated! ===
pause
endlocal

