@echo off
REM CHATGPTSUPERUPLOAD.bat
REM Runs only inside RedditVideoBot folder
REM Creates clean zip for ChatGPT + commits/pushes to GitHub

:: === SETTINGS ===
set PROJECT_DIR=C:\Users\tyler\RedditVideoBot
set ZIPNAME=RedditVideoBot_code_only.zip
set REPO_URL=https://github.com/tylergomes80-ui/RedditVideoBot.git
:: =================

echo === Step 1: Switching to project folder ===
if not exist "%PROJECT_DIR%" exit /b
cd /d "%PROJECT_DIR%"

echo === Step 2: Creating clean zip for ChatGPT ===
if exist "%ZIPNAME%" del "%ZIPNAME%"

powershell -command ^
  "Compress-Archive -Path *.py, *.yaml, *.yml, *.json, *.txt, *.md, requirements.txt, config.yaml -DestinationPath '%ZIPNAME%' -Force"

echo Zip created: %PROJECT_DIR%\%ZIPNAME%

echo === Step 3: Git commit and push ===

if exist ".git\index.lock" del /f /q ".git\index.lock"

if not exist ".git" git init

git add -A

REM Commit only if there are staged changes
git diff --cached --quiet || git commit -m "Auto commit %date% %time%"

git branch -M main
git remote get-url origin >nul 2>&1 || git remote add origin %REPO_URL%

git push -u origin main || git push -u origin main --force

echo === ALL DONE ===
echo - Zip ready: %PROJECT_DIR%\%ZIPNAME%
echo - GitHub updated: %REPO_URL%
