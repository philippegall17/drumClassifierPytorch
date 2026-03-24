@echo off
:: Quickstart runs the drum classifier on the example.mp3 file in this folder.

cd /d "%~dp0\.."

set /p PLOTCHOICE="Show feature map plots? (Y/N): "
if /i "%PLOTCHOICE%"=="Y" (
    python InferenceSingle04.py "quickstart/example.mp3" --enable-plots
) else (
    python InferenceSingle04.py "quickstart/example.mp3"
)
pause
