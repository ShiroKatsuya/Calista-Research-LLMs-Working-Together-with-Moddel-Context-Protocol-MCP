@echo off
echo Starting Minion Terminal...
python minion_terminal.py
if errorlevel 1 (
    echo Error launching Minion Terminal.
    echo Please make sure Python is installed and in your PATH.
    pause
) 