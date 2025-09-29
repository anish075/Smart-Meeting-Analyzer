@echo off
echo Setting up Advanced Video Transcriber Environment...
echo ==================================================

:: Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python is not installed or not in PATH
    pause
    exit /b 1
)

echo Python version:
python --version

:: Install PyTorch
echo.
echo Installing PyTorch...
nvidia-smi >nul 2>&1
if errorlevel 1 (
    echo No CUDA detected. Installing CPU-only PyTorch...
    pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
) else (
    echo CUDA detected. Installing PyTorch with CUDA support...
    pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118
)

:: Install other requirements
echo.
echo Installing other dependencies...
pip install -r requirements.txt

:: Install additional packages for pyannote.audio
echo.
echo Installing additional audio processing libraries...
pip install pyannote.audio[audio]

echo.
echo ==================================================
echo Setup complete!
echo.
echo IMPORTANT NOTES:
echo 1. For pyannote.audio speaker diarization, you may need to:
echo    - Create a free account at https://huggingface.co/
echo    - Accept the terms for pyannote/speaker-diarization-3.1
echo    - Login with: huggingface-cli login
echo.
echo 2. First run may take longer as models are downloaded
echo.
echo 3. For best results, use GPU if available
echo.
echo Usage example:
echo python video_transcriber.py input_video.mp4 output_transcript.xlsx
echo ==================================================
pause