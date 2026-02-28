@echo off
setlocal

:: ==========================================
:: RunPod Download Results Script
:: ==========================================

:: 1. RunPod 접속 정보 (본인의 환경에 맞게 수정하세요)
set RUNPOD_IP=YOUR_RUNPOD_IP
set RUNPOD_PORT=YOUR_RUNPOD_PORT
set RUNPOD_USER=root

:: 2. 경로 설정
set RUNPOD_PROJECT_DIR="/workspace/Decision-Aware-Tail-Risk-Optimization"
set LOCAL_RESULTS_DIR="%~dp0results_runpod"

echo ---------------------------------------------------
echo  📥 Downloading Results from RunPod...
echo  Source: %RUNPOD_USER%@%RUNPOD_IP%:%RUNPOD_PORT%
echo ---------------------------------------------------

:: 로컬 결과 폴더 생성
if not exist %LOCAL_RESULTS_DIR% mkdir %LOCAL_RESULTS_DIR%

:: 3. scp를 사용하여 RunPod의 results 폴더를 로컬로 복사
scp -i "%USERPROFILE%\.ssh\id_ed25519" -r -P %RUNPOD_PORT% "%RUNPOD_USER%@%RUNPOD_IP%:%RUNPOD_PROJECT_DIR%/results/*" %LOCAL_RESULTS_DIR%/

echo.
echo ✅ Download Complete.
echo 다운로드 위치: %LOCAL_RESULTS_DIR%
echo.
pause
