@echo off
setlocal

:: ==========================================
:: RunPod Upload Script
:: ==========================================

:: 1. RunPod 접속 정보 (본인의 환경에 맞게 수정하세요)
:: RunPod Connect 화면에서 보이는 IP와 포트 번호를 입력합니다.
set RUNPOD_IP=YOUR_RUNPOD_IP
set RUNPOD_PORT=YOUR_RUNPOD_PORT
set RUNPOD_USER=root

:: 2. 프로젝트 경로 설정
set LOCAL_PROJECT_DIR="%~dp0"
set RUNPOD_PROJECT_DIR="/workspace/Decision-Aware-Tail-Risk-Optimization"

echo ---------------------------------------------------
echo  🚀 Uploading Project to RunPod...
echo  Target: %RUNPOD_USER%@%RUNPOD_IP%:%RUNPOD_PORT%
echo ---------------------------------------------------

:: 3. RunPod에 프로젝트 폴더 생성 (없으면 생성)
ssh -i "%USERPROFILE%\.ssh\id_ed25519" -p %RUNPOD_PORT% %RUNPOD_USER%@%RUNPOD_IP% "mkdir -p %RUNPOD_PROJECT_DIR%"

:: 4. scp를 사용하여 로컬 파일들을 RunPod으로 전송
:: (주의: --exclude 옵션이 필요하다면 rsync를 사용하는 것이 좋습니다. 
:: 윈도우 10+에서는 tar를 이용해 압축 전송하는 방식이 가장 빠르고 안전합니다.)

echo [1/2] Compressing local files...
tar -cJf project_temp.tar.xz --exclude="data/cache" --exclude=".git" --exclude="__pycache__" --exclude=".ipynb_checkpoints" --exclude="results*/" *

echo [2/2] Transferring and extracting to RunPod...
scp -i "%USERPROFILE%\.ssh\id_ed25519" -P %RUNPOD_PORT% project_temp.tar.xz %RUNPOD_USER%@%RUNPOD_IP%:%RUNPOD_PROJECT_DIR%/
ssh -i "%USERPROFILE%\.ssh\id_ed25519" -p %RUNPOD_PORT% %RUNPOD_USER%@%RUNPOD_IP% "cd %RUNPOD_PROJECT_DIR% && tar -xf project_temp.tar.xz && rm project_temp.tar.xz"

:: 임시 파일 삭제
del project_temp.tar.xz

echo.
echo ✅ Upload Complete.
echo.
echo 접속 후 다음 명령어로 환경을 준비하세요:
echo 1. cd %RUNPOD_PROJECT_DIR%
echo 2. pip install cvxpy cvxpylayers arch pandas_datareader yfinance
echo 3. tmux new -s exp
echo 4. python run_compare.py --label phase7_p3
echo.
pause
