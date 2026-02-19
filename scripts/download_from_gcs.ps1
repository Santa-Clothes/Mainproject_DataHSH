# Google Cloud Storage에서 데이터 다운로드
# ================================================

param(
    [Parameter(Mandatory=$true)]
    [string]$BucketName,

    [string]$ProjectRoot = "c:\FinalProject_v2"
)

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Google Cloud Storage 데이터 다운로드" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# gcloud CLI 확인
$gcloudInstalled = Get-Command gcloud -ErrorAction SilentlyContinue

if (-not $gcloudInstalled) {
    Write-Host "[ERROR] gcloud CLI가 설치되어 있지 않습니다." -ForegroundColor Red
    Write-Host ""
    Write-Host "설치 방법:" -ForegroundColor Yellow
    Write-Host "1. https://cloud.google.com/sdk/docs/install 에서 다운로드" -ForegroundColor Yellow
    Write-Host "2. 설치 후 PowerShell 재시작" -ForegroundColor Yellow
    Write-Host "3. 'gcloud init' 실행" -ForegroundColor Yellow
    exit 1
}

Write-Host "[OK] gcloud CLI 설치됨" -ForegroundColor Green
Write-Host ""

# 인증 확인
Write-Host "[1] Google Cloud 인증 확인..." -ForegroundColor Cyan
$accountInfo = gcloud auth list --filter=status:ACTIVE --format="value(account)" 2>$null

if (-not $accountInfo) {
    Write-Host "  [INFO] 로그인 필요" -ForegroundColor Yellow
    Write-Host "  브라우저가 열립니다..." -ForegroundColor Yellow
    gcloud auth login
}

Write-Host "  [OK] 로그인됨: $accountInfo" -ForegroundColor Green
Write-Host ""

# 버킷 존재 확인
Write-Host "[2] 버킷 존재 확인: gs://$BucketName" -ForegroundColor Cyan
$bucketExists = gsutil ls gs://$BucketName 2>$null

if (-not $bucketExists) {
    Write-Host "  [ERROR] 버킷을 찾을 수 없습니다: $BucketName" -ForegroundColor Red
    Write-Host ""
    Write-Host "  사용 가능한 버킷 목록:" -ForegroundColor Yellow
    gsutil ls
    exit 1
}

Write-Host "  [OK] 버킷 존재 확인" -ForegroundColor Green
Write-Host ""

# 버킷 내용 확인
Write-Host "[3] 버킷 내용 확인..." -ForegroundColor Cyan
gsutil ls -r gs://$BucketName/**
Write-Host ""

# 다운로드 대상 확인
Write-Host "[4] 다운로드 대상 확인..." -ForegroundColor Cyan
Write-Host "  찾을 파일:" -ForegroundColor Yellow
Write-Host "    - FAISS 인덱스: *.index, *.ids.npy" -ForegroundColor Yellow
Write-Host "    - CSV 데이터: *.csv" -ForegroundColor Yellow
Write-Host "    - 체크포인트: *.pt, *.pth" -ForegroundColor Yellow
Write-Host ""

# 사용자 확인
$confirm = Read-Host "다운로드를 진행하시겠습니까? (y/n)"
if ($confirm -ne "y") {
    Write-Host "취소되었습니다." -ForegroundColor Yellow
    exit 0
}

Write-Host ""
Write-Host "[5] 다운로드 시작..." -ForegroundColor Cyan

# 디렉토리 생성
$dataDir = "$ProjectRoot\data"
$indexDir = "$dataDir\indexes"
$csvDir = "$dataDir\csv"
$checkpointDir = "$ProjectRoot\checkpoints"

New-Item -ItemType Directory -Force -Path $indexDir | Out-Null
New-Item -ItemType Directory -Force -Path $csvDir | Out-Null
New-Item -ItemType Directory -Force -Path $checkpointDir | Out-Null

# FAISS 인덱스 다운로드
Write-Host "  [5-1] FAISS 인덱스 다운로드..." -ForegroundColor Cyan
gsutil -m cp -r "gs://$BucketName/data/indexes/*.index" "$indexDir\" 2>$null
gsutil -m cp -r "gs://$BucketName/data/indexes/*.npy" "$indexDir\" 2>$null
gsutil -m cp -r "gs://$BucketName/indexes/*.index" "$indexDir\" 2>$null
gsutil -m cp -r "gs://$BucketName/indexes/*.npy" "$indexDir\" 2>$null

# CSV 데이터 다운로드
Write-Host "  [5-2] CSV 데이터 다운로드..." -ForegroundColor Cyan
gsutil -m cp -r "gs://$BucketName/data/csv/*.csv" "$csvDir\" 2>$null
gsutil -m cp -r "gs://$BucketName/csv/*.csv" "$csvDir\" 2>$null

# 체크포인트 다운로드
Write-Host "  [5-3] 체크포인트 다운로드..." -ForegroundColor Cyan
gsutil -m cp -r "gs://$BucketName/checkpoints/*.pt" "$checkpointDir\" 2>$null
gsutil -m cp -r "gs://$BucketName/checkpoints/*.pth" "$checkpointDir\" 2>$null

Write-Host ""
Write-Host "[6] 다운로드 완료!" -ForegroundColor Green
Write-Host ""

# 결과 확인
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "다운로드 결과" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

Write-Host "FAISS 인덱스:" -ForegroundColor Yellow
Get-ChildItem "$indexDir\*" -Include *.index,*.npy | ForEach-Object {
    $size = [math]::Round($_.Length / 1MB, 2)
    Write-Host "  ✓ $($_.Name) ($size MB)" -ForegroundColor Green
}

Write-Host ""
Write-Host "CSV 데이터:" -ForegroundColor Yellow
Get-ChildItem "$csvDir\*.csv" -ErrorAction SilentlyContinue | ForEach-Object {
    $size = [math]::Round($_.Length / 1MB, 2)
    Write-Host "  ✓ $($_.Name) ($size MB)" -ForegroundColor Green
}

Write-Host ""
Write-Host "체크포인트:" -ForegroundColor Yellow
Get-ChildItem "$checkpointDir\*" -Include *.pt,*.pth -ErrorAction SilentlyContinue | ForEach-Object {
    $size = [math]::Round($_.Length / 1MB, 2)
    Write-Host "  ✓ $($_.Name) ($size MB)" -ForegroundColor Green
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "다음 단계:" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "1. .env 파일 확인" -ForegroundColor Yellow
Write-Host "2. API 재시작: docker-compose restart" -ForegroundColor Yellow
Write-Host "3. 테스트: curl http://localhost:8001/health" -ForegroundColor Yellow
Write-Host ""
