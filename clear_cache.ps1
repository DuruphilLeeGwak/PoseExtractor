# Python 캐시 완전 삭제 스크립트

Write-Host "🗑️  Python 캐시 삭제 중..." -ForegroundColor Cyan

# __pycache__ 폴더 삭제
Get-ChildItem -Path "." -Recurse -Directory -Filter "__pycache__" | ForEach-Object {
    Write-Host "   삭제: $($_.FullName)" -ForegroundColor Yellow
    Remove-Item -Path $_.FullName -Recurse -Force -ErrorAction SilentlyContinue
}

# .pyc 파일 삭제
Get-ChildItem -Path "." -Recurse -File -Filter "*.pyc" | ForEach-Object {
    Write-Host "   삭제: $($_.FullName)" -ForegroundColor Yellow
    Remove-Item -Path $_.FullName -Force -ErrorAction SilentlyContinue
}

# .pyo 파일 삭제
Get-ChildItem -Path "." -Recurse -File -Filter "*.pyo" | ForEach-Object {
    Write-Host "   삭제: $($_.FullName)" -ForegroundColor Yellow
    Remove-Item -Path $_.FullName -Force -ErrorAction SilentlyContinue
}

Write-Host "`n✅ 캐시 삭제 완료!" -ForegroundColor Green
Write-Host "이제 python PoseExtractor.py 를 다시 실행하세요.`n" -ForegroundColor Cyan
