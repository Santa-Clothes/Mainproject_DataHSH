#!/usr/bin/env python3
"""
Fashion JSON Encoder API Server
간단한 API 서버 시작 스크립트
"""

import os
import uvicorn
import sys
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def main():
    """FastAPI 서버 시작"""
    # 환경 변수로 설정 가능
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "8000"))
    reload = os.getenv("API_RELOAD", "true").lower() == "true"

    print("🚀 Fashion JSON Encoder API 서버 시작")
    print(f"📍 http://localhost:{port}")
    print(f"📖 API 문서: http://localhost:{port}/docs")
    print(f"⚙️  Reload: {reload}")

    uvicorn.run(
        "api.main:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info",
        access_log=True,
        timeout_keep_alive=30,
    )

if __name__ == "__main__":
    main()
