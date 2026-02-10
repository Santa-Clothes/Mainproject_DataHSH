#!/usr/bin/env python3
"""
Fashion JSON Encoder API Server
간단한 API 서버 시작 스크립트
"""

import uvicorn
import sys
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def main():
    """FastAPI 서버 시작"""
    print("🚀 Fashion JSON Encoder API 서버 시작")
    print("📍 http://localhost:8000")
    print("📖 API 문서: http://localhost:8000/docs")
    
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )

if __name__ == "__main__":
    main()
