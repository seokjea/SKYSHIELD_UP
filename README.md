# SKYSHIELD_up

LLM 프롬프트에 대한 공격 탐지와 이상 행위를 판정하는 웹 애플리케이션입니다.  
백엔드(FastAPI)와 프론트엔드(React + Vite)로 분리되어 있습니다.

---

## 1. 구성 요약

디렉터리 구조

- backend: Python FastAPI 서버, 분석 로직
- frontend: React + Vite 기반 웹 UI

동작 흐름

1. 사용자가 웹 페이지에서 문장을 입력하고 옵션을 선택한다.
2. 프론트엔드가 backend의 /analyze API로 요청을 보낸다.
3. 백엔드가 임베딩, Adaptive threshold, SkyShield, HDBSCAN 분석을 수행하고 결과를 JSON으로 반환한다.
4. 프론트엔드에서 최종 판정과 상세 정보를 화면에 표시한다.

---

## 2. 사전 요구사항

공통

- WSL 또는 Linux 환경 (이미 사용 중이라고 가정)

백엔드

- Python 3.10 이상

프론트엔드

- Node.js 20.19 이상 또는 22.12 이상 권장
- nvm으로 Node 버전 관리 추천

---

## 3. 백엔드 설정 및 실행

### 3.1. 가상환경 생성 및 패키지 설치

```bash
cd ~/SKYSHIELD_up/backend

# 가상환경 생성 (예시 이름: .venv)
python -m venv .venv
source .venv/bin/activate

# 의존성 설치
pip install -r requirements.txt