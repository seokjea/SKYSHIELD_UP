from typing import Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from dotenv import load_dotenv

from src.utils import load_all, get_length_adaptive_threshold
from src.summarizer import Summarizer
from src.detector import SkyShield
from src.cluster_analyzer import ClusterAnalyzer

# .env 로드
load_dotenv()

app = FastAPI(
    title="SKYSHIELD Backend",
    description="Adaptive LLM Security Defense System - FastAPI backend",
    version="0.1.0",
)

# CORS 설정 (React dev 서버에서 접근 가능하게)
origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AnalysisRequest(BaseModel):
    text: str
    embed_model: str
    summ_model: str
    base_threshold: float
    sensitivity: float


class AnalysisResponse(BaseModel):
    final_decision: str
    summary: str
    adaptive_thr: float
    base_threshold: float

    decision_basic: str
    score_basic: float

    cluster_decision: str
    cluster_id: Optional[int] = None
    cluster_sim: float

    novel_thr: float
    susp_thr: float

    cluster_name: Optional[str] = None


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/analyze", response_model=AnalysisResponse)
def analyze(req: AnalysisRequest):
    """
    React 프론트에서 호출하는 분석 엔드포인트.
    Streamlit 코드의 '분석 🚀' 버튼 부분을 백엔드로 옮긴 버전.
    """
    # 1) 데이터/임베딩 로딩 (load_all은 lru_cache로 캐싱됨)
    embedder, atk_texts, norm_texts, atk_vec, norm_vec = load_all(req.embed_model)

    # 2) 사용자 입력 임베딩
    user_vec = embedder.encode([req.text])[0]

    # 3) Adaptive Threshold 계산
    adaptive_thr = get_length_adaptive_threshold(req.base_threshold, req.text)

    # 4) Summarizer
    summarizer = Summarizer(req.summ_model)
    summary = summarizer.summarize(req.text)

    # 5) SkyShield 기본 유사도 검사
    base_detector = SkyShield(attack_vectors=atk_vec, threshold_block=adaptive_thr)
    decision_basic, score_basic = base_detector.predict(user_vec)

    # 6) HDBSCAN 기반 클러스터 분석
    analyzer = ClusterAnalyzer(summarizer=summarizer)
    analyzer.fit(atk_vec)
    analyzer.generate_cluster_names(atk_texts, summarizer)

    # 민감도 기반 Novel / Suspicious 기준
    novel_thr = 0.05 + 0.15 * req.sensitivity
    susp_thr = novel_thr + (0.20 + 0.20 * req.sensitivity)

    cluster_decision, cluster_id, cluster_sim = analyzer.detect(user_vec)

    # 민감도 기준에 따라 판정 재조정
    if cluster_sim < novel_thr:
        cluster_decision = "NOVEL_ATTACK"
    elif cluster_sim < susp_thr:
        cluster_decision = "SUSPICIOUS"
    else:
        cluster_decision = "KNOWN_ATTACK"

    # 최종 Block/Review/Allow 결정
    if cluster_decision == "NOVEL_ATTACK":
        final_decision = "BLOCK"
    elif cluster_decision == "SUSPICIOUS":
        final_decision = "REVIEW"
    else:
        final_decision = decision_basic

    # 클러스터 의미 태그
    cluster_name = None
    if cluster_id is not None and cluster_id in getattr(analyzer, "cluster_names", {}):
        cluster_name = analyzer.cluster_names[cluster_id]

    return AnalysisResponse(
        final_decision=final_decision,
        summary=summary,
        adaptive_thr=float(adaptive_thr),
        base_threshold=float(req.base_threshold),
        decision_basic=decision_basic,
        score_basic=float(score_basic),
        cluster_decision=cluster_decision,
        cluster_id=int(cluster_id) if cluster_id is not None else None,
        cluster_sim=float(cluster_sim),
        novel_thr=float(novel_thr),
        susp_thr=float(susp_thr),
        cluster_name=cluster_name,
    )
