from typing import Optional
from pathlib import Path
from functools import lru_cache
from pathlib import Path
import pickle
 
from fastapi import FastAPI
from functools import lru_cache
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from dotenv import load_dotenv

from src.utils import (
    load_attack_data,
    get_length_adaptive_threshold,
    get_embedding_client,
    safe_name,
)
from src.embedding import Embedder
from src.summarizer import Summarizer
from src.detector import SkyShield
from src.cluster_analyzer import ClusterAnalyzer

# .env 로드
load_dotenv()

app = FastAPI(
    title="SkyShield Backend",
    version="1.0.0",
    description="Adaptive LLM Jailbreak Detection & Prompt Risk Analysis",
)

# CORS 설정 (개발용: 전부 허용)
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------------
# 캐시 헬퍼들
# --------------------------------------------------------
@lru_cache(maxsize=8)
def get_embedder(model_name: str) -> Embedder:
    """
    임베딩 모델별 Embedder 인스턴스 캐시.
    OpenAI / Mistral / DeepSeek / sentence-transformers 모두 여기로 통일.
    """
    client = get_embedding_client(model_name)
    return Embedder(model_name, client=client)


@lru_cache(maxsize=8)
def get_attack_dataset(model_name: str):
    """
    precompute_jailbreak.py 에서 만든
    precomputed/vectors/attack_{model}.npy 를 로드.
    """
    atk_texts, atk_vec = load_attack_data(model_name)
    return atk_texts, atk_vec


@lru_cache(maxsize=16)
def get_precomputed_analyzer(embed_model: str, summ_model: str) -> ClusterAnalyzer:
    """
    precompute_jailbreak.py 에서 만든
    precomputed/cluster_{embed_model}_{summ_model}.pkl 로부터
    ClusterAnalyzer 인스턴스를 로드.
    """
    base_dir = Path(__file__).resolve().parent
    pre_dir = base_dir / "precomputed"

    fname = f"cluster_{safe_name(embed_model)}_{safe_name(summ_model)}.pkl"
    path = pre_dir / fname

    if not path.exists():
        raise RuntimeError(
            f"사전 계산된 클러스터 파일을 찾을 수 없습니다: {path}\n"
            f"backend 디렉터리에서 precompute_jailbreak.py 를 먼저 실행하세요."
        )

    with open(path, "rb") as f:
        analyzer: ClusterAnalyzer = pickle.load(f)
    return analyzer


# --------------------------------------------------------
# Pydantic 모델
# --------------------------------------------------------
class AnalysisRequest(BaseModel):
    text: str
    embed_model: str          # 예: "OpenAI Embedding"
    summ_model: str           # 예: "OpenAI"
    base_threshold: float     # UI에서 설정하는 Base Threshold
    sensitivity: float        # 0.0 ~ 1.0 민감도 슬라이더 값


class AnalysisResponse(BaseModel):
    final_decision: str       # "ALLOW" / "REVIEW" / "BLOCK"
    summary: str
    adaptive_thr: float
    base_threshold: float

    decision_basic: str
    score_basic: float

    cluster_decision: str
    cluster_id: int | None = None
    cluster_sim: float

    novel_thr: float
    susp_thr: float

    cluster_name: str | None = None


# --------------------------------------------------------
# 헬스체크
# --------------------------------------------------------
@app.get("/health")
def health():
    return {"status": "ok"}


# --------------------------------------------------------
# 분석 엔드포인트
# --------------------------------------------------------
@app.post("/analyze", response_model=AnalysisResponse)
def analyze(req: AnalysisRequest):
    """
    전체 흐름:
    1) 사전 계산된 공격 벡터 + 클러스터 분석기 로드
    2) 사용자 입력 1개만 임베딩
    3) 길이 기반 Adaptive Threshold 계산
    4) SkyShield 기본 유사도 검사
    5) HDBSCAN 클러스터 기반 패턴 분석
    6) 최종 판단(ALLOW / REVIEW / BLOCK) 및 메타 정보 반환
    """

    # 1) Embedder + 공격 데이터 로드 (전부 캐시)
    embedder = get_embedder(req.embed_model)
    atk_texts, atk_vec = get_attack_dataset(req.embed_model)

    # 2) 사용자 입력 임베딩
    user_vec = embedder.encode([req.text])[0]

    # 3) Adaptive Threshold 계산
    adaptive_thr = get_length_adaptive_threshold(req.base_threshold, req.text)

    # 4) Summarizer (사용자 입력 요약만 수행)
    summarizer = Summarizer(req.summ_model)
    summary = summarizer.summarize(req.text)

    # 5) SkyShield 기본 유사도 검사
    base_detector = SkyShield(attack_vectors=atk_vec, threshold_block=adaptive_thr)
    decision_basic, score_basic = base_detector.predict(user_vec)

    # 6) HDBSCAN 기반 클러스터 분석 (사전 계산된 analyzer 사용)
    analyzer = get_precomputed_analyzer(req.embed_model, req.summ_model)
    cluster_decision, cluster_id, cluster_sim = analyzer.detect(user_vec)

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
    if cluster_id is not None and hasattr(analyzer, "cluster_names"):
        names = analyzer.cluster_names
        if isinstance(names, dict) and cluster_id in names:
            cluster_name = names[cluster_id]

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
