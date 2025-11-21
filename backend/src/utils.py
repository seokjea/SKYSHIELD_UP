import os
import math
from functools import lru_cache
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from openai import OpenAI
from mistralai import Mistral

try:
    from deepseek import DeepSeek
except ImportError:
    DeepSeek = None

from .embedding import Embedder

# .env 로부터 API 키 로드
load_dotenv()


def get_embedding_client(model_name: str):
    """
    임베딩 모델 이름에 따라 OpenAI / Mistral / DeepSeek 클라이언트를 자동 로드.
    로컬 sentence-transformers 계열은 None 반환(Embedder 내부에서 처리).
    """
    if "OpenAI" in model_name:
        key = os.getenv("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("OPENAI_API_KEY 가 설정되어 있지 않습니다.")
        return OpenAI(api_key=key)

    if "Mistral" in model_name:
        key = os.getenv("MISTRAL_API_KEY")
        if not key:
            raise RuntimeError("MISTRAL_API_KEY 가 설정되어 있지 않습니다.")
        return Mistral(api_key=key)

    if "DeepSeek" in model_name:
        key = os.getenv("DEEPSEEK_API_KEY")
        if not key:
            raise RuntimeError("DEEPSEEK_API_KEY 가 설정되어 있지 않습니다.")
        if DeepSeek is None:
            raise RuntimeError("deepseek 패키지가 설치되어 있지 않습니다.")
        return DeepSeek(api_key=key)

    # sentence-transformers 등 로컬 모델인 경우
    return None


@lru_cache(maxsize=8)
def load_all(model_name: str):
    """
    임베딩 모델 이름을 입력으로 받아:
    - 데이터셋 로드
    - 공격/정상 텍스트 리스트 분리
    - Embedder 생성
    - 미리 전체 벡터 임베딩 계산

    반환:
        embedder, attack_texts, normal_texts, attack_vectors, normal_vectors
    """
    base_dir = Path(__file__).resolve().parent.parent
    data_dir = base_dir / "data"

    robust_models = [
        "intfloat/multilingual-e5-large (robust)",
        "thenlper/gte-large (robust)",
        "BAAI/bge-m3 (robust)",
    ]

    if model_name in robust_models:
        df_safe = pd.read_csv(data_dir / "jailbreak_safe2.csv")
        df_danger = pd.read_csv(data_dir / "jailbreak_dangerous2.csv")
        data = pd.concat([df_safe, df_danger], ignore_index=True)
    else:
        df_base = pd.read_csv(data_dir / "jailbreak_dataset.csv")
        df_custom = pd.read_csv(data_dir / "jailbreak_customed.csv")
        data = pd.concat([df_base, df_custom], ignore_index=True)

    attacks = data[data["label"] == 1]["text"].tolist()
    normals = data[data["label"] == 0]["text"].tolist()

    client = get_embedding_client(model_name)
    embedder = Embedder(model_name, client=client)

    # 한 번만 전체 임베딩 계산 후 캐싱
    atk_vec = embedder.encode(attacks)
    norm_vec = embedder.encode(normals)

    return embedder, attacks, normals, atk_vec, norm_vec


def get_length_adaptive_threshold(base_thr: float, text: str) -> float:
    """
    입력 텍스트 길이에 따라 Block 임계값을 부드럽게 조정하는 함수.
    Streamlit 버전에서 가져온 S-curve boosting 로직.
    """
    L = len(text)
    boost = 0.18 * (1 / (1 + math.exp(-0.03 * (L - 60))))
    return min(0.90, base_thr + boost)
