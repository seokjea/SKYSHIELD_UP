import os
import math
from functools import lru_cache
from pathlib import Path
import json

import pandas as pd
import numpy as np
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

# === 공통 경로 ===
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
PRE_DIR = BASE_DIR / "precomputed"
VEC_DIR = PRE_DIR / "vectors"


# ------------------------------------------------------------
# 1) 안전한 파일명 변환
# ------------------------------------------------------------
def safe_name(s: str) -> str:
    return s.replace("/", "_").replace(" ", "_")


# ------------------------------------------------------------
# 2) OpenAI / Mistral / DeepSeek embedding client
# ------------------------------------------------------------
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
        df_safe = pd.read_csv(DATA_DIR / "jailbreak_safe2.csv")
        df_danger = pd.read_csv(DATA_DIR / "jailbreak_dangerous2.csv")
        data = pd.concat([df_safe, df_danger], ignore_index=True)
    else:
        df_base = pd.read_csv(DATA_DIR / "jailbreak_dataset.csv")
        df_custom = pd.read_csv(DATA_DIR / "jailbreak_customed.csv")
        data = pd.concat([df_base, df_custom], ignore_index=True)

    attacks = data[data["label"] == 1]["text"].tolist()
    normals = data[data["label"] == 0]["text"].tolist()

    client = get_embedding_client(model_name)
    embedder = Embedder(model_name, client=client)

    # 한 번만 전체 임베딩 계산 후 캐싱
    atk_vec = embedder.encode(attacks)
    norm_vec = embedder.encode(normals)

    return embedder, attacks, normals, atk_vec, norm_vec


# ------------------------------------------------------------
# 4) 전체 데이터 로드 (공격/정상 텍스트만)
# ------------------------------------------------------------
def load_dataset():
    df_base = pd.read_csv(DATA_DIR / "jailbreak_dataset.csv")
    df_custom = pd.read_csv(DATA_DIR / "jailbreak_customed.csv")
    data = pd.concat([df_base, df_custom], ignore_index=True)

    atk = data[data["label"] == 1]["text"].tolist()
    norm = data[data["label"] == 0]["text"].tolist()
    return atk, norm


# ------------------------------------------------------------
# 5) 메모리 터지지 않는 구조:
#    공격 벡터만 로드 (precompute_jailbreak.py 결과)
# ------------------------------------------------------------
def load_attack_data(embed_model: str):
    """
    precompute_jailbreak.py 에서 만든:
        /precomputed/vectors/attack_{model}.npy

    를 불러와서 (공격 텍스트 + 공격 벡터)를 반환한다.
    """
    atk_texts, _ = load_dataset()

    atk_path = VEC_DIR / f"attack_{safe_name(embed_model)}.npy"
    if not atk_path.exists():
        raise RuntimeError(f"공격 벡터 파일이 없습니다: {atk_path}")

    atk_vec = np.load(atk_path)  # float32, shape=(n_atk, dim)
    return atk_texts, atk_vec


# ------------------------------------------------------------
# 6) 정상 벡터는 memmap 으로 부분 로딩 (필요 시)
# ------------------------------------------------------------
def load_normal_memmap(embed_model: str):
    """
    optional: 정상 벡터 전체가 필요할 때만 로딩.
    메모리를 사용하지 않고 디스크에서 직접 접근 가능.
    """
    meta_path = VEC_DIR / f"normal_{safe_name(embed_model)}.meta.json"
    dat_path = VEC_DIR / f"normal_{safe_name(embed_model)}.dat"

    if not meta_path.exists() or not dat_path.exists():
        raise RuntimeError(
            f"정상 벡터 memmap 파일을 찾을 수 없습니다: {dat_path}"
        )

    meta = json.load(open(meta_path))
    n = meta["n_normals"]
    dim = meta["dim"]

    mem = np.memmap(dat_path, dtype="float32", mode="r", shape=(n, dim))
    return mem


# ------------------------------------------------------------
# 7) Adaptive Threshold (기존 Streamlit 부드러운 S-curve)
# ------------------------------------------------------------
def get_length_adaptive_threshold(base_thr: float, text: str) -> float:
    L = len(text)
    boost = 0.18 * (1 / (1 + math.exp(-0.03 * (L - 60))))
    return min(0.90, base_thr + boost)
