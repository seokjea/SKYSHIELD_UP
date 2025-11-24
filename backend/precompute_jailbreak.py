"""
17만개 전체 텍스트를 OpenAI Embedding으로 chunk 단위 임베딩하고,
공격 벡터 + HDBSCAN 클러스터를 미리 계산해 두는 스크립트.

사용법 (backend 디렉터리에서):

    (SKY_venv) $ python precompute_jailbreak.py \
        --embed-model "OpenAI Embedding" \
        --summ-model "OpenAI"

전제:
- backend/.env 에 OPENAI_API_KEY 가 설정되어 있어야 함.
- data/jailbreak_dataset.csv, data/jailbreak_customed.csv 가 존재해야 함.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from src.embedding import Embedder
from src.cluster_analyzer import ClusterAnalyzer
from src.summarizer import Summarizer
from src.utils import get_embedding_client

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
PRE_DIR = BASE_DIR / "precomputed"
VEC_DIR = PRE_DIR / "vectors"


def safe_name(s: str) -> str:
    return s.replace("/", "_").replace(" ", "_")


def load_dataset():
    """jailbreak_dataset + customed 를 합쳐서 공격/정상 텍스트 분리."""
    df_main = pd.read_csv(DATA_DIR / "jailbreak_dataset.csv")
    df_custom = pd.read_csv(DATA_DIR / "jailbreak_customed.csv")
    data = pd.concat([df_main, df_custom], ignore_index=True)

    atk_mask = data["label"] == 1
    atk_texts = data.loc[atk_mask, "text"].tolist()
    norm_texts = data.loc[~atk_mask, "text"].tolist()
    return atk_texts, norm_texts


def precompute_embeddings(embed_model: str, batch_size: int = 128):
    """
    전체 텍스트(공격 + 정상)를 chunk 단위로 임베딩해서 디스크에 저장.

    - 공격 벡터: npy (작음)
    - 정상 벡터: memmap(.dat) + meta.json
    """
    VEC_DIR.mkdir(parents=True, exist_ok=True)

    atk_texts, norm_texts = load_dataset()
    print(f"[1/3] 데이터 로드 완료")
    print(f"  - 공격 텍스트 개수: {len(atk_texts)}")
    print(f"  - 정상 텍스트 개수: {len(norm_texts)}")

    client = get_embedding_client(embed_model)
    embedder = Embedder(embed_model, client=client, batch_size=batch_size)

    # 1) 공격 벡터는 한 번에 임베딩 (개수가 1,000 정도라 메모리 여유)
    print("[1-1] 공격 텍스트 임베딩 중...")
    atk_vec = embedder.encode(atk_texts).astype("float32")
    atk_path = VEC_DIR / f"attack_{safe_name(embed_model)}.npy"
    np.save(atk_path, atk_vec)
    print(f"  - 공격 벡터 저장: {atk_path} (shape={atk_vec.shape})")

    # 2) 정상 텍스트는 memmap으로 chunk 임베딩
    print("[1-2] 정상 텍스트 임베딩 (chunk + memmap) 중...")
    n_norm = len(norm_texts)
    if n_norm == 0:
        print("  - 정상 텍스트가 없습니다. 건너뜀.")
        return atk_path, atk_texts

    # 첫 batch 임베딩해서 차원 확인
    first_batch = norm_texts[:batch_size]
    first_vecs = embedder.encode(first_batch).astype("float32")
    dim = first_vecs.shape[1]

    norm_path = VEC_DIR / f"normal_{safe_name(embed_model)}.dat"
    norm_mem = np.memmap(norm_path, dtype="float32", mode="w+", shape=(n_norm, dim))

    # 첫 batch 기록
    norm_mem[0:first_vecs.shape[0], :] = first_vecs
    idx = first_vecs.shape[0]
    print(f"  - 첫 batch 완료: {idx}/{n_norm}")

    # 나머지 batch
    while idx < n_norm:
        batch_texts = norm_texts[idx: idx + batch_size]
        vecs = embedder.encode(batch_texts).astype("float32")
        end = idx + vecs.shape[0]
        norm_mem[idx:end, :] = vecs
        idx = end
        print(f"  - 진행 중: {idx}/{n_norm}")

    norm_mem.flush()
    del norm_mem

    # meta 정보 저장
    meta = {"n_normals": int(n_norm), "dim": int(dim)}
    meta_path = VEC_DIR / f"normal_{safe_name(embed_model)}.meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"  - 정상 벡터 memmap 저장: {norm_path} (n={n_norm}, dim={dim})")
    print(f"  - meta 저장: {meta_path}")

    return atk_path, atk_texts


def precompute_clusters(embed_model: str, summ_model: str, atk_vec_path: Path, atk_texts):
    """
    공격 벡터 + 텍스트를 이용해 HDBSCAN 클러스터링 + 클러스터 이름 생성 후 pkl에 저장.
    """
    PRE_DIR.mkdir(parents=True, exist_ok=True)

    print("[2/3] 공격 벡터 로드 중...")
    atk_vec = np.load(atk_vec_path)  # float32, shape=(n_atk, dim)
    print(f"  - atk_vec shape: {atk_vec.shape}")

    print(f"[2/3] 클러스터링 + 클러스터 이름 생성 중... (summ_model={summ_model})")
    summarizer = Summarizer(summ_model)
    analyzer = ClusterAnalyzer(summarizer=summarizer)
    analyzer.fit(atk_vec)
    analyzer.generate_cluster_names(atk_texts, summarizer)

    fname = f"cluster_{safe_name(embed_model)}_{safe_name(summ_model)}.pkl"
    out_path = PRE_DIR / fname

    print(f"[3/3] 클러스터 분석기 저장 중... -> {out_path}")
    import pickle

    # 🔥 Summarizer(OpenAI 클라이언트)는 pickle 불가 + 런타임에서 필요 없으므로 제거
    analyzer.summarizer = None

    with open(out_path, "wb") as f:
        pickle.dump(analyzer, f)

    print("완료! 이제 FastAPI에서 사전 계산된 클러스터를 재사용할 수 있습니다.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embed-model", type=str, default="OpenAI Embedding")
    parser.add_argument("--summ-model", type=str, default="OpenAI")
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()

    embed_model = args.embed_model
    summ_model = args.summ_model

    print(f"[0] embed_model={embed_model}, summ_model={summ_model}")
    atk_vec_path, atk_texts = precompute_embeddings(
        embed_model,
        batch_size=args.batch_size,
    )
    precompute_clusters(embed_model, summ_model, atk_vec_path, atk_texts)


if __name__ == "__main__":
    main()
