import os, pickle, hashlib, numpy as np
from sentence_transformers import SentenceTransformer


class Embedder:
    """
    통합 임베딩 엔진:
    - SentenceTransformer (로컬)
    - OpenAI Embedding API
    - Mistral Embedding API
    - DeepSeek Embedding API
    """

    def __init__(self, model_name, client=None, batch_size=128):
        self.model_name = model_name
        self.client = client  # API 클라이언트 (OpenAI / Mistral / DeepSeek)
        self.batch_size = batch_size
        self.input_field = "input"  # 기본값
        self.model = None           # 로컬 모델 or API 모델 문자열

        # ==============================
        # SentenceTransformer (local)
        # ==============================
        if model_name.startswith("sentence-transformers"):
            self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

        elif model_name.startswith("intfloat"):
            self.model = SentenceTransformer("intfloat/multilingual-e5-large")

        elif model_name.lower().startswith("thenlper") or model_name.lower().startswith("gte"):
            self.model = SentenceTransformer("thenlper/gte-large")

        elif model_name.startswith("BAAI"):
            self.model = SentenceTransformer("BAAI/bge-m3")

        # ==============================
        # OpenAI / Mistral / DeepSeek
        # ==============================
        elif "OpenAI" in model_name:
            self.model = "text-embedding-3-large"
            self.input_field = "input"

        elif "Mistral" in model_name:
            self.model = "mistral-embed"
            self.input_field = "inputs"

        elif "DeepSeek" in model_name:
            self.model = "deepseek-embedding"
            self.input_field = "input"

        else:
            raise ValueError(f"Unsupported embedding backend: {model_name}")


    # ----------------------------------------------------------
    # Array chunking
    # ----------------------------------------------------------
    def chunk(self, arr, size):
        for i in range(0, len(arr), size):
            yield arr[i:i + size]


    # ----------------------------------------------------------
    # Cache path
    # ----------------------------------------------------------
    def _cache_path(self, texts):
        key = hashlib.sha256("|||".join(texts).encode()).hexdigest()[:32]
        safe_name = self.model_name.replace("/", "_").replace(" ", "_")
        return f"cache_embed_{safe_name}_{key}.pkl"


    # ----------------------------------------------------------
    # Encoding
    # ----------------------------------------------------------
    def encode(self, texts, use_cache=True, save_cache=True):
        """문장 임베딩 생성 (로컬 모델 또는 API 기반)"""

        # 캐시 로드
        cache_file = self._cache_path(texts)
        if use_cache and os.path.exists(cache_file):
            return pickle.load(open(cache_file, "rb"))

        # ==============================
        # Local transformer encoding
        # ==============================
        if isinstance(self.model, SentenceTransformer):
            vecs = self.model.encode(texts)

        # ==============================
        # API-based embedding
        # ==============================
        else:
            if self.client is None:
                raise RuntimeError(
                    f"[Embedder ERROR] '{self.model_name}'은 API 기반 모델이지만 "
                    f"client=None 상태입니다. app.py에서 client를 전달해야 합니다."
                )

            vecs = []
            for batch in self.chunk(texts, self.batch_size):

                # API 응답
                try:
                    resp = self.client.embeddings.create(
                        model=self.model,
                        **{self.input_field: batch}
                    )
                except Exception as e:
                    raise RuntimeError(f"[Embedder API Error] {str(e)}")

                # 데이터 파싱
                try:
                    vecs.extend([d.embedding for d in resp.data])
                except Exception as e:
                    raise RuntimeError(
                        f"[Embedder Parse Error] API 응답 구조가 예상과 다릅니다: {resp}\n{e}"
                    )

            vecs = np.array(vecs)

        # 캐싱 저장
        if save_cache:
            pickle.dump(vecs, open(cache_file, "wb"))

        return vecs
