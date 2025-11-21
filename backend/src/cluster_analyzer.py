import numpy as np
import hdbscan
from sklearn.metrics.pairwise import cosine_similarity


class ClusterAnalyzer:

    def __init__(self, summarizer=None, min_cluster_size=20, min_samples=5):
        self.summarizer = summarizer

        # HDBSCAN 하이퍼파라미터
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples

        # 모델 변수
        self.clusterer = None
        self.labels = None
        self.probabilities = None
        self.cluster_centers = None
        self.cluster_names = None

    # -----------------------------------------------------
    # HDBSCAN 학습
    # -----------------------------------------------------
    def fit(self, attack_vecs):
        self.clusterer = hdbscan.HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            min_samples=self.min_samples,
            metric="euclidean",
            cluster_selection_method='eom'
        ).fit(attack_vecs)

        self.labels = self.clusterer.labels_                # 클러스터 번호 (-1 포함)
        self.probabilities = self.clusterer.probabilities_  # membership confidence

        # 클러스터 중심 계산 (평균값 기반)
        self.cluster_centers = self._compute_centers(attack_vecs)
        return self

    # -----------------------------------------------------
    # 클러스터 중심 계산
    # -----------------------------------------------------
    def _compute_centers(self, attack_vecs):
        unique = set(self.labels)
        unique.discard(-1)  # noise 제외

        centers = {}
        for cid in unique:
            members = attack_vecs[self.labels == cid]
            centers[cid] = np.mean(members, axis=0)

        return centers

    # -----------------------------------------------------
    # HDBSCAN 기반 anomaly detection
    # -----------------------------------------------------
    def detect(self, user_vec):
        max_sim = -1
        best_cluster = None

        for cid, center in self.cluster_centers.items():
            sim = float(cosine_similarity([user_vec], [center])[0][0])
            if sim > max_sim:
                max_sim = sim
                best_cluster = cid

        # ============================
        # 완화된(less aggressive) 기준
        # ============================

        # 매우 낮은 유사도만 Novel
        if max_sim < 0.10:
            return "NOVEL_ATTACK", best_cluster, max_sim

        # 중간 유사도 → Suspicious
        if max_sim < 0.30:
            return "SUSPICIOUS", best_cluster, max_sim

        # 그 이상은 Known Attack
        return "KNOWN_ATTACK", best_cluster, max_sim

    # -----------------------------------------------------
    # 클러스터 의미 자동 라벨링
    # -----------------------------------------------------
    def generate_cluster_names(self, attack_texts, summarizer):
        names = {}
        unique = set(self.labels)
        unique.discard(-1)

        for cid in unique:
            samples = [
                attack_texts[i]
                for i in range(len(attack_texts))
                if self.labels[i] == cid
            ]
            if len(samples) == 0:
                names[cid] = "기타"
            else:
                names[cid] = self._label_cluster(samples, summarizer)

        # Noise 클러스터는 별도 표기
        names[-1] = "Novel Attack Noise Cluster"
        self.cluster_names = names
        return names

    def _label_cluster(self, samples, summarizer):
        joined = "\n".join(samples[:20])
        prompt = f"""
아래 문장들은 같은 공격 유형의 클러스터입니다.
공통된 공격 의도를 한국어 '2~3 단어 태그'로 요약하세요.

조건:
- 문장 형태 X
- 설명 X
- 태그 하나만
예: 지침 우회, 민감정보 탈취, 모델 조작 등

텍스트:
{joined}
"""
        label = summarizer.summarize(prompt)
        label = label.strip().split("\n")[0]
        return label.replace(":", "").replace("-", "").strip()
