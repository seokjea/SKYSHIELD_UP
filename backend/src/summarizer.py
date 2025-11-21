import os
from dotenv import load_dotenv

# API Clients
try:
    from openai import OpenAI
except:
    OpenAI = None

try:
    from mistralai import Mistral
except:
    Mistral = None

try:
    from deepseek import DeepSeek
except:
    DeepSeek = None


load_dotenv()


class Summarizer:
    def __init__(self, backend="Google Gemini"):
        self.backend = backend
        self.client = self._get_client()

    # ----------------------------------------------------
    # API Client 자동 선택
    # ----------------------------------------------------
    def _get_client(self):
        if self.backend == "OpenAI" and OpenAI:
            key = os.getenv("OPENAI_API_KEY")
            if key:
                return OpenAI(api_key=key)

        if self.backend == "Mistral" and Mistral:
            key = os.getenv("MISTRAL_API_KEY")
            if key:
                return Mistral(api_key=key)

        if self.backend == "DeepSeek" and DeepSeek:
            key = os.getenv("DEEPSEEK_API_KEY")
            if key:
                return DeepSeek(api_key=key)

        # Google Gemini도 동일하게 추가 가능
        # API 없으면 로컬 fallback으로 자동 전환
        return None

    # ----------------------------------------------------
    # 1) 기본 의미 요약: 공격 여부를 사람이 읽기 쉽게
    # ----------------------------------------------------
    def summarize(self, text):

        # 1) LLM 사용 가능하면 고품질 요약
        if self.client:
            prompt = f"""
다음 텍스트의 **의도(intent)** 를 분석하여 한 문단으로 요약하세요.
- 공격/비공격 여부
- 어떤 방향의 행동을 요청하는지
- 우회/사회공학/취약점 악용 등 위험 요소가 있는지
- 사용자 의도에 대한 한줄 결론 포함

텍스트:
{text}
"""

            try:
                resp = self.client.chat.completions.create(
                    model="gpt-4o-mini" if self.backend=="OpenAI" else "mixtral-8x7b",
                    messages=[
                        {"role": "system", "content": "You are an intent analysis expert."},
                        {"role": "user", "content": prompt}
                    ]
                )
                return resp.choices[0].message["content"]

            except Exception as e:
                return f"[LLM 요약 실패 → 로컬 요약 사용]\n{text[:120]}..."

        # 2) API 없으면 로컬 fallback 요약
        return self._local_summary(text)

    # ----------------------------------------------------
    # 로컬 단순 의미 요약기
    # ----------------------------------------------------
    def _local_summary(self, text):
        # 간단한 “리뷰형 요약” 스타일
        lowered = text.lower()
        risk = []

        if any(x in lowered for x in ["hack", "exploit", "bypass", "attack", "jailbreak"]):
            risk.append("잠재적 공격 의도가 포함되어 있습니다.")

        if any(x in lowered for x in ["how to", "방법", "어떻게", "우회"]):
            risk.append("사용자가 특정 행동 수행 방법을 요청하고 있습니다.")

        if not risk:
            risk.append("일반 정보 요청 또는 안전한 프롬프트로 보입니다.")

        return " / ".join(risk)

    # ----------------------------------------------------
    # 2) 임베딩 기반 해설 (유사도 + 의도 이름 포함)
    # ----------------------------------------------------
    def summarize_with_embedding(self, text, cluster_name, similarity_score):

        if self.client:
            prompt = f"""
다음 텍스트가 '{cluster_name}' 의도(클러스터)와 {similarity_score:.3f} 유사도를 보였습니다.

- 입력이 어떤 위험 행동을 시도하는지
- 왜 그 클러스터와 유사하다고 판단되는지
- 보안적 관점에서의 해석
- 짧은 결론 1문장 포함

텍스트:
{text}
"""

            try:
                resp = self.client.chat.completions.create(
                    model="gpt-4o-mini" if self.backend=="OpenAI" else "mixtral-8x7b",
                    messages=[
                        {"role": "system", "content": "You analyze dangerous intentions."},
                        {"role": "user", "content": prompt}
                    ]
                )
                return resp.choices[0].message["content"]

            except:
                pass

        # fallback: 로컬 해석
        return f"'{cluster_name}' 의도와 {similarity_score:.3f} 유사. 입력 내용은 해당 의도 범주와 부분적으로 관련됩니다."

