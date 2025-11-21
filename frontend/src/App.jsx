import React, { useState } from "react";

const EMBED_MODELS = [
  "sentence-transformers (기본)",
  "intfloat/multilingual-e5-large (robust)",
  "thenlper/gte-large (robust)",
  "BAAI/bge-m3 (robust)",
  "OpenAI Embedding",
  "Mistral Embedding",
  "DeepSeek Embedding",
];

const SUMM_MODELS = ["Google Gemini", "OpenAI", "DeepSeek", "Mistral"];

const COLOR_MAP = {
  BLOCK: "#e11d48", // red
  REVIEW: "#f97316", // orange
  ALLOW: "#22c55e", // green
  SUSPICIOUS: "#f97316",
  NOVEL_ATTACK: "#e11d48",
};

function App() {
  const [embedOpt, setEmbedOpt] = useState(EMBED_MODELS[0]);
  const [summOpt, setSummOpt] = useState(SUMM_MODELS[0]);
  const [baseThreshold, setBaseThreshold] = useState(0.4);
  const [sensitivity, setSensitivity] = useState(0.4);
  const [userInput, setUserInput] = useState("");

  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");

  const handleAnalyze = async () => {
    if (!userInput.trim()) {
      setError("분석할 문장을 입력해주세요.");
      setResult(null);
      return;
    }
    setError("");
    setLoading(true);
    setResult(null);

    try {
      // 백엔드 FastAPI 서버가 http://localhost:8000 에 떠 있다고 가정
      const res = await fetch("http://localhost:8000/analyze", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          text: userInput,
          embed_model: embedOpt,
          summ_model: summOpt,
          base_threshold: baseThreshold,
          sensitivity: sensitivity,
        }),
      });

      if (!res.ok) {
        throw new Error(`서버 오류: ${res.status}`);
      }

      const data = await res.json();
      setResult(data);
    } catch (e) {
      setError(e.message || "분석 중 오류가 발생했습니다.");
    } finally {
      setLoading(false);
    }
  };

  const decisionColor =
    result && COLOR_MAP[result.final_decision]
      ? COLOR_MAP[result.final_decision]
      : "#64748b";

  return (
    <div
      style={{
        minHeight: "100vh",
        background: "#020617",
        color: "#e5e7eb",
        fontFamily: "system-ui, -apple-system, BlinkMacSystemFont, sans-serif",
      }}
    >
      {/* 헤더 */}
      <header
        style={{
          padding: "16px 24px",
          borderBottom: "1px solid #1f2937",
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          position: "sticky",
          top: 0,
          zIndex: 10,
          background: "rgba(2,6,23,0.9)",
          backdropFilter: "blur(6px)",
        }}
      >
        <div>
          <h1 style={{ fontSize: "20px", fontWeight: 700 }}>
            🛡️ SKYSHIELD
          </h1>
          <p style={{ fontSize: "12px", color: "#9ca3af", marginTop: 4 }}>
            Adaptive LLM Security Defense System — Intelligent Threshold & Anomaly Detection
          </p>
        </div>
        <span style={{ fontSize: "12px", color: "#6b7280" }}>
          Local UI (React)
        </span>
      </header>

      {/* 메인 영역 */}
      <main
        style={{
          display: "flex",
          gap: "16px",
          padding: "16px 24px 32px",
        }}
      >
        {/* 왼쪽 사이드바: 설정 */}
        <aside
          style={{
            width: "280px",
            flexShrink: 0,
            borderRadius: "12px",
            border: "1px solid #1f2937",
            padding: "16px",
            background:
              "radial-gradient(circle at top left, #0f172a, #020617)",
          }}
        >
          <h2
            style={{
              fontSize: "14px",
              fontWeight: 600,
              marginBottom: "12px",
            }}
          >
            ⚙️ 분석 설정
          </h2>

          {/* 임베딩 모델 */}
          <div style={{ marginBottom: "12px" }}>
            <label
              style={{
                display: "block",
                fontSize: "12px",
                marginBottom: "4px",
                color: "#9ca3af",
              }}
            >
              임베딩 모델
            </label>
            <select
              value={embedOpt}
              onChange={(e) => setEmbedOpt(e.target.value)}
              style={{
                width: "100%",
                padding: "6px 8px",
                borderRadius: "8px",
                border: "1px solid #374151",
                background: "#020617",
                color: "#e5e7eb",
                fontSize: "12px",
              }}
            >
              {EMBED_MODELS.map((m) => (
                <option key={m} value={m}>
                  {m}
                </option>
              ))}
            </select>
          </div>

          {/* 요약기 선택 */}
          <div style={{ marginBottom: "12px" }}>
            <label
              style={{
                display: "block",
                fontSize: "12px",
                marginBottom: "4px",
                color: "#9ca3af",
              }}
            >
              요약기 모델
            </label>
            <select
              value={summOpt}
              onChange={(e) => setSummOpt(e.target.value)}
              style={{
                width: "100%",
                padding: "6px 8px",
                borderRadius: "8px",
                border: "1px solid #374151",
                background: "#020617",
                color: "#e5e7eb",
                fontSize: "12px",
              }}
            >
              {SUMM_MODELS.map((m) => (
                <option key={m} value={m}>
                  {m}
                </option>
              ))}
            </select>
          </div>

          {/* Base Threshold */}
          <div style={{ marginBottom: "12px" }}>
            <div
              style={{
                display: "flex",
                justifyContent: "space-between",
                fontSize: "12px",
                marginBottom: "2px",
              }}
            >
              <span style={{ color: "#9ca3af" }}>Base Block Threshold</span>
              <span style={{ color: "#e5e7eb" }}>
                {baseThreshold.toFixed(2)}
              </span>
            </div>
            <input
              type="range"
              min={0}
              max={1}
              step={0.01}
              value={baseThreshold}
              onChange={(e) => setBaseThreshold(Number(e.target.value))}
              style={{ width: "100%" }}
            />
          </div>

          {/* HDBSCAN 민감도 */}
          <div style={{ marginBottom: "12px" }}>
            <div
              style={{
                display: "flex",
                justifyContent: "space-between",
                fontSize: "12px",
                marginBottom: "2px",
              }}
            >
              <span style={{ color: "#9ca3af" }}>
                HDBSCAN 민감도 (Sensitivity)
              </span>
              <span style={{ color: "#e5e7eb" }}>
                {sensitivity.toFixed(2)}
              </span>
            </div>
            <input
              type="range"
              min={0}
              max={1}
              step={0.05}
              value={sensitivity}
              onChange={(e) => setSensitivity(Number(e.target.value))}
              style={{ width: "100%" }}
            />
          </div>

          {/* 분석 버튼 */}
          <button
            onClick={handleAnalyze}
            disabled={loading}
            style={{
              marginTop: "8px",
              width: "100%",
              padding: "8px 0",
              borderRadius: "999px",
              border: "none",
              background:
                "linear-gradient(to right, #22c55e, #3b82f6, #a855f7)",
              color: "white",
              fontWeight: 600,
              fontSize: "13px",
              cursor: loading ? "wait" : "pointer",
              opacity: loading ? 0.7 : 1,
            }}
          >
            {loading ? "분석 중..." : "분석 🚀"}
          </button>

          {error && (
            <p
              style={{
                marginTop: "8px",
                fontSize: "11px",
                color: "#f97316",
              }}
            >
              {error}
            </p>
          )}
        </aside>

        {/* 오른쪽: 입력 + 결과 */}
        <section
          style={{ flex: 1, display: "flex", flexDirection: "column", gap: "16px" }}
        >
          {/* 입력 영역 */}
          <div
            style={{
              borderRadius: "12px",
              border: "1px solid #1f2937",
              padding: "16px",
              background: "#020617",
            }}
          >
            <h2
              style={{
                fontSize: "14px",
                fontWeight: 600,
                marginBottom: "8px",
              }}
            >
              👤 테스트 문장
            </h2>
            <textarea
              value={userInput}
              onChange={(e) => setUserInput(e.target.value)}
              placeholder="분석할 프롬프트 / 사용자 입력을 여기에 적어주세요."
              rows={6}
              style={{
                width: "100%",
                resize: "vertical",
                borderRadius: "8px",
                border: "1px solid #374151",
                padding: "8px",
                background: "#020617",
                color: "#e5e7eb",
                fontSize: "13px",
                lineHeight: 1.5,
              }}
            />
            <p
              style={{
                marginTop: "4px",
                fontSize: "11px",
                color: "#6b7280",
              }}
            >
              길이와 내용에 따라 Adaptive Threshold와 HDBSCAN 기반 판정이 달라집니다.
            </p>
          </div>

          {/* 결과 영역 */}
          {result && (
            <>
              {/* 최상단: 최종 판정 카드 */}
              <div
                style={{
                  borderRadius: "12px",
                  padding: "16px",
                  background: "#020617",
                  border: "1px solid #1f2937",
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "space-between",
                  gap: "12px",
                }}
              >
                <div>
                  <h2
                    style={{
                      fontSize: "14px",
                      fontWeight: 600,
                      marginBottom: "4px",
                    }}
                  >
                    🎯 최종 판정
                  </h2>
                  <p
                    style={{
                      fontSize: "12px",
                      color: "#9ca3af",
                    }}
                  >
                    SkyShield 기본 임계값 + HDBSCAN 클러스터링 결과를 반영한 결론입니다.
                  </p>
                </div>
                <div
                  style={{
                    padding: "10px 20px",
                    borderRadius: "999px",
                    background: decisionColor,
                    color: "white",
                    fontWeight: 700,
                    fontSize: "14px",
                    textAlign: "center",
                    minWidth: "120px",
                  }}
                >
                  {result.final_decision}
                </div>
              </div>

              {/* Adaptive Threshold + 요약 */}
              <div
                style={{
                  display: "grid",
                  gridTemplateColumns: "1.1fr 1.3fr",
                  gap: "16px",
                }}
              >
                {/* Adaptive Threshold 카드 */}
                <div
                  style={{
                    borderRadius: "12px",
                    border: "1px solid #1f2937",
                    padding: "14px",
                    background: "#020617",
                  }}
                >
                  <h3
                    style={{
                      fontSize: "13px",
                      fontWeight: 600,
                      marginBottom: "8px",
                    }}
                  >
                    📏 Adaptive Threshold
                  </h3>
                  <div
                    style={{
                      display: "grid",
                      gridTemplateColumns: "1fr 1fr",
                      rowGap: "6px",
                      columnGap: "8px",
                      fontSize: "12px",
                    }}
                  >
                    <span style={{ color: "#9ca3af" }}>입력 길이</span>
                    <span style={{ textAlign: "right" }}>
                      {userInput.length}자
                    </span>
                    <span style={{ color: "#9ca3af" }}>Base Threshold</span>
                    <span style={{ textAlign: "right" }}>
                      {result.base_threshold
                        ? result.base_threshold.toFixed(3)
                        : "-"}
                    </span>
                    <span style={{ color: "#9ca3af" }}>
                      Length-Adjusted
                    </span>
                    <span style={{ textAlign: "right" }}>
                      {result.adaptive_thr.toFixed(3)}
                    </span>
                  </div>
                </div>

                {/* 요약 카드 */}
                <div
                  style={{
                    borderRadius: "12px",
                    border: "1px solid #1f2937",
                    padding: "14px",
                    background: "#020617",
                  }}
                >
                  <h3
                    style={{
                      fontSize: "13px",
                      fontWeight: 600,
                      marginBottom: "8px",
                    }}
                  >
                    🧠 의미 요약
                  </h3>
                  <p
                    style={{
                      fontSize: "12px",
                      color: "#e5e7eb",
                      whiteSpace: "pre-wrap",
                    }}
                  >
                    {result.summary}
                  </p>
                </div>
              </div>

              {/* HDBSCAN / SkyShield 상세 */}
              <div
                style={{
                  borderRadius: "12px",
                  border: "1px solid #1f2937",
                  padding: "14px",
                  background: "#020617",
                }}
              >
                <h3
                  style={{
                    fontSize: "13px",
                    fontWeight: 600,
                    marginBottom: "8px",
                  }}
                >
                  🔍 상세 분석 정보
                </h3>

                <div
                  style={{
                    display: "grid",
                    gridTemplateColumns: "repeat(3, minmax(0, 1fr))",
                    gap: "8px",
                    fontSize: "12px",
                    marginBottom: "8px",
                  }}
                >
                  <div>
                    <div style={{ color: "#9ca3af" }}>기본 SkyShield</div>
                    <div>
                      {result.decision_basic} ({result.score_basic?.toFixed(3)})
                    </div>
                  </div>
                  <div>
                    <div style={{ color: "#9ca3af" }}>HDBSCAN 판정</div>
                    <div>{result.cluster_decision}</div>
                  </div>
                  <div>
                    <div style={{ color: "#9ca3af" }}>클러스터 ID</div>
                    <div>{result.cluster_id}</div>
                  </div>
                  <div>
                    <div style={{ color: "#9ca3af" }}>클러스터 유사도</div>
                    <div>{result.cluster_sim?.toFixed(3)}</div>
                  </div>
                  <div>
                    <div style={{ color: "#9ca3af" }}>Novel 기준</div>
                    <div>{result.novel_thr?.toFixed(3)}</div>
                  </div>
                  <div>
                    <div style={{ color: "#9ca3af" }}>Suspicious 기준</div>
                    <div>{result.susp_thr?.toFixed(3)}</div>
                  </div>
                </div>

                {result.cluster_name && (
                  <div
                    style={{
                      marginTop: "4px",
                      fontSize: "12px",
                      color: "#9ca3af",
                    }}
                  >
                    클러스터 의미 태그:{" "}
                    <span style={{ color: "#e5e7eb" }}>
                      {result.cluster_name}
                    </span>
                  </div>
                )}
              </div>

              {/* UMAP Placeholder */}
              <div
                style={{
                  borderRadius: "12px",
                  border: "1px solid #1f2937",
                  padding: "14px",
                  background: "#020617",
                }}
              >
                <h3
                  style={{
                    fontSize: "13px",
                    fontWeight: 600,
                    marginBottom: "8px",
                  }}
                >
                  🌈 의미 공간 UMAP 시각화
                </h3>
                <div
                  style={{
                    height: "260px",
                    borderRadius: "10px",
                    border: "1px dashed #374151",
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                    fontSize: "12px",
                    color: "#6b7280",
                  }}
                >
                  {/* 나중에 Plotly.js나 이미지로 대체 가능 */}
                  UMAP 그래프 영역 (백엔드 연동 후 추가)
                </div>
              </div>
            </>
          )}
        </section>
      </main>
    </div>
  );
}

export default App;
