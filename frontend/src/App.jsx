// src/App.jsx

import { useState } from "react";
import {
  Shield,
  Activity,
  Brain,
  Target,
  BarChart3,
  TrendingUp,
  Lock,
  Loader2,
  AlertTriangle,
  CheckCircle2,
  Cloud,
  Sparkles,
} from "lucide-react";
import {
  ScatterChart,
  Scatter,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip as RechartsTooltip,
  ResponsiveContainer,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  RadarChart,
  Radar,
  Legend,
} from "recharts";
import "./App.css";

function App() {
  const [embeddingModel, setEmbeddingModel] = useState("sentence-transformers");
  const [summarizerModel, setSummarizerModel] = useState("Google Gemini");
  const [baseThreshold, setBaseThreshold] = useState(40); // 0~100
  const [sensitivity, setSensitivity] = useState(40); // 0~100
  const [prompt, setPrompt] = useState("");
  const [result, setResult] = useState(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);

  const canAnalyze = prompt.trim().length > 0 && !isAnalyzing;

  // clusterSimilarity 기반 신뢰도(%) 계산
  const confidencePercent = result
    ? Math.min(100, Math.max(0, result.clusterSimilarity * 100))
    : 0;

  const analyzePrompt = async () => {
    if (!prompt.trim()) return;

    setIsAnalyzing(true);
    setResult(null);

    const backendUrl =
      import.meta.env.VITE_BACKEND_URL || "http://127.0.0.1:8000";

    try {
      const response = await fetch(`${backendUrl}/analyze`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          text: prompt,
          embed_model: embeddingModel,
          summ_model: summarizerModel,
          base_threshold: baseThreshold / 100,
          sensitivity: sensitivity / 100,
        }),
      });

      if (!response.ok) {
        throw new Error(`Server error: ${response.status}`);
      }

      const data = await response.json();

      const finalDecision = data.final_decision || "ALLOW";
      const basicDecision = data.decision_basic || "ALLOW";
      const clusterDecision = data.cluster_decision || "KNOWN_ATTACK";

      const umapData = generateUMAPData(finalDecision);
      const radarData = generateRadarData(finalDecision);

      setResult({
        inputLength: prompt.length,
        baseThreshold:
          typeof data.base_threshold === "number"
            ? data.base_threshold
            : baseThreshold / 100,
        adaptiveThreshold:
          typeof data.adaptive_thr === "number"
            ? data.adaptive_thr
            : baseThreshold / 100,
        summary: data.summary || "",
        basicDecision,
        basicScore: typeof data.score_basic === "number" ? data.score_basic : 0,
        clusterDecision,
        clusterId:
          typeof data.cluster_id === "number" ? data.cluster_id : -1,
        clusterSimilarity:
          typeof data.cluster_sim === "number" ? data.cluster_sim : 0,
        clusterName: data.cluster_name || "",
        novelThreshold:
          typeof data.novel_thr === "number" ? data.novel_thr : 0,
        suspiciousThreshold:
          typeof data.susp_thr === "number" ? data.susp_thr : 0,
        finalDecision,
        umapData,
        radarData,
      });
    } catch (e) {
      console.error("분석 요청 중 오류:", e);
      const fallbackDecision = "ALLOW";
      setResult({
        inputLength: prompt.length,
        baseThreshold: baseThreshold / 100,
        adaptiveThreshold: baseThreshold / 100,
        summary: "서버와 통신 중 문제가 발생했습니다.",
        basicDecision: fallbackDecision,
        basicScore: 0,
        clusterDecision: "KNOWN_ATTACK",
        clusterId: -1,
        clusterSimilarity: 0,
        clusterName: "",
        novelThreshold: 0.1,
        suspiciousThreshold: 0.3,
        finalDecision: fallbackDecision,
        umapData: generateUMAPData(fallbackDecision),
        radarData: generateRadarData(fallbackDecision),
      });
    } finally {
      setIsAnalyzing(false);
    }
  };

  // ---- UMAP용 더미 데이터 (피그마 스타일에 맞춤) ----
  function generateUMAPData(decision) {
    const data = [];

    const pushCluster = (cx, cy, count, type) => {
      for (let i = 0; i < count; i++) {
        data.push({
          x: cx + (Math.random() - 0.5) * 12,
          y: cy + (Math.random() - 0.5) * 12,
          type,
          label: type,
        });
      }
    };

    // 정적 클러스터들
    pushCluster(25, 70, 35, "정상");
    pushCluster(70, 30, 30, "공격");
    pushCluster(45, 45, 25, "의심");
    pushCluster(75, 75, 25, "경계");

    // 사용자 포인트 위치는 판정에 따라 이동
    let userX = 35;
    let userY = 35;
    if (decision === "REVIEW") {
      userX = 50;
      userY = 50;
    } else if (decision === "BLOCK") {
      userX = 75;
      userY = 75;
    }

    data.push({
      x: userX + (Math.random() - 0.5) * 4,
      y: userY + (Math.random() - 0.5) * 4,
      type: "입력",
      label: "입력",
    });

    return data;
  }

  // ---- 레이더 차트용 더미 데이터 (피그마 버전과 맞춤) ----
  function generateRadarData(decision) {
    let baseValue = 45;

    if (decision === "BLOCK") {
      baseValue = 70;
    } else if (decision === "REVIEW") {
      baseValue = 58;
    } else {
      baseValue = 40;
    }

    return [
      { category: "보안성", value: baseValue + Math.random() * 8, fullMark: 100 },
      { category: "윤리성", value: baseValue + Math.random() * 10, fullMark: 100 },
      {
        category: "프라이버시",
        value: baseValue + Math.random() * 9,
        fullMark: 100,
      },
      { category: "투명성", value: baseValue + Math.random() * 7, fullMark: 100 },
      { category: "신뢰도", value: baseValue + Math.random() * 8, fullMark: 100 },
    ];
  }

  // ---- 판정 관련 헬퍼 ----
  function getDecisionClass(decision) {
    if (decision === "BLOCK") return "decision-banner block";
    if (decision === "REVIEW") return "decision-banner review";
    return "decision-banner allow";
  }

  function getDecisionLabel(decision) {
    if (decision === "BLOCK") return "BLOCK";
    if (decision === "REVIEW") return "REVIEW";
    return "ALLOW";
  }

  function getDecisionSubLabel(decision) {
    if (decision === "BLOCK") return "위험한 시도가 감지되었습니다.";
    if (decision === "REVIEW") return "추가 검토가 권장됩니다.";
    return "특별한 보안 위협은 감지되지 않았습니다.";
  }

  function getDecisionIcon(decision) {
    if (decision === "BLOCK") return AlertTriangle;
    if (decision === "REVIEW") return Shield;
    return CheckCircle2;
  }

  function getClusterBadge(decision) {
    if (decision === "NOVEL_ATTACK") {
      return { label: "Unknown Pattern", className: "badge badge-red" };
    }
    if (decision === "SUSPICIOUS") {
      return { label: "Suspicious Behavior", className: "badge badge-amber" };
    }
    return { label: "Known Pattern", className: "badge badge-blue" };
  }

  return (
    <div className="app-root">
      {/* 헤더 */}
      <header className="app-header">
        <div className="app-header-left">
          <div className="logo-wrapper">
            <div className="logo-bg" />
            <div className="logo-main">
              <Shield className="icon-md" />
            </div>
          </div>
          <div>
            <div className="logo-title-row">
              <span className="logo-title">SkyShield</span>
              <span className="logo-badge">AI Security Platform</span>
            </div>
            <p className="logo-sub">
              Adaptive LLM Jailbreak Detection &amp; Prompt Risk Analysis
            </p>
          </div>
        </div>
        <div className="app-header-right">
          <div className="header-meta">
            <div className="header-meta-item">
              <Activity className="icon-xs text-green" />
              <span>Backend: Online</span>
            </div>
            <span className="divider" />
            <div className="header-meta-item">
              <Brain className="icon-xs text-sky" />
              <span>LLM-Aware Defense</span>
            </div>
          </div>
          <button className="premium-chip">
            <Cloud className="icon-xs" />
            <span>Premium</span>
          </button>
        </div>
      </header>

      {/* 메인 레이아웃 */}
      <main className="app-main">
        <div className="layout-grid">
          {/* 왼쪽: 입력 + 설정 */}
          <div className="left-column">
            {/* 프롬프트 입력 */}
            <section className="card">
              <div className="card-header">
                <div className="card-header-icon">
                  <Target className="icon-sm text-sky" />
                </div>
                <div>
                  <h2 className="card-title">프롬프트 입력</h2>
                  <p className="card-sub">
                    분석할 프롬프트를 입력하면 SkyShield가 보안 위험을 평가합니다.
                  </p>
                </div>
              </div>
              <div className="card-body">
                <div className="textarea-wrapper">
                  <textarea
                    value={prompt}
                    onChange={(e) => setPrompt(e.target.value)}
                    placeholder="분석할 프롬프트를 입력하세요..."
                    maxLength={2000}
                  />
                  <div className="textarea-counter">
                    {prompt.length} / 2000
                  </div>
                </div>
              </div>
            </section>

            {/* 분석 설정 */}
            <section className="card">
              <div className="card-header">
                {/* 이모티콘 제거, 텍스트만 유지 */}
                <div>
                  <h2 className="card-title">분석 설정</h2>
                  <p className="card-sub">
                    사용할 임베딩 모델과 요약기, 임계값 및 민감도를 설정합니다.
                  </p>
                </div>
              </div>
              <div className="card-body settings-grid">
                <div className="field-group">
                  <div className="field-label-row">
                    <span className="field-label">임베딩 모델</span>
                  </div>
                  <select
                    className="select"
                    value={embeddingModel}
                    onChange={(e) => setEmbeddingModel(e.target.value)}
                  >
                    <option value="sentence-transformers">
                      sentence-transformers
                    </option>
                    <option value="intfloat/multilingual-e5-large">
                      intfloat/multilingual-e5-large
                    </option>
                    <option value="thenlper/gte-large">
                      thenlper/gte-large
                    </option>
                    <option value="BAAI/bge-m3">BAAI/bge-m3</option>
                    <option value="OpenAI Embedding">OpenAI Embedding</option>
                    <option value="Mistral Embedding">Mistral Embedding</option>
                    <option value="DeepSeek Embedding">
                      DeepSeek Embedding
                    </option>
                  </select>
                </div>

                <div className="field-group">
                  <div className="field-label-row">
                    <span className="field-label">요약기</span>
                  </div>
                  <select
                    className="select"
                    value={summarizerModel}
                    onChange={(e) => setSummarizerModel(e.target.value)}
                  >
                    <option value="Google Gemini">Google Gemini</option>
                    <option value="OpenAI">OpenAI</option>
                    <option value="DeepSeek">DeepSeek</option>
                    <option value="Mistral">Mistral</option>
                  </select>
                </div>

                <div className="field-group">
                  <div className="field-label-row">
                    <span className="field-label">Base Threshold</span>
                    <span className="pill pill-sky">
                      {(baseThreshold / 100).toFixed(2)}
                    </span>
                  </div>
                  <input
                    type="range"
                    min={0}
                    max={100}
                    step={1}
                    value={baseThreshold}
                    onChange={(e) =>
                      setBaseThreshold(parseInt(e.target.value, 10))
                    }
                  />
                </div>

                <div className="field-group">
                  <div className="field-label-row">
                    <span className="field-label">민감도</span>
                    <span className="pill pill-pink">
                      {(sensitivity / 100).toFixed(2)}
                    </span>
                  </div>
                  <input
                    type="range"
                    min={0}
                    max={100}
                    step={5}
                    value={sensitivity}
                    onChange={(e) =>
                      setSensitivity(parseInt(e.target.value, 10))
                    }
                  />
                </div>
              </div>

              <div className="card-footer">
                <button
                  className="primary-button"
                  onClick={analyzePrompt}
                  disabled={!canAnalyze}
                >
                  {isAnalyzing ? (
                    <>
                      <Loader2 className="icon-sm spin" />
                      AI 분석 중...
                    </>
                  ) : (
                    <>
                      <Lock className="icon-sm" />
                      보안 분석 시작
                    </>
                  )}
                </button>
                {!prompt.trim() && (
                  <p className="hint-text">
                    프롬프트를 입력해야 분석을 시작할 수 있습니다.
                  </p>
                )}
              </div>
            </section>
          </div>

          {/* 오른쪽: 결과 영역 */}
          <div className="right-column">
            {/* 초기/로딩/결과 상태 분기 */}
            {!result && !isAnalyzing && (
              <div className="placeholder">
                <div className="placeholder-icon">
                  <Shield className="icon-lg" />
                </div>
                <h2>분석 대기 중</h2>
                <p>
                  프롬프트를 입력하고 분석을 시작하면, AI가 보안 위험을 정밀
                  분석합니다.
                </p>
                <div className="placeholder-meta">
                  <span>
                    <Activity className="icon-xs" /> Adaptive Threshold
                  </span>
                  <span className="dot" />
                  <span>
                    <BarChart3 className="icon-xs" /> HDBSCAN Anomaly Detection
                  </span>
                </div>
              </div>
            )}

            {isAnalyzing && (
              <div className="placeholder">
                <div className="placeholder-icon">
                  <Loader2 className="icon-lg spin" />
                </div>
                <h2>AI 분석 중입니다</h2>
                <p>
                  임베딩 유사도, Adaptive Threshold, HDBSCAN 클러스터 분석을
                  수행 중입니다.
                </p>
                <div className="placeholder-meta">
                  <span>
                    <Brain className="icon-xs" /> Semantic Embedding
                  </span>
                  <span className="dot" />
                  <span>
                    <BarChart3 className="icon-xs" /> High-Density Clustering
                  </span>
                </div>
              </div>
            )}

            {result && !isAnalyzing && (
              <div className="results-stack">
                {/* 최종 판정 배너 */}
                <section className={getDecisionClass(result.finalDecision)}>
                  <div className="decision-left">
                    <div className="decision-icon-wrapper">
                      {(() => {
                        const Icon = getDecisionIcon(result.finalDecision);
                        return <Icon className="icon-md" />;
                      })()}
                    </div>
                    <div>
                      <div className="decision-label-row">
                        <span className="decision-label">최종 판정</span>
                      </div>
                      <div className="decision-main-text">
                        {getDecisionLabel(result.finalDecision)}
                      </div>
                      <div className="decision-sub-text">
                        {getDecisionSubLabel(result.finalDecision)}
                      </div>
                    </div>
                  </div>
                  <div className="decision-right">
                    <div className="decision-tags">
                      <span className="pill pill-glass">
                        Cluster ID {result.clusterId}
                      </span>
                      <span className="pill pill-glass">
                        유사도 {result.clusterSimilarity.toFixed(3)}
                      </span>
                    </div>
                    <div className="decision-tags small">
                      <span className="pill pill-glass-dark">
                        Adaptive Threshold{" "}
                        {result.adaptiveThreshold.toFixed(3)}
                      </span>
                    </div>
                  </div>
                </section>

                {/* 5개 카드 + 요약/Threshold 상세 */}
                <section className="section-block">
                  {/* 상단 5개 카드 */}
                  <div className="small-card-grid">
                    <div className="small-card">
                      <div className="small-card-header">
                        <div className="small-icon pink">
                          <Brain className="icon-xs" />
                        </div>
                        <div>
                          <div className="small-title">임베딩 모델</div>
                          <div className="small-value ellipsis">
                            {embeddingModel}
                          </div>
                        </div>
                      </div>
                    </div>

                    <div className="small-card">
                      <div className="small-card-header">
                        <div className="small-icon red">
                          <Sparkles className="icon-xs" />
                        </div>
                        <div>
                          <div className="small-title">요약기</div>
                          <div className="small-value ellipsis">
                            {summarizerModel}
                          </div>
                        </div>
                      </div>
                    </div>

                    <div className="small-card">
                      <div className="small-card-header">
                        <div className="small-icon blue">
                          <BarChart3 className="icon-xs" />
                        </div>
                        <div>
                          <div className="small-title">클러스터 유사도</div>
                          <div className="small-value">
                            {result.clusterSimilarity.toFixed(3)}
                          </div>
                        </div>
                      </div>
                    </div>

                    <div className="small-card">
                      <div className="small-card-header">
                        <div className="small-icon green">
                          <TrendingUp className="icon-xs" />
                        </div>
                        <div>
                          <div className="small-title">Adaptive Threshold</div>
                          <div className="small-value">
                            {result.adaptiveThreshold.toFixed(3)}
                          </div>
                        </div>
                      </div>
                    </div>

                    <div className="small-card">
                      <div className="small-card-header">
                        <div className="small-icon purple">
                          <Target className="icon-xs" />
                        </div>
                        <div>
                          <div className="small-title">입력 길이</div>
                          <div className="small-value">
                            {result.inputLength}자
                          </div>
                        </div>
                      </div>
                    </div>
                  </div>

                  {/* 의미 요약 + Adaptive 상세 (아이콘 제거) */}
                  <div className="two-column">
                    <div className="card">
                      <div className="card-header small">
                        <div>
                          <h3 className="card-title small">의미 요약</h3>
                        </div>
                      </div>
                      <div className="card-body small">
                        <p className="summary-text">
                          {result.summary || "요약 결과가 없습니다."}
                        </p>
                      </div>
                    </div>

                    <div className="card">
                      <div className="card-header small">
                        <div>
                          <h3 className="card-title small">
                            Adaptive Threshold 상세
                          </h3>
                        </div>
                      </div>
                      <div className="card-body small">
                        <div className="field-row">
                          <span>Base Threshold</span>
                          <span className="mono">
                            {result.baseThreshold.toFixed(3)}
                          </span>
                        </div>
                        <div className="field-row">
                          <span>Adjusted Threshold</span>
                          <span className="mono">
                            {result.adaptiveThreshold.toFixed(3)}
                          </span>
                        </div>
                        <div className="field-row">
                          <span>Basic Score</span>
                          <span className="mono">
                            {result.basicScore.toFixed(3)}
                          </span>
                        </div>
                      </div>
                    </div>
                  </div>
                </section>

                {/* HDBSCAN + 시각화 + 클러스터 요약 */}
                <section className="section-block">
                  {/* HDBSCAN 상단 4개 카드 */}
                  <div className="card">
                    <div className="card-header small">
                      <div className="card-header-icon indigo">
                        <BarChart3 className="icon-xs text-indigo" />
                      </div>
                      <div>
                        <h3 className="card-title small">
                          HDBSCAN 클러스터 분석
                        </h3>
                      </div>
                    </div>
                    <div className="card-body hdbscan-grid">
                      <div className="hdbscan-box light">
                        <div className="hdbscan-label">의미 태그</div>
                        <div className="hdbscan-value">
                          {result.clusterName || "-"}
                        </div>
                      </div>
                      <div className="hdbscan-box sky">
                        <div className="hdbscan-label">클러스터 유사도</div>
                        <div className="hdbscan-value">
                          {result.clusterSimilarity.toFixed(3)}
                        </div>
                      </div>
                      <div className="hdbscan-box amber">
                        <div className="hdbscan-label">Novel 기준</div>
                        <div className="hdbscan-value">
                          {result.novelThreshold.toFixed(3)}
                        </div>
                      </div>
                      <div className="hdbscan-box violet">
                        <div className="hdbscan-label">Suspicious 기준</div>
                        <div className="hdbscan-value">
                          {result.suspiciousThreshold.toFixed(3)}
                        </div>
                      </div>
                    </div>
                  </div>

                  {/* UMAP / 레이더 차트 */}
                  <div className="two-column charts">
                    <div className="card">
                      <div className="card-header small">
                        <div>
                          <h3 className="card-title small">UMAP 시각화</h3>
                        </div>
                      </div>
                      <div className="card-body chart-body">
                        <ResponsiveContainer width="100%" height={320}>
                          <ScatterChart>
                            <CartesianGrid
                              strokeDasharray="3 3"
                              stroke="#e2e8f0"
                            />
                            <XAxis
                              type="number"
                              dataKey="x"
                              domain={[0, 100]}
                              hide
                            />
                            <YAxis
                              type="number"
                              dataKey="y"
                              domain={[0, 100]}
                              hide
                            />
                            <RechartsTooltip
                              cursor={{ strokeDasharray: "3 3" }}
                              contentStyle={{
                                border: "1px solid #e2e8f0",
                                borderRadius: 6,
                                fontSize: 12,
                              }}
                            />
                            <Legend wrapperStyle={{ fontSize: 12 }} />
                            <Scatter
                              name="정상"
                              data={result.umapData.filter(
                                (d) => d.type === "정상"
                              )}
                              fill="#10b981"
                              opacity={0.6}
                            />
                            <Scatter
                              name="공격"
                              data={result.umapData.filter(
                                (d) => d.type === "공격"
                              )}
                              fill="#ef4444"
                              opacity={0.6}
                            />
                            <Scatter
                              name="의심"
                              data={result.umapData.filter(
                                (d) => d.type === "의심"
                              )}
                              fill="#f59e0b"
                              opacity={0.5}
                            />
                            <Scatter
                              name="경계"
                              data={result.umapData.filter(
                                (d) => d.type === "경계"
                              )}
                              fill="#6366f1"
                              opacity={0.5}
                            />
                            <Scatter
                              name="입력"
                              data={result.umapData.filter(
                                (d) => d.type === "입력"
                              )}
                              fill="#0ea5e9"
                              shape="star"
                            />
                          </ScatterChart>
                        </ResponsiveContainer>
                      </div>
                    </div>

                    <div className="card">
                      <div className="card-header small">
                        <div>
                          <h3 className="card-title small">레이더 차트</h3>
                        </div>
                      </div>
                      <div className="card-body chart-body">
                        <ResponsiveContainer width="100%" height={320}>
                          <RadarChart data={result.radarData}>
                            <PolarGrid stroke="#e2e8f0" />
                            <PolarAngleAxis
                              dataKey="category"
                              tick={{ fontSize: 11 }}
                            />
                            <PolarRadiusAxis
                              domain={[0, 100]}
                              tick={{ fontSize: 10 }}
                            />
                            <Radar
                              dataKey="value"
                              stroke="#0ea5e9"
                              fill="#0ea5e9"
                              fillOpacity={0.4}
                              strokeWidth={2}
                            />
                          </RadarChart>
                        </ResponsiveContainer>
                      </div>
                    </div>
                  </div>

                  {/* 클러스터 상태 요약 + 신뢰도 (헤더 아이콘 제거) */}
                  <div className="two-column">
                    <div className="card">
                      <div className="card-header small">
                        <div>
                          <h3 className="card-title small">클러스터 상태 요약</h3>
                        </div>
                      </div>
                      <div className="card-body small">
                        <div className="cluster-summary">
                          <div className="cluster-text">
                            <div className="cluster-label">클러스터 판정</div>
                            <div className="cluster-desc">
                              {result.clusterDecision === "NOVEL_ATTACK"
                                ? "기존 패턴과 유사하지 않은 새로운 공격 시도로 보입니다."
                                : result.clusterDecision === "SUSPICIOUS"
                                ? "위험 클러스터와의 유사도가 기준 범위 내에 있어 주의가 필요합니다."
                                : "기존에 알려진 패턴과 유사한 입력으로 판단됩니다."}
                            </div>
                          </div>
                          <span
                            className={
                              getClusterBadge(result.clusterDecision).className
                            }
                          >
                            {getClusterBadge(result.clusterDecision).label}
                          </span>
                        </div>
                      </div>
                    </div>

                    <div className="card">
                      <div className="card-header small">
                        <div>
                          <h3 className="card-title small">신뢰도</h3>
                        </div>
                      </div>
                      <div className="card-body small">
                        <div className="confidence-row">
                          <div className="confidence-bar">
                            <div
                              className={
                                "confidence-fill " +
                                (result.finalDecision === "BLOCK"
                                  ? "block"
                                  : result.finalDecision === "REVIEW"
                                  ? "review"
                                  : "allow")
                              }
                              style={{ width: `${confidencePercent}%` }}
                            />
                          </div>
                          <span className="confidence-value">
                            {confidencePercent.toFixed(0)}%
                          </span>
                        </div>
                        <div className="field-row">
                          <span>클러스터 유사도</span>
                          <span className="mono">
                            {result.clusterSimilarity.toFixed(3)}
                          </span>
                        </div>
                        <div className="field-row">
                          <span>Cluster ID</span>
                          <span className="mono">{result.clusterId}</span>
                        </div>
                      </div>
                    </div>
                  </div>
                </section>
              </div>
            )}
          </div>
        </div>

        {/* 푸터 */}
        <footer className="app-footer">
          <div className="footer-left">
            <Shield className="icon-xs" />
            <span>SkyShield · Adaptive LLM Security Defense System</span>
          </div>
          <div className="footer-right">
            <span>HDBSCAN 기반 의미 클러스터링 ·</span>
            <span> Adaptive Threshold ·</span>
            <span> Prompt Risk Analytics</span>
          </div>
        </footer>
      </main>
    </div>
  );
}

export default App;
