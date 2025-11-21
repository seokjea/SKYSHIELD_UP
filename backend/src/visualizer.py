import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import umap.umap_ as umap


def plot_radar(center_sims, cluster_ids):
    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=center_sims,
        theta=[f"C{i}" for i in cluster_ids],
        fill='toself'
    ))

    fig.update_layout(
        title="클러스터 유사도 레이더 차트",
        showlegend=False
    )
    return fig


def plot_umap(atk_vec, norm_vec, user_vec, decision, centers):
    reducer = umap.UMAP(metric="cosine", random_state=42)

    atk_s = atk_vec[:1500]
    norm_s = norm_vec[:1500]

    all_vec = np.vstack([atk_s, norm_s, [user_vec]])
    labels = ["공격"] * len(atk_s) + ["정상"] * len(norm_s) + ["입력"]

    points = reducer.fit_transform(all_vec)

    df = pd.DataFrame({
        "x": points[:, 0],
        "y": points[:, 1],
        "type": labels
    })

    fig = px.scatter(df, x="x", y="y", color="type", opacity=0.65)

    # 입력 표시
    color_map = {"BLOCK": "red", "REVIEW": "orange", "ALLOW": "green", "SUSPICIOUS": "orange", "NOVEL_ATTACK": "red"}

    fig.add_trace(
        go.Scatter(
            x=[points[-1, 0]],
            y=[points[-1, 1]],
            mode="markers+text",
            marker=dict(size=22, symbol="star", color=color_map.get(decision, "blue")),
            text=[f"입력 ({decision})"],
            textposition="top center"
        )
    )

    # 클러스터 중심 표시
    for cid, c in centers.items():
        c2d = reducer.transform([c])[0]
        fig.add_trace(
            go.Scatter(
                x=[c2d[0]], y=[c2d[1]],
                mode="markers+text",
                marker=dict(size=16, color="black"),
                text=[f"C{cid}"],
                textposition="top center"
            )
        )

    return fig
