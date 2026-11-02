"""
Plotting and Visualization Helpers using Plotly for FmaxAI Dashboard.
"""

from typing import List, Tuple, Dict, Any, Optional
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def plot_wafer_map(df: pd.DataFrame, color_col: str = "fmax_true_mhz") -> go.Figure:
    """Renders a wafer spatial map with die locations and frequency heatmap."""
    fig = px.scatter(
        df,
        x="die_x",
        y="die_y",
        color=color_col,
        hover_data=["unit_id", "fmax_true_mhz", "true_bin_mhz"],
        color_continuous_scale="Viridis",
        title=f"Wafer Spatial Map — Color: {color_col}",
        labels={"die_x": "Die X Coordinate (norm)", "die_y": "Die Y Coordinate (norm)"}
    )
    # Draw circular wafer boundary
    fig.add_shape(
        type="circle",
        xref="x", yref="y",
        x0=-1.05, y0=-1.05, x1=1.05, y1=1.05,
        line=dict(color="rgba(150, 150, 150, 0.5)", width=2, dash="dash")
    )
    fig.update_layout(
        template="plotly_dark",
        height=520,
        margin=dict(l=20, r=20, t=50, b=20),
        xaxis=dict(scaleanchor="y", scaleratio=1),
        yaxis=dict(constrain="domain")
    )
    return fig


def plot_search_trajectories(
    trajectories: Dict[str, List[Tuple[float, bool]]],
    fmax_true: float,
    window_bounds: Optional[Tuple[float, float]] = None
) -> go.Figure:
    """
    Renders step-by-step test frequency trajectories across search algorithms (Figure 3 in document).
    """
    fig = go.Figure()
    
    # Shaded prediction window if provided
    if window_bounds is not None:
        low, high = window_bounds
        fig.add_hrect(
            y0=low, y1=high,
            fillcolor="rgba(46, 204, 113, 0.15)",
            line_width=1,
            line_color="rgba(46, 204, 113, 0.5)",
            annotation_text=f"AI Calibrated Window [{low:.0f} - {high:.0f} MHz]",
            annotation_position="top left"
        )

    # True Fmax reference line
    fig.add_hline(
        y=fmax_true,
        line_dash="dot",
        line_color="#E74C3C",
        line_width=2,
        annotation_text=f"True Fmax ({fmax_true:.1f} MHz)",
        annotation_position="bottom right"
    )

    colors = {
        "Linear sweep": "#95A5A6",
        "Binary search": "#3498DB",
        "ML-seeded walk": "#E67E22",
        "ML-windowed binary": "#2ECC71",
        "ML + Bayesian": "#9B59B6"
    }

    for method, points in trajectories.items():
        if not points:
            continue
        xs = list(range(1, len(points) + 1))
        ys = [p[0] for p in points]
        outcomes = [p[1] for p in points]
        
        # Line connecting steps
        method_color = colors.get(method, "#FFFFFF")
        fig.add_trace(go.Scatter(
            x=xs,
            y=ys,
            mode="lines+markers",
            name=f"{method} ({len(points)} steps)",
            line=dict(color=method_color, width=2),
            marker=dict(
                size=9,
                color=["#2ECC71" if o else "#E74C3C" for o in outcomes],
                symbol=["circle" if o else "x" for o in outcomes],
                line=dict(width=1, color="#FFFFFF")
            ),
            hovertext=[f"Step {x}: {y:.0f} MHz - {'PASS' if o else 'FAIL'}" for x, y, o in zip(xs, ys, outcomes)]
        ))

    fig.update_layout(
        title="Fmax Search Trajectory Comparison (Green = Pass, Red = Fail)",
        xaxis_title="Test Application Step #",
        yaxis_title="Tested Frequency (MHz)",
        template="plotly_dark",
        height=500,
        margin=dict(l=20, r=20, t=50, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig


def plot_residuals_and_calibration(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    bounds_low: np.ndarray,
    bounds_high: np.ndarray
) -> go.Figure:
    """Renders predicted vs true Fmax scatter with conformal interval bands."""
    df_plot = pd.DataFrame({
        "True Fmax (MHz)": y_true,
        "Predicted Fmax (MHz)": y_pred,
        "Low (MHz)": bounds_low,
        "High (MHz)": bounds_high,
        "In Window": (y_true >= bounds_low) & (y_true <= bounds_high)
    })
    
    fig = px.scatter(
        df_plot.sample(min(800, len(df_plot)), random_state=42),
        x="True Fmax (MHz)",
        y="Predicted Fmax (MHz)",
        color="In Window",
        color_discrete_map={True: "#2ECC71", False: "#E74C3C"},
        title="Model Predictions vs Ground Truth (Calibrated Window Coverage)",
        opacity=0.65
    )
    
    # 1:1 Identity reference line
    min_val = min(y_true.min(), y_pred.min()) - 20
    max_val = max(y_true.max(), y_pred.max()) + 20
    fig.add_trace(go.Scatter(
        x=[min_val, max_val],
        y=[min_val, max_val],
        mode="lines",
        line=dict(color="rgba(255, 255, 255, 0.4)", dash="dash"),
        name="Ideal 1:1 Reference"
    ))
    
    fig.update_layout(
        template="plotly_dark",
        height=480,
        margin=dict(l=20, r=20, t=50, b=20)
    )
    return fig


def plot_drift_control_chart(df_drift: pd.DataFrame, title: str = "EWMA Residual Control Chart") -> go.Figure:
    """Renders EWMA / CUSUM control chart for statistical process control."""
    fig = go.Figure()
    
    steps = list(range(1, len(df_drift) + 1))
    
    if "ewma" in df_drift.columns:
        fig.add_trace(go.Scatter(
            x=steps, y=df_drift["ewma"],
            mode="lines", name="EWMA Residual",
            line=dict(color="#3498DB", width=2)
        ))
        fig.add_trace(go.Scatter(
            x=steps, y=df_drift["ucl"],
            mode="lines", name="Upper Control Limit (+3σ)",
            line=dict(color="#E74C3C", dash="dash")
        ))
        fig.add_trace(go.Scatter(
            x=steps, y=df_drift["lcl"],
            mode="lines", name="Lower Control Limit (-3σ)",
            line=dict(color="#E74C3C", dash="dash")
        ))
        # Highlight alarms
        alarms = df_drift[df_drift["alarm"]]
        if not alarms.empty:
            fig.add_trace(go.Scatter(
                x=alarms.index + 1, y=alarms["ewma"],
                mode="markers", name="Out-of-Control Alarm",
                marker=dict(color="#FF0000", size=10, symbol="triangle-up")
            ))
            
    fig.update_layout(
        title=title,
        xaxis_title="Audited Unit Sequence",
        yaxis_title="Residual Deviation (MHz)",
        template="plotly_dark",
        height=420,
        margin=dict(l=20, r=20, t=50, b=20)
    )
    return fig

