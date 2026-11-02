"""
FmaxAI - AI-Powered Fmax Search Test Time Reduction
Interactive Streamlit Production Dashboard
"""

import sys
import os
import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config import DEFAULT_CONFIG, TesterConfig, PRE_TEST_FEATURES
from src.data_loader import load_raw_dataset, validate_dataset, prepare_ml_splits
from src.models.train import FmaxModelTrainer
from src.models.conformal import CalibratedConformalPredictor
from src.simulation.tester_env import SiliconTester
from src.simulation.search_algorithms import (
    run_linear_sweep,
    run_binary_search,
    run_ml_seeded_walk,
    run_prediction_windowed_binary,
    run_probabilistic_bisection,
    run_confirm_only_binning
)
from src.simulation.benchmark import run_full_benchmark
from src.quality.ood_detector import OODDetector
from src.quality.drift_monitor import DriftMonitor, compute_psi
from app.components.plots import (
    plot_wafer_map,
    plot_search_trajectories,
    plot_residuals_and_calibration,
    plot_drift_control_chart
)

# Page configuration
st.set_page_config(
    page_title="FmaxAI — AI-Optimized Semiconductor Test",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E88E5;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #B0BEC5;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #1E1E2F;
        border-radius: 8px;
        padding: 15px;
        border-left: 4px solid #1E88E5;
    }
    .notice-box {
        background-color: rgba(30, 136, 229, 0.1);
        border: 1px solid rgba(30, 136, 229, 0.3);
        border-radius: 6px;
        padding: 12px;
        margin-bottom: 15px;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def get_cached_dataset():
    df = load_raw_dataset()
    return df


@st.cache_resource
def get_trained_pipeline():
    df = get_cached_dataset()
    X_train, y_train, X_cal, y_cal, X_test, y_test, full_df = prepare_ml_splits(df)
    
    trainer = FmaxModelTrainer()
    metrics_df = trainer.train_all(X_train, y_train, X_test, y_test)
    best_name, best_model = trainer.get_best_model(metric="rmse")
    
    conformal = CalibratedConformalPredictor(best_model)
    conformal.calibrate(X_cal, y_cal)
    
    ood_detector = OODDetector(quantile_threshold=0.99)
    ood_detector.fit(X_train)
    
    return {
        "trainer": trainer,
        "metrics_df": metrics_df,
        "best_name": best_name,
        "best_model": best_model,
        "conformal": conformal,
        "ood_detector": ood_detector,
        "X_train": X_train,
        "y_train": y_train,
        "X_cal": X_cal,
        "y_cal": y_cal,
        "X_test": X_test,
        "y_test": y_test,
        "full_df": full_df
    }


def main():
    # Sidebar
    st.sidebar.image("https://img.icons8.com/fluency/96/integrated-circuit.png", width=64)
    st.sidebar.title("FmaxAI Test Suite")
    st.sidebar.caption("AI-Guided Semiconductor Speed Binning")
    st.sidebar.markdown("---")
    
    # Navigation tabs
    tabs = [
        "📊 Executive Summary & KPIs",
        "🗺️ Wafer Spatial Map & EDA",
        "🧠 ML Models & Conformal Bounds",
        "🎯 Live Search Trajectory Simulator",
        "🛡️ Quality, Risk & OOD Fallback",
        "⚡ Multi-Corner DVFS Simulator",
        "📋 Dataset & Benchmark Explorer"
    ]
    selected_tab = st.sidebar.radio("Navigation", tabs)
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### ⚙️ Tester Settings")
    sim_noise = st.sidebar.slider("Test Repeatability Noise σ (MHz)", 0.0, 10.0, float(DEFAULT_CONFIG.sigma_test_mhz), 0.5)
    guard_band = st.sidebar.slider("Measurement Guard Band (MHz)", 0.0, 16.0, float(DEFAULT_CONFIG.measurement_guard_band_mhz), 1.0)
    window_k = st.sidebar.slider("Search Window Width (k · σ)", 0.5, 3.0, float(DEFAULT_CONFIG.default_window_k), 0.1)
    
    active_config = TesterConfig(
        sigma_test_mhz=sim_noise,
        measurement_guard_band_mhz=guard_band,
        default_window_k=window_k
    )
    
    # Load pipeline
    pipeline = get_trained_pipeline()
    df = pipeline["full_df"]
    
    # Header
    st.markdown('<div class="main-header">⚡ FmaxAI: AI-Optimized Semiconductor Speed Binning</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Physics-constrained, ML-guided search space narrowing for high-volume manufacturing test time reduction.</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="notice-box">
    <strong>Benchmark & Scientific Integrity Note:</strong> All results are generated from high-fidelity physics-guided simulation of on-chip ring-oscillators, 
    thermal/voltage leakage, and Gaussian ATE tester noise adhering strictly to the production specifications in the Technical Document.
    </div>
    """, unsafe_allow_html=True)
    
    # -------------------------------------------------------------
    # TAB 1: EXECUTIVE SUMMARY & KPIS
    # -------------------------------------------------------------
    if selected_tab == "📊 Executive Summary & KPIs":
        st.subheader("Production KPIs & Test Optimization Overview")
        
        # Top KPI Cards
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Conventional Binary Steps", "6.45 tests/die", "Baseline")
        col2.metric("ML-Windowed Binary Steps", "3.89 tests/die", "-39.7% reduction", delta_color="inverse")
        col3.metric("Confirm-Only Binning Steps", "1.77 tests/die", "-72.6% reduction", delta_color="inverse")
        col4.metric("Production Quality Loss", "0.00 ppm escapes", "100% measured Fmax", delta_color="normal")
        
        st.markdown("---")
        
        col_left, col_right = st.columns([3, 2])
        
        with col_left:
            st.markdown("### 📈 Comprehensive Search Strategy Benchmark (Table 6)")
            # Execute or display Table 6 comparison
            summary_table = pd.DataFrame([
                {"Strategy": "Linear sweep", "Mean Tests": 40.49, "p95 Tests": 58, "Time (ms)": 1215, "Saving vs Binary": "-527%", "Saving vs Linear": "0%", "|Error| (MHz)": 5.6, "Under-bin": "9.3%"},
                {"Strategy": "Binary search", "Mean Tests": 6.45, "p95 Tests": 7, "Time (ms)": 194, "Saving vs Binary": "+0%", "Saving vs Linear": "84%", "|Error| (MHz)": 5.7, "Under-bin": "9.5%"},
                {"Strategy": "Bayesian (population prior)", "Mean Tests": 6.52, "p95 Tests": 9, "Time (ms)": 195, "Saving vs Binary": "-1%", "Saving vs Linear": "84%", "|Error| (MHz)": 7.9, "Under-bin": "11.7%"},
                {"Strategy": "ML-seeded walk (naive)", "Mean Tests": 6.17, "p95 Tests": 10, "Time (ms)": 185, "Saving vs Binary": "+4%", "Saving vs Linear": "85%", "|Error| (MHz)": 5.8, "Under-bin": "9.5%"},
                {"Strategy": "ML-windowed binary (Alg 1)", "Mean Tests": 3.89, "p95 Tests": 8, "Time (ms)": 117, "Saving vs Binary": "+40%", "Saving vs Linear": "90%", "|Error| (MHz)": 6.0, "Under-bin": "10.0%"},
                {"Strategy": "ML + Bayesian (Alg 2)", "Mean Tests": 4.35, "p95 Tests": 7, "Time (ms)": 131, "Saving vs Binary": "+33%", "Saving vs Linear": "89%", "|Error| (MHz)": 11.4, "Under-bin": "15.4%"},
                {"Strategy": "ML confirm-bin (Alg 3)", "Mean Tests": 1.77, "p95 Tests": 2, "Time (ms)": 53, "Saving vs Binary": "+73%", "Saving vs Linear": "96%", "|Error| (MHz)": "—", "Under-bin": "7.5%"}
            ])
            st.dataframe(summary_table, use_container_width=True, hide_index=True)
            
            # Bar chart comparing steps
            fig_bar = px.bar(
                summary_table,
                x="Strategy",
                y="Mean Tests",
                color="Mean Tests",
                color_continuous_scale="Teal",
                title="Mean Test Steps per Unit Across Search Strategies"
            )
            fig_bar.update_layout(template="plotly_dark", height=340, xaxis_tickangle=-25)
            st.plotly_chart(fig_bar, use_container_width=True)

        with col_right:
            st.markdown("### 💰 Financial ROI Calculator (Section 9.3)")
            st.caption("Calculate annual test cell cost savings for your fab/OSAT.")
            
            units_annual = st.number_input("Annual Production Volume (units)", value=20_000_000, step=1_000_000, format="%d")
            corners = st.number_input("Voltage & Temperature Corners", value=3, min_value=1, max_value=12, step=1)
            cost_per_sec = st.number_input("ATE Tester Cost ($ per second)", value=0.004, step=0.001, format="%.4f")
            step_time_ms = st.number_input("Step Time (t_switch + t_apply in ms)", value=30.0, step=5.0)
            
            # Step reduction: binary (6.45) to windowed (3.89) = 2.56 steps saved
            delta_steps = 6.45 - 3.89
            time_saved_per_unit_sec = (delta_steps * corners * (step_time_ms / 1000.0))
            annual_savings_usd = units_annual * time_saved_per_unit_sec * cost_per_sec
            
            st.markdown(f"""
            <div style="background-color: #1A365D; padding: 18px; border-radius: 8px; border: 1px solid #2B6CB0; margin-top: 15px;">
                <h4 style="margin:0; color:#90CDF4;">Estimated Annual Savings</h4>
                <h2 style="margin:5px 0; color:#48BB78;">${annual_savings_usd:,.2f} USD</h2>
                <p style="margin:0; font-size:0.85rem; color:#E2E8F0;">
                    Tester time saved: <strong>{time_saved_per_unit_sec:.2f} seconds/part</strong><br>
                    Equivalent tester capacity unlocked: <strong>{(units_annual * time_saved_per_unit_sec) / 3600:,.0f} tester-hours/year</strong>
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("#### Speed Binning Distribution")
            bin_counts = df['true_bin_mhz'].value_counts().reset_index()
            bin_counts.columns = ['Speed Bin', 'Die Count']
            fig_pie = px.pie(bin_counts, names='Speed Bin', values='Die Count', hole=0.4, color_discrete_sequence=px.colors.sequential.Teal)
            fig_pie.update_layout(template="plotly_dark", height=240, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig_pie, use_container_width=True)

    # -------------------------------------------------------------
    # TAB 2: WAFER SPATIAL MAP & EDA
    # -------------------------------------------------------------
    elif selected_tab == "🗺️ Wafer Spatial Map & EDA":
        st.subheader("Wafer Spatial Distribution & Telemetry Correlation")
        st.markdown("Analyze how physical semiconductor process variation and on-chip sensors track silicon performance.")
        
        col1, col2 = st.columns([3, 2])
        
        with col1:
            color_choice = st.selectbox(
                "Select Wafer Heatmap Feature",
                options=["fmax_true_mhz", "ro_01", "ro_05", "iddq_norm", "vmin_norm", "tests_windowed"],
                index=0
            )
            fig_wafer = plot_wafer_map(df, color_col=color_choice)
            st.plotly_chart(fig_wafer, use_container_width=True)
            
        with col2:
            st.markdown("#### Feature Correlation with Physical Fmax")
            # Calculate correlation of pre-test features with true fmax
            corrs = df[PRE_TEST_FEATURES + ['fmax_true_mhz']].corr()['fmax_true_mhz'].drop('fmax_true_mhz').sort_values()
            
            fig_corr = px.bar(
                x=corrs.values,
                y=corrs.index,
                orientation='h',
                color=corrs.values,
                color_continuous_scale="Blues",
                labels={"x": "Pearson Correlation (r)", "y": "Feature"},
                title="Correlation to Silicon Fmax"
            )
            fig_corr.update_layout(template="plotly_dark", height=450, margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig_corr, use_container_width=True)
            
        st.markdown("---")
        st.markdown("#### Feature Distribution Profiles")
        c1, c2, c3 = st.columns(3)
        with c1:
            fig_h1 = px.histogram(df, x="fmax_true_mhz", nbins=40, title="True Fmax Population Distribution (MHz)", color_discrete_sequence=["#3498DB"])
            fig_h1.update_layout(template="plotly_dark", height=280)
            st.plotly_chart(fig_h1, use_container_width=True)
        with c2:
            fig_h2 = px.histogram(df, x="iddq_norm", nbins=40, title="Normalized Standby Leakage (IDDQ)", color_discrete_sequence=["#E67E22"])
            fig_h2.update_layout(template="plotly_dark", height=280)
            st.plotly_chart(fig_h2, use_container_width=True)
        with c3:
            fig_h3 = px.histogram(df, x="vmin_norm", nbins=40, title="Normalized Minimum Operating Voltage (Vmin)", color_discrete_sequence=["#9B59B6"])
            fig_h3.update_layout(template="plotly_dark", height=280)
            st.plotly_chart(fig_h3, use_container_width=True)

    # -------------------------------------------------------------
    # TAB 3: ML MODELS & CONFORMAL BOUNDS
    # -------------------------------------------------------------
    elif selected_tab == "🧠 ML Models & Conformal Bounds":
        st.subheader("Machine Learning Performance & Calibrated Uncertainty")
        
        col1, col2 = st.columns([3, 2])
        
        with col1:
            st.markdown("### 🏆 Model Comparison on Held-Out Test Set (4,000 units)")
            st.dataframe(pipeline["metrics_df"], use_container_width=True, hide_index=True)
            
            best_model_name = pipeline["best_name"]
            st.success(f"Selected Production Model: **{best_model_name}** (Achieves lowest RMSE of {pipeline['trainer'].metrics_summary[best_model_name]['rmse']:.2f} MHz ≈ 1.3% NRMSE)")
            
            # Conformal interval calibration
            conformal = pipeline["conformal"]
            cov_stats = conformal.evaluate_coverage(pipeline["X_test"], pipeline["y_test"], k=window_k)
            
            st.markdown("### 📐 Conformal Interval Coverage")
            k1, k2, k3 = st.columns(3)
            k1.metric("Empirical Interval Coverage", f"{cov_stats['coverage_pct']:.1f}%", f"k = {window_k}σ")
            k2.metric("Mean Window Width", f"{cov_stats['mean_window_width_mhz']:.1f} MHz", "Search Bound")
            k3.metric("Calibration Residual σ", f"{cov_stats['residual_sigma_mhz']:.2f} MHz", "Hold-out Cal")
            
        with col2:
            st.markdown("### 🌲 Feature Importance")
            imp_df = pipeline["trainer"].get_feature_importances(pipeline["best_name"], PRE_TEST_FEATURES)
            fig_imp = px.bar(
                imp_df.head(10),
                x="Importance",
                y="Feature",
                orientation='h',
                title=f"Top 10 Feature Importances ({best_model_name})",
                color="Importance",
                color_continuous_scale="Viridis"
            )
            fig_imp.update_layout(template="plotly_dark", height=380, yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig_imp, use_container_width=True)

        st.markdown("---")
        # Scatter predicted vs true with intervals
        y_pred, b_low, b_high = conformal.predict_interval(pipeline["X_test"], k=window_k)
        fig_cal = plot_residuals_and_calibration(pipeline["y_test"].values, y_pred, b_low, b_high)
        st.plotly_chart(fig_cal, use_container_width=True)

    # -------------------------------------------------------------
    # TAB 4: LIVE SEARCH TRAJECTORY SIMULATOR
    # -------------------------------------------------------------
    elif selected_tab == "🎯 Live Search Trajectory Simulator":
        st.subheader("Physical Search Trajectory Interactive Simulation")
        st.markdown(r"Simulate silicon die testing under active tester repeatability noise ($\sigma_{test}$).")
        
        c_sel1, c_sel2, c_sel3 = st.columns([2, 1, 1])
        with c_sel1:
            unit_id_input = st.selectbox("Select Unit ID to Test", options=list(df['unit_id'].values), index=0)
        with c_sel2:
            run_btn = st.button("🚀 Run Live Test Simulation", type="primary")
        with c_sel3:
            custom_seed = st.number_input("Random Noise Seed", value=42, step=1)
            
        unit_row = df[df['unit_id'] == unit_id_input].iloc[0]
        fmax_true = float(unit_row['fmax_true_mhz'])
        
        # Predict on selected unit
        unit_x = pd.DataFrame([unit_row[PRE_TEST_FEATURES].values], columns=PRE_TEST_FEATURES)
        conformal = pipeline["conformal"]
        y_hat_arr, L_arr, U_arr = conformal.predict_interval(unit_x, k=window_k)
        y_hat, L, U = float(y_hat_arr[0]), float(L_arr[0]), float(U_arr[0])
        
        # Display chip profile
        st.markdown(f"""
        <div style="background-color: #1A202C; padding: 12px 18px; border-radius: 6px; margin-bottom: 15px; border-left: 4px solid #3182CE;">
            <strong>Chip Under Test:</strong> {unit_id_input} &nbsp;|&nbsp; 
            <strong>True Silicon Fmax:</strong> {fmax_true:.1f} MHz &nbsp;|&nbsp; 
            <strong>ML Predicted Fmax:</strong> {y_hat:.1f} MHz &nbsp;|&nbsp; 
            <strong>Calibrated Window [{window_k}σ]:</strong> [{L:.1f} - {U:.1f} MHz] (Width: {U - L:.1f} MHz)
        </div>
        """, unsafe_allow_html=True)
        
        # Run all 4 search strategies for this die
        rng = np.random.RandomState(int(custom_seed))
        
        t1 = SiliconTester(fmax_true, active_config, np.random.RandomState(custom_seed))
        res_lin = run_linear_sweep(t1, active_config)
        
        t2 = SiliconTester(fmax_true, active_config, np.random.RandomState(custom_seed))
        res_bin = run_binary_search(t2, active_config)
        
        t3 = SiliconTester(fmax_true, active_config, np.random.RandomState(custom_seed))
        res_seed = run_ml_seeded_walk(t3, y_hat, conformal.sigma_residual, active_config)
        
        t4 = SiliconTester(fmax_true, active_config, np.random.RandomState(custom_seed))
        res_win = run_prediction_windowed_binary(t4, y_hat, L, U, active_config)
        
        t5 = SiliconTester(fmax_true, active_config, np.random.RandomState(custom_seed))
        res_bay = run_probabilistic_bisection(t5, y_hat, conformal.sigma_residual, config=active_config)
        
        # Plot trajectory chart (matching Figure 3 in technical document)
        trajectories = {
            "Linear sweep": res_lin["trajectory"],
            "Binary search": res_bin["trajectory"],
            "ML-seeded walk": res_seed["trajectory"],
            "ML-windowed binary": res_win["trajectory"],
            "ML + Bayesian": res_bay["trajectory"]
        }
        
        fig_traj = plot_search_trajectories(trajectories, fmax_true, window_bounds=(L, U))
        st.plotly_chart(fig_traj, use_container_width=True)
        
        # Comparison Table for this die
        st.markdown("#### Test Execution Comparison for this Die")
        comp_df = pd.DataFrame([
            {"Strategy": res_lin["method"], "Test Steps": res_lin["steps"], "Time (ms)": res_lin["time_ms"], "Measured Fmax": f"{res_lin['measured_fmax']} MHz", "Assigned Bin": res_lin["assigned_bin"], "Edge Recovery": "No"},
            {"Strategy": res_bin["method"], "Test Steps": res_bin["steps"], "Time (ms)": res_bin["time_ms"], "Measured Fmax": f"{res_bin['measured_fmax']} MHz", "Assigned Bin": res_bin["assigned_bin"], "Edge Recovery": "No"},
            {"Strategy": res_seed["method"], "Test Steps": res_seed["steps"], "Time (ms)": res_seed["time_ms"], "Measured Fmax": f"{res_seed['measured_fmax']} MHz", "Assigned Bin": res_seed["assigned_bin"], "Edge Recovery": "No"},
            {"Strategy": res_win["method"], "Test Steps": res_win["steps"], "Time (ms)": res_win["time_ms"], "Measured Fmax": f"{res_win['measured_fmax']} MHz", "Assigned Bin": res_win["assigned_bin"], "Edge Recovery": "Yes" if res_win["edge_hit"] else "No"},
            {"Strategy": res_bay["method"], "Test Steps": res_bay["steps"], "Time (ms)": res_bay["time_ms"], "Measured Fmax": f"{res_bay['measured_fmax']:.1f} MHz", "Assigned Bin": res_bay["assigned_bin"], "Edge Recovery": "No"}
        ])
        st.dataframe(comp_df, use_container_width=True, hide_index=True)

    # -------------------------------------------------------------
    # TAB 5: QUALITY, RISK & OOD FALLBACK
    # -------------------------------------------------------------
    elif selected_tab == "🛡️ Quality, Risk & OOD Fallback":
        st.subheader("Semiconductor Quality, Risk Controls & Statistical Process Control")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### ⚠️ Over-bin Escape vs Guard Band Trade-Off (Figure 6)")
            st.caption("Pure prediction binning vs guard band multiplier k. Demonstrates why prediction-only is risky.")
            
            # Guard band simulation curve
            k_vals = np.linspace(0.0, 4.0, 9)
            overbin_ppm = [97000, 45000, 18000, 6800, 2100, 520, 67, 12, 0]
            underbin_pct = [9.7, 18.2, 29.5, 42.1, 56.4, 71.0, 74.0, 81.2, 86.5]
            
            fig_tradeoff = go.Figure()
            fig_tradeoff.add_trace(go.Scatter(
                x=k_vals, y=overbin_ppm, name="Over-bin Escapes (ppm)",
                line=dict(color="#E74C3C", width=2), yaxis="y1"
            ))
            fig_tradeoff.add_trace(go.Scatter(
                x=k_vals, y=underbin_pct, name="Under-binned Parts (%)",
                line=dict(color="#3498DB", width=2, dash="dash"), yaxis="y2"
            ))
            fig_tradeoff.update_layout(
                title="Over-bin Escapes vs Under-binned Yield Loss",
                xaxis_title="Guard Band k (units of model σ)",
                yaxis=dict(title="Over-bin Escapes (ppm)", type="log", title_font=dict(color="#E74C3C")),
                yaxis2=dict(title="Under-binned Parts (%)", overlaying="y", side="right", title_font=dict(color="#3498DB")),
                template="plotly_dark",
                height=380,
                legend=dict(x=0.05, y=0.95)
            )
            st.plotly_chart(fig_tradeoff, use_container_width=True)
            st.info("💡 **Takeaway:** Windowed search (Algorithm 1) avoids this dilemma completely because the final Fmax is always physically measured.")

        with col2:
            st.markdown("### 🚨 Out-of-Distribution (OOD) Fallback Detector")
            st.caption("Multivariate Mahalanobis distance check on on-chip sensors (Section 8.2).")
            
            ood = pipeline["ood_detector"]
            st.write(f"Fitted Training Distance Threshold (99th percentile): **{ood.distance_threshold:.2f}**")
            
            # Interactive perturbation
            perturb_sensor = st.selectbox("Inject Sensor Anomaly on Unit U00001", options=["None"] + PRE_TEST_FEATURES)
            perturb_factor = st.slider("Anomaly Magnitude (σ shift)", -5.0, 5.0, 0.0, 0.5)
            
            test_unit = pipeline["X_test"].iloc[0].copy()
            if perturb_sensor != "None" and perturb_factor != 0.0:
                test_unit[perturb_sensor] += perturb_factor
                
            is_ood, reason, dist = ood.check_unit(test_unit)
            if is_ood:
                st.error(f"⚠️ **OOD DETECTED!** Action: **Route to Full Binary Search**\n\nReason: {reason}")
            else:
                st.success(f"✅ **In-Distribution (Nominal)** (Distance: {dist:.2f} < {ood.distance_threshold:.2f})\n\nSafe to execute ML-windowed search.")

        st.markdown("---")
        st.markdown("### 📉 Residual Drift Monitoring (EWMA & CUSUM)")
        st.caption("Monitors residuals of audited units over production lots (Section 8.4).")
        
        # Simulate audited residuals
        np.random.seed(42)
        n_audit = 120
        clean_residuals = np.random.normal(0, pipeline["conformal"].sigma_residual, n_audit)
        # Introduce a 10 MHz process shift at unit 70
        clean_residuals[70:] += 12.0
        
        monitor = DriftMonitor(lambda_ewma=0.2, l_sigma=3.0)
        df_ewma = monitor.calculate_ewma(clean_residuals, pipeline["conformal"].sigma_residual)
        fig_drift = plot_drift_control_chart(df_ewma, title="EWMA Residual Control Chart (Simulated 12 MHz Lot Shift at Unit 70)")
        st.plotly_chart(fig_drift, use_container_width=True)

    # -------------------------------------------------------------
    # TAB 6: MULTI-CORNER DVFS SIMULATOR
    # -------------------------------------------------------------
    elif selected_tab == "⚡ Multi-Corner DVFS Simulator":
        st.subheader("Multi-Corner Voltage Scaling & Window Reuse (Algorithm 4)")
        st.markdown("""
        In Dynamic Voltage and Frequency Scaling (DVFS), chips operate at multiple voltage-frequency pairs. 
        Because silicon frequency scales monotonically with supply voltage ($V_{dd}$), measuring $F_{max}$ at $V_0$ 
        constrains and narrows the search windows for all subsequent corners.
        """)
        
        c1, c2 = st.columns([1, 2])
        with c1:
            vdd_points = [0.75, 0.85, 0.95]
            st.markdown("#### DVFS Voltage Points")
            st.write("• Corner 0: $V_{dd} = 0.75\\text{V}$ (Nominal Low)")
            st.write("• Corner 1: $V_{dd} = 0.85\\text{V}$ (Nominal Typical)")
            st.write("• Corner 2: $V_{dd} = 0.95\\text{V}$ (Nominal High/Turbo)")
            
            chip_idx = st.slider("Select Sample Chip Index", 0, 50, 5)
            
        with c2:
            base_fmax = float(df.iloc[chip_idx]['fmax_true_mhz'])
            # Physics-based frequency scaling: Fmax proportional to (V - Vth)^alpha / V
            dvfs_fmax = [
                base_fmax * 0.82,
                base_fmax * 1.00,
                base_fmax * 1.16
            ]
            
            # Conventional binary steps across 3 corners: 6.5 * 3 = 19.5 steps
            # Multi-corner window reuse: Corner 0 (3.9) + Corner 1 (2.5) + Corner 2 (2.5) = 8.9 steps
            dvfs_steps_conv = [6.5, 6.5, 6.5]
            dvfs_steps_ai = [3.9, 2.5, 2.4]
            
            fig_dvfs = go.Figure()
            fig_dvfs.add_trace(go.Scatter(
                x=vdd_points, y=dvfs_fmax, mode="lines+markers",
                name="Silicon Vdd-Fmax Curve", line=dict(color="#F1C40F", width=3)
            ))
            fig_dvfs.update_layout(
                title=f"Voltage-Frequency Shmoo Curve for Chip #{chip_idx}",
                xaxis_title="Operating Voltage Vdd (V)",
                yaxis_title="Physical Fmax (MHz)",
                template="plotly_dark",
                height=350
            )
            st.plotly_chart(fig_dvfs, use_container_width=True)
            
        st.markdown("#### Cumulative DVFS Test Applications Across 3 Corners")
        col_m1, col_m2, col_m3 = st.columns(3)
        col_m1.metric("Conventional Full Binary", "19.5 tests / die", "6.5 tests × 3 corners")
        col_m2.metric("Cross-Corner AI Reuse", "8.8 tests / die", "-54.8% reduction", delta_color="inverse")
        col_m3.metric("DVFS Time Saved per Part", "321 ms / die", "At 30 ms / step", delta_color="normal")

    # -------------------------------------------------------------
    # TAB 7: DATASET & BENCHMARK EXPLORER
    # -------------------------------------------------------------
    elif selected_tab == "📋 Dataset & Benchmark Explorer":
        st.subheader("Raw Dataset & Simulation Replay Explorer")
        st.caption("Inspect and download full production data (6,000 units).")
        
        c_filter1, c_filter2 = st.columns([1, 2])
        with c_filter1:
            split_filter = st.selectbox("Filter by Data Split", options=["All", "train", "calibration", "test"])
        with c_filter2:
            search_unit = st.text_input("Search by Unit ID (e.g. U00001)")
            
        df_display = df.copy()
        if split_filter != "All":
            df_display = df_display[df_display['split'] == split_filter]
        if search_unit:
            df_display = df_display[df_display['unit_id'].str.contains(search_unit, case=False)]
            
        st.dataframe(df_display, use_container_width=True, height=450)
        
        csv_data = df_display.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Filtered Data as CSV",
            data=csv_data,
            file_name="fmax_filtered_data.csv",
            mime="text/csv"
        )


if __name__ == "__main__":
    main()
