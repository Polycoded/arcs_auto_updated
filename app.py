import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from utils.data_loader import (
    load_csv,
    detect_datetime_column,
    detect_frequency,
    validate_timeseries,
    prepare_timeseries,
)
from utils.feature_extractor import compute_all_features
from utils.visualizer import (
    plot_timeseries,
    plot_decomposition,
    plot_acf_pacf,
    create_stats_cards,
    plot_trend_simple,
    plot_seasonality_simple,
)
from utils.rule_engine import get_recommendations
from utils.code_generator import generate_code_snippet
from utils.preprocessor import TimeSeriesPreprocessor
from utils.model_trainer import ModelTrainer
from utils.evidence_generator import (
    generate_trend_evidence,
    generate_seasonality_evidence,
    generate_stationarity_evidence,
    generate_autocorrelation_evidence,
    summarize_patterns,
)
from utils.model_io import save_model, load_model

# -----------------------------------------------------------------------------
# Page configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Time Series Algorithm Recommender",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------------------------------------------------------
# Custom CSS
# -----------------------------------------------------------------------------
st.markdown(
    """
<style>
    .main > div {
        padding-top: 2rem;
    }
    .stAlert {
        margin-top: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .best-model {
        border-left: 4px solid #2ecc71;
        background-color: #e8f8f5;
    }
    .evidence-box {
        background-color: #fff9e6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #f39c12;
        margin: 1rem 0;
    }
</style>
""",
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# Session state
# -----------------------------------------------------------------------------
for key, default in [
    ("analyzed", False),
    ("features", None),
    ("df", None),
    ("df_original", None),
    ("preprocessed", False),
    ("models_trained", False),
    ("training_results", None),
    ("best_model", None),
    ("current_task", "Forecasting"),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# -----------------------------------------------------------------------------
# Title & Intro
# -----------------------------------------------------------------------------
st.title("🤖 Time Series Algorithm Recommender")
st.markdown("### 🏆 AI-Powered Model Selection, Benchmarking, and Code Generation")
st.markdown(
    "Upload a time series dataset, get intelligent algorithm recommendations, **see real model performance**, "
    "and generate ready-to-use code — all in one place!"
)

# -----------------------------------------------------------------------------
# Sidebar – Data upload & configuration
# -----------------------------------------------------------------------------
st.sidebar.header("📁 Data Upload")

demo_option = st.sidebar.selectbox(
    "Choose a demo dataset or upload your own",
    ["Upload your own", "Airline Passengers (Monthly)", "Daily Sales", "Stock Prices"],
)

uploaded_file = None
if demo_option == "Upload your own":
    uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])
else:
    demo_paths = {
        "Airline Passengers (Monthly)": "data/sample_datasets/airline_passengers.csv",
        "Daily Sales": "data/sample_datasets/daily_sales.csv",
        "Stock Prices": "data/sample_datasets/stock_prices.csv",
    }
    path = demo_paths.get(demo_option)
    uploaded_file = path
    st.sidebar.success(f"✅ Loaded demo: {demo_option}")

# Task selection (used for preprocessing defaults)
st.sidebar.header("🎯 Task")
task_sidebar = st.sidebar.selectbox(
    "What do you want to do?",
    ["Forecasting", "Classification", "Anomaly Detection", "Clustering"],
)
st.session_state.current_task = task_sidebar

# -----------------------------------------------------------------------------
# Main logic – if data is provided
# -----------------------------------------------------------------------------
if uploaded_file is not None:
    # Load data
    if isinstance(uploaded_file, str):
        df, error = pd.read_csv(uploaded_file), None
    else:
        df, error = load_csv(uploaded_file)

    if error:
        st.error(f"Error loading file: {error}")
    else:
        st.success("✅ File loaded successfully!")

        # Preview
        with st.expander("📋 Data Preview", expanded=False):
            st.dataframe(df.head(10))
            st.write(f"**Shape:** {df.shape[0]} rows × {df.shape[1]} columns")

        # Column selection
        st.sidebar.header("⚙️ Columns")

        detected_time_col = detect_datetime_column(df)
        time_col = st.sidebar.selectbox(
            "Time/Date Column",
            options=df.columns.tolist(),
            index=df.columns.tolist().index(detected_time_col) if detected_time_col else 0,
        )

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if not numeric_cols:
            st.error("No numeric columns found. Please provide a numeric target column.")
            st.stop()

        target_col = st.sidebar.selectbox(
            "Target Column (series to model)",
            options=numeric_cols,
            index=0,
        )

        # Task-aware preprocessing options
        st.sidebar.header("🔧 Preprocessing")

        with st.sidebar.expander("Missing Values", expanded=False):
            missing_method = st.selectbox(
                "Method",
                ["ffill", "bfill", "interpolate", "mean", "drop"],
                help="Choose how to handle missing values in the target series.",
            )

        with st.sidebar.expander("Outliers", expanded=False):
            if task_sidebar == "Anomaly Detection":
                outlier_help = "For anomaly detection you usually keep raw outliers (none) or use Z-score gently."
                default_outlier_idx = 0
            else:
                outlier_help = "Clip or replace extreme values to stabilize models."
                default_outlier_idx = 1

            outlier_method = st.selectbox(
                "Method",
                ["none", "clip", "zscore"],
                index=default_outlier_idx,
                help=outlier_help,
            )
            outlier_threshold = st.slider("Threshold", 1.5, 5.0, 3.0, 0.5)

        with st.sidebar.expander("Scaling / Transform", expanded=False):
            scaling_method = st.selectbox(
                "Method",
                ["none", "standard", "minmax", "log"],
                index=0,
                help="Scaling is helpful for ML/deep models; log transform stabilizes variance.",
            )

        # Extra task-specific knobs
        if task_sidebar == "Forecasting":
            with st.sidebar.expander("Trend / Stationarity", expanded=False):
                apply_diff = st.checkbox(
                    "Apply differencing (make series more stationary)", value=False
                )
                diff_order = st.slider("Differencing order", 0, 2, 1)
        else:
            apply_diff = False
            diff_order = 0

        if task_sidebar == "Anomaly Detection":
            with st.sidebar.expander("Smoothing (optional)", expanded=False):
                smooth_window = st.slider("Rolling mean window", 1, 60, 1)
        else:
            smooth_window = 1

        # Analyze
        analyze_button = st.sidebar.button(
            "🔍 Analyze & Extract Features", type="primary", use_container_width=True
        )

        if analyze_button:
            with st.spinner("Analyzing time series and applying preprocessing..."):
                # Validate
                is_valid, errors = validate_timeseries(df, time_col, target_col)
                if not is_valid:
                    st.error("❌ Validation failed:")
                    for e in errors:
                        st.error(f"- {e}")
                    st.stop()

                # Prepare base series (sorted, indexed by time)
                df_base = prepare_timeseries(df, time_col, target_col)

                # Preprocess
                pre = TimeSeriesPreprocessor(df_base, target_col)
                pre.handle_missing_values(missing_method)
                pre.handle_outliers(outlier_method, outlier_threshold)
                pre.apply_scaling(scaling_method)
                if apply_diff and diff_order > 0:
                    pre.apply_differencing(diff_order)
                if smooth_window and smooth_window > 1 and task_sidebar == "Anomaly Detection":
                    pre.apply_rolling_smoothing(smooth_window)

                df_proc = pre.get_processed_data()
                preprocessing_log = pre.get_log()

                # Frequency
                freq_code, freq_name = detect_frequency(df, time_col)

                # Features
                features = compute_all_features(df_proc, target_col, freq_code)

                # Store in session state
                st.session_state.analyzed = True
                st.session_state.df = df_proc
                st.session_state.df_original = df_base
                st.session_state.features = features
                st.session_state.time_col = time_col
                st.session_state.target_col = target_col
                st.session_state.freq_code = freq_code
                st.session_state.preprocessing_log = preprocessing_log
                st.session_state.preprocessed = True
                st.session_state.models_trained = False
                st.success("✅ Analysis and feature extraction complete!")
                st.rerun()

# -----------------------------------------------------------------------------
# When analysis is done – show tabs
# -----------------------------------------------------------------------------
if st.session_state.analyzed:
    df_proc = st.session_state.df
    df_orig = st.session_state.df_original
    features = st.session_state.features
    time_col = st.session_state.time_col
    target_col = st.session_state.target_col
    freq_code = st.session_state.freq_code

    # Preprocessing log
    if st.session_state.preprocessed:
        with st.expander("✅ Preprocessing Steps Applied", expanded=False):
            for log_line in st.session_state.preprocessing_log:
                st.write(f"• {log_line}")

    tab_vis, tab_stats, tab_recs, tab_train, tab_code = st.tabs(
        ["📊 Visualization", "🔬 Evidence & Summary", "🎯 Recommendations", "🏋️ Benchmark Models", "💻 Code"]
    )

    # -------------------------------------------------------------------------
    # TAB 1: Visualization
    # -------------------------------------------------------------------------
    with tab_vis:
        st.header("📈 Time Series Visualization")

        # Original vs Processed comparison
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Original Series")
            fig_o = plot_timeseries(df_orig, target_col, "Original data")
            st.plotly_chart(fig_o, use_container_width=True)

        with c2:
            st.subheader("Processed Series")
            fig_p = plot_timeseries(df_proc, target_col, "Processed data")
            st.plotly_chart(fig_p, use_container_width=True)

        st.markdown("<br><br>", unsafe_allow_html=True)

        # Simple Trend View (for layman)
        st.subheader("📈 Simple Trend View (for non-technical users)")
        st.caption("The blue line shows the overall direction by smoothing out short-term fluctuations.")
        
        trend_fig = plot_trend_simple(df_proc, target_col, window=7)
        st.plotly_chart(trend_fig, use_container_width=True)

        st.markdown("<br><br>", unsafe_allow_html=True)

        # Simple Seasonality View
        st.subheader("🔄 Simple Seasonality View")
        st.caption("This shows if certain days/months consistently have higher or lower values.")
        
        season_fig = plot_seasonality_simple(df_proc, time_col, target_col, freq_code)
        st.plotly_chart(season_fig, use_container_width=True)

        st.markdown("<br><br>", unsafe_allow_html=True)

        # Advanced Decomposition (for power users)
        st.subheader("🔍 Advanced Decomposition (for technical users)")
        with st.expander("ℹ️ What is decomposition?", expanded=False):
            st.write("""
            **Decomposition** breaks down the time series into:
            - **Trend**: Long-term increase or decrease
            - **Seasonal**: Regular repeating patterns (e.g., weekly, yearly)
            - **Residual**: Random noise or irregularities
            """)
        
        fig_dec = plot_decomposition(df_proc[target_col], freq_code)
        if fig_dec:
            st.plotly_chart(fig_dec, use_container_width=True)
        else:
            st.info("⚠️ Not enough data to perform seasonal decomposition (need at least 2 complete seasonal cycles).")

        st.markdown("<br><br>", unsafe_allow_html=True)

        # ACF/PACF (for technical users)
        st.subheader("🔗 Autocorrelation Analysis (ACF & PACF)")
        with st.expander("ℹ️ What is autocorrelation?", expanded=False):
            st.write("""
            **Autocorrelation** measures how much past values influence current values.
            - **ACF (Autocorrelation Function)**: Shows correlation with all past lags
            - **PACF (Partial Autocorrelation Function)**: Shows direct correlation removing intermediate effects
            
            Bars crossing the red dashed lines indicate **statistically significant** relationships.
            """)
        
        fig_acf = plot_acf_pacf(df_proc[target_col])
        if fig_acf:
            st.plotly_chart(fig_acf, use_container_width=True)
        else:
            st.info("⚠️ Could not compute ACF/PACF for this series.")

    # -------------------------------------------------------------------------
    # TAB 2: Statistical Evidence & Plain-English Summary
    # -------------------------------------------------------------------------
    with tab_stats:
        st.header("🔬 Statistical Evidence & Plain-English Summary")

        # Cards
        st.subheader("📊 Diagnostic Snapshot")
        cards = create_stats_cards(features)
        cols = st.columns(3)
        for i, card in enumerate(cards):
            with cols[i % 3]:
                st.markdown(f"#### {card['title']}")
                for item in card["items"]:
                    if item:
                        st.write(f"• {item}")
                st.markdown("---")

        # Evidence sections
        st.subheader("📈 Trend Evidence")
        with st.container():
            st.markdown('<div class="evidence-box">', unsafe_allow_html=True)
            for line in generate_trend_evidence(df_proc[target_col]):
                st.markdown(line)
            st.markdown("</div>", unsafe_allow_html=True)

        st.subheader("🔄 Seasonality Evidence")
        if features.get("has_seasonality") and features.get("seasonal_period"):
            with st.container():
                st.markdown('<div class="evidence-box">', unsafe_allow_html=True)
                for line in generate_seasonality_evidence(
                    df_proc[target_col], features["seasonal_period"]
                ):
                    st.markdown(line)
                st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.info("No strong seasonal component detected.")

        st.subheader("⚖️ Stationarity Evidence")
        with st.container():
            st.markdown('<div class="evidence-box">', unsafe_allow_html=True)
            for line in generate_stationarity_evidence(df_proc[target_col]):
                st.markdown(line)
            st.markdown("</div>", unsafe_allow_html=True)

        st.subheader("🔗 Autocorrelation Evidence")
        with st.container():
            st.markdown('<div class="evidence-box">', unsafe_allow_html=True)
            for line in generate_autocorrelation_evidence(df_proc[target_col]):
                st.markdown(line)
            st.markdown("</div>", unsafe_allow_html=True)

        st.subheader("🧠 Plain-English Summary")
        for line in summarize_patterns(features):
            st.write(f"• {line}")

        with st.expander("📋 All extracted features (JSON)", expanded=False):
            st.json(features)

    # -------------------------------------------------------------------------
    # TAB 3: Recommendations (AI + Rule-Based Hybrid)
    # -------------------------------------------------------------------------
    with tab_recs:
        st.header("🎯 Algorithm Recommendations")
        st.markdown("Get intelligent algorithm suggestions using **AI meta-learning** (80% weight) + **rule-based heuristics** (20% weight).")

        # Task selection
        task = st.selectbox(
            "Select Task",
            ["Forecasting", "Classification", "Anomaly Detection", "Clustering"],
            index=["Forecasting", "Classification", "Anomaly Detection", "Clustering"].index(
                st.session_state.current_task
            ),
        )
        st.session_state.current_task = task

        st.markdown("---")

        # -------------------------------------------------------------------------
        # AI Meta-Model Prediction
        # -------------------------------------------------------------------------
        st.subheader("🤖 AI Meta-Model Prediction")
        st.caption("Trained on 300+ time series with known best-performing algorithms.")

        from utils.meta_predictor import MetaPredictor
        
        meta = MetaPredictor()
        ai_prediction = None
        ai_confidence = 0.0
        ai_available = False

        if meta.enabled:
            # Extract features in same format as training
            try:
                meta_features = {
                    'length': features.get('length', 0),
                    'mean': features.get('mean', 0),
                    'std': features.get('std', 0),
                    'cv': features.get('coefficient_of_variation', 0),
                    'min': features.get('min', 0),
                    'max': features.get('max', 0),
                    'range': features.get('range', 0),
                    'skewness': features.get('skewness', 0),
                    'kurtosis': features.get('kurtosis', 0),
                    'has_trend': int(features.get('has_trend', False)),
                    'trend_strength': features.get('trend_strength', 0),
                    'trend_slope': features.get('trend_slope', 0),
                    'has_seasonality': int(features.get('has_seasonality', False)),
                    'seasonal_period': features.get('seasonal_period', 0) or 0,
                    'seasonal_strength': features.get('seasonal_strength', 0),
                    'is_stationary': int(features.get('is_stationary', False)),
                    'adf_pvalue': features.get('adf_pvalue', 1.0),
                    'autocorr_lag1': features.get('autocorrelation_lag1', 0),
                    'pacf_lag1': features.get('pacf_lag1', 0),
                    'entropy': features.get('entropy', 0),
                    'complexity_score': features.get('data_complexity_score', 0),
                    'volatility': features.get('volatility', 0),
                    'diff_needed': features.get('differencing_needed', 0) or 0,
                    'missing_pct': features.get('missing_percentage', 0),
                    'outlier_pct': features.get('outliers_percentage', 0),
                    'mean_abs_change': np.mean(np.abs(np.diff(df_proc[target_col].dropna()))),
                    'std_1st_diff': np.std(np.diff(df_proc[target_col].dropna())),
                    'zero_crossing_rate': np.sum(np.diff(np.sign(df_proc[target_col].dropna() - df_proc[target_col].mean())) != 0) / len(df_proc),
                }
                
                ai_prediction, ai_confidence = meta.predict(meta_features)
                
                if ai_prediction and ai_confidence > 0.3:  # Minimum confidence threshold
                    ai_available = True
                    
                    # Display AI prediction
                    conf_color = "#2ecc71" if ai_confidence > 0.6 else "#f39c12" if ai_confidence > 0.4 else "#e74c3c"
                    
                    st.markdown(
                        f"""
<div style="background-color: {conf_color}20; padding: 1.5rem; border-radius: 0.5rem; border-left: 4px solid {conf_color};">
    <h3 style="margin: 0; color: {conf_color};">🤖 {ai_prediction}</h3>
    <p style="margin: 0.5rem 0 0 0;"><strong>AI Confidence:</strong> {ai_confidence*100:.1f}%</p>
    <p style="margin: 0.5rem 0 0 0; font-size: 0.9em;">Model trained with {meta.config['accuracy']*100:.1f}% test accuracy on diverse patterns</p>
</div>
""",
                        unsafe_allow_html=True,
                    )
                    
                    if ai_confidence < 0.6:
                        st.warning(
                            f"⚠️ AI confidence is moderate ({ai_confidence*100:.1f}%). "
                            "Rule-based recommendations will have higher weight."
                        )
                else:
                    st.info("ℹ️ AI prediction confidence too low. Using rule-based approach instead.")
                    
            except Exception as e:
                st.error(f"❌ Error generating AI prediction: {e}")
                ai_available = False
        else:
            st.info(
                "ℹ️ **Meta-model not available.** Using rule-based recommendations only.\n\n"
                "To enable AI predictions, run: `python train_meta_model.py`"
            )

        st.markdown("---")

        # -------------------------------------------------------------------------
        # Rule-Based Recommendations
        # -------------------------------------------------------------------------
        st.subheader("📋 Rule-Based Recommendations")
        st.caption("Traditional heuristics based on statistical tests and time series characteristics.")

        with st.spinner("Computing rule-based recommendations..."):
            recs = get_recommendations(features, task)

        rule_primary = None
        rule_confidence = 0.0

        if recs["primary"]:
            rule_primary = recs["primary"]["name"]
            rule_confidence = recs["primary"]["confidence"] / 100.0  # Convert to 0-1
            
            st.markdown(
                f"""
<div style="background-color: #3498db20; padding: 1.5rem; border-radius: 0.5rem; border-left: 4px solid #3498db;">
    <h3 style="margin: 0; color: #3498db;">📋 {rule_primary}</h3>
    <p style="margin: 0.5rem 0 0 0;"><strong>Rule Confidence:</strong> {rule_confidence*100:.0f}%</p>
    <p style="margin: 0.5rem 0 0 0;"><strong>Category:</strong> {recs['primary']['category']}</p>
</div>
""",
                unsafe_allow_html=True,
            )

        st.markdown("---")

        # -------------------------------------------------------------------------
        # Hybrid Final Recommendation (80% AI + 20% Rule)
        # -------------------------------------------------------------------------
        st.subheader("🎯 Final Hybrid Recommendation")
        st.caption("Intelligent combination: 80% weight to AI model, 20% to rule-based heuristics.")

        final_recommendation = None
        final_confidence = 0.0
        reasoning = []

        if ai_available and ai_confidence >= 0.6:
            # HIGH AI CONFIDENCE: Use AI primarily (80% weight)
            final_recommendation = ai_prediction
            final_confidence = 0.8 * ai_confidence + 0.2 * rule_confidence
            reasoning.append(f"✅ AI model has high confidence ({ai_confidence*100:.1f}%)")
            reasoning.append(f"🎯 **Primary recommendation:** {ai_prediction} (AI-driven)")
            
            if rule_primary and rule_primary != ai_prediction:
                reasoning.append(f"⚠️ Note: Rule-based system suggested {rule_primary}, but AI has stronger evidence")
            elif rule_primary == ai_prediction:
                reasoning.append(f"✅ Rule-based system agrees: {rule_primary}")
                final_confidence = min(0.95, final_confidence + 0.1)  # Boost confidence when both agree
            
        elif ai_available and ai_confidence >= 0.4:
            # MODERATE AI CONFIDENCE: Weighted average
            if rule_primary == ai_prediction:
                # Both agree
                final_recommendation = ai_prediction
                final_confidence = 0.8 * ai_confidence + 0.2 * rule_confidence + 0.1
                reasoning.append(f"✅ Both AI ({ai_confidence*100:.1f}%) and rules agree on {ai_prediction}")
                reasoning.append(f"🎯 **High consensus recommendation**")
            else:
                # Disagree - use weighted score
                ai_score = 0.8 * ai_confidence
                rule_score = 0.2 * rule_confidence
                
                if ai_score > rule_score:
                    final_recommendation = ai_prediction
                    final_confidence = ai_score
                    reasoning.append(f"🤖 AI suggestion ({ai_prediction}, {ai_confidence*100:.1f}%) has higher weighted score")
                    reasoning.append(f"📋 Rule-based suggested {rule_primary} as alternative")
                else:
                    final_recommendation = rule_primary
                    final_confidence = 0.5 * ai_confidence + 0.5 * rule_confidence
                    reasoning.append(f"📋 Using rule-based recommendation due to low AI confidence")
                    reasoning.append(f"🤖 AI suggested {ai_prediction} but with only {ai_confidence*100:.1f}% confidence")
        
        else:
            # LOW/NO AI CONFIDENCE: Use rule-based
            final_recommendation = rule_primary
            final_confidence = rule_confidence
            reasoning.append(f"📋 Using rule-based recommendation (AI unavailable or low confidence)")
            if ai_prediction:
                reasoning.append(f"⚠️ AI prediction ({ai_prediction}) had insufficient confidence ({ai_confidence*100:.1f}%)")

        # Display final recommendation
        if final_recommendation:
            conf_color = "#2ecc71" if final_confidence > 0.7 else "#f39c12" if final_confidence > 0.5 else "#e74c3c"
            
            st.markdown(
                f"""
<div class="metric-card best-model" style="border-left: 5px solid {conf_color}; background: linear-gradient(135deg, {conf_color}15, {conf_color}05);">
    <h2 style="color: {conf_color}; margin: 0;">🏆 {final_recommendation}</h2>
    <p style="margin: 0.5rem 0; font-size: 1.1em;"><strong>Final Confidence:</strong> {final_confidence*100:.1f}%</p>
    <hr style="margin: 1rem 0; border: none; border-top: 1px solid #ddd;">
    <p style="margin: 0; font-size: 0.95em;"><strong>Decision Logic:</strong></p>
</div>
""",
                unsafe_allow_html=True,
            )
            
            for line in reasoning:
                st.write(f"• {line}")

            st.markdown("---")

            # Show details of final recommendation
            st.subheader(f"📖 About {final_recommendation}")
            
            # Find details from recommendations
            if recs["primary"] and recs["primary"]["name"] == final_recommendation:
                details = recs["primary"]
            else:
                details = next(
                    (alt for alt in recs.get("alternatives", []) if alt["name"] == final_recommendation),
                    None
                )
            
            if details:
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("**Why this algorithm?**")
                    for r in details["reasons"]:
                        st.write(f"✓ {r}")
                with c2:
                    with st.expander("👍 Pros", expanded=True):
                        for p in details["pros"]:
                            st.write(f"• {p}")
                    with st.expander("👎 Cons", expanded=True):
                        for c in details["cons"]:
                            st.write(f"• {c}")

        st.markdown("---")

        # -------------------------------------------------------------------------
        # Alternative Options
        # -------------------------------------------------------------------------
        if recs["alternatives"]:
            st.subheader("🥈 Alternative Options")
            st.caption("Consider these if the primary recommendation doesn't work well.")
            
            for i, alt in enumerate(recs["alternatives"]):
                # Don't show final recommendation again in alternatives
                if alt["name"] == final_recommendation:
                    continue
                    
                with st.expander(f"{i+2}. {alt['name']} (rule confidence: {alt['confidence']:.0f}%)"):
                    st.markdown(f"**Category:** {alt['category']}")
                    st.markdown("**Reasons:**")
                    for r in alt["reasons"]:
                        st.write(f"• {r}")
                    c1, c2 = st.columns(2)
                    with c1:
                        st.markdown("**Pros:**")
                        for p in alt["pros"]:
                            st.write(f"✓ {p}")
                    with c2:
                        st.markdown("**Cons:**")
                        for c in alt["cons"]:
                            st.write(f"✗ {c}")

        # -------------------------------------------------------------------------
        # Hybrid Strategy Suggestion
        # -------------------------------------------------------------------------
        if recs.get("hybrid_suggestion"):
            st.markdown("---")
            h = recs["hybrid_suggestion"]
            st.subheader("🚀 Advanced: Hybrid Strategy")
            st.info(
                f"**Combination:** {h['combo']}\n\n"
                f"**Why:** {h['rationale']}\n\n"
                f"**How:** {h['approach']}"
            )

        st.markdown("---")

        # -------------------------------------------------------------------------
        # Pattern Summary (Plain English)
        # -------------------------------------------------------------------------
        st.subheader("🔍 Your Data Pattern Summary")
        st.caption("What we detected in your time series (in plain English):")
        
        for line in summarize_patterns(features):
            st.write(f"• {line}")

        # -------------------------------------------------------------------------
        # Comparison Table: AI vs Rule-Based
        # -------------------------------------------------------------------------
        st.markdown("---")
        st.subheader("📊 Prediction Comparison")
        
        comparison_data = []
        
        if ai_available:
            comparison_data.append({
                "Method": "🤖 AI Meta-Model",
                "Prediction": ai_prediction or "N/A",
                "Confidence": f"{ai_confidence*100:.1f}%",
                "Weight": "80%",
                "Basis": "Trained on 300+ benchmarked series"
            })
        
        if rule_primary:
            comparison_data.append({
                "Method": "📋 Rule-Based",
                "Prediction": rule_primary,
                "Confidence": f"{rule_confidence*100:.0f}%",
                "Weight": "20%",
                "Basis": "Statistical tests + heuristics"
            })
        
        comparison_data.append({
            "Method": "🎯 Final Hybrid",
            "Prediction": final_recommendation or "N/A",
            "Confidence": f"{final_confidence*100:.1f}%",
            "Weight": "100%",
            "Basis": "Weighted combination"
        })
        
        comparison_df = pd.DataFrame(comparison_data)
        st.dataframe(comparison_df, use_container_width=True, hide_index=True)

        # -------------------------------------------------------------------------
        # Quick Actions
        # -------------------------------------------------------------------------
        st.markdown("---")
        st.subheader("⚡ Quick Actions")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🏋️ Train This Algorithm", use_container_width=True):
                st.session_state.quick_train_algo = final_recommendation
                st.info(f"Go to the 'Benchmark Models' tab and select '{final_recommendation}' to train it!")
        
        with col2:
            if st.button("💻 Generate Code", use_container_width=True):
                st.session_state.quick_code_algo = final_recommendation
                st.info(f"Go to the 'Code' tab to generate starter code for '{final_recommendation}'!")
        
        with col3:
            if st.button("📊 See All Algorithms", use_container_width=True):
                st.session_state.show_all_algos = True
                st.rerun()

    # -------------------------------------------------------------------------
    # TAB 4: Model Training & Benchmarking
    # -------------------------------------------------------------------------
    with tab_train:
        st.header("🏋️ Train & Benchmark Forecasting Models")

        st.markdown(
            "Run several models (including **hybrid algorithms**), compare MAE/RMSE/MAPE/R², "
            "and visualize forecasts side-by-side."
        )

        # Configuration
        c1, c2, c3 = st.columns(3)
        with c1:
            train_pct = st.slider("Train size (%)", 60, 90, 80, 5)
        with c2:
            mode = st.radio(
                "Which models to train?",
                ["Top 3 (fast)", "All main + hybrids"],
                index=0,
            )
        with c3:
            train_len = int(len(df_proc) * train_pct / 100)
            test_len = len(df_proc) - train_len
            st.metric("Series length", len(df_proc))
            st.metric("Train", train_len)
            st.metric("Test", test_len)

        st.markdown("---")

        # -------------------------------------------------------------------------
        # Section 1: Run a single preferred algorithm
        # -------------------------------------------------------------------------
        st.subheader("🎛 Run a Single Preferred Algorithm")
        st.caption("Quick test: run just one algorithm with current settings.")

        # Build list of available algorithms
        preferred_algos = [
            "SARIMA",
            "ARIMA",
            "Prophet",
            "Exponential Smoothing",
            "XGBoost",
            "STL + ETS (Hybrid)",
            "ARIMA + ETS + GBDT (Hybrid)",
        ]

        # Initialize with best recommended algorithm for Forecasting
        initial_index = 0
        if st.session_state.current_task == "Forecasting":
            recs_pref = get_recommendations(features, "Forecasting")
            primary_algo = recs_pref["primary"]["name"] if recs_pref.get("primary") else None
            if primary_algo in preferred_algos:
                initial_index = preferred_algos.index(primary_algo)

        user_algo = st.selectbox(
            "Pick algorithm to run",
            preferred_algos,
            index=initial_index,
            help="Pre-selected based on our recommendation for your data."
        )

        run_single = st.button("▶ Run selected algorithm", type="secondary")

        if run_single:
            trainer_single = ModelTrainer(df_proc, target_col, train_size=train_pct / 100.0)
            
            # Mapping algorithms to trainer methods
            mapping = {
                "SARIMA": lambda: trainer_single.train_sarima(features.get("seasonal_period", 12)),
                "ARIMA": lambda: trainer_single.train_arima(),
                "Prophet": lambda: trainer_single.train_prophet(),
                "Exponential Smoothing": lambda: trainer_single.train_ets(features.get("seasonal_period", 12)),
                "XGBoost": lambda: trainer_single.train_xgboost(features.get("recommended_lags", [1, 2, 3])),
                "STL + ETS (Hybrid)": lambda: trainer_single.train_stl_ets(features.get("seasonal_period", 12)),
                "ARIMA + ETS + GBDT (Hybrid)": lambda: trainer_single.train_arima_ets_gbdt(features.get("seasonal_period", 12)),
            }

            with st.spinner(f"Training {user_algo}..."):
                res = mapping[user_algo]()

            if res.get("status") == "success":
                st.success(f"✅ {user_algo} trained successfully!")
                
                # Show metrics
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("MAE", f"{res['MAE']:.4f}")
                col2.metric("RMSE", f"{res['RMSE']:.4f}")
                col3.metric("MAPE", f"{res['MAPE']:.2f}%")
                col4.metric("R²", f"{res['R²']:.4f}")
                
                # Show forecast plot
                test_index = df_proc.index[trainer_single.train_size:]
                actual = df_proc[target_col].iloc[trainer_single.train_size:]
                preds = np.asarray(res["predictions"])

                fig_single = go.Figure()
                fig_single.add_trace(go.Scatter(
                    x=test_index, y=actual,
                    mode="lines", name="Actual",
                    line=dict(color="black", width=2)
                ))
                fig_single.add_trace(go.Scatter(
                    x=test_index, y=preds,
                    mode="lines", name=user_algo,
                    line=dict(color="#e74c3c", width=2, dash="dash")
                ))
                fig_single.update_layout(
                    title=f"Forecast vs Actual - {user_algo}",
                    xaxis_title="Time",
                    yaxis_title="Value",
                    hovermode="x unified",
                    height=450,
                    template="plotly_white"
                )
                st.plotly_chart(fig_single, use_container_width=True)
            else:
                st.error(f"❌ {user_algo} failed: {res.get('error', 'Unknown error')}")

        st.markdown("---")

        # -------------------------------------------------------------------------
        # Section 2: Full Benchmark (multiple models)
        # -------------------------------------------------------------------------
        st.subheader("🏆 Full Benchmark: Compare Multiple Models")
        st.caption("Train multiple algorithms and see which performs best on your data.")

        train_btn = st.button("🚀 Run Full Benchmark", type="primary", use_container_width=True)

        if train_btn:
            trainer = ModelTrainer(df_proc, target_col, train_size=train_pct / 100.0)

            # Define which models to train based on mode
            if mode == "Top 3 (fast)":
                models_to_train = [
                    ("SARIMA", lambda: trainer.train_sarima(features.get("seasonal_period", 12))),
                    ("Prophet", lambda: trainer.train_prophet()),
                    ("XGBoost", lambda: trainer.train_xgboost(features.get("recommended_lags", [1, 2, 3]))),
                ]
            else:  # All main + hybrids
                models_to_train = [
                    ("SARIMA", lambda: trainer.train_sarima(features.get("seasonal_period", 12))),
                    ("ARIMA", lambda: trainer.train_arima()),
                    ("Prophet", lambda: trainer.train_prophet()),
                    ("Exponential Smoothing", lambda: trainer.train_ets(features.get("seasonal_period", 12))),
                    ("XGBoost", lambda: trainer.train_xgboost(features.get("recommended_lags", [1, 2, 3]))),
                    ("STL + ETS (Hybrid)", lambda: trainer.train_stl_ets(features.get("seasonal_period", 12))),
                    ("ARIMA + ETS + GBDT (Hybrid)", lambda: trainer.train_arima_ets_gbdt(features.get("seasonal_period", 12))),
                ]

            # Train models with progress bar
            results = []
            progress = st.progress(0)
            status = st.empty()

            for i, (name, fn) in enumerate(models_to_train):
                status.text(f"Training {name}... ({i+1}/{len(models_to_train)})")
                res = fn()
                results.append(res)
                progress.progress((i + 1) / len(models_to_train))

            status.text("✅ All models trained!")
            progress.empty()

            # Separate successful and failed
            success = [r for r in results if r.get("status") == "success"]
            failed = [r for r in results if r.get("status") == "failed"]

            # Show failed models
            if failed:
                with st.expander("⚠️ Models that failed to train", expanded=False):
                    for r in failed:
                        st.error(f"**{r['name']}**: {r.get('error', 'Unknown error')}")
                    st.info(
                        "**Note on Prophet errors:** If you see `'stan_backend'` errors, this is a known "
                        "compatibility issue. Try: `pip install prophet==1.1.5 cmdstanpy==1.1.0` "
                        "and use Python 3.9 or 3.10."
                    )

            if success:
                st.session_state.models_trained = True
                st.session_state.training_results = success

                st.markdown("---")
                st.subheader("📊 Performance Comparison Table")

                # Build metrics dataframe
                metrics_df = pd.DataFrame(
                    [
                        {
                            "Model": r["name"],
                            "MAE": r["MAE"],
                            "RMSE": r["RMSE"],
                            "MAPE (%)": r["MAPE"],
                            "R²": r["R²"],
                        }
                        for r in success
                    ]
                ).sort_values("MAE")

                best_model_name = metrics_df.iloc[0]["Model"]
                st.session_state.best_model = best_model_name

                st.markdown(f"**🏆 Best model (lowest MAE): {best_model_name}**")

                # Style the table to highlight best model
                def highlight(row):
                    color = "background-color: #d4edda" if row["Model"] == best_model_name else ""
                    return [color] * len(row)

                styled_df = metrics_df.style.apply(highlight, axis=1).format(
                    {"MAE": "{:.4f}", "RMSE": "{:.4f}", "MAPE (%)": "{:.2f}", "R²": "{:.4f}"}
                )

                st.dataframe(styled_df, use_container_width=True)

                st.markdown("---")
                st.subheader("📈 Metrics Comparison (Visual)")

                # Create subplots for all metrics
                fig = make_subplots(
                    rows=2,
                    cols=2,
                    subplot_titles=(
                        "MAE (lower is better)",
                        "RMSE (lower is better)",
                        "MAPE % (lower is better)",
                        "R² (higher is better)",
                    ),
                )

                colors = [
                    "#2ecc71" if m == best_model_name else "#3498db" for m in metrics_df["Model"]
                ]

                # Add bars to subplots
                fig.add_trace(
                    go.Bar(x=metrics_df["Model"], y=metrics_df["MAE"], marker_color=colors, showlegend=False),
                    row=1,
                    col=1,
                )
                fig.add_trace(
                    go.Bar(x=metrics_df["Model"], y=metrics_df["RMSE"], marker_color=colors, showlegend=False),
                    row=1,
                    col=2,
                )
                fig.add_trace(
                    go.Bar(x=metrics_df["Model"], y=metrics_df["MAPE (%)"], marker_color=colors, showlegend=False),
                    row=2,
                    col=1,
                )
                fig.add_trace(
                    go.Bar(x=metrics_df["Model"], y=metrics_df["R²"], marker_color=colors, showlegend=False),
                    row=2,
                    col=2,
                )

                fig.update_layout(height=650, showlegend=False, template="plotly_white")
                st.plotly_chart(fig, use_container_width=True)

                st.markdown("---")
                st.subheader("🔮 Forecast Comparison: All Models vs Actual")

                # Plot actual values and all model predictions
                test_index = df_proc.index[trainer.train_size:]
                actual = df_proc[target_col].iloc[trainer.train_size:]

                fig_fc = go.Figure()

                # Actual values (thick black line)
                fig_fc.add_trace(
                    go.Scatter(
                        x=test_index,
                        y=actual,
                        mode="lines",
                        name="Actual",
                        line=dict(color="black", width=3),
                    )
                )

                # Model predictions (dashed colored lines)
                color_map = {
                    "SARIMA": "#e74c3c",
                    "ARIMA": "#9b59b6",
                    "Prophet": "#3498db",
                    "Exponential Smoothing": "#1abc9c",
                    "XGBoost": "#f39c12",
                    "STL + ETS (Hybrid)": "#e67e22",
                    "ARIMA + ETS + GBDT (Hybrid)": "#16a085",
                }

                for r in success:
                    preds = np.asarray(r["predictions"])
                    if len(preds) == len(test_index):
                        # Highlight best model with thicker line
                        width = 3 if r["name"] == best_model_name else 1.5
                        fig_fc.add_trace(
                            go.Scatter(
                                x=test_index,
                                y=preds,
                                mode="lines",
                                name=r["name"],
                                line=dict(
                                    color=color_map.get(r["name"], "#95a5a6"),
                                    width=width,
                                    dash="dash" if r["name"] != best_model_name else "dot",
                                ),
                            )
                        )

                fig_fc.update_layout(
                    title="Model Predictions vs Actual Values (Test Set)",
                    xaxis_title="Time",
                    yaxis_title="Value",
                    hovermode="x unified",
                    height=500,
                    template="plotly_white",
                )
                st.plotly_chart(fig_fc, use_container_width=True)

                st.markdown("---")

                # -------------------------------------------------------------------------
                # Save best model
                # -------------------------------------------------------------------------
                st.subheader("💾 Save Best Model")
                st.write(f"Best model: **{best_model_name}** (MAE: {metrics_df.iloc[0]['MAE']:.4f})")

                best_record = next((r for r in success if r["name"] == best_model_name), None)
                if best_record and "model_obj" in best_record:
                    if st.button("💾 Save best model to disk"):
                        path = save_model(best_record["model_obj"], best_model_name)
                        st.success(f"✅ Model saved to: `{path}`")
                        st.info("You can reload this model later using the section below.")
                else:
                    st.info("ℹ️ This model type does not expose a serializable object yet.")

                # Download results CSV
                st.markdown("---")
                st.subheader("📥 Export Results")
                csv_data = metrics_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download metrics as CSV",
                    data=csv_data,
                    file_name="model_comparison_results.csv",
                    mime="text/csv",
                )

        st.markdown("---")

        # -------------------------------------------------------------------------
        # Load and test saved model
        # -------------------------------------------------------------------------
        st.subheader("📂 Load Saved Model & Test on New Data")
        st.caption("Load a previously saved model and run it on a new test CSV.")

        load_name = st.text_input(
            "Saved model name (e.g., `sarima`, `xgboost`)",
            value=st.session_state.best_model.replace(" ", "_").lower() if st.session_state.best_model else "",
            help="Enter the exact name used when saving (spaces replaced with underscores)."
        )
        new_test_file = st.file_uploader(
            "Upload new test CSV (must have same time & target columns)",
            type=["csv"],
            key="test_file_uploader",
        )

        if st.button("🔄 Run saved model on new data"):
            try:
                model_loaded = load_model(load_name)
                if new_test_file is None:
                    st.warning("⚠️ Please upload a test dataset first.")
                else:
                    df_new, err = load_csv(new_test_file)
                    if err:
                        st.error(f"Error loading test file: {err}")
                    else:
                        # Prepare new data
                        df_new[time_col] = pd.to_datetime(df_new[time_col])
                        df_new = df_new.sort_values(time_col).set_index(time_col)

                        # Try forecasting (works for statsmodels models)
                        if hasattr(model_loaded, "forecast"):
                            steps = len(df_new)
                            preds = model_loaded.forecast(steps=steps)
                            comp = pd.DataFrame(
                                {
                                    "Actual": df_new[target_col].values,
                                    "Predicted": np.asarray(preds),
                                },
                                index=df_new.index,
                            )
                            st.success("✅ Predictions generated!")
                            st.line_chart(comp)
                        else:
                            st.info(
                                "ℹ️ This model type requires custom prediction logic. "
                                "Currently only statsmodels-based models (ARIMA, SARIMA, ETS) support direct forecasting."
                            )
            except FileNotFoundError:
                st.error(f"❌ No saved model found with name: `{load_name}`")
            except Exception as e:
                st.error(f"❌ Failed to load or run model: {e}")

    # -------------------------------------------------------------------------
    # TAB 5: Code Generator
    # -------------------------------------------------------------------------
    with tab_code:
        st.header("💻 Code Generator")

        # Need latest recommendations again for list of algorithms
        recs_for_code = get_recommendations(features, st.session_state.current_task)
        algos = []
        if recs_for_code["primary"]:
            algos.append(recs_for_code["primary"]["name"])
        algos.extend([a["name"] for a in recs_for_code["alternatives"]])

        if not algos:
            st.info("Run recommendations first to generate code.")
        else:
            algo_sel = st.selectbox("Choose algorithm to generate starter code", algos)
            if st.button("🎨 Generate Code", type="primary"):
                code = generate_code_snippet(
                    algo_sel, features, st.session_state.current_task, time_col, target_col
                )
                st.markdown(f"### Code for: {algo_sel}")
                st.code(code, language="python")
                st.download_button(
                    label="📥 Download code file",
                    data=code,
                    file_name=f"{algo_sel.replace(' ', '_').lower()}_starter.py",
                    mime="text/plain",
                )
                
                st.markdown("---")
                st.markdown("### 💡 Next Steps")
                st.info("""
                1. Copy or download the code above
                2. Install required libraries: `pip install [library_name]`
                3. Adjust hyperparameters based on your needs
                4. Run in Jupyter notebook or Python script
                5. Validate with proper cross-validation
                """)

else:
    # -----------------------------------------------------------------------------
    # Landing screen
    # -----------------------------------------------------------------------------
    st.info("👆 Upload a CSV file or select a demo dataset from the sidebar to get started.")

    st.markdown("---")
    st.markdown("### 🎯 How This App Works")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown("#### 1️⃣ Upload")
        st.write("Provide a time series CSV or pick a demo.")
    with c2:
        st.markdown("#### 2️⃣ Analyze")
        st.write("We profile the series and detect patterns automatically.")
    with c3:
        st.markdown("#### 3️⃣ Recommend & Train")
        st.write("We suggest algorithms, then **actually train and compare** them.")
    with c4:
        st.markdown("#### 4️⃣ Deploy")
        st.write("Generate production-ready code or save models.")

    st.markdown("---")
    st.markdown("### ✨ Key Features")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        **📊 Analysis**
        - Automatic trend detection
        - Seasonality identification
        - Stationarity tests
        - Plain-English summaries
        
        **🤖 Smart Recommendations**
        - AI meta-learning (80% weight)
        - Rule-based heuristics (20% weight)
        - Hybrid confidence scoring
        """)
    
    with col2:
        st.markdown("""
        **🏋️ Real Model Training**
        - Actual benchmarking with metrics
        - Hybrid algorithms (STL+ETS, ARIMA+GBDT)
        - Visual forecast comparisons
        - Model save/load
        
        **💻 Code Generation**
        - Ready-to-use Python code
        - Customized for your data
        - Download instantly
        """)

    st.markdown("---")
    st.markdown(
        "<div style='text-align: center;'>Built for hackathons 🏆 | "
        "Time Series Algorithm Recommender v3.0</div>",
        unsafe_allow_html=True,
    )

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #888;'>"
    "💡 Tip: Start with a demo dataset to see the full workflow"
    "</div>",
    unsafe_allow_html=True,
)
