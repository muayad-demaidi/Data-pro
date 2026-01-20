import streamlit as st
import pandas as pd
import numpy as np
from utils.ui import apply_theme, card_container, end_card
from predictions import (
    simple_forecast, analyze_trend, build_ml_prediction_model,
    create_risk_clusters, create_categorical_distribution,
    create_feature_importance_chart, create_cluster_scatter,
    analyze_categorical_insights, create_categorical_bar_chart,
    create_outlier_visualization, create_trend_chart, detect_outliers
)
from models import get_user_limits

# Apply Theme
apply_theme()

def main():
    if 'user' not in st.session_state or not st.session_state.user:
        st.warning("Please sign in to access this page.")
        st.stop()
        
    if st.session_state.df is None:
        st.info("Please upload a dataset in the Dashboard first.")
        st.stop()

    limits = get_user_limits()
    if not limits['predictions_enabled']:
        st.warning("This feature is available for Premium users only.")
        st.stop()

    st.markdown('<h1 class="glow-header">🧠 Predictions & AI Models</h1>', unsafe_allow_html=True)
    
    df_pred = st.session_state.df_cleaned if st.session_state.df_cleaned is not None else st.session_state.df
    numeric_cols = df_pred.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df_pred.select_dtypes(include=['object', 'category']).columns.tolist()

    tabs = st.tabs(["🔮 Forecasting", "🎯 ML Prediction", "🧩 Clustering"])
    
    with tabs[0]:
        st.subheader("Time Series Forecasting")
        card_container()
        if numeric_cols:
            col1, col2 = st.columns(2)
            with col1:
                target_col = st.selectbox("Target Column", numeric_cols, key="pred_target")
            with col2:
                periods = st.slider("Forecast Periods", 1, 12, 6)
            
            if st.button("Generate Forecast", type="primary"):
                with st.spinner("Calculating..."):
                    values = df_pred[target_col].dropna().tolist()
                    forecast = simple_forecast(values, periods)
                    
                    if forecast is not None:
                         fig = create_trend_chart(df_pred, target_col, forecast)
                         st.plotly_chart(fig, use_container_width=True)
                         
                         analysis = analyze_trend(df_pred, target_col)
                         st.info(f"Trend Analysis: {analysis}")
        else:
            st.warning("No numeric columns available for forecasting.")
        end_card()

    with tabs[1]:
        st.subheader("Target Variable Prediction")
        card_container()
        if len(numeric_cols) >= 3:
            target_ml = st.selectbox("Select Target Variable", numeric_cols, key="ml_target")
            
            if st.button("Train Model", type="primary"):
                with st.spinner("Training model..."):
                    result = build_ml_prediction_model(df_pred, target_ml)
                    
                    if 'error' in result:
                        st.error(result['error'])
                    else:
                        st.success("Model Trained Successfully!")
                        col1, col2, col3 = st.columns(3)
                        col1.metric("Model Type", result['model_type'].title())
                        if result['model_type'] == 'classification':
                            col2.metric("Accuracy", f"{result['accuracy']}%")
                        else:
                            col2.metric("R² Score", f"{result['r2_score']}%")
                        col3.metric("Train Size", f"{result['train_size']:,}")
                        
                        if 'feature_importance' in result:
                            st.caption("Feature Importance")
                            fig = create_feature_importance_chart(result['feature_importance'])
                            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Need at least 3 numeric columns for ML.")
        end_card()

    with tabs[2]:
        st.subheader("Data Clustering")
        card_container()
        if len(numeric_cols) >= 2:
            n_clusters = st.slider("Number of Clusters", 2, 6, 4)
            
            if st.button("Create Clusters", type="primary"):
                with st.spinner("Clustering..."):
                    result = create_risk_clusters(df_pred, n_clusters)
                    
                    if 'error' in result:
                        st.error(result['error'])
                    else:
                        st.success(f"Created {n_clusters} clusters!")
                        
                        # Cluster Stats
                        for c_name, stats in result['cluster_stats'].items():
                            with st.expander(f"{c_name} - {stats['percentage']}%"):
                                st.write(stats['characteristics'])
                        
                        # Viz
                        col1, col2 = st.columns(2)
                        with col1:
                            x = st.selectbox("X Axis", numeric_cols, key="clust_x")
                        with col2:
                            y = st.selectbox("Y Axis", [c for c in numeric_cols if c != x], key="clust_y")
                        
                        df_viz = df_pred[numeric_cols].dropna()
                        if len(df_viz) == len(result['cluster_labels']):
                            fig = create_cluster_scatter(df_viz, x, y, result['cluster_labels'])
                            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Need at least 2 numeric columns for clustering.")
        end_card()

if __name__ == "__main__":
    main()
