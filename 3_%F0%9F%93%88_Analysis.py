import streamlit as st
import pandas as pd
import numpy as np
from app_utils.ui import apply_theme, card_container, end_card, neon_metric
from data_analyzer import (
    get_numeric_stats, get_categorical_stats, find_strong_correlations, detect_outliers
)
from visualizations import (
    create_distribution_overview, create_correlation_heatmap, create_bar_chart,
    create_scatter_plot, create_box_plot, create_pie_chart, create_line_chart, create_histogram
)

# Apply Theme
apply_theme()

def main():
    if 'user' not in st.session_state or not st.session_state.user:
        st.warning("Please sign in to access this page.")
        st.stop()
        
    if st.session_state.df is None:
        st.info("Please upload a dataset in the Dashboard first.")
        st.stop()

    df_viz = st.session_state.df_cleaned if st.session_state.df_cleaned is not None else st.session_state.df
    numeric_cols = df_viz.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df_viz.select_dtypes(include=['object', 'category']).columns.tolist()

    st.markdown('<h1 class="glow-header">📈 Analysis & Visualizations</h1>', unsafe_allow_html=True)
    
    tabs = st.tabs(["📊 Statistics", "🎨 Custom Plots", "🔥 Correlations"])
    
    with tabs[0]:
        # Numeric Stats
        st.subheader("Descriptive Statistics")
        card_container()
        numeric_stats = get_numeric_stats(df_viz)
        if not numeric_stats.empty:
            st.dataframe(numeric_stats, use_container_width=True)
        else:
            st.info("No numeric columns found")
        end_card()
        
        # Categorical Stats
        if categorical_cols:
            st.subheader("Categorical Breakdown")
            cat_stats = get_categorical_stats(df_viz)
            
            for col, stats in cat_stats.items():
                with st.expander(f"📁 {col}"):
                    c1, c2 = st.columns(2)
                    with c1:
                        st.write(f"**Unique Values:** {stats['unique_count']}")
                        st.write(f"**Most Common:** {stats['most_common']}")
                    with c2:
                        st.write(f"**Least Common:** {stats['least_common']}")
                        st.write(f"**Missing:** {stats['missing']}")
    
    with tabs[1]:
        st.subheader("Create Your Chart")
        card_container()
        
        chart_type = st.selectbox(
            "Select Chart Type",
            ["Bar Chart", "Scatter Plot", "Box Plot", "Pie Chart", "Line Chart", "Histogram"]
        )
        
        col1, col2 = st.columns(2)
        
        fig = None
        
        if chart_type == "Bar Chart" and categorical_cols:
            with col1:
                x_col = st.selectbox("Category (X)", categorical_cols, key="bar_x")
            with col2:
                y_col = st.selectbox("Value (Y)", numeric_cols if numeric_cols else [None], key="bar_y")
            if x_col:
                fig = create_bar_chart(df_viz, x_col, y_col)
        
        elif chart_type == "Scatter Plot" and len(numeric_cols) >= 2:
            with col1:
                x_col = st.selectbox("X Axis", numeric_cols, key="scatter_x")
            with col2:
                y_col = st.selectbox("Y Axis", numeric_cols, key="scatter_y", index=1 if len(numeric_cols) > 1 else 0)
            fig = create_scatter_plot(df_viz, x_col, y_col)
        
        elif chart_type == "Box Plot" and numeric_cols:
            with col1:
                y_col = st.selectbox("Numeric Column", numeric_cols, key="box_y")
            with col2:
                x_col = st.selectbox("Group By (Optional)", [None] + categorical_cols, key="box_x")
            fig = create_box_plot(df_viz, y_col, x_col)
        
        elif chart_type == "Pie Chart" and categorical_cols:
            with col1:
                col_select = st.selectbox("Column", categorical_cols, key="pie_col")
            fig = create_pie_chart(df_viz, col_select)
        
        elif chart_type == "Line Chart" and numeric_cols:
            with col1:
                x_col = st.selectbox("X Axis", df_viz.columns.tolist(), key="line_x")
            with col2:
                y_col = st.selectbox("Y Axis", numeric_cols, key="line_y")
            fig = create_line_chart(df_viz, x_col, y_col)
        
        elif chart_type == "Histogram" and numeric_cols:
            with col1:
                col_select = st.selectbox("Column", numeric_cols, key="hist_col")
            fig = create_histogram(df_viz, col_select)

        if fig:
            st.plotly_chart(fig, use_container_width=True)
        else:
            if not categorical_cols and chart_type in ["Bar Chart", "Pie Chart"]:
                 st.warning("This chart type requires categorical columns.")
            elif len(numeric_cols) < 2 and chart_type == "Scatter Plot":
                 st.warning("Scatter plot requires at least 2 numeric columns.")
        
        end_card()

    with tabs[2]:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Correlation Heatmap")
            card_container()
            corr_heatmap = create_correlation_heatmap(df_viz)
            if corr_heatmap:
                st.plotly_chart(corr_heatmap, use_container_width=True)
            end_card()
            
        with col2:
            st.subheader("Top Correlations")
            card_container()
            correlations = find_strong_correlations(df_viz)
            if correlations:
                for corr in correlations[:5]:
                    emoji = "🟢" if corr['correlation'] > 0 else "🔴"
                    st.markdown(f"{emoji} **{corr['column1']}** & **{corr['column2']}**")
                    st.caption(f"Correlation: {corr['correlation']:.3f}")
            else:
                st.info("No strong correlations found")
            end_card()
            
            st.subheader("Outliers")
            outliers = detect_outliers(df_viz)
            if outliers:
                for col, info in outliers.items():
                    st.error(f"⚠️ **{col}**: {info['count']} outliers ({info['percentage']}%)")
            else:
                st.success("No significant outliers detected")

if __name__ == "__main__":
    main()
