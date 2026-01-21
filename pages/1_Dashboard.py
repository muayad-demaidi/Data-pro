import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from app_utils.ui import apply_theme, card_container, end_card, neon_metric
from app_utils.data_loader import load_file
from models import (
    get_db, save_dataset_record, find_similar_datasets, 
    increment_analysis_count, get_user_limits
)
from app_utils.data_cleaner import clean_data, get_data_quality_score, detect_column_types
from app_utils.data_analyzer import generate_summary_report
import hashlib
import math

# Apply Theme
apply_theme()

def sanitize_for_json(obj):
    """Recursively replace NaN and Inf values with None for JSON serialization"""
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize_for_json(item) for item in obj]
    elif isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    return obj

def calculate_data_hash(df):
    columns_str = '_'.join(sorted(df.columns.tolist()))
    return hashlib.md5(columns_str.encode()).hexdigest()

def main():
    if 'user' not in st.session_state or not st.session_state.user:
        st.warning("Please sign in to access the dashboard.")
        st.stop()

    limits = get_user_limits()
    
    st.markdown('<h1 class="glow-header">📊 Dashboard</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-title">Upload and manage your datasets</p>', unsafe_allow_html=True)

    # --- Upload Section ---
    card_container()
    st.header("📤 Upload Data")
    
    uploaded_file = st.file_uploader(
        "Droag & Drop Data File Here",
        type=['csv', 'xlsx', 'xls'],
        help="Upload your CSV or Excel file"
    )
    
    if uploaded_file:
        file_size_mb = uploaded_file.size / (1024 * 1024)
        
        if file_size_mb > limits['max_file_size_mb']:
            st.error(f"File size ({file_size_mb:.1f} MB) exceeds the limit ({limits['max_file_size_mb']} MB)")
        else:
            # Metadata Grid
            col1, col2, col3 = st.columns(3)
            with col1:
                period_month = st.selectbox("Month", range(1, 13), index=datetime.now().month - 1)
            with col2:
                period_year = st.selectbox("Year", range(2020, 2030), index=datetime.now().year - 2020)
            with col3:
                dataset_name = st.text_input("Dataset Name", value=uploaded_file.name.split('.')[0])

            if st.button("🚀 Start Analysis", type="primary", use_container_width=True):
                with st.spinner("Loading and analyzing..."):
                    df = load_file(uploaded_file)
                    
                    if df is not None:
                        if len(df) > limits['max_rows']:
                            st.error(f"Row count ({len(df):,}) exceeds the limit ({limits['max_rows']:,})")
                        else:
                            st.session_state.df = df
                            
                            # Auto-clean
                            df_cleaned, cleaning_report = clean_data(df)
                            st.session_state.df_cleaned = df_cleaned
                            st.session_state.cleaning_report = cleaning_report
                            
                            # Analyze
                            analysis_results = generate_summary_report(df_cleaned)
                            st.session_state.analysis_results = analysis_results
                            
                            # Save Record
                            data_hash = calculate_data_hash(df)
                            columns_info = {col: str(df[col].dtype) for col in df.columns}
                            
                            db = get_db()
                            try:
                                record = save_dataset_record(
                                    db,
                                    filename=uploaded_file.name,
                                    dataset_name=dataset_name,
                                    period_month=period_month,
                                    period_year=period_year,
                                    row_count=len(df),
                                    column_count=len(df.columns),
                                    columns_info=columns_info,
                                    data_hash=data_hash,
                                    summary_stats=sanitize_for_json(analysis_results.get('numeric_summary', {}))
                                )
                                st.session_state.current_dataset_id = record.id
                                
                                # Find Similar
                                similar = find_similar_datasets(db, columns_info)
                                similar = [s for s in similar if s['record'].id != record.id]
                                st.session_state.similar_datasets = similar
                                
                                if st.session_state.user:
                                    increment_analysis_count(db, st.session_state.user.get('id'))
                            finally:
                                db.close()
                            
                            st.success("Analysis completed! Explore the tabs in the sidebar.")
                            st.rerun()
    end_card()

    # --- Data Overview Section ---
    if st.session_state.df is not None:
        st.markdown("---")
        
        # Quality Metrics Row
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            neon_metric("Total Rows", f"{len(st.session_state.df):,}")
        with col2:
            neon_metric("Total Columns", len(st.session_state.df.columns))
        with col3:
            if st.session_state.df_cleaned is not None:
                quality = get_data_quality_score(st.session_state.df_cleaned)
                neon_metric("Data Quality", f"{quality['overall_score']}%")
        with col4:
             missing_pct = (st.session_state.df.isnull().sum().sum() / st.session_state.df.size) * 100
             neon_metric("Missing Values", f"{missing_pct:.1f}%")

        # Preview
        st.subheader("📋 Data Preview")
        card_container()
        st.dataframe(st.session_state.df.head(10), use_container_width=True)
        end_card()
        
        # Column Types
        st.subheader("ℹ️ Column Details")
        col_types = detect_column_types(st.session_state.df)
        col_types_df = pd.DataFrame({
            'Column': list(col_types.keys()),
            'Type': list(col_types.values())
        })
        st.dataframe(col_types_df, use_container_width=True)

if __name__ == "__main__":
    main()
