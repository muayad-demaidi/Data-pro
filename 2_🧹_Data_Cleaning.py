import streamlit as st
import pandas as pd
from utils.ui import apply_theme, card_container, end_card, neon_metric
from data_cleaner import get_data_quality_score, clean_data
from visualizations import create_missing_values_chart

# Apply Theme
apply_theme()

def main():
    if 'user' not in st.session_state or not st.session_state.user:
        st.warning("Please sign in to access this page.")
        st.stop()
        
    if st.session_state.df is None:
        st.info("Please upload a dataset in the Dashboard first.")
        st.stop()

    st.markdown('<h1 class="glow-header">🧹 Data Cleaning</h1>', unsafe_allow_html=True)
    
    # Cleaning Report
    if st.session_state.cleaning_report:
        report = st.session_state.cleaning_report
        
        card_container()
        st.subheader("📝 Cleaning Summary")
        col1, col2, col3 = st.columns(3)
        with col1:
            neon_metric("Original Rows", f"{report['original_rows']:,}")
        with col2:
            neon_metric("Cleaned Rows", f"{report['final_rows']:,}")
        with col3:
            neon_metric("Rows Removed", report['rows_removed'])
        end_card()
        
        col_main1, col_main2 = st.columns([1, 1])
        
        with col_main1:
            card_container()
            st.subheader("Actions Taken")
            if report['changes']:
                for change in report['changes']:
                    st.success(f"✓ {change}")
            else:
                st.success("Data is clean! No modifications needed.")
            end_card()

        with col_main2:
            card_container()
            st.subheader("Data Quality Score")
            if st.session_state.df_cleaned is not None:
                quality = get_data_quality_score(st.session_state.df_cleaned)
                
                col_q1, col_q2 = st.columns(2)
                with col_q1:
                    neon_metric("Completeness", f"{quality['completeness']}%")
                with col_q2:
                    neon_metric("Uniqueness", f"{quality['uniqueness']}%")
                
                st.progress(quality['overall_score'] / 100)
                st.caption(f"Overall Score: {quality['overall_score']}%")
            end_card()

    # Missing Values Chart
    st.subheader("🧐 Missing Values Analysis")
    card_container()
    missing_chart = create_missing_values_chart(st.session_state.df)
    if missing_chart:
        st.plotly_chart(missing_chart, use_container_width=True)
    else:
        st.success("No missing values found in the original dataset.")
    end_card()
    
    # Manual Reclean Option (Optional Feature)
    if st.button("🔄 Re-run Cleaning Process"):
        with st.spinner("Cleaning data..."):
            df_cleaned, new_report = clean_data(st.session_state.df)
            st.session_state.df_cleaned = df_cleaned
            st.session_state.cleaning_report = new_report
            st.rerun()

if __name__ == "__main__":
    main()
