import streamlit as st
import numpy as np
import pandas as pd
from utils.ui import apply_theme, card_container, end_card
from ai_assistant import chat_about_data, get_model
from models import get_user_limits, save_chat_message, get_db

# Apply Theme
apply_theme()

def main():
    if 'user' not in st.session_state or not st.session_state.user:
        st.warning("Please sign in to access this page.")
        st.stop()
        
    limits = get_user_limits()
    if not limits['ai_chat_enabled']:
        st.warning("AI features are available for Premium users only.")
        st.stop()

    if not get_model():
        st.error("AI Model not configured. Please add your GEMINI_API_KEY to secrets.")
        st.stop()

    st.markdown('<h1 class="glow-header">💬 AI Data Assistant</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-title">Powered by Google Gemini 1.5 Flash</p>', unsafe_allow_html=True)
    
    # Initialize Chat History
    if 'chat_messages' not in st.session_state:
        st.session_state.chat_messages = []

    # Display Chat
    card_container()
    st.markdown('<div style="height: 400px; overflow-y: auto;">', unsafe_allow_html=True)
    for msg in st.session_state.chat_messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
    st.markdown('</div>', unsafe_allow_html=True)
    end_card()

    # Chat Input
    if prompt := st.chat_input("Ask a question about your data..."):
        if st.session_state.df is None:
            st.error("Please upload a dataset in the Dashboard first.")
        else:
            st.session_state.chat_messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.write(prompt)
            
            with st.chat_message("assistant"):
                with st.spinner("Analyzing data..."):
                    df_chat = st.session_state.df_cleaned if st.session_state.df_cleaned is not None else st.session_state.df
                    
                    df_info = {
                        'row_count': len(df_chat),
                        'column_count': len(df_chat.columns),
                        'columns': df_chat.columns.tolist(),
                        'dtypes': df_chat.dtypes.astype(str).to_dict(),
                        'numeric_summary': df_chat.describe().to_dict() if not df_chat.select_dtypes(include=[np.number]).empty else {}
                    }
                    
                    response = chat_about_data(prompt, df_info, st.session_state.chat_messages)
                    st.write(response)
                    st.session_state.chat_messages.append({"role": "assistant", "content": response})
                    
                    # Save to DB
                    if st.session_state.current_dataset_id:
                        db = get_db()
                        try:
                            save_chat_message(db, st.session_state.current_dataset_id, prompt, response)
                        finally:
                            db.close()

if __name__ == "__main__":
    main()
