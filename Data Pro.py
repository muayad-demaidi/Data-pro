import streamlit as st
import pandas as pd
from datetime import datetime
from models import init_db, get_db, authenticate_user, create_user, user_to_dict
from utils.ui import apply_theme, card_container, end_card

# Page Config
st.set_page_config(
    page_title="DataVision Pro",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply Theme
apply_theme()

# Initialize State
if 'user' not in st.session_state:
    st.session_state.user = None
if 'df' not in st.session_state:
    st.session_state.df = None
if 'df_cleaned' not in st.session_state:
    st.session_state.df_cleaned = None
if 'cleaning_report' not in st.session_state:
    st.session_state.cleaning_report = None
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None
if 'chat_messages' not in st.session_state:
    st.session_state.chat_messages = []
if 'current_dataset_id' not in st.session_state:
    st.session_state.current_dataset_id = None

# Initialize DB
init_db()

def show_login():
    col1, col2, col3 = st.columns([1, 1.5, 1])
    with col2:
        card_container()
        st.markdown('<h2 style="text-align: center;">👋 Welcome Back</h2>', unsafe_allow_html=True)
        
        with st.form("login_form"):
            email = st.text_input("Email / Username")
            password = st.text_input("Password", type="password")
            submit = st.form_submit_button("Sign In", use_container_width=True)
            
            if submit:
                if email and password:
                    db = get_db()
                    try:
                        user = authenticate_user(db, email, password)
                        if user:
                            st.session_state.user = user_to_dict(user)
                            st.success("Signed in successfully!")
                            st.rerun()
                        else:
                            st.error("Invalid credentials")
                    finally:
                        db.close()
                else:
                    st.warning("Please fill in all fields")
        
        if st.button("Create New Account", type="secondary", use_container_width=True):
            st.session_state.auth_page = 'register'
            st.rerun()
        end_card()

def show_register():
    col1, col2, col3 = st.columns([1, 1.5, 1])
    with col2:
        card_container()
        st.markdown('<h2 style="text-align: center;">🚀 Create Account</h2>', unsafe_allow_html=True)
        
        with st.form("register_form"):
            full_name = st.text_input("Full Name")
            username = st.text_input("Username")
            email = st.text_input("Email")
            password = st.text_input("Password", type="password")
            confirm = st.text_input("Confirm Password", type="password")
            submit = st.form_submit_button("Sign Up", use_container_width=True)
            
            if submit:
                if password != confirm:
                    st.error("Passwords do not match")
                elif len(password) < 6:
                    st.error("Password too short")
                elif all([full_name, username, email, password]):
                    db = get_db()
                    try:
                        user = create_user(db, email, username, password, full_name)
                        if user:
                            st.session_state.user = user_to_dict(user)
                            st.success("Account created!")
                            st.rerun()
                        else:
                            st.error("User already exists")
                    finally:
                        db.close()
                else:
                    st.warning("Fill all fields")
        
        if st.button("Back to Login", type="secondary", use_container_width=True):
            st.session_state.auth_page = 'login'
            st.rerun()
        end_card()

def main():
    st.markdown('<h1 class="glow-header" style="text-align: center; font-size: 3.5rem; margin-bottom: 30px;">🔮 DataVision Pro</h1>', unsafe_allow_html=True)
    
    if st.session_state.user:
        # Landing Page for Logged In User
        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown(f"## Welcome back, **{st.session_state.user['full_name']}**! 👋")
            st.write("Ready to analyze your data? Select a tool from the sidebar to begin.")
            
            card_container()
            st.markdown("### 🚀 Quick Actions")
            c1, c2 = st.columns(2)
            with c1:
                st.info("📊 **Go to Dashboard**\n\nUpload and manage your datasets.")
            with c2:
                st.info("💬 **Ask AI Assistant**\n\nChat with your data using Gemini.")
            end_card()

        with col2:
            card_container()
            st.write("### 👤 Profile Status")
            st.caption(f"Plan: {st.session_state.user['subscription_type'].upper()}")
            if st.session_state.get('current_dataset_id'):
                st.success("✅ Dataset Loaded")
            else:
                st.warning("⚠️ No Dataset Loaded")
            
            if st.button("Sign Out", type="primary", use_container_width=True):
                st.session_state.user = None
                st.rerun()
            end_card()
            
    else:
        # Auth Flow
        if 'auth_page' not in st.session_state:
            st.session_state.auth_page = 'login'
            
        if st.session_state.auth_page == 'login':
            show_login()
        else:
            show_register()

if __name__ == "__main__":
    main()
