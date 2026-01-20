import streamlit as st
import pandas as pd
from app_utils.ui import apply_theme, card_container, end_card, neon_metric
from models import get_db, get_admin_stats, get_all_users, get_all_datasets, get_chat_history

# Apply Theme
apply_theme()

def main():
    if 'user' not in st.session_state or not st.session_state.user:
        st.warning("Please sign in to access this page.")
        st.stop()
        
    if not st.session_state.user.get('is_admin'):
        st.error("🚫 Access Denied: Admin privileges required.")
        st.stop()

    st.markdown('<h1 class="glow-header">⚙️ Admin Dashboard</h1>', unsafe_allow_html=True)
    
    db = get_db()
    try:
        stats = get_admin_stats(db)
        
        # Stats Overview
        card_container()
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            neon_metric("Total Users", stats['total_users'])
        with col2:
            neon_metric("Premium Users", stats['premium_users'])
        with col3:
            neon_metric("Datasets", stats['total_datasets'])
        with col4:
            neon_metric("Analyses", stats['total_analyses'])
        end_card()
        
        tabs = st.tabs(["👥 Users", "📊 Datasets", "💬 Chat Logs"])
        
        with tabs[0]:
            st.subheader("User Management")
            card_container()
            users = get_all_users(db)
            if users:
                users_data = []
                for u in users:
                    users_data.append({
                        'ID': u.id,
                        'Name': u.full_name or u.username,
                        'Email': u.email,
                        'Plan': '💎 Premium' if u.subscription_type == 'premium' else '🆓 Free',
                        'Analyses': u.analysis_count or 0,
                        'Joined': u.created_at.strftime('%Y-%m-%d') if u.created_at else '-',
                        'Last Login': u.last_login.strftime('%Y-%m-%d %H:%M') if u.last_login else '-'
                    })
                st.dataframe(pd.DataFrame(users_data), use_container_width=True)
            else:
                st.info("No users registered yet")
            end_card()
        
        with tabs[1]:
            st.subheader("Dataset Analytics")
            card_container()
            datasets = get_all_datasets(db)
            if datasets:
                datasets_data = []
                for d in datasets:
                    datasets_data.append({
                        'ID': d.id,
                        'Filename': d.filename,
                        'Dataset Name': d.dataset_name,
                        'Rows': f"{d.row_count:,}",
                        'Columns': d.column_count,
                        'Uploaded': d.upload_date.strftime('%Y-%m-%d %H:%M') if d.upload_date else '-'
                    })
                st.dataframe(pd.DataFrame(datasets_data), use_container_width=True)
            else:
                st.info("No datasets uploaded yet")
            end_card()
            
        with tabs[2]:
            st.subheader("Recent Conversations")
            chats = get_chat_history(db, limit=50)
            if chats:
                for chat in chats:
                    with st.expander(f"💬 {chat.timestamp.strftime('%Y-%m-%d %H:%M')} - {chat.user_message[:50]}"):
                        st.write(f"**User:** {chat.user_message}")
                        st.write(f"**AI:** {chat.ai_response}")
            else:
                st.info("No chat history available")

    finally:
        db.close()

if __name__ == "__main__":
    main()
