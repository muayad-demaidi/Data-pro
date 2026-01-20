import streamlit as st

def get_neon_css():
    """Returns the CSS string for the Neon Glassmorphism theme."""
    return """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');

    :root {
        --bg-color: #1a1033;
        --card-bg: rgba(255, 255, 255, 0.05);
        --sidebar-bg: rgba(26, 16, 51, 0.95);
        --neon-primary: #a855f7;  /* Purple */
        --neon-secondary: #06b6d4; /* Cyan */
        --neon-accent: #ec4899;   /* Pink */
        --text-primary: #ffffff;
        --text-secondary: #94a3b8;
    }

    /* General Reset */
    * {
        font-family: 'Outfit', sans-serif;
    }

    /* Main Background */
    .stApp {
        background-color: var(--bg-color);
        background-image: 
            radial-gradient(circle at 10% 20%, rgba(168, 85, 247, 0.2) 0%, transparent 20%),
            radial-gradient(circle at 90% 80%, rgba(6, 182, 212, 0.2) 0%, transparent 20%);
    }

    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: var(--sidebar-bg);
        border-right: 1px solid rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
    }
    
    [data-testid="stSidebar"] [data-testid="stMarkdown"] {
        color: var(--text-primary);
    }

    /* Card / Glassmorphism Container */
    .glass-card {
        background: var(--card-bg);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 20px;
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }

    .glass-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 40px rgba(168, 85, 247, 0.2);
        border-color: rgba(168, 85, 247, 0.3);
    }

    /* Metric Cards */
    [data-testid="stMetric"] {
        background-color: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 15px;
        backdrop-filter: blur(5px);
    }
    
    [data-testid="stMetricLabel"] {
        color: var(--text-secondary);
        font-weight: 500;
    }

    [data-testid="stMetricValue"] {
        color: var(--text-primary);
        text-shadow: 0 0 10px rgba(168, 85, 247, 0.5);
    }

    /* Custom File Uploader */
    [data-testid="stFileUploader"] {
        background: rgba(255, 255, 255, 0.02);
        border-radius: 20px;
        padding: 20px;
        border: 2px dashed rgba(168, 85, 247, 0.4);
        transition: all 0.3s ease;
    }

    [data-testid="stFileUploader"]:hover {
        border-color: var(--neon-secondary);
        background: rgba(6, 182, 212, 0.05);
        box-shadow: 0 0 20px rgba(6, 182, 212, 0.2);
    }
    
    .uploder-text {
        color: var(--text-secondary);
        text-align: center;
        font-size: 1.1rem;
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, var(--neon-primary), var(--neon-accent));
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(168, 85, 247, 0.4);
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(168, 85, 247, 0.6);
    }

    /* Headers */
    h1, h2, h3 {
        color: white !important;
        font-weight: 700;
        letter-spacing: 0.5px;
    }
    
    .glow-header {
        text-shadow: 0 0 20px rgba(168, 85, 247, 0.6);
    }

    /* DataFrame */
    [data-testid="stDataFrame"] {
        background: transparent;
    }
    
    /* Plotly Charts Background */
    .js-plotly-plot .plotly .main-svg {
        background: transparent !important;
    }
    </style>
    """

def apply_theme():
    """Applies the custom theme to the current Streamlit app."""
    st.markdown(get_neon_css(), unsafe_allow_html=True)

def card_container(key=None):
    """Returns a container that looks like a glassmorphism card."""
    container = st.container()
    st.markdown(f'<div class="glass-card">', unsafe_allow_html=True)
    return container

def end_card():
    """Closes the card div. Use after filling the container."""
    st.markdown('</div>', unsafe_allow_html=True)

def neon_metric(label, value, delta=None, help=None):
    """Custom metric display."""
    st.metric(label=label, value=value, delta=delta, help=help)
