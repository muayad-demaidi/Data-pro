import os
import json
import pandas as pd
from typing import Dict, Any, List, Optional
import google.generativeai as genai
import streamlit as st

# Configure API Key
try:
    GENAI_API_KEY = st.secrets.get("GEMINI_API_KEY")
except Exception:
    GENAI_API_KEY = None

if not GENAI_API_KEY:
    GENAI_API_KEY = os.environ.get("GEMINI_API_KEY")

if GENAI_API_KEY:
    genai.configure(api_key=GENAI_API_KEY)
    model = genai.GenerativeModel('gemini-1.5-flash')
else:
    model = None

def get_model():
    """Returns the configured Gemini model or None if not configured."""
    return model

def generate_data_insights(df_summary: Dict, analysis_results: Dict) -> str:
    """Generate AI-powered insights from data analysis"""
    if not model:
        return "AI model not configured. Please check your API key."

    prompt = f"""You are a professional data analyst. Analyze the following data and provide useful insights and recommendations.

Data Summary:
- Rows: {df_summary.get('row_count', 'Unknown')}
- Columns: {df_summary.get('column_count', 'Unknown')}
- Column names: {', '.join(df_summary.get('columns', [])[:20])}

Analysis Results:
{json.dumps(analysis_results, ensure_ascii=False, indent=2, default=str)[:3000]}

Provide:
1. Top 5 key observations from the data
2. 3 actionable recommendations
3. Strengths and weaknesses of the data
4. Suggestions for improvement

Write the response in a clear and organized manner."""

    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Sorry, an error occurred while generating analysis: {str(e)}"

def chat_about_data(user_question: str, df_info: Dict, chat_history: List[Dict] = None) -> str:
    """Interactive chat about the data"""
    if not model:
        return "AI model not configured. Please check your API key."

    context = f"""Available data information:
- Rows: {df_info.get('row_count', 'Unknown')}
- Columns: {df_info.get('column_count', 'Unknown')}
- Column names: {', '.join(df_info.get('columns', []))}
- Data types: {json.dumps(df_info.get('dtypes', {}), ensure_ascii=False)}
- Statistical summary: {json.dumps(df_info.get('numeric_summary', {}), ensure_ascii=False, default=str)[:2000]}"""

    history_context = ""
    if chat_history:
        history_context = "Chat History:\n" + "\n".join([f"{msg['role']}: {msg['content']}" for msg in chat_history[-5:]])

    full_prompt = f"""You are an intelligent data analysis assistant.
You have information about the user's dataset.
Answer their questions accurately and helpfully.

{context}

{history_context}

User Question: {user_question}"""
    
    try:
        response = model.generate_content(full_prompt)
        return response.text
    except Exception as e:
        return f"Sorry, an error occurred: {str(e)}"

def generate_comparison_insights(comparison_data: Dict) -> str:
    """Generate insights from period comparison"""
    if not model:
        return "AI model not configured."

    prompt = f"""Analyze the following data comparison between two periods and provide useful insights:

Comparison Data:
{json.dumps(comparison_data, ensure_ascii=False, indent=2, default=str)[:3000]}

Provide:
1. Key changes between periods
2. Noticeable trends
3. Recommendations based on changes"""

    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error: {str(e)}"

def generate_prediction_insights(prediction_data: Dict, historical_context: str = "") -> str:
    """Generate insights about predictions"""
    if not model:
        return "AI model not configured."

    prompt = f"""Analyze the final prediction results:

Prediction Results:
{json.dumps(prediction_data, ensure_ascii=False, indent=2, default=str)}

{f"Historical Context: {historical_context}" if historical_context else ""}

Provide:
1. Interpretation of predictions
2. Reliability assessment
3. Recommendations for the future"""

    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error: {str(e)}"
