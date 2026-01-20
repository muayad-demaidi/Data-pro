import pandas as pd
import streamlit as st
import io

def load_file(uploaded_file):
    """Load CSV or Excel file with multi-encoding support for Arabic and other languages"""
    ENCODINGS = [
        'utf-8',
        'utf-8-sig',
        'cp1256',
        'windows-1256', 
        'iso-8859-6',
        'utf-16',
        'latin-1',
        'cp1252'
    ]
    
    try:
        if uploaded_file.name.endswith('.csv'):
            file_bytes = uploaded_file.read()
            
            df = None
            successful_encoding = None
            last_error = None
            
            for encoding in ENCODINGS:
                try:
                    df = pd.read_csv(io.BytesIO(file_bytes), encoding=encoding)
                    
                    if df.empty or len(df.columns) == 0:
                        continue
                    
                    col_text = ''.join(str(c) for c in df.columns)
                    if '\ufffd' in col_text or '?' * 5 in col_text:
                        continue
                    
                    successful_encoding = encoding
                    break
                except (UnicodeDecodeError, UnicodeError) as e:
                    last_error = e
                    continue
                except Exception as e:
                    last_error = e
                    continue
            
            if df is None or df.empty:
                st.error(f"Could not read file with any supported encoding. The file may be corrupted or use an unsupported format.")
                return None
            
            return df
            
        elif uploaded_file.name.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(uploaded_file)
            return df
        else:
            st.error("Unsupported file type. Please upload a CSV or Excel file.")
            return None
            
    except Exception as e:
        st.error(f"Error reading file: {str(e)}")
        return None
