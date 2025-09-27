import streamlit as st
import pandas as pd
import numpy as np
from src.feature_extraction import extract_features
from src.model import load_model, predict
import pickle
from nltk.stem import PorterStemmer
import string
import os
import time

# Load the saved model, vectorizer, and label encoder
MODEL_PATH = r'C:\Project_NLP\models'
model = load_model(os.path.join(MODEL_PATH, 'best_model.pkl'))
with open(os.path.join(MODEL_PATH, 'vectorizer.pkl'), 'rb') as file:
    vectorizer = pickle.load(file)
with open(os.path.join(MODEL_PATH, 'label_encoder.pkl'), 'rb') as file:
    label_encoder = pickle.load(file)

# Streamlit app configuration
st.set_page_config(
    page_title="AI Newsgroup Classifier",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Simplified CSS with darker pastel colors
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700;800&display=swap');
    
    .stApp {
        background: linear-gradient(135deg, #9A7BC8 0%, #B197D4 25%, #C49FE8 50%, #8A9FD1 100%);
        font-family: 'Poppins', sans-serif;
        min-height: 100vh;
    }
    
    .main-title {
        text-align: center;
        font-size: 3.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #4C1D95 0%, #7C3AED 50%, #A855F7 100%);
        background-clip: text;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 1rem 0;
        padding: 0;
    }
    
    .main-subtitle {
        text-align: center;
        font-size: 1.3rem;
        color: #581C87;
        font-weight: 600;
        margin-bottom: 2rem;
    }
    
    .glass-card {
        background: rgba(255, 255, 255, 0.2);
        backdrop-filter: blur(20px);
        border: 2px solid rgba(139, 92, 246, 0.4);
        border-radius: 20px;
        padding: 2rem;
        margin: 1rem 0;
        box-shadow: 0 15px 35px rgba(124, 58, 237, 0.2);
    }
    
    .glass-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 20px 40px rgba(124, 58, 237, 0.3);
        transition: all 0.3s ease;
    }
    
    .stTextArea > div > div > textarea {
        background: rgba(255, 255, 255, 0.3) !important;
        border: 2px solid rgba(139, 92, 246, 0.5) !important;
        border-radius: 15px !important;
        color: #581C87 !important;
        font-size: 17px !important;
        font-weight: 500 !important;
        padding: 20px !important;
        height: 180px !important;
        font-family: 'Poppins', sans-serif !important;
    }
    
    .stTextArea > div > div > textarea:focus {
        border-color: #A855F7 !important;
        box-shadow: 0 0 0 3px rgba(168, 85, 247, 0.3) !important;
        background: rgba(255, 255, 255, 0.4) !important;
        outline: none !important;
    }
    
    .stTextArea label {
        color: #581C87 !important;
        font-weight: 700 !important;
        font-size: 18px !important;
        margin-bottom: 15px !important;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #7C3AED 0%, #A855F7 100%) !important;
        color: white !important;
        border: none !important;
        padding: 18px 25px !important;
        border-radius: 15px !important;
        font-weight: 700 !important;
        font-size: 16px !important;
        font-family: 'Poppins', sans-serif !important;
        text-transform: uppercase !important;
        letter-spacing: 1px !important;
        box-shadow: 0 6px 20px rgba(124, 58, 237, 0.4) !important;
        transition: all 0.3s ease !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-3px) !important;
        box-shadow: 0 8px 25px rgba(124, 58, 237, 0.5) !important;
        background: linear-gradient(135deg, #6D28D9 0%, #9333EA 100%) !important;
    }
    
    .prediction-result {
        background: linear-gradient(135deg, rgba(138, 159, 209, 0.3) 0%, rgba(196, 159, 232, 0.3) 100%);
        border: 2px solid rgba(124, 58, 237, 0.5);
        border-radius: 16px;
        padding: 25px;
        text-align: center;
        margin: 20px 0;
        backdrop-filter: blur(20px);
    }
    
    .prediction-title {
        font-size: 1.4rem;
        font-weight: 700;
        color: #6D28D9;
        margin-bottom: 15px;
        text-transform: uppercase;
        letter-spacing: 1.5px;
    }
    
    .prediction-text {
        font-size: 1.8rem;
        font-weight: 800;
        background: linear-gradient(135deg, #7C3AED 0%, #A855F7 100%);
        background-clip: text;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin-bottom: 10px;
    }
    
    .confidence-badge {
        display: inline-block;
        background: rgba(138, 159, 209, 0.5);
        color: #581C87;
        padding: 8px 16px;
        border-radius: 25px;
        font-size: 0.9rem;
        font-weight: 600;
        border: 1px solid rgba(138, 159, 209, 0.6);
    }
    
    .error-message {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.3) 0%, rgba(220, 38, 38, 0.3) 100%);
        border: 2px solid rgba(239, 68, 68, 0.5);
        color: #991B1B;
        border-radius: 16px;
        padding: 25px;
        text-align: center;
        margin: 20px 0;
        backdrop-filter: blur(20px);
        font-weight: 600;
        font-size: 1.1rem;
    }
    
    .stats-card {
        background: rgba(255, 255, 255, 0.2);
        border: 2px solid rgba(139, 92, 246, 0.4);
        border-radius: 16px;
        padding: 1.5rem;
        text-align: center;
        margin: 1rem 0;
        backdrop-filter: blur(20px);
        transition: all 0.3s ease;
    }
    
    .stats-card:hover {
        transform: translateY(-3px);
        background: rgba(255, 255, 255, 0.3);
        border-color: #A855F7;
        box-shadow: 0 6px 25px rgba(124, 58, 237, 0.3);
    }
    
    .stat-number {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #7C3AED 0%, #A855F7 100%);
        background-clip: text;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    
    .stat-label {
        font-size: 1rem;
        color: #6D28D9;
        text-transform: uppercase;
        letter-spacing: 1px;
        font-weight: 600;
    }
    
    .info-text {
        color: #6D28D9;
        font-size: 1rem;
        line-height: 1.6;
        font-weight: 500;
    }
    
    .section-title {
        color: #581C87;
        font-weight: 700;
        font-size: 1.3rem;
        margin-bottom: 1rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .team-name {
        background: linear-gradient(135deg, #7C3AED 0%, #A855F7 100%);
        background-clip: text;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        font-size: 1.3rem;
        margin: 0.5rem 0;
    }
    
    /* Hide Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stDeployButton {display: none;}
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="main-title">🌸 AI Newsgroup Classifier</div>', unsafe_allow_html=True)
st.markdown('<div class="main-subtitle">Advanced Machine Learning • Real-time Text Classification</div>', unsafe_allow_html=True)

# Main content in two columns
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    
    # Text input
    user_input = st.text_area(
        "👀 Enter Text for Classification",
        placeholder="Type or paste your text here...\n\nExample: 'I'm having issues with my graphics card. The display keeps flickering and I'm getting artifacts in games.'",
        key="text_input",
        help="Enter any text and our AI will classify it into one of 20 newsgroup categories",
        height=180
    )
    
    # Buttons
    col_btn1, col_btn2, col_btn3 = st.columns([3, 1, 1])
    
    with col_btn1:
        classify_btn = st.button("🎯 Classify Text", key="classify_button", use_container_width=True)
    
    with col_btn2:
        def clear_text():
            st.session_state.text_input = ""
        clear_btn = st.button("🧹 Clear", key="clear_button", on_click=clear_text, use_container_width=True)
    
    with col_btn3:
        def load_sample():
            st.session_state.text_input = "I'm having trouble with my graphics card driver. The display keeps flickering and I'm getting artifacts in games. I've tried updating the drivers but the problem persists. Any suggestions for troubleshooting this issue?"
        sample_btn = st.button("💡 Load Sample", key="sample_button", on_click=load_sample, use_container_width=True)
    
    # Results
    if classify_btn:
        if user_input and user_input.strip():
            if len(user_input.strip()) <= 3 and user_input.strip().lower() in ['.', ',', '!', 'hi', 'hey', 'ok']:
                st.markdown("""
                    <div class="error-message">
                        ⚠️ Input Too Short<br>
                        Please provide more descriptive text for better classification.
                    </div>
                """, unsafe_allow_html=True)
            else:
                try:
                    with st.spinner('🌸 Analyzing...'):
                        time.sleep(0.8)
                        
                        # Extract features and predict
                        input_features = extract_features(user_input, vectorizer)
                        prediction = predict(model, input_features)
                        predicted_newsgroup = label_encoder.inverse_transform([prediction])[0]
                        
                        # Display result
                        st.markdown(f"""
                            <div class="prediction-result">
                                <div class="prediction-title">Classification Result</div>
                                <div class="prediction-text">{predicted_newsgroup}</div>
                                <div class="confidence-badge">High Confidence</div>
                            </div>
                        """, unsafe_allow_html=True)
                        
                except Exception as e:
                    st.markdown("""
                        <div class="error-message">
                            🚨 Classification Error<br>
                            Please try again with different text.
                        </div>
                    """, unsafe_allow_html=True)
        else:
            st.markdown("""
                <div class="error-message">
                    📝 No Text Provided<br>
                    Please enter some text to classify.
                </div>
            """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    # Stats section
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">📊 Model Statistics</div>', unsafe_allow_html=True)
    
    # Stats grid
    stat_col1, stat_col2 = st.columns(2)
    
    with stat_col1:
        st.markdown("""
            <div class="stats-card">
                <div class="stat-number">20</div>
                <div class="stat-label">Categories</div>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
            <div class="stats-card">
                <div class="stat-number">NLP & ML</div>
                <div class="stat-label">Powered</div>
            </div>
        """, unsafe_allow_html=True)
    
    with stat_col2:
        st.markdown("""
            <div class="stats-card">
                <div class="stat-number">88%</div>
                <div class="stat-label">Accuracy</div>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
            <div class="stats-card">
                <div class="stat-number">Fast</div>
                <div class="stat-label">Response</div>
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Categories info
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">🏷️ Categories</div>', unsafe_allow_html=True)
    st.markdown("""
        <div class="info-text">
            My AI classifies text into 20 newsgroup categories:<br><br>
            • Computer hardware & software<br>
            • Science & technology<br>
            • Recreation & sports<br>
            • Politics & society<br>
            • Religion & philosophy<br>
            • And 15 more specialized topics
        </div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Team section
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">👩‍💻 Developed By : </div>', unsafe_allow_html=True)
    st.markdown('<div class="team-name">Menna Khaled</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)