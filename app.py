import streamlit as st
import joblib
import pandas as pd
import numpy as np
import os
from text_preprocessor import ArabicTextPreprocessor
from emoji_handler import EmojiHandler

# Set page config
st.set_page_config(
    page_title="Arabic Text Analysis",
    page_icon="📝",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        background-color: #f5f5f5;
    }
    .stTextInput>div>div>input {
        font-size: 18px;
    }
    .result-box {
        background-color: #ffffff;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .title {
        color: #1E88E5;
        font-size: 2.5em;
        text-align: center;
        margin-bottom: 30px;
    }
    .subtitle {
        color: #424242;
        font-size: 1.5em;
        text-align: center;
        margin-bottom: 20px;
    }
    .error-box {
        background-color: #ffebee;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    </style>
    """, unsafe_allow_html=True)

# Initialize preprocessors
@st.cache_resource
def load_preprocessors():
    return ArabicTextPreprocessor(), EmojiHandler()

# Load models
@st.cache_resource
def load_models():
    try:
        models_dir = 'models'
        if not os.path.exists(models_dir):
            st.error("Models directory not found. Please ensure the models are available.")
            return None, None, None, None

        sentiment_model = joblib.load(os.path.join(models_dir, 'sentiment_rf_model.joblib'))
        sentiment_encoder = joblib.load(os.path.join(models_dir, 'sentiment_label_encoder.joblib'))
        sarcasm_model = joblib.load(os.path.join(models_dir, 'sarcasm_rf_model.joblib'))
        sarcasm_encoder = joblib.load(os.path.join(models_dir, 'sarcasm_label_encoder.joblib'))
        return sentiment_model, sentiment_encoder, sarcasm_model, sarcasm_encoder
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None, None, None, None

# Load TF-IDF vectorizer
@st.cache_resource
def load_vectorizer():
    try:
        return joblib.load(os.path.join('models', 'tfidf_vectorizer.joblib'))
    except Exception as e:
        st.error(f"Error loading vectorizer: {str(e)}")
        return None

def main():
    # Load resources
    text_preprocessor, emoji_handler = load_preprocessors()
    models = load_models()
    vectorizer = load_vectorizer()

    if None in models or vectorizer is None:
        st.error("""
        Unable to load required models. Please ensure the following files are available in the 'models' directory:
        - sentiment_rf_model.joblib
        - sentiment_label_encoder.joblib
        - sarcasm_rf_model.joblib
        - sarcasm_label_encoder.joblib
        - tfidf_vectorizer.joblib
        """)
        return

    sentiment_model, sentiment_encoder, sarcasm_model, sarcasm_encoder = models

    # Header
    st.markdown('<h1 class="title">Arabic Text Analysis</h1>', unsafe_allow_html=True)
    st.markdown('<h2 class="subtitle">Sentiment and Sarcasm Detection</h2>', unsafe_allow_html=True)

    # Text input
    text = st.text_area("Enter your Arabic text here:", height=150)

    if st.button("Analyze Text", type="primary"):
        if text:
            with st.spinner("Processing text..."):
                try:
                    # Preprocess text
                    processed_text = text_preprocessor.preprocess(text)
                    processed_text = emoji_handler.process_emojis(processed_text)

                    # Vectorize text
                    text_vectorized = vectorizer.transform([processed_text])

                    # Get predictions
                    sentiment_pred = sentiment_model.predict(text_vectorized)
                    sentiment_label = sentiment_encoder.inverse_transform(sentiment_pred)[0]

                    sarcasm_pred = sarcasm_model.predict(text_vectorized)
                    sarcasm_label = sarcasm_encoder.inverse_transform(sarcasm_pred)[0]

                    # Display results
                    st.markdown('<div class="result-box">', unsafe_allow_html=True)
                    st.markdown("### Results")
                    
                    # Sentiment result
                    sentiment_color = "green" if sentiment_label == "positive" else "red" if sentiment_label == "negative" else "orange"
                    st.markdown(f"**Sentiment:** <span style='color: {sentiment_color}'>{sentiment_label}</span>", unsafe_allow_html=True)
                    
                    # Sarcasm result
                    sarcasm_color = "red" if sarcasm_label == "sarcastic" else "green"
                    st.markdown(f"**Sarcasm:** <span style='color: {sarcasm_color}'>{sarcasm_label}</span>", unsafe_allow_html=True)
                    
                    st.markdown('</div>', unsafe_allow_html=True)

                    # Show processed text
                    with st.expander("View Processed Text"):
                        st.write(processed_text)
                except Exception as e:
                    st.error(f"Error processing text: {str(e)}")
        else:
            st.warning("Please enter some text to analyze.")

if __name__ == "__main__":
    main() 
