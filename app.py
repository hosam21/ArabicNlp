import streamlit as st
import joblib
import pandas as pd
import numpy as np
import os
from text_preprocessor import ArabicTextPreprocessor
from emoji_handler import EmojiHandler
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix, hstack
from sklearn.preprocessing import StandardScaler
import arabic_reshaper
from bidi.algorithm import get_display

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
    .arabic-text {
        direction: rtl;
        text-align: right;
        font-family: 'Arial', sans-serif;
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
        
        # Print available classes for debugging
        st.write("Available sentiment classes:", sentiment_encoder.classes_)
        st.write("Available sarcasm classes:", sarcasm_encoder.classes_)
        
        return sentiment_model, sentiment_encoder, sarcasm_model, sarcasm_encoder
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None, None, None, None

# Load TF-IDF vectorizer and scaler
@st.cache_resource
def load_vectorizer_and_scaler():
    try:
        vectorizer = joblib.load(os.path.join('models', 'tfidf_vectorizer.joblib'))
        scaler = joblib.load(os.path.join('models', 'scaler.joblib'))
        
        # Ensure the vectorizer is fitted and has the correct vocabulary
        if not hasattr(vectorizer, 'vocabulary_'):
            st.error("TF-IDF vectorizer is not properly fitted")
            return None, None
        
        # Store the expected feature count and vocabulary
        vectorizer.expected_features = len(vectorizer.vocabulary_)
        vectorizer.expected_vocabulary = vectorizer.vocabulary_
        return vectorizer, scaler
    except Exception as e:
        st.error(f"Error loading vectorizer or scaler: {str(e)}")
        return None, None

def prepare_features(text, vectorizer, scaler):
    """Prepare features exactly as done during training"""
    # Get TF-IDF features
    text_vectorized = vectorizer.transform([text])
    
    # Calculate additional features
    text_length = len(text)
    word_count = len(text.split())
    avg_word_length = np.mean([len(word) for word in text.split()]) if word_count > 0 else 0
    
    # Create additional features array
    additional_features = np.array([[text_length, word_count, avg_word_length]])
    additional_features_scaled = scaler.transform(additional_features)
    
    # Combine features
    combined_features = hstack([text_vectorized, additional_features_scaled])
    
    return combined_features

def display_arabic_text(text):
    """Display Arabic text properly"""
    reshaped_text = arabic_reshaper.reshape(text)
    bidi_text = get_display(reshaped_text)
    return f'<div class="arabic-text">{bidi_text}</div>'

def get_sentiment_color(label):
    """Get color based on sentiment label"""
    if label == 'positive':
        return "green"
    elif label == 'negative':
        return "red"
    else:  # neutral
        return "orange"

def get_sarcasm_color(label):
    """Get color based on sarcasm label"""
    return "red" if label == 'sarcastic' else "green"

def main():
    # Load resources
    text_preprocessor, emoji_handler = load_preprocessors()
    models = load_models()
    vectorizer, scaler = load_vectorizer_and_scaler()

    if None in models or vectorizer is None or scaler is None:
        st.error("""
        Unable to load required models. Please ensure the following files are available in the 'models' directory:
        - sentiment_rf_model.joblib
        - sentiment_label_encoder.joblib
        - sarcasm_rf_model.joblib
        - sarcasm_label_encoder.joblib
        - tfidf_vectorizer.joblib
        - scaler.joblib
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

                    # Prepare features
                    try:
                        features = prepare_features(processed_text, vectorizer, scaler)
                    except Exception as e:
                        st.error(f"Error preparing features: {str(e)}")
                        return

                    # Get predictions
                    sentiment_pred = sentiment_model.predict(features)
                    sentiment_label = sentiment_encoder.inverse_transform(sentiment_pred)[0]

                    sarcasm_pred = sarcasm_model.predict(features)
                    sarcasm_label = sarcasm_encoder.inverse_transform(sarcasm_pred)[0]

                    # Display results
                    st.markdown('<div class="result-box">', unsafe_allow_html=True)
                    st.markdown("### Results")
                    
                    # Sentiment result
                    sentiment_color = get_sentiment_color(sentiment_label)
                    st.markdown(f"**Sentiment:** <span style='color: {sentiment_color}'>{sentiment_label}</span>", unsafe_allow_html=True)
                    
                    # Sarcasm result
                    sarcasm_color = get_sarcasm_color(sarcasm_label)
                    st.markdown(f"**Sarcasm:** <span style='color: {sarcasm_color}'>{sarcasm_label}</span>", unsafe_allow_html=True)
                    
                    st.markdown('</div>', unsafe_allow_html=True)

                    # Show processed text
                    with st.expander("View Processed Text"):
                        st.markdown(display_arabic_text(processed_text), unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Error processing text: {str(e)}")
        else:
            st.warning("Please enter some text to analyze.")

if __name__ == "__main__":
    main() 
