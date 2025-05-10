import streamlit as st
import joblib
import pandas as pd
import numpy as np
from nlp_practicala8eaa6982c import ArabicTextPreprocessor, EmojiHandler

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
    </style>
    """, unsafe_allow_html=True)

# Initialize preprocessors
@st.cache_resource
def load_preprocessors():
    return ArabicTextPreprocessor(), EmojiHandler()

# Load models
@st.cache_resource
def load_models():
    sentiment_model = joblib.load('models/sentiment_rf_model.joblib')
    sentiment_encoder = joblib.load('models/sentiment_label_encoder.joblib')
    sarcasm_model = joblib.load('models/sarcasm_rf_model.joblib')
    sarcasm_encoder = joblib.load('models/sarcasm_label_encoder.joblib')
    return sentiment_model, sentiment_encoder, sarcasm_model, sarcasm_encoder

# Load TF-IDF vectorizer
@st.cache_resource
def load_vectorizer():
    return joblib.load('models/tfidf_vectorizer.joblib')

def main():
    # Load resources
    text_preprocessor, emoji_handler = load_preprocessors()
    sentiment_model, sentiment_encoder, sarcasm_model, sarcasm_encoder = load_models()
    vectorizer = load_vectorizer()

    # Header
    st.markdown('<h1 class="title">Arabic Text Analysis</h1>', unsafe_allow_html=True)
    st.markdown('<h2 class="subtitle">Sentiment and Sarcasm Detection</h2>', unsafe_allow_html=True)

    # Text input
    text = st.text_area("Enter your Arabic text here:", height=150)

    if st.button("Analyze Text", type="primary"):
        if text:
            with st.spinner("Processing text..."):
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
        else:
            st.warning("Please enter some text to analyze.")

if __name__ == "__main__":
    main() 