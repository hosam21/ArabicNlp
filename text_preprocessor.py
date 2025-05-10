import re
import string
import nltk
import os
from nltk.corpus import stopwords
import arabic_reshaper
from bidi.algorithm import get_display
import logging
from emoji_handler import EmojiHandler

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def download_nltk_data():
    try:
        # Create a directory for NLTK data in the current working directory
        nltk_data_dir = os.path.join(os.getcwd(), 'nltk_data')
        os.makedirs(nltk_data_dir, exist_ok=True)
        
        # Add the directory to NLTK's data path
        nltk.data.path.append(nltk_data_dir)
        
        # Only download stopwords
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            logger.info("Downloading NLTK stopwords data...")
            nltk.download('stopwords', download_dir=nltk_data_dir, quiet=True)
            logger.info("Successfully downloaded stopwords")
    except Exception as e:
        logger.error(f"Error downloading NLTK data: {str(e)}")
        raise

class ArabicTextPreprocessor:
    def __init__(self):
        # Download required NLTK data
        download_nltk_data()
        
        # Initialize Arabic stopwords
        try:
            with open("stop_list.txt", encoding="utf8") as f:
                filtered_sw = {w.strip() for w in f if w.strip()}
        except FileNotFoundError:
            logger.warning("stop_list.txt not found, using NLTK Arabic stopwords")
            filtered_sw = set(stopwords.words('arabic'))
        
        self.stop_words = set(filtered_sw)
        
        # Define positive and negative words to keep
        self.positive_words = {
            "جميل", "رائع", "ممتاز", "لذيذ", "مرح", "سعيد", "مبتهج", "متفائل",
            "متحمس", "مبهر", "مذهل", "مبشر", "مفعم", "مشرق", "منير", "مضيء",
            "مبهرج", "ذكي", "لطيف", "كريم", "موهوب", "ودود", "مسؤول", "مفيد",
            "ملهم", "متحمس"
        }
        
        self.negative_words = {
            "سيء", "فظيع", "قبيح", "بغيض", "حزين", "شرير", "غاضب", "خائف",
            "متوتر", "كسول", "قذر", "مقرف", "بشع", "متعجرف", "متعصب",
            "مثير للاشمئزاز", "كارثي", "مؤلم", "محبط", "مظلم", "قاتم", "مخيف",
            "مؤذي", "متخلف", "بائس", "مريض", "حقير", "غبي", "أحمق", "متعفن",
            "مزعج", "مثير للقلق", "مغرض", "وقح", "عدواني", "غير موثوق"
        }
        
        # Remove sentiment words from stopwords
        self.stop_words = self.stop_words - self.positive_words - self.negative_words
        
        logger.info("Initialized ArabicTextPreprocessor")
        self.emoji_handler = EmojiHandler()

    def remove_duplicate_phrases(self, text, max_phrase_length=3):
        """Remove duplicate consecutive phrases"""
        words = text.split()
        if len(words) <= max_phrase_length:
            return text

        for length in range(max_phrase_length, 0, -1):
            i = 0
            while i < len(words) - length:
                phrase = ' '.join(words[i:i+length])
                next_phrase = ' '.join(words[i+length:i+2*length])
                if phrase == next_phrase:
                    words = words[:i+length] + words[i+2*length:]
                else:
                    i += 1
        return ' '.join(words)

    def remove_duplicate_words(self, text):
        """Remove duplicate consecutive words"""
        words = text.split()
        return ' '.join(word for i, word in enumerate(words) if i == 0 or word != words[i-1])

    def remove_diacritics(self, text):
        """Remove Arabic diacritics"""
        return re.sub(r'[\u064B-\u065F\u0670]', '', text)

    def remove_urls(self, text):
        """Remove URLs from text"""
        return re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)

    def remove_mentions(self, text):
        """Remove @mentions from text"""
        return re.sub(r'@\w+', '', text)

    def remove_hashtags(self, text):
        """Remove #hashtags from text"""
        return re.sub(r'#\w+', '', text)

    def remove_numbers(self, text):
        """Remove numbers from text"""
        return re.sub(r'\d+', '', text)

    def remove_punctuations(self, text):
        """Remove punctuations from text"""
        translator = str.maketrans('', '', string.punctuation)
        return text.translate(translator)

    def remove_stopwords(self, text):
        """Remove stopwords from text using simple split"""
        words = text.split()
        return ' '.join(word for word in words if word not in self.stop_words)

    def normalize_arabic_text(self, text):
        """Normalize Arabic text"""
        try:
            # Reshape Arabic text
            reshaped_text = arabic_reshaper.reshape(text)
            # Handle bidirectional text
            bidi_text = get_display(reshaped_text)
            return bidi_text
        except Exception as e:
            logger.error(f"Error in normalize_arabic_text: {str(e)}")
            return text

    def remove_noise(self, text):
        """Remove noise from text"""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters
        text = re.sub(r'[^\w\s]', '', text)
        return text.strip()

    def preprocess(self, text):
        """Apply all preprocessing steps to the text"""
        if not isinstance(text, str):
            return ""

        # Process emojis first
        original_text = text
        text = self.emoji_handler.process_emojis(text)

        if text != original_text:
            logger.info(f"Emoji processing changed text from '{original_text}' to '{text}'")

        try:
            # Apply preprocessing steps
            text = self.remove_urls(text)
            text = self.remove_mentions(text)
            text = self.remove_hashtags(text)
            text = self.remove_numbers(text)
            text = self.remove_punctuations(text)
            text = self.remove_diacritics(text)
            text = self.normalize_arabic_text(text)
            text = self.remove_duplicate_phrases(text)
            text = self.remove_duplicate_words(text)
            text = self.remove_stopwords(text)
            text = self.remove_noise(text)

            logger.debug(f"Preprocessed text: {text}")
            return text
        except Exception as e:
            logger.error(f"Error in preprocess: {str(e)}")
            return text 
