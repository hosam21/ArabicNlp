import re
import string
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import arabic_reshaper
from bidi.algorithm import get_display
import logging
from emoji_handler import EmojiHandler
# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ArabicTextPreprocessor:
    def __init__(self):
        # Initialize Arabic stopwords
        with open("stop_list.txt", encoding="utf8") as f:
            filtered_sw = {w.strip() for w in f if w.strip()}
        
        self.stop_words = set(filtered_sw)
                self.punctuations = string.punctuation + '،؛؟'
        # Define Arabic diacritics pattern
        self.arabic_diacritics = re.compile("""
                                         ّ    | # Tashdid
                                         َ    | # Fatha
                                         ً    | # Tanwin Fath
                                         ُ    | # Damma
                                         ٌ    | # Tanwin Damm
                                         ِ    | # Kasra
                                         ٍ    | # Tanwin Kasr
                                         ْ    | # Sukun
                                         ـ     # Tatwil/Kashida
                                         """, re.VERBOSE)

        # Define common Arabic noise patterns
        self.noise_patterns = {
            r'[^\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF\s]': '',  # Keep only Arabic characters
            r'\s+': ' ',  # Replace multiple spaces with single space
            r'[^\w\s]': '',  # Remove special characters
            r'[a-zA-Z]': '',  # Remove English characters
        }

        # Initialize emoji handler
        self.emoji_handler = EmojiHandler()
        
    def remove_duplicate_words(self, text):

        if not text:
            return text

        words = text.split()
        if not words:
            return text

        # Only remove consecutive duplicates
        result = []
        prev_word = None

        for word in words:
            # Only add the word if it's different from the previous word
            if word != prev_word:
                result.append(word)
                prev_word = word

        return ' '.join(result)

    def remove_diacritics(self, text):
        """Remove Arabic diacritics from text"""
        text = re.sub(self.arabic_diacritics, '', text)
        return text

    def remove_diacritics(self, text):
        """Remove Arabic diacritics"""
        return re.sub(r'[\u064B-\u065F\u0670]', '', text)

    def remove_urls(self, text):
        """Remove URLs from text"""
        url_pattern = re.compile(r'https?://\S+|www\.\S+')
        return url_pattern.sub('', text)

    def remove_mentions(self, text):
        """Remove Twitter mentions"""
        return re.sub(r'@\w+', '', text)

    def remove_hashtags(self, text):
        """Remove hashtags"""
        return re.sub(r'#\w+', '', text)

    def remove_numbers(self, text):
        """Remove numbers"""
        return re.sub(r'\d+', '', text)

    def remove_punctuations(self, text):
        """Remove punctuations"""
        translator = str.maketrans('', '', self.punctuations)
        return text.translate(translator)

    def remove_stopwords(self, text):
        """Remove Arabic stopwords"""
        words = text.split()
        return ' '.join([word for word in words if word not in self.stop_words])

    def normalize_arabic_text(self, text):
        """Normalize Arabic text by standardizing characters"""
        # Normalize Alef variations
        text = re.sub("[إأآا]", "ا", text)
        # Normalize Ya variations
        text = re.sub("ى", "ي", text)
        # Normalize Ta Marbuta
        text = re.sub("ة", "ه", text)
        # Normalize Kaf
        text = re.sub("گ", "ك", text)
        # Normalize Waw
        text = re.sub("ؤ", "و", text)
        # Normalize Ya with Hamza
        text = re.sub("ئ", "ي", text)
        return text

    def remove_noise(self, text):
        """Remove noise patterns from text"""
        for pattern, replacement in self.noise_patterns.items():
            text = re.sub(pattern, replacement, text)
        return text

    def preprocess(self, text):
        """Apply all preprocessing steps to the text"""
        if not isinstance(text, str):
            return ""

        # Process emojis first
        original_text = text
        text = self.emoji_handler.process_emojis(text)

        if text != original_text:
            logger.info(f"Emoji processing changed text from '{original_text}' to '{text}'")




        # Apply other preprocessing steps
        text = self.remove_urls(text)
        text = self.remove_mentions(text)
        text = self.remove_hashtags(text)
        text = self.remove_numbers(text)
        text = self.remove_punctuations(text)
        text = self.normalize_arabic_text(text)
        text = self.remove_diacritics(text)
        text = self.remove_noise(text)
        text = self.remove_stopwords(text)
        text = self.remove_duplicate_words(text)
        # Remove duplicate phrases
        text = self.remove_duplicate_phrases(text)
        # Remove extra whitespace
        text = ' '.join(text.split())
        return text
