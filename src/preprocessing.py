"""
Text preprocessing utilities for sentiment analysis.
Includes tokenization, stop-words removal, stemming, lemmatization, and vectorization.
"""
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, SnowballStemmer
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


# Download required NLTK data
def download_nltk_data():
    """Download required NLTK datasets."""
    required_data = ['punkt', 'stopwords', 'wordnet', 'omw-1.4']
    for data in required_data:
        try:
            nltk.download(data, quiet=True)
        except:
            print(f"Warning: Could not download {data}")


class TextPreprocessor:
    """Text preprocessing pipeline for sentiment analysis."""
    
    def __init__(self, use_stemming=False, use_lemmatization=True, 
                 remove_stopwords=True, lowercase=True):
        """
        Initialize TextPreprocessor.
        
        Args:
            use_stemming: Whether to apply stemming
            use_lemmatization: Whether to apply lemmatization
            remove_stopwords: Whether to remove stop words
            lowercase: Whether to convert to lowercase
        """
        download_nltk_data()
        
        self.use_stemming = use_stemming
        self.use_lemmatization = use_lemmatization
        self.remove_stopwords = remove_stopwords
        self.lowercase = lowercase
        
        # Initialize tools
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
    
    def clean_text(self, text):
        """
        Basic text cleaning.
        
        Args:
            text: Input text string
            
        Returns:
            str: Cleaned text
        """
        if not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        if self.lowercase:
            text = text.lower()
        
        # Remove HTML tags
        text = re.sub(r'<.*?>', '', text)
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+', '', text)
        
        # Remove special characters and numbers
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    def tokenize(self, text):
        """
        Tokenize text into words.
        
        Args:
            text: Input text string
            
        Returns:
            list: List of tokens
        """
        try:
            tokens = word_tokenize(text)
        except:
            # Fallback to simple split
            tokens = text.split()
        return tokens
    
    def remove_stopwords_func(self, tokens):
        """
        Remove stop words from token list.
        
        Args:
            tokens: List of tokens
            
        Returns:
            list: Filtered tokens
        """
        return [token for token in tokens if token not in self.stop_words]
    
    def stem_tokens(self, tokens):
        """
        Apply stemming to tokens.
        
        Args:
            tokens: List of tokens
            
        Returns:
            list: Stemmed tokens
        """
        return [self.stemmer.stem(token) for token in tokens]
    
    def lemmatize_tokens(self, tokens):
        """
        Apply lemmatization to tokens.
        
        Args:
            tokens: List of tokens
            
        Returns:
            list: Lemmatized tokens
        """
        return [self.lemmatizer.lemmatize(token) for token in tokens]
    
    def preprocess(self, text):
        """
        Complete preprocessing pipeline.
        
        Args:
            text: Input text string
            
        Returns:
            str: Preprocessed text
        """
        # Clean text
        text = self.clean_text(text)
        
        # Tokenize
        tokens = self.tokenize(text)
        
        # Remove stopwords
        if self.remove_stopwords:
            tokens = self.remove_stopwords_func(tokens)
        
        # Apply stemming or lemmatization
        if self.use_stemming:
            tokens = self.stem_tokens(tokens)
        elif self.use_lemmatization:
            tokens = self.lemmatize_tokens(tokens)
        
        # Join back to string
        return ' '.join(tokens)
    
    def preprocess_corpus(self, texts):
        """
        Preprocess a corpus of texts.
        
        Args:
            texts: List or Series of texts
            
        Returns:
            list: Preprocessed texts
        """
        return [self.preprocess(text) for text in texts]


class TextVectorizer:
    """Wrapper for different text vectorization methods."""
    
    def __init__(self, method='tfidf', max_features=5000, ngram_range=(1, 2)):
        """
        Initialize TextVectorizer.
        
        Args:
            method: Vectorization method ('tfidf' or 'count')
            max_features: Maximum number of features
            ngram_range: N-gram range for vectorizer
        """
        self.method = method
        self.max_features = max_features
        self.ngram_range = ngram_range
        
        if method == 'tfidf':
            self.vectorizer = TfidfVectorizer(
                max_features=max_features,
                ngram_range=ngram_range
            )
        elif method == 'count':
            self.vectorizer = CountVectorizer(
                max_features=max_features,
                ngram_range=ngram_range
            )
        else:
            raise ValueError(f"Unknown vectorization method: {method}")
    
    def fit_transform(self, texts):
        """Fit vectorizer and transform texts."""
        return self.vectorizer.fit_transform(texts)
    
    def transform(self, texts):
        """Transform texts using fitted vectorizer."""
        return self.vectorizer.transform(texts)
    
    def get_vectorizer(self):
        """Get the underlying vectorizer object."""
        return self.vectorizer
