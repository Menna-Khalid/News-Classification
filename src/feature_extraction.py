import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import nltk
import os

# Set NLTK data path
NLTK_DATA_PATH = 'C:\\nltk_data'
if not os.path.exists(NLTK_DATA_PATH):
    os.makedirs(NLTK_DATA_PATH)
nltk.data.path.append(NLTK_DATA_PATH)

# Download NLTK data with error handling
for corpus in ['punkt', 'punkt_tab', 'stopwords']:
    try:
        nltk.download(corpus, download_dir=NLTK_DATA_PATH, quiet=False)
    except Exception as e:
        print(f"Error downloading NLTK corpus '{corpus}': {e}")
        print(f"Please manually download '{corpus}' using nltk.download('{corpus}')")
        raise SystemExit("Exiting due to missing NLTK data.")

# Verify that the data is accessible
try:
    stopwords.words('english')
    word_tokenize("test sentence")
except LookupError as e:
    print(f"NLTK data not found: {e}")
    raise SystemExit("Exiting due to missing NLTK data.")

stop_words = set(stopwords.words('english'))
ps = PorterStemmer()

def extract_features(text, vectorizer=None, debug=False):
    """
    Preprocess text and convert to TF-IDF features.
    If vectorizer is provided, transform the text; otherwise, return preprocessed text.
    """
    # Clean text: remove special characters, numbers, and convert to lowercase
    text = re.sub(r'[^a-zA-Z\s]', '', text.lower())
    # Tokenize
    tokens = word_tokenize(text)
    # Remove stopwords and stem
    tokens = [ps.stem(word) for word in tokens if word not in stop_words]
    cleaned_text = ' '.join(tokens)
    
    if vectorizer is None:
        return cleaned_text  # For training phase
    else:
        # Transform using the provided vectorizer
        return vectorizer.transform([cleaned_text])
    