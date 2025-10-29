# AI Newsgroup Classifier: Advanced Real-time Text Classification📰

## Project Overview

The **AI Newsgroup Classifier** is a comprehensive machine learning project designed for the real-time classification of text documents into one of 20 distinct newsgroup categories. This project demonstrates a complete end-to-end workflow, from data parsing and preprocessing to model training, evaluation, and deployment via an interactive Streamlit web application.

The core objective is to accurately categorize unseen text data, providing a robust solution for automated content moderation, topic routing, and information organization.
<p align="center">
  <img src="https://github.com/user-attachments/assets/c769e42a-c8ff-4131-8eeb-a5b21249d8b8" width="500">
  <img src="https://github.com/user-attachments/assets/b19e129b-672b-4a36-af28-5846adbba168" width="500">
  
### Key Features📍

*   **20 Newsgroups Dataset:** Utilizes a classic and widely-used dataset for text classification benchmarking.
*   **TF-IDF Vectorization:** Employs Term Frequency-Inverse Document Frequency (TF-IDF) for effective feature extraction.
*   **Logistic Regression Model:** A highly performant and interpretable model is trained for multi-class classification.
*   **Comprehensive Evaluation:** Includes metrics such as accuracy, precision, recall, F1-score, and visual aids like Confusion Matrices and ROC Curves (saved in `models/`).
*   **Interactive Streamlit App:** A user-friendly web interface for real-time text classification demos.
*   **Modular Codebase:** Clean separation of concerns with dedicated modules for feature extraction and model handling (`src/`).

---

## 🚀 Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

You need to have Python installed on your system. This project was developed using Python 3.x.

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Menna-Khalid/News-Classification.git
    cd News-Classification
    ```

2.  **Create and activate a virtual environment (Recommended):**
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Linux/macOS
    # venv\Scripts\activate   # On Windows
    ```

3.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

### Running the Application

The project includes a pre-trained model and a Streamlit application for immediate use.

1.  **Run the Streamlit web app:**
    ```bash
    streamlit run app.py
    ```

2.  The application will open in your default web browser (usually at `http://localhost:8501`). You can now paste any text into the input box and click "Classify Text" to see the model's prediction.

---

## 📂 Project Structure

The repository is organized into a clean and logical structure to separate data, code, models, and notebooks.

```
AI-Resume-Matcher/
├── app.py                      # The main file for the interactive Streamlit web application.
├── requirements.txt            # Lists all necessary Python dependencies for the project.
├── data/                       # Contains the raw and processed data.
│   └── 20newsgroups/           # Stores the individual text files and a CSV of the 20 Newsgroups dataset.
├── models/                     # Stores the trained model artifacts and evaluation visuals.
│   ├── best_model.pkl          # The serialized Logistic Regression classifier.
│   ├── vectorizer.pkl          # The fitted TF-IDF vectorizer object.
│   ├── label_encoder.pkl       # The fitted label encoder for mapping class indices to newsgroup names.
│   ├── confusion_matrix.png    # Visual representation of the model's performance.
│   └── roc_curve.png           # Receiver Operating Characteristic (ROC) curve plot.
├── notebooks/                  # Jupyter notebooks used for data exploration, training, and evaluation.
│   └── train_20newsgroups.ipynb# The primary notebook detailing the end-to-end ML pipeline.
└── src/                        # Source code for modular functions.
    ├── feature_extraction.py   # Functions for text preprocessing, cleaning, stemming, and feature generation.
    └── model.py                # Utility functions for loading, saving, and making predictions with the model.
```

## 🧠 Machine Learning Pipeline

The entire machine learning workflow is documented and executed in the `notebooks/train_20newsgroups.ipynb` file.

### 1. Data Acquisition and Preprocessing

*   **Dataset:** The project uses the **20 Newsgroups dataset**, a collection of approximately 20,000 newsgroup documents, partitioned across 20 different newsgroups.
*   **Parsing:** A custom function in the notebook is used to parse the raw text files in `data/20newsgroups/`, extracting the document ID, text content, and the newsgroup label.
*   **Text Cleaning (in `src/feature_extraction.py`):**
    *   Conversion to lowercase.
    *   Removal of punctuation, numbers, and special characters.
    *   Tokenization using `nltk.word_tokenize`.
    *   Stop-word removal using `nltk.corpus.stopwords`.
    *   **Stemming** using the `PorterStemmer`.

### 2. Feature Engineering

*   **Vectorization:** The preprocessed text is converted into a numerical feature matrix using the **TF-IDF Vectorizer** from `scikit-learn`. This technique weights words based on their frequency in a document relative to their frequency across all documents, highlighting important, discriminative terms.

### 3. Model Training and Selection

*   **Model:** A **Logistic Regression** classifier is chosen for its balance of high performance and interpretability in multi-class text classification tasks.
*   **Training:** The model is trained on the TF-IDF features derived from the training split of the dataset.

### 4. Evaluation and Results

The model achieves a high level of performance, as evidenced by the following metrics (extracted from the notebook output):

| Metric | Score |
| :--- | :--- |
| **Overall Accuracy** | **88%** |
| **Macro Average F1-Score** | **88%** |
| **Weighted Average F1-Score** | **88%** |

#### Classification Report Summary

The model performs exceptionally well across most categories, with high precision and recall, especially in technical and sports-related groups.

*   **Highest F1-Scores :** `rec.sport.hockey` (0.97), `talk.politics.mideast` (0.96), `sci.crypt` (0.96).
*   **Lowest F1-Scores :** `talk.religion.misc` (0.70), `alt.atheism` (0.87).

The detailed evaluation plots (Confusion Matrix and ROC Curve) are saved in the `models/` directory.

---

## 💻 Code Implementation Details

The project adheres to software engineering best practices by separating core logic into dedicated Python modules.

### `src/feature_extraction.py`

This module handles all text preprocessing steps, ensuring consistency between training and deployment.

### `src/model.py`

This module provides reusable functions for model persistence and inference, abstracting away the file handling logic.

### `app.py` - Web Application Logic

The Streamlit application loads the necessary artifacts and uses the modular functions for real-time inference, featuring a custom, visually appealing "glass-card" design with a pastel color palette.

---

## 🛠 Dependencies

The project relies on standard Python libraries for data science and web deployment.

| Package | Purpose |
| :--- | :--- |
| `streamlit` | For building the interactive web application. |
| `scikit-learn` | Core machine learning library (Logistic Regression, TF-IDF, metrics). |
| `pandas`, `numpy` | Data manipulation and numerical operations. |
| `nltk` | Natural Language Toolkit for text preprocessing (tokenization, stemming, stop-words). |
| `joblib`, `pickle` | For model persistence (saving and loading artifacts). |
| `matplotlib`, `seaborn` | For data visualization and plotting evaluation results. |

The full list of dependencies can be found in `requirements.txt`.

---

## 🤝 Contribution

This project is maintained by **Menna-Khalid**.

Feel free to fork the repository, submit pull requests, or open issues for bugs and feature suggestions.

---
