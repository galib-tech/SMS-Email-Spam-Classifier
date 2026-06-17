# 📧 Email Spam Detection — Machine Learning Project

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-blue?style=for-the-badge&logo=python)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.0+-orange?style=for-the-badge&logo=scikit-learn)
![NLP](https://img.shields.io/badge/NLP-NLTK-green?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://sms-email-spam-classifier-pidtrcxmmirwmv6ymgfiz3.streamlit.app/)

**A complete end-to-end NLP pipeline to classify emails as Spam or Ham using multiple ML algorithms.**

[🌐 Live Demo](#-live-demo) • [📊 Results](#-model-results) • [🚀 Quick Start](#-quick-start) • [📁 Project Structure](#-project-structure) • [🔍 How It Works](#-how-it-works)

</div>

---

## 📌 Project Overview

This project builds a **text classification system** that detects whether an email/SMS message is **spam** or **ham (legitimate)**. It covers the full ML pipeline — from raw data to comparing 11 different algorithms.

| | |
|---|---|
| **Problem Type** | Binary Text Classification |
| **Dataset** | SMS Spam Collection (5,572 messages) |
| **Best Model** | Multinomial Naive Bayes (MNB) |
| **Best Precision** | ~100% |
| **Best Accuracy** | ~97% |
| **Text Vectorization** | TF-IDF |

---

## 🌐 Live Demo

🚀 **Try the live application here:**

### 👉 https://sms-email-spam-classifier-pidtrcxmmirwmv6ymgfiz3.streamlit.app/

Test the classifier by entering any Email or SMS message and instantly see whether it is classified as **Spam** or **Ham** using the trained Machine Learning model.

---

```
Beginner ████████░░ Intermediate
```

| Skill Area | Level | Notes |
|---|---|---|
| Python | ⭐⭐⭐ Intermediate | Pandas, NumPy, string processing |
| NLP / Text Processing | ⭐⭐⭐ Intermediate | Tokenization, stemming, stopwords |
| Machine Learning | ⭐⭐⭐ Intermediate | 11 classifiers compared |
| Data Visualization | ⭐⭐ Beginner-Mid | Matplotlib, Seaborn, WordCloud |
| Overall | ⭐⭐⭐ Intermediate | Great for portfolio |

> **Ideal for:** Someone who knows Python basics and wants a strong NLP project for their portfolio.

---

## 📊 Model Results

### All 11 Algorithms Compared

| Rank | Algorithm | Accuracy | Precision | Recommended? |
|------|-----------|----------|-----------|--------------|
| 🥇 1 | **Multinomial NB** | ~97.1% | ~100.0% | ✅ Best |
| 🥈 2 | **Bernoulli NB** | ~97.9% | ~98.6% | ✅ Great |
| 🥉 3 | **SVC (Sigmoid)** | ~97.5% | ~97.4% | ✅ Good |
| 4 | Extra Trees | ~97.5% | ~97.3% | ✅ Good |
| 5 | Random Forest | ~97.1% | ~97.2% | ✅ Good |
| 6 | Logistic Regression | ~96.2% | ~96.8% | ⚠️ Decent |
| 7 | XGBoost | ~96.8% | ~95.7% | ⚠️ Decent |
| 8 | Gradient Boosting | ~95.5% | ~93.8% | ⚠️ Decent |
| 9 | Bagging Classifier | ~95.8% | ~93.5% | ⚠️ Decent |
| 10 | AdaBoost | ~95.5% | ~90.5% | ❌ Weaker |
| 11 | KNN | ~90.5% | ~89.3% | ❌ Weakest |
| 12 | Gaussian NB | ~86.6% | ~57.2% | ❌ Poor |

> **Why Precision over Accuracy?**
> For spam detection, a **False Positive** (blocking a real email) is more costly than a **False Negative** (letting spam through). So we optimise for **Precision**.

### Naive Bayes — Why It Wins for Text

- Assumes word independence (works surprisingly well for text)
- Extremely fast to train
- Works great with TF-IDF features
- Proven approach for email spam since the 1990s

---

## 🔍 How It Works

### Pipeline Overview

```
Raw Email Text
      │
      ▼
1. DATA CLEANING
   • Remove duplicate rows (415 removed)
   • Drop unnamed columns
   • Label encode: ham=0, spam=1
      │
      ▼
2. EDA (Exploratory Data Analysis)
   • Class distribution: 87.4% ham / 12.6% spam (imbalanced)
   • Spam messages are longer (avg 138 chars vs 71 for ham)
   • Spam has more words (avg 28 vs 15)
   • WordCloud visualization
      │
      ▼
3. TEXT PREPROCESSING
   • Lowercase conversion
   • Tokenization (NLTK)
   • Remove special characters
   • Remove stopwords & punctuation
   • Porter Stemming
      │
      ▼
4. VECTORIZATION
   • TF-IDF (Term Frequency-Inverse Document Frequency)
   • Converts text → numerical matrix
      │
      ▼
5. MODEL TRAINING & COMPARISON
   • 11 algorithms trained and evaluated
   • Metrics: Accuracy + Precision
      │
      ▼
6. BEST MODEL: Multinomial Naive Bayes
   • ~97% accuracy, ~100% precision
```

### Text Preprocessing Example

```
Input:  "WINNER!! You have been selected for a FREE prize. Call NOW!"

Step 1 (lowercase):   "winner!! you have been selected for a free prize. call now!"
Step 2 (tokenize):    ["winner", "you", "have", "been", "selected", "free", "prize", "call", "now"]
Step 3 (stopwords):   ["winner", "selected", "free", "prize", "call"]
Step 4 (stemming):    ["winner", "select", "free", "prize", "call"]

Output: "winner select free prize call"  →  SPAM 🚨
```

---

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/email-spam-detection.git
cd email-spam-detection
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Notebook

```bash
jupyter notebook email_spam_detection.ipynb
```

### 4. Try a Quick Prediction

```python
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import string

# Load model
model = pickle.load(open('model.pkl', 'rb'))
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))

ps = PorterStemmer()
stop_words = stopwords.words("english")

def predict_spam(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    text = [ps.stem(w) for w in text
            if w.isalnum() and w not in stop_words and w not in string.punctuation]
    text = " ".join(text)
    vec = tfidf.transform([text]).toarray()
    result = model.predict(vec)[0]
    return "🚨 SPAM" if result == 1 else "✅ HAM"

# Test it
print(predict_spam("Congratulations! You've won a FREE iPhone. Click here NOW!"))
# Output: 🚨 SPAM

print(predict_spam("Hey, are you coming to the meeting tomorrow?"))
# Output: ✅ HAM
```

---

## 📁 Project Structure

```
email-spam-detection/
│
├── 📓 email_spam_detection.ipynb   # Main notebook
├── 📄 spam.csv                     # Dataset (5,572 messages)
├── 📄 requirements.txt             # Dependencies
├── 📄 README.md                    # This file
│
├── 📊 outputs/                     # Generated visualizations
│   ├── class_distribution.png
│   ├── spam_wordcloud.png
│   ├── ham_wordcloud.png
│   └── model_comparison.png
│
└── 💾 models/                      # Saved models (after training)
    ├── model.pkl
    └── vectorizer.pkl
```

---

## 📦 Requirements

```txt
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
nltk>=3.6.0
xgboost>=1.5.0
matplotlib>=3.4.0
seaborn>=0.11.0
wordcloud>=1.8.0
jupyter>=1.0.0
```

Install all:
```bash
pip install pandas numpy scikit-learn nltk xgboost matplotlib seaborn wordcloud jupyter
```

Download NLTK data (run once):
```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
```

---

## 📈 Key Findings from EDA

### Dataset Statistics
- **Total messages:** 5,572 (after cleaning: 5,157)
- **Duplicates removed:** 415
- **Class balance:** 87.4% Ham vs 12.6% Spam (imbalanced)

### Spam vs Ham Patterns

| Feature | Ham (avg) | Spam (avg) |
|---|---|---|
| Characters | ~71 | ~138 |
| Words | ~15 | ~28 |
| Sentences | ~1.7 | ~2.2 |

### Top Spam Keywords
`free`, `call`, `text`, `claim`, `prize`, `mobile`, `stop`, `reply`, `win`, `cash`

### Top Ham Keywords
`ok`, `go`, `get`, `come`, `know`, `good`, `time`, `day`, `got`, `want`

---

## 🧠 Concepts Used

| Concept | Used Where |
|---|---|
| Label Encoding | Target column (ham/spam → 0/1) |
| TF-IDF Vectorization | Text → numbers |
| Porter Stemming | "running" → "run" |
| Stop Word Removal | Remove "the", "is", "at" etc. |
| Tokenization | Split text into words |
| Train-Test Split (80/20) | Model evaluation |
| Precision Score | Primary metric |
| Confusion Matrix | Error analysis |

---

## 🔮 Future Improvements

- [ ] Add deep learning model (LSTM / BERT)
- [ ] Handle class imbalance with SMOTE
- [ ] Add email header features (sender, subject line)
- [ ] Build REST API with FastAPI
- [ ] Add real-time prediction interface

---

## 🤝 Contributing

Contributions are welcome! Feel free to:
- Open an issue for bugs or suggestions
- Submit a pull request for improvements
- Star ⭐ the repo if you found it helpful

---

## 📄 License

This project is licensed under the MIT License.

---

## 👨‍💻 Author

**Galib**
- GitHub: [@yourusername](https://github.com/yourusername)

---

<div align="center">
  <sub>Built with ❤️ using Python & Scikit-Learn</sub>
</div>
