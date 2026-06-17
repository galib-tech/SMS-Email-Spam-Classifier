import  streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer

nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("stopwords")


stopwords = stopwords.words("english")
ps = PorterStemmer()


def text_transformer(text):
    # for lower case
    text = text.lower()
    text = nltk.word_tokenize(text)

    # for tokenization
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    # for stop words and punctuation
    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords and i not in string.punctuation:
            y.append(i)

    # for stemming
    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

tfidf = pickle.load(open("vectorizer.pkl", "rb"))
model = pickle.load(open("model.pkl", "rb"))

st.title("Email/SMS Spam Classification")

input_sms = st.text_area("Enter the message: ")

if st.button("Predict"):

    # 1. preprocessing
    transform_sms = text_transformer(input_sms)
    # 2. vectorize
    vector_input= tfidf.transform([transform_sms])
    # 3. predict
    result = model.predict(vector_input)[0]
    # 4. display
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")
