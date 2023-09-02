import pickle
import string
import nltk
import streamlit as st
from nltk.stem.porter import PorterStemmer

nltk.download('stopwords')
from nltk.corpus import stopwords

ps = PorterStemmer()

tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('Model.pkl', 'rb'))


def text_preprocess(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)


st.title("Email/SMS Spam Classifier")

input_text = st.text_area("Enter The Message")

if st.button("Predict"):

    Transformed_text = text_preprocess(input_text)

    vector_input = tfidf.transform([Transformed_text])

    result = model.predict(vector_input)[0]

    if result == 1:
        st.warning('This Mail is Spam', icon="⚠️")
        st.header(':red[SPAM]')
    else:
        st.header(':blue[HAM]')


