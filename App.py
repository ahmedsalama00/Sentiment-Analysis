import streamlit as st
import re
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

model = tf.keras.models.load_model("sentiment_lstm_model.h5")

with open("tokenizer.pkl", "rb") as handle:
    tokenizer = pickle.load(handle)

def clean_text(text):
    text = str(text)
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)  
    text = re.sub(r"[^a-zA-Z\s]", '', text)  
    return text.lower().strip()

st.title("Sentiment Analysis App")
st.write("Enter a Comment and We Will analyze its feelings 😊")

user_input = st.text_area("Enter a Comment here:")

if st.button("Analyze"):
    if user_input.strip() != "":
        cleaned = clean_text(user_input)
        
        seq = tokenizer.texts_to_sequences([cleaned])
        
        padded = pad_sequences(seq, maxlen=150, padding='post', truncating='post')

        pred_probs = model.predict(padded)
        pred_class = np.argmax(pred_probs, axis=1)[0]      

        mapping = {0: "Negative 😡", 1: "Neutral 😐", 2: "Positive 😀"}
        
        st.success(f"Classification: {mapping[pred_class]}")
    else:
        st.warning("Please Write a comment first ✍️")