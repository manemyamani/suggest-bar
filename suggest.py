import streamlit as st
import nltk
from nltk.corpus import reuters
from collections import defaultdict
import random

#------------------ Download NLTK data for 1st time of running
nltk.download('punkt')
nltk.download('reuters')

reuters_sentences = reuters.sents()

flatten = lambda l: [item for sublist in l for item in sublist]
reuters_words = flatten(reuters_sentences)

def create_ngrams(words, n):
    ngrams = []
    for i in range(len(words) - n + 1):
        ngrams.append(tuple(words[i:i + n]))
    return ngrams

def build_ngram_model(ngrams):
    model = defaultdict(lambda: defaultdict(lambda: 0))
    for gram in ngrams:
        prefix = gram[:-1]
        suffix = gram[-1]
        model[prefix][suffix] += 1
    return model

def predict_next_words(model, prefix, k=3):
    options = model[prefix]
    if not options:
        return random.sample(reuters_words, k)  
    total_count = sum(options.values())
    probs = {word: count / total_count for word, count in options.items()}
    sorted_words = sorted(probs, key=probs.get, reverse=True)
    return sorted_words[:k]

n = 3 
ngrams = create_ngrams(reuters_words, n)
model = build_ngram_model(ngrams)

st.title("AI Word Predictor")

user_input = st.text_input("Type your sentence here:")

if user_input:
    prefix = tuple(user_input.split()[-(n - 1):])
    suggested_words = predict_next_words(model, prefix, k=5) 
    
    if suggested_words:
        st.write("Predicted next words:")
        for word in suggested_words:
            word = str(word)
            st.write(user_input + ' ' + word)
    else:
        st.write("No predictions available. Please try a different input.")

