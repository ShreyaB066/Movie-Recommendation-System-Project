import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import streamlit as st

# Load the dataset
@st.cache_data
def load_data():
    return pd.read_csv("movies.csv")

movies = load_data()

# Clean titles
def clean_title(title):
    title = re.sub("[^a-zA-Z0-9 ]", "", str(title))
    return title

movies["clean_title"] = movies["title"].apply(clean_title)

# TF-IDF Vectorizer
@st.cache_resource
def train_vectorizer():
    vectorizer = TfidfVectorizer(ngram_range=(1, 2))
    tfidf_matrix = vectorizer.fit_transform(movies["clean_title"])
    return vectorizer, tfidf_matrix

vectorizer, tfidf_matrix = train_vectorizer()

# Search Function
def search_movies(title):
    cleaned_title = clean_title(title)
    query_vec = vectorizer.transform([cleaned_title])
    similarity = cosine_similarity(query_vec, tfidf_matrix).flatten()
    indices = similarity.argsort()[-5:][::-1]  # Top 5 most similar movies
    results = movies.iloc[indices]
    return results if not results.empty else None

# Streamlit interface
st.title("🎥 Movie Maze ")

movie_input = st.text_input("Enter Movie Title:", value='Toy Story')

if len(movie_input) > 2:
    with st.spinner("Searching for similar movies..."):
        results = search_movies(movie_input)
    
    if results is not None:
        st.success("Here are the movies we found:")
        st.table(results[['title']])
    else:
        st.error("No similar movies found. Please try a different title.")
else:
    st.info("Enter a movie title to start the search!")
