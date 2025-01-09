import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import streamlit as st

# Load the dataset
movies = pd.read_csv("C:\\Users\\Shreya\\OneDrive\\Desktop\\coding\\IGDTUW ML Internships\\Final_Project_And_Research_Paper\\DATASET\\movies.csv")

# Clean titles
def clean_title(title):
    title = re.sub("[^a-zA-Z0-9 ]", "", str(title))
    return title

movies["clean_title"] = movies["title"].apply(clean_title)

# TF-IDF Vectorizer
vectorizer = TfidfVectorizer(ngram_range=(1, 2))
tfidf = vectorizer.fit_transform(movies["clean_title"])

# Search Function
def search(title):
    title = clean_title(title)
    query_vec = vectorizer.transform([title])
    similarity = cosine_similarity(query_vec, tfidf).flatten()
    indices = np.argpartition(similarity, -5)[-5:]  # Get top 5 indices
    results = movies.iloc[indices].iloc[::-1]  # Reverse order for descending similarity
    return results if not results.empty else "No similar movies found."

# Streamlit interface
st.title("Movie Search App")

movie_input = st.text_input("Enter Movie Title:", value='Toy Story')

if len(movie_input) > 2:
    results = search(movie_input)
    if isinstance(results, str):
        st.write(results)
    else:
        st.write(results[['title']])
