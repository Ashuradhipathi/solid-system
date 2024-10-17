import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

# Load data (replace with your actual data loading)
movies_df = pd.read_csv('/workspaces/solid-system/data/tmdb_5000_movies.csv')

# Function to preprocess data and create recommendation model
def create_model():
    # Combine relevant features
    movies_df['features'] = movies_df['genres'] + ' ' + movies_df['keywords']
    
    # Create count matrix
    count = CountVectorizer(stop_words='english')
    count_matrix = count.fit_transform(movies_df['features'])
    
    # Compute cosine similarity
    cosine_sim = cosine_similarity(count_matrix, count_matrix)
    
    return cosine_sim

# Function to get movie recommendations
def get_recommendations(title, cosine_sim):
    idx = movies_df[movies_df['title'] == title].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]  # Top 10 similar movies
    movie_indices = [i[0] for i in sim_scores]
    return movies_df['title'].iloc[movie_indices]

# Create the model
cosine_sim = create_model()

# Streamlit app
st.title('Movie Recommendation System')

# User input
user_movie = st.text_input('Enter a movie you like:')

if user_movie:
    if user_movie in movies_df['title'].values:
        recommendations = get_recommendations(user_movie, cosine_sim)
        st.write('Here are some movies you might like:')
        for i, movie in enumerate(recommendations, 1):
            st.write(f"{i}. {movie}")
    else:
        st.write('Sorry, we don\'t have that movie in our database. Please try another one.')

# Add more features like multi-select for genres, year range, etc.