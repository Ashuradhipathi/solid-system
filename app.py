import streamlit as st
from utils import *

movies_df = prep_data('/workspaces/solid-system/data/tmdb_5000_movies.csv')

# Create count matrix
# count = CountVectorizer(stop_words='english')
# vectorized_data = vectorize_data(movies_df, count)


# Create Tfidf matrix
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
vectorized_data = vectorize_data(movies_df, tfidf_vectorizer)

# Create the model
cosine_sim = create_model(movies_df, vectorized_data)

# Streamlit app
st.title('Movie Recommendation System')

# User input
user_movie = st.text_input('Enter a movie you like:')

if user_movie:
    if user_movie in movies_df['title'].values:
        recommendations = get_recommendations(user_movie, cosine_sim, movies_df)
        st.write('Here are some movies you might like:')
        for i, movie in enumerate(recommendations, 1):
            st.write(f"{i}. {movie}")
    else:
        st.write('Sorry, we don\'t have that movie in our database. Please try another one.')

# Add more features like multi-select for genres, year range, etc.