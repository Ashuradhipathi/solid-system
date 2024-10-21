import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

#function to initalize and feature engineer dataset
def prep_data(path: str):
    # Load data (replace with your actual data loading)
    movies_df = pd.read_csv(path)
    # Combine relevant features
    movies_df['features'] = movies_df['genres'] + ' ' + movies_df['keywords']

    return movies_df

#Function to vectorize data
def vectorize_data(movies_df, vectorizer):
    vectorized_data = vectorizer.fit_transform(movies_df['features'])
    
    return vectorized_data

# Function to preprocess data and create recommendation model
def create_model(movies_df, vectorized_data):
    
    # Compute cosine similarity
    cosine_sim = cosine_similarity(vectorized_data, vectorized_data)
    
    return cosine_sim

# Function to get movie recommendations
def get_recommendations(title, cosine_sim, movies_df):
    idx = movies_df[movies_df['title'] == title].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]  # Top 10 similar movies
    movie_indices = [i[0] for i in sim_scores]
    return movies_df['title'].iloc[movie_indices]