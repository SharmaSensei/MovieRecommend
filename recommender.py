# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
import streamlit as st

# Load the datasets
movies = pd.read_csv(
    'C:\\Users\\ShyGu\\OneDrive\\Desktop\\Movie Recommender System\\pyfiles\\movies_metadata.csv',
    low_memory=False
)
ratings = pd.read_csv('C:\\Users\\ShyGu\\OneDrive\\Desktop\\Movie Recommender System\\pyfiles\\ratings_small.csv')

# Rename the 'id' column in movies to 'movieId'
movies.rename(columns={'id': 'movieId'}, inplace=True)

# Convert relevant columns to numeric, coercing errors
numeric_columns = ['budget', 'revenue', 'runtime']
for column in numeric_columns:
    movies[column] = pd.to_numeric(movies[column], errors='coerce')

# Clean the dataset
movies_cleaned = movies.dropna(subset=['genres'])

# Function to plot rating distribution
def plot_rating_distribution(ratings):
    plt.figure(figsize=(10, 6))
    sns.histplot(ratings['rating'], bins=30, kde=True, color='skyblue')
    plt.title('Distribution of Movie Ratings')
    plt.xlabel('Rating')
    plt.ylabel('Count')
    st.pyplot(plt)

# Function to create a content-based recommender
def recommend_movies_content(movie_title, num_recommendations=5):
    count_vectorizer = CountVectorizer(tokenizer=lambda x: x.split('|'))
    genre_matrix = count_vectorizer.fit_transform(movies_cleaned['genres'])
    cosine_sim = cosine_similarity(genre_matrix, genre_matrix)

    # Check if the movie title exists
    if movie_title not in movies_cleaned['title'].values:
        st.error("Movie title not found. Please check the title and try again.")
        return pd.DataFrame()  # Return an empty DataFrame

    # Find the index of the movie
    idx = movies_cleaned[movies_cleaned['title'] == movie_title].index[0]
    
    # Get similarity scores
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    # Get the indices of the most similar movies
    sim_indices = [i[0] for i in sim_scores[1:num_recommendations + 1]]
    
    # Return the titles of the recommended movies
    return movies_cleaned.iloc[sim_indices][['title', 'genres', 'overview']]

# Function to create a collaborative filtering recommender
def recommend_movies_collaborative(user_id, num_recommendations=5):
    user_movie_matrix = ratings.pivot(index='userId', columns='movieId', values='rating').fillna(0)

    # Use KNN to find similar users
    knn = NearestNeighbors(metric='cosine', algorithm='brute')
    knn.fit(user_movie_matrix)

    # Find similar users to the specified user
    distances, indices = knn.kneighbors(user_movie_matrix.loc[user_id].values.reshape(1, -1), n_neighbors=num_recommendations + 1)

    # Get recommended movie IDs based on nearest users
    similar_users = indices.flatten()[1:]  # Skip the first user (itself)
    recommended_movie_ids = ratings[ratings['userId'].isin(similar_users)]['movieId'].value_counts().head(num_recommendations).index.tolist()

    # Return recommended movie titles
    return movies[movies['movieId'].isin(recommended_movie_ids)][['title', 'genres', 'overview']]

# Streamlit app
st.title("Movie Recommender System")

# Display a sample of available movie titles
st.write("### Available Movies:")
st.write(movies_cleaned['title'].sample(10).tolist())  # Display 10 random movie titles

# Input: Movie title from user for content-based filtering
movie_title = st.text_input("Enter a movie title you like (for content-based recommendations):")

if movie_title:
    with st.spinner('Finding recommendations...'):
        recommendations = recommend_movies_content(movie_title, 5)
        if not recommendations.empty:
            st.write("Recommended movies based on content:")
            st.table(recommendations)

# Input: User ID for collaborative filtering
st.write("### Available User IDs:")
st.write(ratings['userId'].unique().tolist())  # Display all unique user IDs

user_id = st.number_input("Enter your User ID (for collaborative recommendations):", min_value=1)

if user_id:
    with st.spinner('Finding recommendations...'):
        collaborative_recommendations = recommend_movies_collaborative(user_id, 5)
        if not collaborative_recommendations.empty:
            st.write("Recommended movies based on user collaboration:")
            st.table(collaborative_recommendations)
        else:
            st.error("No recommendations available based on user collaboration.")

# Display the rating distribution
if st.button("Show Rating Distribution"):
    plot_rating_distribution(ratings)

# Optional: Show additional movie details
if st.checkbox("Show Movie Details"):
    st.write("### Movie Details")
    selected_movie = st.selectbox("Select a movie to see details:", movies_cleaned['title'].unique())
    movie_info = movies_cleaned[movies_cleaned['title'] == selected_movie]
    st.write(f"**Title:** {movie_info['title'].values[0]}")
    st.write(f"**Genres:** {movie_info['genres'].values[0]}")
    st.write(f"**Overview:** {movie_info['overview'].values[0]}")
