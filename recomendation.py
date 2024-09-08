import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Sample data: Movies and their features (genres)
movies = {
    'Movie1': {'Action': 1, 'Comedy': 0, 'Drama': 0},
    'Movie2': {'Action': 1, 'Comedy': 1, 'Drama': 0},
    'Movie3': {'Action': 0, 'Comedy': 1, 'Drama': 1},
    'Movie4': {'Action': 1, 'Comedy': 0, 'Drama': 1},
    'Movie5': {'Action': 0, 'Comedy': 1, 'Drama': 0}
}

# User preferences
user_preferences = {'Action': 1, 'Comedy': 1, 'Drama': 0}
def build_feature_matrix(items):
    feature_names = sorted(set(feature for item_features in items.values() for feature in item_features))
    matrix = []
    for item, features in items.items():
        row = [features.get(feature, 0) for feature in feature_names]
        matrix.append(row)
    return np.array(matrix), feature_names

def build_user_profile(preferences):
    return np.array([preferences.get(feature, 0) for feature in sorted(preferences)])

# Build the movie feature matrix and user profile
movie_matrix, feature_names = build_feature_matrix(movies)
user_profile = build_user_profile(user_preferences)
def recommend_movies(user_profile, movie_matrix, movie_list):
    similarities = cosine_similarity([user_profile], movie_matrix)
    recommendations = sorted(zip(movie_list, similarities[0]), key=lambda x: x[1], reverse=True)
    return recommendations

# Get recommendations
movie_list = list(movies.keys())
recommendations = recommend_movies(user_profile, movie_matrix, movie_list)
def display_recommendations(recommendations):
    print("Recommended Movies:")
    for movie, score in recommendations:
        print(f"{movie}: {score:.2f}")

display_recommendations(recommendations)
