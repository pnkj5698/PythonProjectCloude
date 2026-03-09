import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

movies = pd.read_csv("ml-latest-small/movies.csv")
ratings = pd.read_csv("ml-latest-small/ratings.csv")

# print(movies.shape)
# print(ratings.shape)
# print(movies.head())
# print(ratings.head())

# Merge movies and ratings together
data = ratings.merge(movies, on="movieId")

# Create a user-movie matrix
# Rows = users, Columns = movies, Values = ratings
movie_matrix = data.pivot_table(index="userId", columns="title", values="rating")

# print(movie_matrix.shape)
# print(movie_matrix.head())

# Merge movies and ratings together
data = ratings.merge(movies, on="movieId")

# Create a user-movie matrix
# Rows = users, Columns = movies, Values = ratings
movie_matrix = data.pivot_table(index="userId", columns="title", values="rating")

# print(movie_matrix.shape)
# print(movie_matrix.head())

# Fill NaN with 0 (unwatched = 0 rating)
movie_matrix_filled = movie_matrix.fillna(0)

# Calculate similarity between every movie
movie_similarity = cosine_similarity(movie_matrix_filled.T)
# print(movie_similarity)

# Convert to dataframe for easy lookup
similarity_df = pd.DataFrame(
    movie_similarity, index=movie_matrix.columns, columns=movie_matrix.columns
)


# Recommendation function
def recommend(movie_name, n=5):
    if movie_name not in similarity_df.columns:
        print("Movie not found!")
        return

    similar_movies = similarity_df[movie_name].sort_values(ascending=False)
    similar_movies = similar_movies.drop(movie_name)  # remove the movie itself
    print(f"\nTop {n} movies similar to '{movie_name}':")
    print(similar_movies.head(n))


# Test it!
recommend("Forrest Gump (1994)")
