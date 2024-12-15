import streamlit as st
import pandas as pd
import numpy as np
import requests
from io import StringIO

st.set_page_config(layout="wide")
# Apply custom CSS for the container width
st.markdown("""
    <style>
        /* Set the width of the container */
        .block-container {
            width: 75%;  /* Adjust the percentage to control width */
            max-width: 75%  /* Optional: Set a maximum width */
            margin: 0 auto;  /* Center the content */
        }
    </style>
""", unsafe_allow_html=True)


st.title("Movie Recommender")


# In this section, we need to list the 100 movies the user needs to rank.

# Format of rating dataset:
#    movie_id
#    user_rating

# Things that need to get done:
# 1. List the movies in columns and rows (dynamically changeable)
# - Each movie section needs to have a interactable section oh reviews out of 5 stars
# - No need to half stars
# - Retrieve movie information (thumnail, movie name) from 
# 2. Each movie rating needs to be stored into a DataFrame to use for computing the recommendations




with st.spinner("Loading information..."):
    # RETRIEVE MOVIE DATA
    # Assign each movie data into a pandas dataframe

    movie_data_url = 'https://liangfgithub.github.io/MovieData/movies.dat?raw=true'

    # Fetch the file content from github
    movie_response = requests.get(movie_data_url)
    movie_response.raise_for_status()  # Raise an exceptionif the request fails

    # Split the file content into lines
    lines = movie_response.text.strip().split("\n")

    movie_data = [line.split("::") for line in lines]

    movies_df = pd.DataFrame(movie_data, columns=['movie_id', 'title', 'genres'])

    # Convert movie_id to integer for better usability
    movies_df["movie_id"] = movies_df["movie_id"].astype(int)

    # Split the genre column into lists
    movies_df["genres"] = movies_df["genres"].apply(lambda x: x.split("|"))

    # Add a new column for movie thumbnail URLs
    base_url = "https://liangfgithub.github.io/MovieImages/"
    movies_df["thumbnail_url"] = movies_df["movie_id"].apply(lambda x: f"{base_url}{x}.jpg")



    # RETRIEVE RATINGS DATA
    # Create a matrix for ratings for each user with movies and the columns with values assigned to them

    rating_data_url = 'https://liangfgithub.github.io/MovieData/ratings.dat?raw=true'
    rating_response = requests.get(rating_data_url)

    if rating_response.status_code == 200:
        rating_data = pd.read_csv(StringIO(rating_response.text), sep="::", engine="python", names=["UserID", "MovieID", "Rating", "Timestamp"])

        rating_matrix = rating_data.pivot(index='UserID', columns='MovieID', values='Rating')
    else:
        print('failed')








## ------------ THIS SECTION OF CODE WAS USED TO GENERATE THE TOP 30 S MATRIX ----------------

# # Step 1: Normalize the rating matrix
# def normalize_matrix(matrix):
#     """
#     Normalize the matrix by centering each row (user).
#     Ensures that user biases (e.g., some users always rate higher or lower) are accounted for.
#     """
#     """Normalize the matrix by centering each row (user)."""
#     return matrix.sub(matrix.mean(axis=1, skipna=True), axis=0)

# start_time = datetime.now()
# print(f"Step 1 started")
# normalized_ratings = normalize_matrix(rating_matrix)
# normalized_ratings.to_csv('app_normalized_ratings.csv', index=True)
# print(f"Step 1 completed")


# # Step 3
# def compute_movie_similarity(matrix):
#     # Initialize empty similarity matrix
#     n_movies = matrix.shape[1]
#     movie_ids = matrix.columns
#     similarity = pd.DataFrame(np.nan, index=movie_ids, columns=movie_ids)
    
#     # Iterate through each pair of movies
#     for i in range(n_movies):
#         for j in range(i, n_movies):  # Only compute upper triangle (including diagonal)
#             movie_i = matrix.iloc[:, i]
#             movie_j = matrix.iloc[:, j]
            
#             # Find users who rated both movies (non-NA values)
#             common_users = movie_i.notna() & movie_j.notna()
#             n_common = common_users.sum()
            
#             # Skip if less than 3 common ratings
#             if n_common <= 2:
#                 continue
            
#             # Get ratings for common users
#             ratings_i = movie_i[common_users]
#             ratings_j = movie_j[common_users]
            
#             # Calculate denominators
#             denom_i = np.sqrt((ratings_i ** 2).sum())
#             denom_j = np.sqrt((ratings_j ** 2).sum())
            
#             # Skip if either denominator is zero
#             if denom_i == 0 or denom_j == 0:
#                 continue
            
#             # Calculate cosine similarity
#             numerator = (ratings_i * ratings_j).sum()
#             cos_sim = numerator / (denom_i * denom_j)
            
#             # Transform to [0,1] range
#             transformed_sim = (1 + cos_sim) / 2
            
#             # Store in similarity matrix (both positions due to symmetry)
#             similarity.iloc[i, j] = transformed_sim
#             similarity.iloc[j, i] = transformed_sim

#     # Set diagonal elements to NaN
#     np.fill_diagonal(similarity.values, np.nan)
    
#     return similarity

# start_time = datetime.now()
# print(f"Step 2 started at: {start_time.strftime('%Y-%m-%d %H:%M')}")
# raw_similarity = compute_movie_similarity(normalized_ratings)
# raw_similarity.to_csv('app_raw_similarity.csv', index=True)
# print(f"Step 2 completed in: {(datetime.now() - start_time).seconds / 60} minutes ({datetime.now()})")


# def top_k_similarity(similarity_matrix, k=30):
#     # Create a copy of the similarity matrix to modify
#     filtered_similarity = similarity_matrix.copy()
    
#     # Iterate over each row (movie)
#     for index in filtered_similarity.index:
#         # Get the row
#         row = filtered_similarity.loc[index]
        
#         # Get the top k largest values
#         top_k = row.nlargest(k)
        
#         # Set all values that are not in the top k to NaN, but only for the current row
#         filtered_similarity.loc[index, ~filtered_similarity.columns.isin(top_k.index)] = np.nan

#     return filtered_similarity

# print(f"Step 3 started at: {start_time.strftime('%Y-%m-%d %H:%M')}")
# top_30_similarity = top_k_similarity(raw_similarity)
# top_30_similarity.to_csv('s_matrix.csv', index=True)
# print(f"Step 3 completed in: {(datetime.now() - start_time).seconds / 60} minutes ({datetime.now()})")

## -----------------------------------------------------------------------------------------------------


# Step 5: Define the myIBCF function
def myIBCF(newuser, similarity_matrix):
    # Load the similarity matrix
    S = similarity_matrix
    S_values = S.values  # Extract the numeric values of the similarity matrix
    print(len(S_values[1]))

    # Ensure newuser is a numpy array
    w = newuser  # Convert the user's ratings to a numpy array


    # Find the indices of the movies that the user has rated (non-NaN ratings)
    rated_indices = np.where(~np.isnan(w))[0]

    # Initialize an array to store the predicted ratings, filled with NaN initially
    score = np.full_like(w, np.nan, dtype=np.float64)

    # Iterate over each movie in the user's ratings
    for i in range(len(w)):
        if np.isnan(w[i]):  # Only predict for movies that the user has not rated
            # Find the indices where similarities for this movie are available (non-NaN)
            S_i = np.where(~np.isnan(S_values[i]))[0]

            # Compute the intersection of the rated indices and the valid similarity indices
            valid_indices = np.intersect1d(S_i, rated_indices)

            # Compute the predicted rating if there are valid indices
            if len(valid_indices) > 0:
                # Get the similarity weights and the ratings for the valid indices
                weights = S_values[i, valid_indices]
                ratings = w[valid_indices]

                # Calculate the predicted rating using a weighted average of the ratings
                score[i] = np.sum(weights * ratings) / np.sum(weights)

    # Get the column names (movie names) from the similarity matrix
    movie_names = S.columns

    # Create a pandas Series with the predicted ratings, using movie names as indices
    predictions = pd.Series(score, index=movie_names)

    # Get the top 10 movies based on the predicted ratings
    top_10_movies = predictions.nlargest(10)


    # # Check if fewer than 10 predictions are available (non-NA)
    # if top_10_movies.isna().sum() >= 10:  # If there are fewer than 10 predictions
    #     # Load popularity data from GitHub raw link
    #     popularity_data = pd.read_csv(popularity_url)
        
    #     # Save the popularity ranking to a file (to avoid recomputing)
    #     popularity_data.to_csv('popularity_ranking.csv', index=False)
        
    #     # Exclude the movies the user has already rated
    #     remaining_movies = popularity_data[~popularity_data['movie_name'].isin(movie_names[rated_indices])]
        
    #     # Add the top popular movies to fill the remaining slots
    #     remaining_movies = remaining_movies.nlargest(10 - len(top_10_movies), 'popularity')

    #     # Combine the top 10 predictions with the most popular remaining movies
    #     top_10_movies = pd.concat([top_10_movies, remaining_movies['movie_name']])


    # Return the top 10 movies
    return top_10_movies








# DISPLY 100 MOVIES ONTO THE WEBPAGE FOR USERS TO RATE
# What needs to be displayed:
# - Movie Numbnail
# - Movie Name
# - Interactable Option for users to rate the movie out of 5 stars

movie_100 = movies_df.head(100)

# Dictionary for the user movie ratings
ratings_dict = {}

# Container for aligning submit button
top_columns = st.columns((10,2))
with top_columns[1]:
    submit_rating = st.button("Finish Rating", use_container_width=True)

# Container to set the output 10 movie reviews
output_container = st.container(border=True)


st.markdown("<h3 style='text-align: center; padding-bottom: 20px'>Rate These Movies </h3>", unsafe_allow_html=True)
rating_container = st.container(height=800)

# Create the scrollable container for movie ratings
with rating_container:
    num_cols = 5
    rows = len(movie_100) // num_cols + (len(movie_100) % num_cols > 0)

    for row_id in range(rows):
        cols = st.columns(num_cols)
        for col_id, col in enumerate(cols):
            movie_id = row_id * num_cols + col_id
            if movie_id >= len(movie_100):
                break
            movie = movie_100.iloc[movie_id]
            with col:
                st.image(movie['thumbnail_url'], use_container_width=True)
                st.text(movie['title'])

                # # Capture the star rating selection for each movie
                # rating = st_star_rating("Rating", maxValue=5, defaultValue=0, key=f"rating_{movie['movie_id']}", size=20)

                rating = st.radio(f"Rating", options=["Select a rating", "★", "★★", "★★★", "★★★★", "★★★★★"], index=0, key=f"rating_{movie['movie_id']}")

                # # Store the rating in the ratings_dict, or set to NaN if not rated
                ratings_dict[movie['movie_id']] = rating if rating != 0 else np.nan

    for movie_id in ratings_dict:
        star_rating = ratings_dict[movie_id]
        ratings_dict[movie_id] = {"★": 1, "★★": 2, "★★★": 3, "★★★★": 4, "★★★★★": 5}.get(star_rating, np.nan)

if submit_rating:
    # Create a DataFrame from the ratings_dict, ensuring that unrated movies are set to NaN
    user_rating = {movie: ratings_dict.get(movie, np.nan) for movie in movie_100['movie_id']}
    ratings_df = pd.DataFrame(list(user_rating.items()), columns=["Movie ID", "Rating"])

    # RETRIEVE TOP 30 S MATRIX FROM GITHUB
    # This section is added to speed up the application run time so that the cosine similarity computation does not need to be repeated every time (~67 minutes for this dataset)

    top30_s_matrix_url = 'https://github.com/yh12chang/Cs_598_PSL_Project_4/raw/refs/heads/main/app_top30_s_matrix.csv'
    top30_s_matrix_response = requests.get(top30_s_matrix_url)

    csv_data = StringIO(top30_s_matrix_response.text)

    top30_s_matrix = pd.read_csv(csv_data, index_col=0)

    w = ratings_df.values  # Convert the user's ratings to a numpy array
    user_array = w[:, 1]

    additional_nan = np.full(3706 - len(user_array), np.nan)

    user_rating_array = np.concatenate((user_array, additional_nan))



    top_movies_df = myIBCF(user_rating_array, top30_s_matrix)

    st.write(top_movies_df.index)


    # Display a thank you message
    with output_container:
        st.write("Thank you for your ratings!")

        # Display the top 10 rated movies
        num_cols = 5
        rows = 2

        for row_id in range(rows):
            cols = st.columns(num_cols)
            for col_id, col in enumerate(cols):
                movie_idx = row_id * num_cols + col_id
                if movie_idx >= len(top_movies_df):
                    break
                movie_id = int(top_movies_df.index[movie_idx])

                movie = movies_df.loc[movie_id]
                with col:
                    st.image(movie['thumbnail_url'], use_container_width=True)
                    st.text(movie['title'])