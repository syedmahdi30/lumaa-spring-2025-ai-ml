import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# This function samples and preprocesses the dataset
def sample_and_preprocess_dataset():
    # Load the dataset
    df = pd.read_csv('Data/movies_metadata.csv')
    
    # Extract only the necessary columns
    subset_df = df[['original_title', 'overview']]
    
    # Sample 500 rows from the dataset
    sample_subset_df = subset_df.sample(n=500, random_state=42)
    
    # Save the sampled dataset
    sample_subset_df.to_csv('Data/sampled_movies.csv', index=False)
    
    return


# This function computes TF-IDf vectors
def compute_idf_vectors(query, documents):
    # Initialize the TF-IDF Vectorizer
    vectorizer = TfidfVectorizer(stop_words='english')
    
    # Compute TF-IDF Vectors for documents and query
    movie_vector = vectorizer.fit_transform(documents)
    query_vector = vectorizer.transform([query])
    
    return movie_vector, query_vector

# This function computes cosine similarity between query and movies
def get_cosine_similarity(query_vector, movie_vector):
    
    cosine_sim = cosine_similarity(query_vector, movie_vector)
    
    return cosine_sim

def main():
    #Preprocess the dataset
    sample_and_preprocess_dataset()
    # Load the sampled dataset
    file_path = 'Data/sampled_movies.csv'
    with open(file_path, "r", encoding="utf-8") as file:
        documents = file.readlines()

    # Extract titles from documents
    titles = [line.split(',')[0] for line in documents]

    # Get user query for recommendation
    query = input("Enter the types of movie you like: ")
    
    # Get TF-IDF vectors for user query and movies
    movie_vector, query_vector = compute_idf_vectors(query, documents)

    # Compute cosine similarity between user query and movies
    cosine_sim = get_cosine_similarity(query_vector, movie_vector)

    # Find the top 5 most similar movies
    similar_movies = cosine_sim.argsort()[0][-5:][::-1]

    # Print the title of top 5 most similar movies and similarity score
    print("Top 5 recommended movies for you:")
    for movie in similar_movies:
        print(titles[movie])
        print(f"Similarity Score: {cosine_sim[0][movie]}")
        print("-------------------------------------------------")

if __name__ == "__main__":
    main()



