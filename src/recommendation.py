# Description: This file contains the functions to compute the TF-IDF vectors, cosine similarity, and retrieve the top similar items.
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# This function computes the TF-IDF vectors for the movies and the user query
def compute_tfidf_vectors(query, documents):
    
    vectorizer = TfidfVectorizer(stop_words='english')
    
    # Compute TF-IDF Vectors for documents and query
    movie_vector = vectorizer.fit_transform(documents)
    query_vector = vectorizer.transform([query])
    
    return movie_vector, query_vector

# Computes the cosine similarity between the user query vector and the movie vectors
def get_cosine_similarity(query_vector, movie_vector):

    cosine_sim = cosine_similarity(query_vector, movie_vector)
    
    return cosine_sim

# Retrieves the indices of the top_n most similar items based on cosine similarity
def get_top_similar_items(cosine_sim, top_n=5):
    
    indices = cosine_sim.argsort()[0][-top_n:][::-1]
    
    return indices

