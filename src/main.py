# Description: Main script to run the movie recommendation system
from data_processing import sample_and_preprocess_dataset, load_sampled_dataset
from recommendation import compute_tfidf_vectors, get_cosine_similarity, get_top_similar_items

def main():
    input_path = 'Data/movies_metadata.csv'
    output_path = 'Data/sampled_movies.csv'
    
    # Sample and preprocess dataset 
    sample_and_preprocess_dataset(input_path, output_path)
    
    # Load the sampled dataset
    df = load_sampled_dataset(output_path)
    documents = df['overview'].tolist()
    titles = df['original_title'].tolist()
    
    # Get user query for recommendation
    query = input("Enter the types of movie you like: ")
    
    # Compute TF-IDF vectors for the documents and query
    movie_vector, query_vector = compute_tfidf_vectors(query, documents)
    
    # Compute cosine similarity
    cosine_sim = get_cosine_similarity(query_vector, movie_vector)
    
    # Get indices for top 5 similar movies
    top_indices = get_top_similar_items(cosine_sim, top_n=5)
    
    # Print the title of the top 5 similar movies with similarity scores
    print("\nTop 5 recommended movies for you:")
    for idx in top_indices:
        print(f"Title: {titles[idx]}")
        print(f"Similarity Score: {cosine_sim[0][idx]:.4f}")
        print("-------------------------------------------------")

if __name__ == '__main__':
    main()
