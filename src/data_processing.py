# Description: This file contains functions to sample and preprocess the dataset for the recommender system.
import pandas as pd

# Loads dataset from input_path, extracts necessary columns, samples subset of rows, and saves sampled dataset to output_path
def sample_and_preprocess_dataset(input_path, output_path, sample_size=500, random_state=42):
    
    df = pd.read_csv(input_path, encoding='utf-8')
    
    # Extract only the necessary columns: title and overview
    subset_df = df[['original_title', 'overview']]
    
    # Sample rows and handle missing values in 'overview'
    sample_subset_df = subset_df.sample(n=sample_size, random_state=random_state)
    sample_subset_df['overview'] = sample_subset_df['overview'].fillna("")
    
    # Save the sampled dataset
    sample_subset_df.to_csv(output_path, index=False)
    
    return sample_subset_df

# Loads the sampled dataset from file_path and makes sure that missing values are handled
def load_sampled_dataset(file_path):

    df = pd.read_csv(file_path, encoding='utf-8')
    df['overview'] = df['overview'].fillna("")
    return df
