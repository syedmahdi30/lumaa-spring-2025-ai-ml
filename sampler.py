import pandas as pd

# This is how I sampled the dataset to get a smaller dataset
# I sampled 500 rows from the orginal dataset and got rid of unnecessary columns

df = pd.read_csv('Data/movies_metadata.csv')

subset_df = df[['original_title', 'overview']]

sample_subset_df = subset_df.sample(n=500, random_state=42)

sample_subset_df.to_csv('Data/sampled_movies.csv', index=False)


