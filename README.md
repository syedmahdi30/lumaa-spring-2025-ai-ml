# AI/Machine Learning Intern Challenge: Simple Content-Based Recommendation

**Dataset** 
 * Dataset is from [Kaggle] (https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset?select=movies_metadata.csv)
 * This has about 50,000 rows and 24 columns
 * I sampled 500 Rows from this dataset
 * Dataset has already been placed in the Data directory so no need to download and load it

**Setup**
 * Python Version: 3.10.2
 * Download Dependencies through this command: **pip install -r requirements.txt**

**Running**
* Clone the repository 
* Make sure you cd into the right directory should be **cd lumma-spring-2025-ai-ml**
* From there run this command **python3 src/main.py**

**Results**

* Sample Query: I like action movies set in space with a little bit of comedy
   * Output: Top 5 recommended movies for you:
   * Title: Daniel Tosh: Completely Serious | Similarity Score: 0.1649

   * Title: Šakalí léta |
    Similarity Score: 0.1254

   * Title: Space Cop |
   Similarity Score: 0.1178

   * Title: Monsters vs Aliens |
   Similarity Score: 0.1151

   * Title: Φτηνά Τσιγάρα |
   Similarity Score: 0.1099
* Sample Query: I like Romantic Comedies with a bit of thriller mixed in
   * Top 5 recommended movies for you:
   * Title: Cyrano de Bergerac |
   Similarity Score: 0.1185

   * Title: Ya Shagayu po Moskve |
   Similarity Score: 0.1127

   * Title: Bon appétit |
   Similarity Score: 0.1102

   * Title: Escape |
   Similarity Score: 0.1022

   * Title: Palmipédarium |
   Similarity Score: 0.0985

**Salary Expectation**
* Using the 20-30$ per hour on the listing my expectations are within the ranges of [$2000, $3000]


**Extra Commentary**
* When I first looked at this assignment I wasn't sure if I wanted to use TF-IDF vectors and compute cosine similarity, I thought I can implement something more efficient
* I am currently taking a class in Natural Language Processing and we are currently learning about word2vec models
* So at first I thought why don't I train a skip-gram model on this dataset which would most likely give me a better result than TF-IDF vectors
* Here's why I did'nt go with that idea:
   * Training a skip-gram model from scratch isn’t ideal given the small dataset and limited time
* But let's say for instance that my dataset was bigger, such as the one I sampled from
* We can absolutely use the skip-gram model to implement an even more efficient Content-Based Recommnedation System
* Just some thoughts and I appreciate the team from Lumaa for giving out such a fun and simple challenge
