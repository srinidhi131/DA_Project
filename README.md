This is the Project repository of Abraca-data

The repo contains the following files

1. netflix_titles.csv: This was the dataset used. It contains information about shows and movies on Netflix along with features like name, director, description(plot), listed_in(genre), country, release_date and rating
2. EDA.ipynb: This Jupyter notebook has code for pre-processing and visualization of data. it handles missing values and removes duplicated if any.
3. Model.ipynb: This jupyter notebook has the code for the three models developed:
   Model 1: Random Model: This recomends seven movies at random in a given genre.
   Model 2: Content-based Classifier: This recomends seven movies based on the plot of the movie. It uses TF-IDF
   Model 3: Content-based Classifier: For a more accurate reccomendation, we use a combination of three features based on which the recommendation will be done. The features used are Director, Cast and Genre. Create a word soup and use cosine similarity for generating recomendations.
4. Model.py: Python file of the model.ipynb
