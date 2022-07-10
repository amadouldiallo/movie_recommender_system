
"""
Create your conda environment:
In your terminal: conda env create -f environment.yml

USAGE: python recommender_system.py
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from helpers import *
import pickle
# CSV files
movies_path = "./data/tmdb_5000_movies.csv"
credits_path = "./data/tmdb_5000_credits.csv"

pd.set_option('mode.chained_assignment',None)

# read files using pandas
print('Read files_____: ')
movies = pd.read_csv(movies_path)
_credits = pd.read_csv(credits_path)
print("Merge files by title column_____")
movies = movies.merge(_credits, on='title')
print(movies.head(5))

movies = movies[['movie_id', 'title','overview','genres','keywords','cast','crew']]
print("Remove NA and duplicated rows_____")

movies.dropna(inplace=True)


print("Get genres and keywords_____")

movies['genres'] = movies['genres'].apply(convert)

movies['keywords'] = movies['keywords'].apply(convert)

print('Get 3 first characters f the movie____')
movies['cast'] = movies['cast'].apply(casts)

print("Directors of movies _____")
movies['crew'] = movies['crew'].apply(fetch_crew_director)

print("Split overview in list____")
movies['overview'] = movies['overview'].apply(lambda x : x.split())

## remove space between tag
movies['genres'] = movies['genres'].apply(lambda x : [i.replace(" ", "") for i in x])
movies['keywords'] = movies['keywords'].apply(lambda x : [i.replace(" ", "") for i in x])
movies['cast'] = movies['cast'].apply(lambda x : [i.replace(" ", "") for i in x])
movies['crew'] = movies['crew'].apply(lambda x : [i.replace(" ", "") for i in x])

print("Create new column : tag")
movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] +  movies['cast'] + movies['crew']
print("New dataframe")
new_df = movies[['movie_id','title', 'tags']]
print(f"Shape :{new_df.shape}")
print(new_df.head(5))

print("Join tags in one")
new_df['tags'] = new_df.tags.apply(lambda x: ' '.join(x))

print(new_df.head(1))
print("After stemmization")
print(stem(new_df['tags'][0]))
print('Apply stemmization')
new_df.loc[:,'tags'].apply(stem)
new_df.drop_duplicates(subset=['movie_id'], inplace=True)

print("Create Counvectorizer")

cv = CountVectorizer(stop_words='english', ngram_range=(1, 2))

vectors = cv.fit_transform(new_df['tags']).toarray()

print("Compute similarity ...")
similarity  = cosine_similarity(vectors)
similarity.shape


def recommend(title):
    """Get recommend movies based of movie title"""
    movie_index = new_df[new_df['title']== title].index[0]
    distances = similarity[movie_index]
    movie_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:10]
    return [new_df.iloc[mv[0]].title for mv in movie_list]


print("Recommended based of 'Batman Begins'")
print(recommend("Batman Begins"))

print("Save Movies and Similarity")
pickle.dump(new_df.to_dict(), open('data/movies_1.pkl', 'wb'))
pickle.dump(similarity, open('data/similarity_1.pkl', 'wb'))




