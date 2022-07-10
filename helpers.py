"""
Helpers
"""
import ast

from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()


def convert(obj):
    """ get all in one list"""
    L = []
    for i in ast.literal_eval(obj):
        L.append(i['name'])
    return L


def casts(obj):
    """ get 3 first characters of movie"""
    counter = 0
    L = []
    for i in ast.literal_eval(obj):
        if counter != 3:
            L.append(i['name'])
            counter += 1
        else:
            break
    return L


def fetch_crew_director(obj):
    """Name of director of movie's crew"""
    director = []
    for i in ast.literal_eval(obj):
        if i['job'] == 'Director':
            director.append(i['name'])
            break
    return director


def stem(text):
    """Stemmatization of tags get root ow words"""
    L = []
    for w in text.split():
        L.append(ps.stem(w))
    return " ".join(L)

