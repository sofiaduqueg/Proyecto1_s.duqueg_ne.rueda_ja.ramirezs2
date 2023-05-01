import pandas as pd
import numpy as np
import joblib
import seaborn as sns; sns.set()
import nltk
import re, string, unicodedata

from nltk import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import LancasterStemmer, WordNetLemmatizer
from nltk.stem.snowball import SpanishStemmer

from sklearn import metrics
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, HashingVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.svm import SVC
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import classification_report, confusion_matrix, mean_squared_error, mean_absolute_error, r2_score
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LinearRegression, LogisticRegression, SGDClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.datasets import make_classification

import matplotlib as mplt
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn import tree
import pickle
from langdetect import detect

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')


df = pd.read_csv('./static/assets/csv/MovieReviews.csv', encoding = "ISO-8859-1", sep = ',')
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

def detect_language(text):
    try:
        lang = detect(text)
    except:
        lang = 'unknown'
    return lang

df['language'] = df['review_es'].apply(detect_language)

duplicated_rows = df[df.duplicated(subset=['review_es'])]

df = df.drop_duplicates(subset=['review_es'])

df = df[df['language'] == 'es']

df = df.drop('language', axis=1)

def replace_punctuation(review):
    """Replace punctuation with space"""
    return re.sub(r"""
               [,.;@#?!&$]+  # Accept one or more copies of punctuation
               \ *           # plus zero or more copies of a space,
               """,
               " ",          # and replace it with a single space
               review, flags=re.VERBOSE)

def to_lowercase(words):
    """Convert all characters to lowercase from list of tokenized words"""
    new_words = []
    for word in words:
        new_words.append(word.lower())
    return new_words

def remove_punctuation(words):
    """Remove punctuation from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = re.sub(r'[^\w\s]', '', word)
        if new_word != '':
            new_words.append(new_word)
    return new_words

def remove_stopwords(words):
    """Remove stop words from list of tokenized words"""
    stop_words = set(stopwords.words('spanish'))
    new_words = []
    for word in words:
        if word not in stop_words:
            new_words.append(word)
    return new_words

def replace_spanish_words(words):
    """Cambiar palabras tildadas por sus equivalentes sin tildes"""
    new_words = []
    replacements = (
        ("á", "a"),
        ("é", "e"),
        ("í", "i"),
        ("ó", "o"),
        ("ú", "u"),
    )
    for word in words:
        new_word = word
        for a, b in replacements:
            new_word = new_word.replace(a, b).replace(a.upper(), b.upper())
        new_words.append(new_word)
    return new_words
    
def preprocessing(words):
    words = to_lowercase(words)
    words = remove_punctuation(words)
    words = remove_stopwords(words)
    words = replace_spanish_words(words)
    return words

df['words'] = df['review_es'].apply(replace_punctuation).apply(word_tokenize).apply(preprocessing) #Aplica la eliminación del ruido

df['sentimiento'] = df['sentimiento'].replace({'negativo': 0, 'positivo': 1})

df['words'] = df['words'].apply(lambda x: ' '.join(map(str, x)))

X_train, X_test, y_train, y_test = train_test_split(df['words'], df['sentimiento'], test_size = 0.3, random_state = 1)

text_svm = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('svm', SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, random_state=42, max_iter=5, tol=None)) ])


text_svm.fit(X_train, y_train)

with open('./static/assets/txt/review.txt') as f:
    lines = f.readlines()

predicted = text_svm.predict(lines)
with open('./static/assets/txt/sentiment.txt', 'w') as f:
    f.write(str(predicted[0]))

