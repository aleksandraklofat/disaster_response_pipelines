# Import libraries
import sys
import sqlite3
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import nltk
nltk.download(['punkt', 'wordnet'])
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
nltk.download('words')
nltk.download('stopwords')
nltk.download('wordnet')
englishwords=set(nltk.corpus.words.words())
#Fixing a random seed
import random
random.seed(42)
import pickle
import re
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, make_scorer
from sklearn.externals import joblib


def load_data(database_filepath):
    '''Loading saved dataframe from the sqlite database'''
    engine = create_engine('sqlite:///DisasterResponse.db')
    df = pd.read_sql('DisasterResponse', engine)
    X = df['message']
    Y = df.iloc[:,4:]
    category_names=Y.columns
    return X, Y, category_names


def tokenize(text):
    '''Tockenizing function processing text data'''
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower()) # removing case, punctation characters
    words = word_tokenize(text) # tokenizing for words
    tokens = [w for w in words if w not in stopwords.words("english")] # removing stopwords
    lemmatizer = WordNetLemmatizer()
    
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
        return clean_tokens
    

def build_model():
    
    ''' Building the ML Model'''
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    
    parameters = {
        'tfidf__use_idf': (True, False),
        'clf__estimator__n_estimators': [10,20]
    }
    cv = GridSearchCV(pipeline, param_grid=parameters)
    
    return cv
    
   
def evaluate_model(model, X_test, Y_test, category_names):
    y_pred = model.predict(X_test)
    
    
    for i in range(35):
        print ('classification report of {}:'.format(category_names[i]))
        print(classification_report(Y_test[:,i], Y_pred[:,i]))
    



def save_model(model, model_filepath):
    '''Saving model as a pickle file'''
    with open(model_filepath, 'wb') as f:
        pickle.dump(model, f)
   


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()