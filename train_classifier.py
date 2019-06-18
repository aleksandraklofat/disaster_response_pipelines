# Import libraries
import sys
from sqlalchemy import create_engine
import pandas as pd
import numpy as np

import nltk
nltk.download(['punkt', 'wordnet','stopwords','words'])
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

import re

from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, make_scorer

import pickle




def load_data(database_filepath):
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('disaster_messages',con=engine)
    
    X = df['message']
    Y = df.iloc[:,4:]
    category_names = list(df.columns[4:]) # check
    
    return X,Y,category_names
    


def tokenize(text):
    '''Tockenizing function processing text data'''
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower()) # removing case, punctation characters
    words = word_tokenize(text) # tokenizing for words
    words = stopwords.words("english")
    lemmatizer = WordNetLemmatizer()
    lemmed_words = [lemmatizer.lemmatize(w, pos='n').strip() for w in words]
    tokens = [lemmatizer.lemmatize(w, pos='v').strip() for w in lemmed_words] 
    
    return tokens
    
          


def build_model():
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
    
    Y_pred = model.predict(X_test)
    
    for i in range(len(category_names)):
        print("Category:", category_names[i],"\n", classification_report(Y_test.iloc[:, i].values,Y_pred[:, i]))
        print('Accuracy %25s: %.2f' %(category_names[i], accuracy_score(Y_test.iloc[:, i].values, Y_pred[:,i])))
    
    

def save_model(model, model_filepath):
    pickle.dump(model, open(model_filepath, "wb"))
    


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