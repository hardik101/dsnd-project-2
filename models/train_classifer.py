import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import re
import nltk
nltk.download(['punkt', 'stopwords','wordnet']) # download 'wordnet'for lemmatization
from nltk.tokenize import word_tokenize,sent_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
#from sklearn.externals import joblib
import pickle
import warnings
warnings.filterwarnings("ignore")
#lemmatize and remove stop words
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def load_data(database_filepath):
    '''Load data  from the database

    Params:
    database_filepath : path relative to application directory where .db file is located

    Return Value:
    X: Dataframe which includes messages
    Y: Dataframe which includes categories
    categoy_names: List of categories
    '''
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql(sql='select * from DisasterResponse',con=engine)
    X = df.message.values
    Y = df.iloc[:,3:]
    category_names = Y.columns
    return X,Y,category_names

def tokenize(text):
    '''Tokenize given text (assuming it is written in English language)

    Params:
    text : text string to be tokenized

    Return Value:
    final_tokens: List of token words which are cleaned
    '''
    #Lower case and remove any punctuation
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())

    #tokenize text
    tokens = word_tokenize(text)
    final_tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]

    return final_tokens

def build_model():
    '''Build model for classification (multi class)
    We will use RandomForestClassifier first and build our model
    Alternatively, OneVsRestClassifier can also be used to build our model

    Return Value:
        Multi class classification Model
    '''

    # Below code builds a pipline for RandomForestClassifier
    # If you are excuting this on local machine, consider changing value of n_jobs to 1 or 2
    # n_jobs = -1 will deploy all the cores available in CPU for processing which may stop the execution in between.
    # n_jobs value more than 1 in local machine also leads to error saying "indices and data should have the same size""
    # check https://joblib.readthedocs.io/en/latest/parallel.html#joblib.parallel_backend for more details
    pipeline_rf = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize, max_df=0.75)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(estimator=RandomForestClassifier(),n_jobs = -1))
    ])

    parameters_rf = {
        'vect__max_df': (0.5,0.75),
        'clf__estimator__n_estimators': [25,50]
    }

    # Below code builds a pipline for OneVsRestClassifier
    # Replace corresponding calling parmeters in GridSearchCV function call below to build model using it
    pipeline_one_vs_rest  = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize, max_df=0.75)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(OneVsRestClassifier(LinearSVC(random_state = 0)), n_jobs = -1))
    ])

    parameters_one_vs_rest = {
        'vect__max_df': (0.5,0.75),
        'tfidf__smooth_idf':[True, False],
        'clf__estimator__estimator__C': [1, 2, 5]
    }

    cv = GridSearchCV(pipeline_rf, param_grid= parameters_rf, n_jobs= -1, verbose=2)
    return cv

def evaluate_model(model, X_test, Y_test, category_names):
    '''Model Evaluation using standard matrics

    Params:
    1. model : model under evaluation
    2. X_test : test dataset (messages)
    3. Y_test : test labels (categories)
    4. catergories : list of categories
    '''

    Y_pred = model.predict(X_test)

    # change data's type to DataFrame
    Y_pred_df = pd.DataFrame(Y_pred,columns=category_names,index=range(Y_pred.shape[0]))
    Y_test_df = pd.DataFrame(Y_test,columns=category_names,index=range(Y_pred.shape[0]))
    Y_test_df.fillna(0,inplace=True)
    Y_test_df = Y_test_df.applymap(lambda x:int(x))

    '''print(classification_report(Y_test, Y_pred, target_names = category_names))
       print('---------------------------------')
       for i in range(Y_test.shape[1]):
           print('%25s accuracy : %.2f' %(category_names[i], accuracy_score(Y_test[:,i], Y_pred[:,i])))'''

    # for each category
    for column in category_names:
        # show f1_score,precision,recall
        curr_f1_report = classification_report(Y_test_df[[column]],Y_pred_df[[column]])
        print("*"*80)
        print("F1 score table for '%s' column:\n" %column)
        print(curr_f1_report)


def save_model(model, model_filepath):
    '''Save model at given path

    Params:
    1. model : model to be saved
    2. model_filepath : path where to save model
    '''
    model_pickle_file = open(model_filepath,'wb')
    pickle.dump(model,model_pickle_file)

def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print('Building model...')
        #model = build_model()

        print('Training model...')
        #model.fit(X_train, Y_train)

        print('Evaluating model...')
        #evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        #save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
