# -*- coding: utf-8 -*-
"""
Created on Mon Jun 19 01:03:46 2022

@author: SNEHA
"""

# Importing data analysis packages
import pandas as pd
import numpy as np
import glob
import bs4
from wordcloud import wordcloud
import matplotlib.pyplot as plt
import warnings


# Importing natural language processing packages
import re
import emoji
import string
from nltk.corpus import stopwords
from ekphrasis.dicts.emoticons import emoticons


# Importing model selection and feature extraction packages
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline

# Importing machine learning packages
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, precision_score, confusion_matrix


warnings.filterwarnings("ignore")

# Deifining Global Variables

NUMERIC_PATTERN = '\w*\d\w*'
HTTP_URL_PATTERN = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
URL_PATTERN = '[/]?watch(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
HTML_APOSTROPHE_PATTERN = '&#39'
COLUMN_CONTENT = 'CONTENT'


# Reading all the csv files and merging them into a single dataframe
def get_dataset(path):
    '''
    A helper method which merges the dataset into a single dataframe.

    Parameters
    ----------
    path : str
        Path of the data folder from which the separate files can be accessed

    Returns
    -------
    data : pandas.DataFrame
        Merged dataset.

    '''
    files = glob.glob(path + '/*.csv')

    data = list()

    for file in files:
        data.append(pd.read_csv(file))

    return pd.concat(data)


# Replace urls with 'url' token and replace '&#39' in the data
def replace_url(comments, replacement_word):
    '''Helper function to replace urls in the data'''
    comments = str(comments)
    comments = re.sub(HTTP_URL_PATTERN, replacement_word, comments)
    comments = re.sub(URL_PATTERN, replacement_word, comments)
    comments = re.sub(HTML_APOSTROPHE_PATTERN, "'", comments)
    return comments


# Remove extra white space
def remove_extra_spaces(comments):
    '''Helper function to remove extra whitespace from the data'''
    comments = re.sub(r'\s+', ' ', comments)
    return comments


# Remove characters with encoding like \ufeff
# (The first 128 Unicode code point values are the same as ASCII)
def remove_encoding(text):
    '''Helper function to remove non ascii characters'''
    return ''.join(i for i in text if ord(i) < 128)


# Replace emojis with '<emoji>' token
def replace_emojis(s):
    '''Helper function to replace emojis with '<emoji>' token'''
    return ''.join(
        '<emoji>' if c in emoji.UNICODE_EMOJI['en']
        else c for c in s
    )


# Replace emoticons with their actual description tag using ekphrasis package
def replace_emoticons(s):
    '''Helper function to replace emoticons with their actual description tag'''
    tokens = s.split(' ')
    for i, token in enumerate(tokens):
        if token in emoticons:
            tokens[i] = emoticons[token]
    return ' '.join(tokens)


def preprocess(data):
    '''
    A Helper function for cleaning the text data

    Parameters
    ----------
    data : pandas.DataFrame
        Dataframe to be cleaned

    Returns
    -------
    data : pandas.DataFrame
        Dataframe with cleaned text data

    '''

    # Remove HTML tags
    data[COLUMN_CONTENT] = data[COLUMN_CONTENT].apply(
        lambda x: bs4.BeautifulSoup(x, 'lxml').get_text()
    )

    # Other Preprocessing
    data[COLUMN_CONTENT] = data[COLUMN_CONTENT].apply(
        lambda x: replace_url(x, 'url')
    )
    data[COLUMN_CONTENT] = data[COLUMN_CONTENT].apply(
        lambda x: remove_extra_spaces(x)
    )
    data[COLUMN_CONTENT] = data[COLUMN_CONTENT].apply(
        lambda x: remove_encoding(x)
    )
    data[COLUMN_CONTENT] = data[COLUMN_CONTENT].apply(
        lambda x: replace_emojis(x)
    )
    data[COLUMN_CONTENT] = data[COLUMN_CONTENT].apply(
        lambda x: replace_emoticons(x)
    )

    # Remove numeric digits
    data[COLUMN_CONTENT] = data[COLUMN_CONTENT].apply(
        lambda x: re.sub(NUMERIC_PATTERN, '', x)
    )

    # Remove punctuations
    punctuations_to_remove = set(string.punctuation)
    data[COLUMN_CONTENT] = data[COLUMN_CONTENT].apply(
        lambda x: re.sub(
            '[%s]' % re.escape(''.join(punctuations_to_remove)),
            '',
            x
        )
    )

    return data


def transform_data(X, y):
    '''
    Function to extract features before model building

    Parameters
    ----------
    X : pandas.DataFrame
        DESCRIPTION.
    y : pandas.DataFrame
        DESCRIPTION.

    Returns
    -------
    sparse matrix
        Fit all transformers, transform the data and concatenate results
    preprocessor : sklearn.compose.ColumnTransformer
        preprocessor object

    '''
    # Creating stop words set and adding words that are common in
    # both Spam and Non Spam Comments
    edited_stop_words = set(stopwords.words('english'))
    edited_stop_words.add('video')
    edited_stop_words.add('like')
    edited_stop_words.add('br')

    # Segregating features
    text_features = COLUMN_CONTENT

    # Building a TF-IDF Vectorizer with Standard Scaler
    preprocessor = make_column_transformer(
        (TfidfVectorizer(
            stop_words='english',
            lowercase=True
        ), text_features)
    )

    return preprocessor.fit_transform(X, y), preprocessor


def mean_std_cross_val_scores(model, X_train, y_train, **kwargs):
    '''
    Returns mean and std of cross validation

    Parameters
    ----------
    model :
        scikit-learn model
    X_train : numpy array or pandas DataFrame
        X in the training data
    y_train :
        y in the training data

    Returns
    ----------
        pandas Series with mean scores from cross_validation
    '''

    scores = cross_validate(model, X_train, y_train, **kwargs)

    mean_scores = pd.DataFrame(scores).mean()
    std_scores = pd.DataFrame(scores).std()
    out_col = []

    for i in range(len(mean_scores)):
        out_col.append(('%0.3f (+/- %0.3f)' % (mean_scores[i], std_scores[i])))

    return pd.Series(data=out_col, index=mean_scores.index)


def train_models(preprocessor, X, y):
    '''
    A Function to train multiple candidate models 
    
    Parameters
    ----------
    preprocessor : sklearn.compose.ColumnTransformer
        preprocessor objects
    X : pandas.DataFrame
        Train Data
    y : pandas.DataFrame
        Train Labels

    Returns
    -------
    models : List
        List of dictionaries with sklearn.pipeline.Pipeline objects
    results : dict
        Stores results with the scores of all models

    '''
    results = {}

    # Defining the scoring metrics
    scoring = ['accuracy', 'precision']

    # Building Dummy Classifier as Baseline
    dummy_classifier_pipeline = make_pipeline(preprocessor, DummyClassifier())

    results['Dummy Classifier'] = mean_std_cross_val_scores(
        dummy_classifier_pipeline,
        X,
        y,
        scoring=scoring,
        return_train_score=True
    )

    # Building Logistic Regression
    logistic_regression_pipeline = make_pipeline(
        preprocessor,
        LogisticRegression(max_iter=2000)
    )

    results['Logistic Regression'] = mean_std_cross_val_scores(
        logistic_regression_pipeline,
        X,
        y,
        scoring=scoring,
        return_train_score=True
    )

    # Building Naive Bayes
    naive_bayes_pipeline = make_pipeline(preprocessor, MultinomialNB())

    results['Naive Bayes'] = mean_std_cross_val_scores(
        naive_bayes_pipeline,
        X,
        y,
        scoring=scoring,
        return_train_score=True
    )

    # Building Random Forest Classifier
    random_forest_pipeline = make_pipeline(
        preprocessor,
        RandomForestClassifier()
    )

    results['Random Forest'] = mean_std_cross_val_scores(
        random_forest_pipeline,
        X,
        y,
        scoring=scoring,
        return_train_score=True
    )

    models = [
        ('Random Forest', random_forest_pipeline),
        ('Logistic Regression', logistic_regression_pipeline)
    ]

    return models, results


def tune_models(models, X, y):
    '''A Function to perform hyperparameter optimization on the models'''

    index = [model[0] for model in models]

    # Parameter grid for hyperparameter optimization
    param_grid = {
        'Logistic Regression': {
            'logisticregression__C': np.arange(2, 7.5, 0.25),
            'logisticregression__solver': [
                'newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'
            ],
            'logisticregression__random_state': [0]
        },
        'Random Forest': {
            'randomforestclassifier__n_estimators': [200, 250, 300, 350],
            'randomforestclassifier__max_depth': [45, 50, 55, 60, 70],
            'randomforestclassifier__criterion': ['gini', 'entropy']
        }
    }

    # Storing best results in a dataframe
    optimized_results = pd.DataFrame(
        columns=['Selected Hyperparameters', 'Accuracy'],
        index=index
    )

    # Storing cv_results in a dataframe for further analysis
    cv_results = {}

    # Performing grid-search hyperparameter optimization to optimize results
    for i, model in enumerate(models):
        grid_search = GridSearchCV(
            model[1],
            param_grid=param_grid[model[0]],
            cv=5,
            scoring='accuracy',
            n_jobs=-1,
            return_train_score=True
        )

        grid_search.fit(X, y)

        optimized_results.loc[
            model[0], 'Selected Hyperparameters'
        ] = [grid_search.best_params_]

        optimized_results.loc[model[0], 'Accuracy'] = grid_search.best_score_

        cv_results[model[0]] = grid_search.cv_results_

    return cv_results, optimized_results


def train_best_model(preprocessor, optimized_results):
    '''A function to automate picking of best results from the optimized results'''
    rf_best_params = optimized_results["Selected Hyperparameters"]["Random Forest"][0]

    tuned_random_forest_pipeline = make_pipeline(
        preprocessor,
        RandomForestClassifier(
            criterion=rf_best_params['randomforestclassifier__criterion'],
            max_depth=rf_best_params['randomforestclassifier__max_depth'],
            n_estimators=rf_best_params['randomforestclassifier__n_estimators'],
        )
    )

    tuned_random_forest_pipeline.fit(X_train, y_train)

    predictions = tuned_random_forest_pipeline.predict(X_test)

    predictions = pd.DataFrame(
        predictions,
        index=X_test.index
    )

    conf_mat = confusion_matrix(y_test, predictions)

    return (
        tuned_random_forest_pipeline,
        accuracy_score(y_test, predictions),
        precision_score(y_test, predictions),
        conf_mat
    )


def predict(trained_model, X, y):
    '''A function to generate the final predictions on test set and evaluate the results'''
    predictions = trained_model.predict(X)

    predictions = pd.DataFrame(
        predictions,
        index=X_test.index
    )

    result_dataframe = pd.concat(
        [X[COLUMN_CONTENT], pd.Series(y), predictions],
        axis=1,
        ignore_index=True
    )

    result_dataframe.columns = ['Comments', 'True Class', 'Predicted Class']

    return result_dataframe


if __name__ == "__main__":
    youtube_data = get_dataset('data')

    # Dropping DATE and COMMENT_ID Column
    youtube_data = youtube_data.drop(['DATE', 'AUTHOR', 'COMMENT_ID'], axis=1)

    # Splitting the data and the labels
    comments_data = youtube_data.drop(columns=['CLASS'])
    labels = youtube_data['CLASS']

    # Splitting data to test and train before we do the EDA
    X_train, X_test, y_train, y_test = train_test_split(
        comments_data,
        labels,
        test_size=0.2,
        shuffle=True,
        random_state=0,
        stratify=labels
    )

    X_train = preprocess(X_train)
    X_test = preprocess(X_test)

    transformed_data, preprocessor = transform_data(X_train, y_train)

    models, results = train_models(preprocessor, X_train, y_train)

    cv_results, optimized_results = tune_models(models, X_train, y_train)

    tuned_model, accuracy, precision, conf_mat = train_best_model(
        preprocessor,
        optimized_results
    )

    print('The accuracy of the model based on the test set: {}'.format(
        accuracy
    ))

    print('The precision of the model based on the test set: {}'.format(
        precision
    ))

    print('The confusion matrix on the test set:\n{}'.format(
        conf_mat
    ))

    preds_df = predict(tuned_model, X_test, y_test)

    print(preds_df)
