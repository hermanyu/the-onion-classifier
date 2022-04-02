import pandas as pd

from nltk.tokenize import sent_tokenize, word_tokenize, RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer

from collections import Counter

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_score, accuracy_score, recall_score, f1_score


"""
Text-based Helpers, Overwriters, and Readers (T.H.O.R)
"""


def tokenize(ser, pattern = r'(\b\w\w+\b)', regex=True):
    """
    Returns a list of tokens for each text entry in a series.

    Parameters
    ----------
    ser : pd.Series
        Series containing text/sentences to tokenize.
    pattern : regular expression, optional
        regular expression to extract tokens. The default is '\b\w\w+\b'.
    regex : boolean, optional
        If True, uses regex to extract tokens. Otherwise, calls word_tokenize from nltk.tokenize package. The default is True.

    Returns
    -------
    copy : pd.Series
        returns series with all text replaced by list of tokens.

    """
    tokenizer = word_tokenize
    if regex == True:
        regextokenizer = RegexpTokenizer(pattern)
        tokenizer = regextokenizer.tokenize
        
    copy = ser.copy(deep=True)
    
    for i in copy.index:
        copy[i] = tokenizer(ser[i])
    
    return copy



def lemmatize(ser):
    lem = WordNetLemmatizer()
    copy = ser.copy(deep=True)

    for i in copy.index:
        copy[i] = ' '.join([lem.lemmatize(token) for token in ser[i]])
            
    return copy
            

def stem(ser):
    
    p_stemmer = PorterStemmer()
    copy = ser.copy(deep=True)

    for i in copy.index:
        copy[i] = ' '.join([p_stemmer.stem(token) for token in ser[i]])
            
    return copy
            


def vaderize(ser, column_label = 'text'):
    sia = SentimentIntensityAnalyzer()
    
    return pd.DataFrame([ {**sia.polarity_scores(text), **{column_label:text}} for text in ser ])




def remove_stop_words(ser):
    stop_words = set(stopwords.words('english'))
    return tokenize(ser).map(lambda x: ' '.join([word for word in x if word not in stop_words]))




def count_words(ser):
    counter = Counter()
    tokenize(ser).apply(counter.update)
    
    return pd.Series(counter)


def change_threshold(probs, threshold):
    pos_probs = pd.Series([p[1] for p in probs])
    return pos_probs.map(lambda p: 1 if p>=threshold else 0)

def verbose_eval(estimator, data, actual, threshold=0.5):
    preds = change_threshold(estimator.predict_proba(data), threshold)
    ConfusionMatrixDisplay(confusion_matrix(actual, preds)).from_predictions(actual, preds);
    print(" ")
    print("Accuracy: ", accuracy_score(actual, preds))
    print(" ")
    print("Recall: ", recall_score(actual, preds))
    print(" ")
    print("Precision: ", precision_score(actual, preds))
    print(" ")
    print("F1: ", f1_score(actual, preds))
    print(" ")
