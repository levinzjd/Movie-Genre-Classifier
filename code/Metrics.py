from keras import backend as K
import numpy as np
import keras


def precision(y_true, y_pred):
    '''
    Customized precision metric for Keras model log
    Input: y_true, y_pred
    Output: precision score
    '''
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def recall(y_true, y_pred):
    '''
    Customized recall metric for Keras model log
    Input: y_true, y_pred
    Output: recall score
    '''
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def f1(y_true, y_pred):
    '''
    Customized f1 metric for Keras model log
    Input: y_true, y_pred
    Output: f1 score
    '''
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall))
