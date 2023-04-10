import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import os
from src.components.data_cleaner import DataCleaner
import numpy as np
from src.exception import CustomException
from src.logger import logging
import pandas as pd

class PredictPipeline:
    # load the tokenizer
    tokenizer_path = os.path.join(os.getcwd(),'models','tokenizer.pkl')
    with open(tokenizer_path, 'rb') as f:
        tokenizer = pickle.load(f)

    # load the model
    model = tf.keras.models.load_model('models/tweet_model')

    @classmethod
    def get_prediction(cls,review):
        try:
            logging.info('Prediction started')

            # clean the data
            review_1 = DataCleaner.clean_custom_data(str(review))

            # tokenize the data
            # take the review in a list or else the tokenizer takes each word as a token
            X_test_pad = cls.tokenizer.texts_to_sequences([review_1])

            # pad the data
            X_test_pad = pad_sequences(X_test_pad, maxlen=50)

            # predict the data
            y_pred = cls.model.predict(X_test_pad)

            label = ['Negative','Neutral','Positive']

            # return the label
            return label[np.argmax(y_pred)]
        
        except Exception as e:
            logging.info('Error occurred in predicting the data')
            raise CustomException(e)

    @classmethod    
    def get_prediction_dataframe(cls,data,pos):
        try:
            logging.info('Prediction started')

            # clean the data
            data_1 = DataCleaner.clean_data(data, pos)

            # get the review column
            data_2 = data_1[pos]

            # tokenize the data
            X_test_pad = cls.tokenizer.texts_to_sequences(data_2)

            # pad the data
            X_test_pad = pad_sequences(X_test_pad, maxlen=40)

            # predict the data
            y_pred = cls.model.predict(X_test_pad)

            label = ['Negative','Neutral','Positive']

            # return the label
            predictions = []
            for i in range(len(y_pred)):
                predictions.append(label[np.argmax(y_pred[i])])
            
            return predictions
        
        except Exception as e:
            logging.info('Error occurred in predicting the data')
            raise CustomException(e)