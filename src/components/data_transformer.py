from keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import pandas as pd
from src.exception import CustomException
from src.logger import logging
import os
import pickle
class DataTransformer:

    def split_data(data):
        
        try :
            logging.info('Data splitting started')
            # convert y to one hot encoding
            y = pd.get_dummies(data[2])

            X = data[3]

            # split the data into train and test
            X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=0.2, random_state=42)

            # return the train and test data
            logging.info('Data splitting completed')
            return X_train,X_test,y_train,y_test
        
        except Exception as e:
            logging.info('Error occurred in data splitting')
            raise CustomException(e)

    

    def tokenize_data(X_train,X_test,max_len):
        try:
            logging.info('Data tokenization started')
            # tokenize the data
            tokenizer = Tokenizer(oov_token = '<OOV>')
            tokenizer.fit_on_texts(X_train)

            # convert the text to sequences
            X_train_seq = tokenizer.texts_to_sequences(X_train)
            X_test_seq = tokenizer.texts_to_sequences(X_test)

            # create a directory to store the tokenizer
            model_folder = r'models'
            os.makedirs(os.path.join(os.getcwd(),model_folder), exist_ok=True)

            # save the tokenizer

            tokenizer_path = os.path.join(os.getcwd(),model_folder,'tokenizer.pkl')
            with open(tokenizer_path, 'wb') as f:
                pickle.dump(tokenizer, f)

            # pad the sequences
            X_train_pad = pad_sequences(X_train_seq, maxlen=max_len,padding='post')
            X_test_pad = pad_sequences(X_test_seq, maxlen=max_len,padding='post')

            logging.info('Data tokenization completed')

            # return the padded sequences and tokenizer
            return X_train_pad,X_test_pad,tokenizer
        
        except Exception as e:
            logging.info('Error occurred in data tokenization')
            raise CustomException(e)
        
