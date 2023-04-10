import pandas as pd
import numpy as np
from src.exception import CustomException
from src.logger import logging
from src.components.data_cleaner import DataCleaner
from src.components.data_transformer import DataTransformer
from src.components.model_trainer import ModelTrainer

from tensorflow.keras.preprocessing.sequence import pad_sequences

class TrainPipeline:
    def train_pipeline(data_path):
        try:
            logging.info('Training pipeline started')
            
            logging.info('Reading the data')
            # Read the data
            data = pd.read_csv(data_path,encoding='utf-8',header=None)
            logging.info('Data read successfully')
            
            logging.info('Data prevalidation started')
            # removing rows which has sentiment as Irrelavant
            data_2 = data[data[2] != 'Irrelevant']

            # removing rows having NaN values
            data_2 = data_2.dropna()

            # dropping the columns which are not required
            data_3 = data_2.drop(columns = [0,1],axis=1)
            logging.info('Data prevalidation completed')

            # clean the data
            data_4 = DataCleaner.clean_data(data_3, 1)

            # split the data into train and test
            X_train,X_test,y_train,y_test = DataTransformer.split_data(data_4)

            # max length of the sequence
            max_len = 50
            
            # tokenize the data
            X_train_pad,X_test_pad,tokenizer = DataTransformer.tokenize_data(X_train,X_test,max_len)

            # find the number of words in the vocabulary
            n_words = len(tokenizer.word_index) + 1
            n_dim = 50

            print(X_train_pad.shape,n_words,n_dim)
            #train the model
            ModelTrainer.train_model(X_train_pad,X_test_pad,y_train,y_test,n_words,n_dim)
            
            logging.info('Training pipeline completed')
        
        except Exception as e:
            logging.info('Error occurred in training pipeline')
            raise CustomException(e)
        
# run main function
if __name__ == '__main__':

    TrainPipeline.train_pipeline(r'C:\Users\jvkch\OneDrive\Desktop\extra\python\tweet_data\Data\twitter_training.csv')
    print('Completed')