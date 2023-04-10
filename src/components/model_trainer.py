import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SpatialDropout1D, Embedding, LSTM
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from src.logger import logging
from src.exception import CustomException
from sklearn.metrics import r2_score
import os
class ModelTrainer:
    def train_model(X_train_pad,X_test_pad,y_train,y_test,n_words,n_dim):
        try:
            logging.info('Model training started')
            # define the model
            logging.info('Defining the model')
            model = Sequential()
            model.add(Embedding(n_words,n_dim,input_length = X_train_pad.shape[1]))
            model.add(SpatialDropout1D(0.25))
            model.add(LSTM(100,dropout=0.25,recurrent_dropout=0.25))
            model.add(Dense(50,activation='relu'))
            model.add(Dense(25,activation='relu'))
            model.add(Dense(3,activation='softmax'))

            model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
            logging.info('Model defined successfully')

            logging.info('Creating a directory to store the model')

            # create a filepath to store the model
            model_filepath = os.path.join(os.getcwd(),'models','tweet_model')
            logging.info('Directory created successfully')

            # define the callbacks
            checkpoint = ModelCheckpoint(model_filepath, monitor='val_loss', verbose=1, save_best_only=True,save_format='tf')
            early = EarlyStopping(monitor='val_loss', patience=5)

            logging.info('Training the model')
            # train the model
            history = model.fit(X_train_pad, y_train,validation_split = 0.1, batch_size = 128 ,callbacks=[checkpoint,early],epochs=30)
            logging.info('Model trained successfully')
           
            # evaluate the model
            loss, accuracy = model.evaluate(X_test_pad, y_test, verbose=0)
            print('Accuracy: %f' % (accuracy*100))
            print('Loss: %f' % (loss*100))

            
            logging.info('Model training completed')
        
        except Exception as e:
            logging.info('Error occurred in model training')
            raise CustomException(e)
