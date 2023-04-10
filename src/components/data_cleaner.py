import pandas as pd
from src.exception import CustomException
from src.logger import logging
import regex as re
from bs4 import BeautifulSoup
'''import nltk
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer'''
import spacy


class DataCleaner:
    '''
        New Concept: @classmethod
        @classmethod is a decorator that is used to define a method that is bound to the class and not to an instance of the class. A class method receives the class as its first argument, conventionally named cls, and it can access class-level data.
        A class method is called using the class name, rather than an instance of the class.
        eg:
        class MyClass:
            class_variable = "Hello"

            @classmethod
            def my_class_method(cls, param1, param2):
                # do something with cls.class_variable, param1, and param2
                return result

        here I can directly call MyClass.my_class_method(param1, param2) instead of
        my_instance = MyClass()
        my_instance.my_class_method(param1, param2)

        cls is a variable that is used to refer to the class. It acts as same as self, but if you self the you have to create instance.
        usefull when you want to acess class multiple times.
        
        @staticmethod
        @staticmethod is a decorator that is used to define a method that is bound to the class rather than to an instance of the class.
        eg:
        class MyClass:
            class_variable = "Hello"

            @staticmethod
            def my_static_method(param1, param2):
                # do something with param1, param2, and MyClass.class_variable
                return result

        above if you see I am using a class_varibale as MyClass.class_variable, which is ok if you use class only once.
        its like using a class without self.
        
    '''

    # creating a lemmatizer object
    nlp = spacy.load('en_core_web_sm')
    
    # converting stop words to a set for faster processing
    stopwords = spacy.lang.en.stop_words.STOP_WORDS
    stopwords.difference_update({'not', 'no'})
    new_stopwords = set(stopwords.copy())

    @classmethod
    def clean_data(cls,data, pos):
        try:
            logging.info('Data cleaning started')
    
    
            # now we will interate through each review in the list and clean the data
            for i in range(data.shape[0]):
                # get the review
                review = data.iloc[i,pos]
    
                # remove the html tags
                clean_1_review = BeautifulSoup(review, features="html.parser").get_text()
    
                # convert to lower case
                clean_2_review = clean_1_review.lower()
    
                # remove any url's
                clean_3_review_1 = re.sub(r'http\S+', '', clean_2_review)
                clean_3_review = re.sub(r'www\S+', '', clean_3_review_1)
    
                # remove any non-letters
                clean_4_review = re.sub('[^a-zA-Z]', ' ', clean_3_review)
                clean_4_review = ' '.join(clean_4_review.split())
    
                # use spacy to tokenize the words
                clean_5_review = cls.nlp(clean_4_review)
    
                # removing stopwords and lemmatizing the words
                clean_6_review = [word.lemma_ for word in clean_5_review if word not in cls.new_stopwords]
    
                # join the words back into one string
                clean_7_review = ' '.join(clean_6_review)
    
                # update the review list with the cleaned review
                data.iloc[i,pos] = clean_7_review
    
            logging.info('Data cleaning completed')
            return data
    
        except Exception as e:
            logging.info('Error occurred in data cleaning')
            raise CustomException(e)

    @classmethod    
    def clean_custom_data(cls,review):
        try:
            logging.info('Data cleaning started')

            # remove the html tags
            clean_1_review = BeautifulSoup(review, features="html.parser").get_text()
    
            # convert to lower case
            clean_2_review = clean_1_review.lower()
    
            # remove any url's
            clean_3_review_1 = re.sub(r'http\S+', '', clean_2_review)
            clean_3_review = re.sub(r'www\S+', '', clean_3_review_1)
    
            # remove any non-letters
            clean_4_review = re.sub('[^a-zA-Z]', ' ', clean_3_review)
            clean_4_review = ' '.join(clean_4_review.split())
    
            # use spacy to tokenize the words
            clean_5_review = cls.nlp(clean_4_review)
    
            # removing stopwords and lemmatizing the words
            clean_6_review = [word.lemma_ for word in clean_5_review if word not in cls.new_stopwords]
    
            # join the words back into one string
            clean_7_review = ' '.join(clean_6_review)

            logging.info('Data cleaning completed')

            return clean_7_review
        
        except Exception as e:
            logging.info('Error occurred in data cleaning')
            raise CustomException(e)

'''        
# class inot main
if __name__ == '__main__':
    try:
        sample_data = pd.DataFrame({'review':['This is a 1234sample review', 'It is no!!!!!!!!!!!!t a good review.',' It https//google.com is a bad review.'],'label':[1,0,1]})
        print(sample_data)
        cleaned_data = DataCleaner.clean_data(sample_data, 0)
        print(cleaned_data.iloc[:,0])
    except Exception as e:
        logging.info('Error occured in main')
        raise CustomException(e)

'''
