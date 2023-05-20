# Twitter Sentiment Classification using LSTM
The goal of this project is to build a Twitter sentiment classification model using LSTM, with the aim of accurately predicting the sentiment of a given tweet as positive, negative, or neutral. This model can have practical applications in social media monitoring, brand reputation management, and customer service.

# About the Dataset
The dataset used in this project is the Twitter US Airline Sentiment dataset. It contains 69,491 tweets related to various Sentiment categories (i.e Positive,Negative,Neutral & Irrelevant). The dataset is available on Kaggle and can be downloaded from [here](https://www.kaggle.com/datasets/jp797498e/twitter-entity-sentiment-analysis).

## Python Libraries Used
We have used various Python libraries in this project. Here's a brief overview of each library:

Pandas: Used for data manipulation and analysis.

Matplotlib: Used for data visualization.

Beautiful Soup: Used for parsing HTML and XML documents.

Spacy: Used for natural language processing tasks such as tokenization and lemmatization.

Re: Used for regular expression operations.

Tensorflow: Used for building and training deep learning models.

## Data Preprocessing
Before training the model, we need to perform some data preprocessing to remove any irrelevant data and make it suitable for our model. We have performed the following data preprocessing tasks:

Removed null values.
Removed tweets that were tagged as Irrelevant.

## Text Cleaning
In natural language processing, text cleaning plays a vital role in achieving better accuracy in our model. In this project, we have defined a function called cleaned_review which is used to clean our data. The function performs the following operations:

Removes any HTML tags from the review.
Removes any URLs from the review.
Removes any non-letter characters from the review.
Converts the whole sentence to lowercase and splits the words.
Removes irrelevant words (stopwords) and lemmatizes the final output.

## Tokenization and Padding
Tokenization is the process of breaking down text into smaller chunks or tokens. In this project, we have used the Tensorflow Keras tokenizer to tokenize the text data. After tokenization, we pad the sequences to ensure that they are of equal length. Padding ensures that all the input sequences are of the same length, which is required for training our LSTM model.

## Splitting of Data
To evaluate our model's performance, we need to split our data into training and testing sets. In this project, we have used the train_test_split method from Scikit-learn to split our data into training and testing sets.

## Model Architecture
Our model architecture consists of the following layers:

Embedding layer: Used to convert words into dense vectors.

SpatialDropout1D layer: Used for regularization.

LSTM layer: Used for learning the sequence dependencies.

Dense layers: Used for increasing the model's capacity.

Output layer: Used to produce a probability distribution over the three classes.

## Model Training
We have trained our model using the Adam optimizer and categorical cross-entropy loss. To prevent overfitting, we have used two techniques - model checkpoint and early stopping. The model checkpoint saves the best model during training, and the early stopping stops the training if the model's performance does not improve after a certain number of epochs. We have also used a validation step during training to evaluate the model's performance on the validation set.

## Prediction
To predict the sentiment of a new tweet, we have defined a function called "find_sentiment". This function takes a review as input, tokenizes it, and pads it to ensure that it is of the same length as the input sequences used during training. Finally, the function returns the predicted sentiment - positive, negative, or neutral.

## Conclusion
In this project, we have shown how to perform sentiment analysis on tweets using an LSTM-based model. Our model achieved an accuracy of 87.98% on the test set. This model can be used to analyze customer feedback, monitor brand reputation, predict future trends, and improve products and services accordingly.

## Video and Notebook for reference : 
Below is a Video of our model in action:

https://user-images.githubusercontent.com/83896570/230912764-7373e5c0-5f89-4cc5-923c-a88e9f7bac6a.mp4

\
You can find the notebook for this project 
1) By navigating to the [notebooks](https://github.com/jvk1chaitanya/twitter-sentiment-model/blob/4ec3568e5f31f929c0227eba5d340cd31d5506ee/Notebook/twitter-sentiment-classification-using-lstm.ipynb)
2) You can also find the notebook on [Kaggle](https://www.kaggle.com/code/jvkchaitanya410/twitter-sentiment-classification-using-lstm)
