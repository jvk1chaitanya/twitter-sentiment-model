from flask import Flask,request,render_template

from src.pipeline.predict_pipeline import PredictPipeline
from src.exception import CustomException
from src.logger import logging

app = Flask(__name__)

@app.route('/')
def home():
    try:
        logging.info('Home page loaded')
        return render_template('index.html')
    
    except Exception as e:
        logging.info('Error occurred in loading home page')
        raise CustomException(e)

@app.route('/predict',methods=['GET','POST'])
def predict():
    try:
        if request.method == 'POST':
            logging.info('Prediction started')
            review = request.form.get('tweet')
            prediction = PredictPipeline.get_prediction(review)
            logging.info('Prediction completed')
            return render_template('home.html',prediction=prediction,my_tweet=review)
        else:
            return render_template('home.html')
    
    except Exception as e:
        logging.info('Error occurred in predicting the data')
        raise CustomException(e)
    

if __name__=="__main__":
    app.run(host="0.0.0.0",debug=True) 