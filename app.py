from TextPreprocess import textPreprocessing, vectorization ,load_model , predictions
import pickle
import numpy as np
from flask import Flask,render_template,url_for,request




ohe= pickle.load(open('ohe.pkl','rb'))
classes=ohe.categories_[0].tolist()

app = Flask(__name__)

@app.route('/')

def home():
    return render_template('home.html')

@app.route('/predict',methods=['POST'])

def predict():

    if request.method == 'POST':
        message = request.form['message']
        message= textPreprocessing(message)
        output= vectorization([message])
        model = load_model()
        prediction_array = predictions(model,output)

        my_prediction = classes[np.argsort(prediction_array[0])[-1]]
        # print(my_prediction)

    return render_template('result.html',prediction = my_prediction.upper())


if __name__ == '__main__':
    app.run(host='0.0.0.0',port=8080)
