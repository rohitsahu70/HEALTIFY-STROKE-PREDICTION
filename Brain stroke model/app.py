from flask import Flask, render_template, request
import pandas as pd
import pickle
import numpy as np

filename ='D:\PROGRAMING\PROJECT\WEB DEVELOPMENT\HEALTIFY WEBSITE\Brain stroke model\model.pkl'
classifier = pickle.load(open(filename, 'rb'))

app = Flask(__name__)

@app.route('/')
def home():
	return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    input_features = [float(x) for x in request.form.values()]
    features_value = [np.array(input_features)]
    features_name = ['age', 'hypertension', 'heart_disease', 'ever_married',
       'work_type', 'Residence_type', 'avg_glucose_level', 'bmi',
       'smoking_status']
    df = pd.DataFrame(features_value, columns=features_name)
    output = classifier.predict(df)
    return render_template('result.html', prediction=output)

if __name__ == '__main__':
	app.run(debug=True)