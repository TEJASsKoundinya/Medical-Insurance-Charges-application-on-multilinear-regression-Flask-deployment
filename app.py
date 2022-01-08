from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

model = pickle.load(open('reg.pkl','rb'))
@app.route('/')
def main():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    d1 = request.form['1']
    d2 = request.form['2']
    d3 = request.form['3']
    d4 = request.form['4']
    d5 = request.form['5']
    d6 = request.form['6']
    arr = np.array([[d1,d2,d3,d4,d5,d6]])
    pred = model.predict(arr)
    return render_template('result.html', data = pred) 


if __name__ == '__main__':
    app.run(port=5001,debug=True)