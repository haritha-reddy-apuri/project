import pickle
from flask import Flask,request,app,jsonify,url_for,render_template
import numpy as np
import sklearn
app=Flask(__name__)
reg=pickle.load(open('regmodel.pkl','rb'))
scale=pickle.load(open('scaler.pkl','rb'))
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api',methods=['POST'])
def predict_api():
    data=request.json['data']
    print(data)
    print(np.array(list(data.values())).reshape(1,-1))
    new_data=scale.transform(np.array(list(data.values())).reshape(1,-1))
    output=reg.predict(new_data)
    print(output[0])
    return jsonify(output[0])
@app.route('/predict',methods=['post'])
def predict():
    data=[float(x) for x in request.form.values()]
    final_input=scale.transform(np.array(data).reshape(1,-1))
    print(final_input)
    output=reg.predict(final_input)
    return render_template("home.html",prediction_text="the house price prediction is{}".format(output[0]))

if __name__=="__main__":
    app.run(debug=True)


    

