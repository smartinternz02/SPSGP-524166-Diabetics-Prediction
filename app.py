import numpy as np
final_feature=[np.array([1,1,1,40,1,0,0,0,0,1,0,1,0,5,18,15,1,0,9,4,3])]

from flask import Flask, request, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html', **locals())

@app.route('/predict',methods=['POST','GET'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features=[]
    for x in request.form.values():
        int_features.append(int(x))
    final_features = [np.array(int_features)]
    prediction = model.predict(final_feature)
    if prediction[0]==0:
        dia="No diabetes"
    elif prediction[0]==1:
        dia="Prediabetes"
    elif prediction[0]==2:
        dia="Diabetes"
    return render_template('index.html', **locals())


if __name__ == "__main__":
    app.run(debug=True)
    
    
    
    
    
    
    
    
    
    
    
    
    
    