import os.path
import requests
import json
from flask import Flask,request,render_template,url_for,Markup,redirect
import pickle
import pandas as pd
import numpy as np
import io
import torch
from torchvision import transforms
from PIL import Image
from utils.model import ResNet9
from utils.plant_disease import disease_dic
from utils.fertilizer_dic import fertilizer_dic
app = Flask(__name__)
crop_recommendation_path='nrfc.pkl'
crop_recommendation_model=pickle.load(open('nrfc.pkl','rb'))
fert_recommendation_path='fert-rfc.pkl'
fert_recommendation_model=pickle.load(open('fert-rfc.pkl','rb'))

disease_model_path="plant-disease-model.pth"
disease_model=ResNet9(3,38)
disease_model.load_state_dict(torch.load(disease_model_path,map_location=torch.device('cpu')))
disease_model.eval()

disease_classes = ['Apple___Apple_scab',
                   'Apple___Black_rot',
                   'Apple___Cedar_apple_rust',
                   'Apple___healthy',
                   'Blueberry___healthy',
                   'Cherry_(including_sour)___Powdery_mildew',
                   'Cherry_(including_sour)___healthy',
                   'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
                   'Corn_(maize)___Common_rust_',
                   'Corn_(maize)___Northern_Leaf_Blight',
                   'Corn_(maize)___healthy',
                   'Grape___Black_rot',
                   'Grape___Esca_(Black_Measles)',
                   'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
                   'Grape___healthy',
                   'Orange___Haunglongbing_(Citrus_greening)',
                   'Peach___Bacterial_spot',
                   'Peach___healthy',
                   'Pepper,_bell___Bacterial_spot',
                   'Pepper,_bell___healthy',
                   'Potato___Early_blight',
                   'Potato___Late_blight',
                   'Potato___healthy',
                   'Raspberry___healthy',
                   'Soybean___healthy',
                   'Squash___Powdery_mildew',
                   'Strawberry___Leaf_scorch',
                   'Strawberry___healthy',
                   'Tomato___Bacterial_spot',
                   'Tomato___Early_blight',
                   'Tomato___Late_blight',
                   'Tomato___Leaf_Mold',
                   'Tomato___Septoria_leaf_spot',
                   'Tomato___Spider_mites Two-spotted_spider_mite',
                   'Tomato___Target_Spot',
                   'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
                   'Tomato___Tomato_mosaic_virus',
                   'Tomato___healthy']

def predict_image(img,model=disease_model):
    transform=transforms.Compose([
        transforms.Resize(256),
        transforms.ToTensor()
    ])
    image=Image.open(io.BytesIO(img))
    img_t=transform(image)
    img_u=torch.unsqueeze(img_t,0)
    yb=model(img_u)
    _,preds=torch.max(yb,dim=1)
    prediction=disease_classes[preds[0].item()]
    return prediction

def weather(city):
    site = "http://api.openweathermap.org/data/2.5/weather?q="
    appid = "&appid=f82fe65f074195fafd536df296e5f538"
    url = site+city+appid
    response = requests.get(url)
    x=response.json()
    if x["cod"]!="404":
       y=x["main"]
       temperature=round((y['temp']-273.15),2)
       humidity=y["humidity"]
       return temperature,humidity
    else:
        return None
def simple_fert(crop):
    l=[]
    asl=['Maize','Sugarcane','Cotton','Tobacco','Paddy','Barley','Wheat','Millets','Oil seeds','Pulses','Ground Nuts']
    for i in range(len(asl)):
        if asl[i]==crop:
            l.append(1)
        else:
            l.append(0)
    return l
@ app.route('/')
def home():
    return render_template('index.html')
@app.route("/crop-recomendation")
def crop_recomendation():
    return render_template('crop-rec.html')
@app.route("/fertilizer-recomendation")
def fertilizer_recomendation():
    return render_template('fert-rec.html')
@app.route("/Plant-Disease")
def Plant_Disease():
    return render_template('upload.html')
@app.route("/crop-predicted",methods=["GET","POST"])
def crop_predicted():
    if request.method=="POST":
        N=int(request.form["nitrogen"])
        P=int(request.form["phosphorous"])
        K=int(request.form["potassium"])
        ph=float(request.form['ph'])
        city=request.form["city"]
        rainfall=float(request.form['rainfall'])
        if weather(city)!=None:
            temp,hum=weather(city)
            data=np.array([[N,P,K,temp,hum,ph,rainfall]])
            #cpred=crop_recommendation_model.predict(data)
            final_cpred=cpred[0]
            return render_template('crop-predicted.html',prediction=final_cpred)
        else:
            return render_template('try_again.html')
@app.route('/fert-predicted',methods=["GET","POST"])
def fert_predicted():
    if request.method=="POST":
        N = int(request.form["nitrogen"])
        P = int(request.form["phosphorous"])
        K = int(request.form["potassium"])
        moisture=int(request.form['moisture'])
        city = request.form["city"]
        crop=request.form['crop']
        temp,hum=weather(city)
        l=simple_fert(crop)
        asli=[[temp, hum, moisture, N, K, P] + l]
        data=np.array(asli)
        fpred=fert_recommendation_model.predict(data)
        final_fpred=fpred[0]
        df=pd.read_csv('Data/Fertilizer_prediction.csv')
        nr=df[df['Crop Type'] == crop]['Nitrogen'].iloc[0]
        pr=df[df['Crop Type'] == crop]['Phosphorous'].iloc[0]
        kr=df[df['Crop Type'] == crop]['Potassium'].iloc[0]
        n=nr-N
        p=pr-P
        k=kr-K
        temp={abs(n):"N",abs(p):"P",abs(k):"K"}
        max_value=temp[max(temp.keys())]
        if max_value == "N":
            if n < 0:
                key = 'NHigh'
            else:
                key = "Nlow"
        elif max_value == "P":
            if p < 0:
                key = 'PHigh'
            else:
                key = "Plow"
        else:
            if k < 0:
                key = 'KHigh'
            else:
                key = "Klow"

        response = Markup(str(fertilizer_dic[key]))
        return render_template('fpredicted.html',fprediction=final_fpred,dic=response)
    else:
        return render_template('try_again.html')
@app.route('/disease',methods=["GET","POST"])
def disease_prediction():
    if request.method=='POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file=request.files['file']
        if file:
            img=file.read()
            prediction=predict_image(img)
            prediction=Markup(str(disease_dic[prediction]))
            return  render_template('Output.html',disease=prediction)
    return render_template('upload.html')
if __name__ == "__main__":
    app.run(debug=True)
