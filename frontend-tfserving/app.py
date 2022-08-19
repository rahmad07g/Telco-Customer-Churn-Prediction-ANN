import json
import pandas as pd
import pickle
import streamlit as st
import requests

#load pipeline
preprocessor = pickle.load(open("model/preprocess_churn1.pkl" ,"rb"))

st.write("""
# Predict Telco Customer Churn

This app predicts the **Telco Customer Churn**!

Data obtained from the [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) in Kaggle by BlastChar.
""")

st.sidebar.header('User Input Features')

# Collects user input features into dataframe
sc = st.sidebar.selectbox('Is the customer a Senior Citizen ?',[0, 1])
part = st.sidebar.selectbox('Do the customer has partner?',["No","Yes"])
depend = st.sidebar.selectbox('Does the customer has dependent?',["No","Yes"])
tnr = st.sidebar.slider('How many months the customer has been a customer - Tenure (month)', 1,10,36)
multiple = st.sidebar.selectbox('Multiple Lines', ['Yes' ,'No' ,'No phone service'])
internet = st.sidebar.selectbox('Internet Service Provider', ['Fiber optic', 'DSL', 'No'])
if internet != "No":
    security = st.sidebar.selectbox("Online Security", ['Yes', 'No', 'No internet service'])
    backup = st.sidebar.selectbox("Online Backup", ['Yes', 'No' ,'No internet service'])
    protection = st.sidebar.selectbox("Device Protection", ['No' ,'Yes' ,'No internet service'])
    support =st.sidebar.selectbox("Tech Support", ['No', 'Yes' ,'No internet service'])
    tv = st.sidebar.selectbox("Streaming TV", ['No', 'Yes' ,'No internet service'])
    movies = st.sidebar.selectbox("Streaming Movies", ['No', 'Yes' ,'No internet service'])
else:
    security = (
    backup
    ) = (
        protection
    ) = support = tv = movies = "No internet service"
contr = st.sidebar.selectbox("Contract", ['Month-to-month', 'One year' ,'Two year'])
billing = st.sidebar.selectbox("Paperless Billing", ['Yes' ,'No'])
payment = st.sidebar.selectbox("Payment Method", ['Electronic check' ,'Credit card (automatic)' ,'Bank transfer (automatic)' ,'Mailed check'])
mc = st.sidebar.number_input("Monthly Charges", min_value=20)
tc = st.sidebar.number_input("Total Charges", min_value=0)
new_data = {"SeniorCitizen" : sc,
        "Partner" : part,
        "Dependents" : depend,
        "tenure" : tnr,
        "MultipleLines" : multiple,
        "InternetService" : internet,
        "OnlineSecurity" : security,
        "OnlineBackup" : backup,
        "DeviceProtection" : protection,
        "TechSupport" : support,
        "StreamingTV" : tv,
        "StreamingMovies" : movies,
        "Contract" : contr,
        "PaperlessBilling": billing,
        "PaymentMethod" : payment,
        "MonthlyCharges" : mc,
        "TotalCharges" : tc }

new_data = pd.DataFrame([new_data])

# build feature
new_data = preprocessor.transform(new_data)
new_data = new_data.tolist()


# inference 
URL = "http://deploy-ml1p2ragun.herokuapp.com/v1/models/model_churn:predict"
param = json.dumps({
        "signature_name":"serving_default",
        "instances":new_data
    })
r = requests.post(URL, data =param)

if r.status_code == 200:
    res = r.json()
    if res['predictions'][0][0] > 0.5 :
        st.title("Customer Churn")
    else :
        st.title("Customer Not Churn/likely Stay")
else :
    st.title("Unexpected Error")