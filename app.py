import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

@st.cache_data
def load_data():
    df = pd.read_csv("adult.csv")
    df.replace(' ?', pd.NA, inplace=True)
    df.dropna(inplace=True)
    return df

def preprocess_data(df):
    label_encoders = {}
    for column in df.select_dtypes(include='object').columns:
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column])
        label_encoders[column] = le
    return df, label_encoders

@st.cache_resource
def train_model():
    df = load_data()
    df, encoders = preprocess_data(df)
    X = df.drop("income", axis=1)
    y = df["income"]
    model = RandomForestClassifier()
    model.fit(X, y)
    return model, encoders, X.columns

def predict(model, encoders, input_data, feature_names):
    df = pd.DataFrame([input_data], columns=feature_names)
    return model.predict(df)[0]

st.title("🧠 AI Employee Salary Prediction")

model, encoders, feature_names = train_model()

with st.form("input_form"):
    age = st.slider("Age", 18, 70, 30)
    workclass = st.selectbox("Workclass", encoders['workclass'].classes_)
    education = st.selectbox("Education", encoders['education'].classes_)
    marital = st.selectbox("Marital Status", encoders['marital-status'].classes_)
    occupation = st.selectbox("Occupation", encoders['occupation'].classes_)
    relationship = st.selectbox("Relationship", encoders['relationship'].classes_)
    race = st.selectbox("Race", encoders['race'].classes_)
    sex = st.selectbox("Sex", encoders['sex'].classes_)
    hours = st.slider("Hours per week", 1, 100, 40)
    native = st.selectbox("Native Country", encoders['native-country'].classes_)

    submitted = st.form_submit_button("Predict Salary")

if submitted:
    input_data = {
        "age": age,
        "workclass": encoders['workclass'].transform([workclass])[0],
        "education": encoders['education'].transform([education])[0],
        "marital-status": encoders['marital-status'].transform([marital])[0],
        "occupation": encoders['occupation'].transform([occupation])[0],
        "relationship": encoders['relationship'].transform([relationship])[0],
        "race": encoders['race'].transform([race])[0],
        "sex": encoders['sex'].transform([sex])[0],
        "hours-per-week": hours,
        "native-country": encoders['native-country'].transform([native])[0]
    }

    for col in feature_names:
        if col not in input_data:
            input_data[col] = 0

    prediction = predict(model, encoders, input_data, feature_names)
    result = encoders["income"].inverse_transform([prediction])[0]
    st.success(f"✅ Predicted Salary Category: **{result}**")
