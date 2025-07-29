import streamlit as st

import pandas as pd

import joblib


model joblib.load("best_model.pkl")

st.set_page_config(page_title="Employee Salary Classification", page_icon="", layout="centered")

st.title(" Employee Salary Classification App")

st.markdown ("Predict whether an employee earns >50K or 550k based on input features.")

Sidebar inputs (these must match your training feature columns) st.sidebar.header("Input Employee Details")

Replace these fields with your dataset's actual input columns

age st.sidebar.slider("Age", 18, 65, 38) education st.sidebar.selectbox("Education Level", [ 1) "Bachelors", "Masters", "PhD", "HS-grad", "Assoc", "Some-college"

occupation = st.sidebar.selectbox("Job Role", [