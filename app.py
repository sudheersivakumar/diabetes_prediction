# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Title and description
st.set_page_config(page_title="Diabetes Risk Predictor", layout="centered")
st.title("🩺 Diabetes Risk Prediction Tool")
st.markdown("""
    Answer a few health-related questions to estimate your risk of having diabetes.
    This tool uses a machine learning model trained on real health survey data.
""")

# Load model
@st.cache_resource
def load_model():
    return joblib.load('models/trained_model.pkl')

model = load_model()

# Input form
st.subheader("📋 Your Health Information")

col1, col2 = st.columns(2)

with col1:
    high_bp = st.selectbox("Have you ever been told you have high blood pressure?", ["No", "Yes"])
    high_chol = st.selectbox("Have you ever been told your cholesterol is high?", ["No", "Yes"])
    chol_check = st.selectbox("Had cholesterol check in past 5 years?", ["No", "Yes"])
    bmi = st.number_input("Body Mass Index (BMI)", min_value=10, max_value=60, value=25)
    smoker = st.selectbox("Have you smoked at least 100 cigarettes in your lifetime?", ["No", "Yes"])
    stroke = st.selectbox("Did you ever have a stroke?", ["No", "Yes"])
    heart_disease = st.selectbox("Do you have heart disease or heart attack?", ["No", "Yes"])

with col2:
    phys_activity = st.selectbox("Physical activity in past 30 days (not including job)?", ["No", "Yes"])
    fruits = st.selectbox("Consume fruit 1+ times per day?", ["No", "Yes"])
    veggies = st.selectbox("Consume vegetables 1+ times per day?", ["No", "Yes"])
    hvy_alcohol = st.selectbox("Heavy alcohol consumption (adult men >14 drinks/week, adult women >7)?", ["No", "Yes"])
    any_healthcare = st.selectbox("Do you have any health care coverage?", ["No", "Yes"])
    no_doc_cost = st.selectbox("Was there a time in past 12 months when you needed to see a doctor but couldn't because of cost?", ["No", "Yes"])
    gen_hlth = st.selectbox("General health (1=Excellent → 5=Poor)", [1, 2, 3, 4, 5], index=2)
    ment_hlth = st.slider("Days of poor mental health in past 30 days?", 0, 30, 5)
    phys_hlth = st.slider("Days of physical illness/injury in past 30 days?", 0, 30, 5)
    diff_walk = st.selectbox("Do you have serious difficulty walking or climbing stairs?", ["No", "Yes"])
    sex = st.selectbox("Sex assigned at birth", ["Female", "Male"])
    age = st.selectbox("Age category", list(range(1, 14)), index=8)  # 1=18-24, 13=80+
    education = st.selectbox("Education level", list(range(1, 7)), index=4)  # 1=Never, 6=College grad
    income = st.selectbox("Household income", list(range(1, 9)), index=5)  # 1=<$10k, 8=>$75k

# Convert inputs to numeric
input_data = {
    'HighBP': 1 if high_bp == "Yes" else 0,
    'HighChol': 1 if high_chol == "Yes" else 0,
    'CholCheck': 1 if chol_check == "Yes" else 0,
    'BMI': float(bmi),
    'Smoker': 1 if smoker == "Yes" else 0,
    'Stroke': 1 if stroke == "Yes" else 0,
    'HeartDiseaseorAttack': 1 if heart_disease == "Yes" else 0,
    'PhysActivity': 1 if phys_activity == "Yes" else 0,
    'Fruits': 1 if fruits == "Yes" else 0,
    'Veggies': 1 if veggies == "Yes" else 0,
    'HvyAlcoholConsump': 1 if hvy_alcohol == "Yes" else 0,
    'AnyHealthcare': 1 if any_healthcare == "Yes" else 0,
    'NoDocbcCost': 1 if no_doc_cost == "Yes" else 0,
    'GenHlth': int(gen_hlth),
    'MentHlth': int(ment_hlth),
    'PhysHlth': int(phys_hlth),
    'DiffWalk': 1 if diff_walk == "Yes" else 0,
    'Sex': 1 if sex == "Male" else 0,
    'Age': int(age),
    'Education': int(education),
    'Income': int(income)
}

# Predict button
if st.button("🎯 Predict Diabetes Risk"):
    input_df = pd.DataFrame([input_data])
    pred_proba = model.predict_proba(input_df)[0, 1]
    prediction = model.predict(input_df)[0]

    # Display result
    st.markdown("### 🔍 Prediction Result")
    if prediction == 1:
        st.error(f"⚠️ High Risk of Diabetes")
        st.write(f"Estimated probability: **{pred_proba:.2f}**")
    else:
        st.success(f"✅ Lower Risk of Diabetes")
        st.write(f"Estimated probability: **{pred_proba:.2f}**")

    # Visualize probability
    st.progress(float(pred_proba))
    st.caption("Higher progress bar = higher risk.")

    st.info("""
        **Note**: This is a predictive tool for informational purposes only and does not replace medical advice. 
        Consult a healthcare provider for diagnosis and care.
    """)

# Footer
st.markdown("---")
st.markdown("💡 *Model trained on BRFSS 2015 dataset using Random Forest. Deployed with Streamlit.*")