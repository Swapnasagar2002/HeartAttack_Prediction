import streamlit as st
import pickle
import numpy as np

model = pickle.load(open('svm_model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

def main():
    st.title("Heart Attack Prediction Model")

    age = st.slider("Age", 29, 77, 50)
    sex = st.selectbox("Sex", ["Male", "Female"])
    chest_pain = st.selectbox("Chest Pain Type", ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"])
    resting_bp = st.slider("Resting Blood Pressure (mm Hg)", 94, 200, 120)
    cholesterol = st.slider("Cholesterol (mg/dl)", 126, 564, 240)
    fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["False", "True"])
    rest_ecg = st.selectbox("Resting Electrocardiographic Results", ["Normal", "ST-T Wave Normality", "Left Ventricular Hypertrophy"])
    max_heart_rate = st.slider("Maximum Heart Rate Achieved", 71, 202, 150)
    oldpeak = st.slider("Oldpeak", 0.0, 6.2, 3.0) 
    slp = st.slider("Slope", 0, 2, 1) 
    caa = st.slider("Number of Major Vessels", 0, 3, 1)  
    thall = st.selectbox("Thalium Stress Test Result", ["Normal", "Fixed Defect", "Reversible Defect"])  
    
    sex = 1 if sex == "Male" else 0
    chest_pain_mapping = {"Typical Angina": 0, "Atypical Angina": 1, "Non-anginal Pain": 2, "Asymptomatic": 3}
    chest_pain = chest_pain_mapping[chest_pain]
    fasting_bs = 1 if fasting_bs == "True" else 0
    rest_ecg_mapping = {"Normal": 0, "ST-T Wave Normality": 1, "Left Ventricular Hypertrophy": 2}
    rest_ecg = rest_ecg_mapping[rest_ecg]
    
    thall_mapping = {"Normal": 0, "Fixed Defect": 1, "Reversible Defect": 2}
    thall = thall_mapping[thall]
    

    if st.button("Predict"):
        input_data = [[age, sex, chest_pain, resting_bp, cholesterol, fasting_bs, rest_ecg, max_heart_rate, oldpeak, slp, caa, thall]]
        
        input_data_scaled = scaler.transform(input_data)

        prediction = model.predict(input_data_scaled)

        st.subheader("Prediction Result:")
        if prediction[0] == 1:
            st.success("The model predicts that the patient has a heart attack.")
        else:
            st.success("The model predicts that the patient does not have a heart attack.")

if __name__ == "__main__":
    main()
