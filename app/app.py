import streamlit as st
import joblib
import numpy as np

# -------------------- Load Model --------------------
model = joblib.load("../models/heart_model.pkl")
scaler = joblib.load("../models/scaler.pkl")

# -------------------- Page Config --------------------
st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="wide"
)

# -------------------- Banner --------------------
st.image("../images/banner.jpg", use_container_width=True)

st.markdown(
    "<h1 style='text-align:center;color:#D32F2F;'>❤️ Heart Disease Risk Prediction System</h1>",
    unsafe_allow_html=True
)

st.markdown(
    "<p style='text-align:center;font-size:18px;'>Predict the likelihood of heart disease using Machine Learning.</p>",
    unsafe_allow_html=True
)

st.markdown("---")

# -------------------- Sidebar --------------------
st.sidebar.image("../images/doctor.png", use_container_width=True)

st.sidebar.title("About")

st.sidebar.write("""
This application predicts the likelihood of heart disease based on patient health parameters.

Fill in the patient's details and click **Predict Heart Disease**.
""")

st.sidebar.markdown("---")

st.sidebar.success("""
### ❤️ Heart Health Tips

✔ Exercise regularly

✔ Eat a balanced diet

✔ Avoid smoking

✔ Maintain healthy weight

✔ Monitor blood pressure
""")

st.sidebar.info("""
**Disclaimer**

This application is for educational purposes only and should not replace professional medical advice.
""")

# -------------------- Layout --------------------
col1, col2 = st.columns([2,1])

with col1:

    st.subheader("📝 Patient Details")

    age = st.number_input("Age", 20, 100, 50)

    sex = st.selectbox("Gender", ["Female", "Male"])
    sex = 1 if sex == "Male" else 0

    cp = st.selectbox(
        "Chest Pain Type",
        [0,1,2,3],
        help="0=Typical Angina\n1=Atypical Angina\n2=Non-anginal Pain\n3=Asymptomatic"
    )

    trestbps = st.number_input(
        "Resting Blood Pressure",
        value=120
    )

    chol = st.number_input(
        "Serum Cholesterol",
        value=200
    )

    fbs = st.selectbox(
        "Fasting Blood Sugar >120 mg/dl",
        [0,1]
    )

    restecg = st.selectbox(
        "Resting ECG",
        [0,1,2]
    )

    thalach = st.number_input(
        "Maximum Heart Rate Achieved",
        value=150
    )

    exang = st.selectbox(
        "Exercise Induced Angina",
        [0,1]
    )

    oldpeak = st.number_input(
        "Oldpeak",
        value=1.0
    )

    slope = st.selectbox(
        "Slope",
        [0,1,2]
    )

    ca = st.selectbox(
        "Number of Major Vessels",
        [0,1,2,3]
    )

    thal = st.selectbox(
        "Thal",
        [0,1,2]
    )

with col2:

    st.image("../images/heart.png", use_container_width=True)

    st.info("""
### ❤️ Risk Factors

🩸 High Blood Pressure

🍔 High Cholesterol

🚬 Smoking

⚖️ Obesity

🏃 Physical Inactivity

😰 Stress
""")

st.markdown("---")

# -------------------- Prediction --------------------
if st.button("🔍 Predict Heart Disease"):

    input_data = np.array([[

        age,
        sex,
        cp,
        trestbps,
        chol,
        fbs,
        restecg,
        thalach,
        exang,
        oldpeak,
        slope,
        ca,
        thal

    ]])

    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)
    

    probability = model.predict_proba(input_scaled)[0][1]

    st.subheader("Prediction Result")

    if prediction[0] == 1:

        st.error("⚠️ High Risk of Heart Disease")

        st.image("../images/warning.png", width=300)

    else:

        st.success("✅ Low Risk of Heart Disease")

        st.image("../images/healthy.png", width=300)

    st.markdown("---")

    st.subheader("Prediction Probability")

    st.progress(float(probability))

    st.write(f"### Probability of Heart Disease: **{probability:.2%}**")

if probability < 0.30:
    st.success("🟢 Risk Level: LOW")
elif probability < 0.70:
    st.warning("🟡 Risk Level: MODERATE")
else:
    st.error("🔴 Risk Level: HIGH")
    

    st.markdown("---")

    st.subheader("Health Recommendations")

    if prediction[0] == 1:

        st.warning("""
- 🩺 Consult a Cardiologist

- 🥗 Follow a Heart-Healthy Diet

- 🏃 Exercise Regularly

- 💊 Take Medications as Prescribed

- 🩸 Monitor Blood Pressure & Cholesterol

- 🚭 Quit Smoking
""")

    else:

        st.success("""
- 💚 Maintain a Healthy Lifestyle

- 🥗 Continue Balanced Nutrition

- 🏃 Exercise Daily

- 😴 Get Adequate Sleep

- 🩺 Attend Regular Health Check-ups
""")

st.markdown("---")

st.caption("Developed using ❤️ Python | Streamlit | Scikit-learn")