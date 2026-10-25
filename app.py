
import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)

st.title("Customer Churn Prediction Dashboard")
st.write(
    "Predict whether a telecom customer is likely to leave "
    "the service using Machine Learning."
)

# Load trained model
MODEL_PATH = "logistic_regression_model.pkl"

if not os.path.exists(MODEL_PATH):
    st.error("Model file not found. Run train_model.py first.")
    st.stop()

model = joblib.load(MODEL_PATH)

# Load dataset for dashboard statistics
df = pd.read_csv(
    "dataset/WA_Fn-UseC_-Telco-Customer-Churn.csv"
)

# Dashboard metrics
total_customers = len(df)
churned_customers = (df["Churn"] == "Yes").sum()
churn_rate = churned_customers / total_customers * 100

col1, col2, col3 = st.columns(3)

col1.metric("Total Customers", total_customers)
col2.metric("Customers Churned", int(churned_customers))
col3.metric("Churn Rate", f"{churn_rate:.2f}%")

st.divider()

# Dataset visualization
st.subheader("Customer Churn Distribution")

churn_counts = df["Churn"].value_counts()
st.bar_chart(churn_counts)

st.divider()

# Prediction section
st.subheader("Predict Customer Churn")
st.write("Enter customer details below.")

with st.form("prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        gender = st.selectbox(
            "Gender", ["Female", "Male"]
        )
        senior = st.selectbox(
            "Senior Citizen", [0, 1]
        )
        partner = st.selectbox(
            "Has Partner?", ["Yes", "No"]
        )
        dependents = st.selectbox(
            "Has Dependents?", ["Yes", "No"]
        )
        tenure = st.slider(
            "Tenure (months)", 0, 72, 12
        )
        phone = st.selectbox(
            "Phone Service", ["Yes", "No"]
        )
        multiple_lines = st.selectbox(
            "Multiple Lines",
            ["No", "Yes", "No phone service"]
        )

    with col2:
        internet = st.selectbox(
            "Internet Service",
            ["DSL", "Fiber optic", "No"]
        )
        online_security = st.selectbox(
            "Online Security",
            ["Yes", "No", "No internet service"]
        )
        online_backup = st.selectbox(
            "Online Backup",
            ["Yes", "No", "No internet service"]
        )
        contract = st.selectbox(
            "Contract",
            ["Month-to-month", "One year", "Two year"]
        )
        paperless = st.selectbox(
            "Paperless Billing", ["Yes", "No"]
        )
        payment = st.selectbox(
            "Payment Method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)"
            ]
        )
        monthly_charges = st.number_input(
            "Monthly Charges",
            min_value=0.0,
            max_value=2000.0,
            value=70.0
        )
        total_charges = st.number_input(
            "Total Charges",
            min_value=0.0,
            max_value=100000.0,
            value=840.0
        )

    # Other services
    tech_support = st.selectbox(
        "Tech Support",
        ["Yes", "No", "No internet service"]
    )
    device_protection = st.selectbox(
        "Device Protection",
        ["Yes", "No", "No internet service"]
    )
    streaming_tv = st.selectbox(
        "Streaming TV",
        ["Yes", "No", "No internet service"]
    )
    streaming_movies = st.selectbox(
        "Streaming Movies",
        ["Yes", "No", "No internet service"]
    )
    online_backup = online_backup

    submitted = st.form_submit_button("Predict Churn")

if submitted:
    customer = pd.DataFrame([{
        "gender": gender,
        "SeniorCitizen": senior,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone,
        "MultipleLines": multiple_lines,
        "InternetService": internet,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless,
        "PaymentMethod": payment,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges
    }])

    prediction = model.predict(customer)[0]
    probability = model.predict_proba(customer)[0][1]

    st.subheader("Prediction Result")

    if prediction == 1:
        st.error("High Risk: Customer is predicted to churn.")
    else:
        st.success("Low Risk: Customer is predicted to stay.")

    st.metric("Estimated Churn Probability", f"{probability:.2%}")
    st.progress(float(probability))
    st.caption(
        "This is a model estimate, not a guarantee of customer behavior."
    )
