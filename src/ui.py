import streamlit as st
import requests
import time
st.set_page_config(page_title="Credit Risk AI", page_icon="🏦", layout="centered")

st.title("🏦 AI Credit Risk Portal")
st.markdown("Enter the applicant's details below to get a real-time default risk prediction powered by our Random Forest model.")
st.divider()

# Create a beautiful form for user input
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Applicant Age", min_value=18, max_value=100, value=30)
    income = st.number_input("Annual Income ($)", min_value=0, value=65000, step=5000)

with col2:
    credit_score = st.slider("Credit Score", min_value=300, max_value=850, value=720)
    employment_type = st.selectbox("Employment Status", ["Salaried", "Self-Employed", "Unemployed", "Gig_Worker"])

st.divider()

# The "Analyze" button
if st.button("🧠 Analyze Application Risk", type="primary", use_container_width=True):
    
    # 1. Prepare the exact JSON payload our FastAPI expects
    payload = {
        "age": age,
        "income": income,
        "credit_score": credit_score,
        "employment_type": employment_type
    }
    
    # Show a cinematic loading spinner
    with st.spinner('Running applicant data through the Random Forest pipeline...'):
        time.sleep(1) # Fake delay for dramatic effect
        
        try:
            # 2. Send the data to your locally running FastAPI server
            response = requests.post("http://127.0.0.1:8000/predict", json=payload)
            response.raise_for_status()
            result = response.json()
            
            # 3. Display the results beautifully based on the prediction
            if result["prediction_class"] == 1:
                st.error(f"### 🚨 {result['risk_status']}")
                st.markdown("⚠️ **Warning:** This profile matches historical patterns of loan defaults. Application denied.")
            else:
                st.success(f"### ✅ {result['risk_status']}")
                st.markdown("🎉 **Approved:** This profile indicates high financial stability.")
                
            with st.expander("View Raw API Response (JSON)"):
                st.json(result)
                
        except requests.exceptions.ConnectionError:
            st.error("❌ Could not connect to the AI Server. Is your FastAPI server running?")