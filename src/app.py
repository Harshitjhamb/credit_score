from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import os

# 1. Define the exact input data format we expect from users
# Pydantic will act as a firewall, automatically rejecting bad data (like letters in the age field)
class CreditApplication(BaseModel):
    age: int
    income: float
    credit_score: int
    employment_type: str

# 2. Initialize the FastAPI server
app = FastAPI(
    title="Credit Risk Prediction API", 
    description="An MLOps API that predicts loan default risk."
)

# Global variable to hold our model
model = None
MODEL_PATH = "models/model_latest.joblib"

# 3. Load the model into memory the moment the server starts
@app.on_event("startup")
def load_model():
    global model
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
        print(f"✅ Model loaded successfully from {MODEL_PATH}")
    else:
        print(f"❌ Error: Model not found at {MODEL_PATH}. Did you run train.py?")

# 4. A simple health-check endpoint
@app.get("/")
def home():
    return {"message": "Credit Risk API is Live! Send a POST request to /predict to score an application."}

# 5. The core prediction endpoint
@app.post("/predict")
def predict_risk(application: CreditApplication):
    if model is None:
        raise HTTPException(status_code=500, detail="Model is currently offline.")
    
    # Convert the clean, validated JSON into a Pandas DataFrame
    # We use model_dump() instead of dict() for Pydantic v2 compatibility
    input_data = pd.DataFrame([application.model_dump()])
    
    # Run the data through our leak-proof pipeline
    prediction = model.predict(input_data)[0]
    
    # Translate the 0 or 1 into human-readable text
    risk_status = "High Risk (Default)" if prediction == 1 else "Low Risk (Approved)"
    
    return {
        "prediction_class": int(prediction),
        "risk_status": risk_status,
        "applicant_data": application.model_dump()
    }