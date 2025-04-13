from fastapi import FastAPI, HTTPException, Body, Depends, Header, Security
from fastapi.security.api_key import APIKeyHeader, APIKey
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import pandas as pd
import joblib
import uvicorn
import os

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the saved artifacts (removed success message)
kmeans_model = joblib.load('kmeans_model.pkl')
scaler = joblib.load('scaler.pkl') 
cluster_to_bank = joblib.load('cluster_to_bank.pkl')
features_order = joblib.load('features_order.pkl')
print("Artifacts loaded successfully.")
print("Expected features:", features_order)

# API Key security setup
API_KEY = os.environ.get("API_KEY", "Enzktyionw6798-adwWSAoPsAA")  # Set a default key for development
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
# Set to True to disable API key validation for development/testing
DISABLE_API_KEY_VALIDATION = True

async def get_api_key(api_key: str = Security(api_key_header)):
    if DISABLE_API_KEY_VALIDATION:
        return "development_mode"
        
    print(f"Received API key: '{api_key}' (Expected: '{API_KEY}')")
    if not api_key:
        raise HTTPException(
            status_code=403, detail="API Key header is missing"
        )
    if api_key != API_KEY:
        raise HTTPException(
            status_code=403, detail="Invalid API Key"
        )
    return api_key

# Define the input model using aliases to map underscore names 
# to the original training column names.
class BankInput(BaseModel):
    Digital_Interface_Rank: int = Field(..., alias="Digital Interface Rank")
    Number_of_Branches: int = Field(..., alias="Number of Branches")
    Green_Initiatives_Rank: int = Field(..., alias="Green Initiatives Rank")
    Fee_Level_Rank: int = Field(..., alias="Fee Level Rank")
    International_Support_Rank: int = Field(..., alias="International Support Rank")
    Interest_Rate_Range_Rank: int = Field(..., alias="Interest Rate Range Rank")
    Customer_Service_Quality_Rank: int = Field(..., alias="Customer Service Quality Rank")
    Capital_Adequacy_Rank: int = Field(..., alias="Capital Adequacy Rank")
    
    Auto_Loans: int = Field(0, alias="Auto Loans")
    Credit_Cards: int = Field(0, alias="Credit Cards")
    Global_Banking: int = Field(0, alias="Global Banking")
    Investments: int = Field(0, alias="Investments")
    Loans: int = Field(0, alias="Loans")
    Mortgages: int = Field(0, alias="Mortgages")
    Savings_Accounts: int = Field(0, alias="Savings Accounts")
    Global_Customers: int = Field(0, alias="Global Customers")
    Professionals: int = Field(0, alias="Professionals")
    SMEs: int = Field(0, alias="SMEs")
    Seniors: int = Field(0, alias="Seniors")
    Students: int = Field(0, alias="Students")
    Tech_Savvy: int = Field(0, alias="Tech-Savvy")

    class Config:
        # Allow population by field name or by alias from the request body.
        allow_population_by_field_name = True

@app.get("/")
async def root():
    return {"message": "Bank Recommendation API. Use /predict endpoint with valid API key."}

@app.post("/predict")
def predict_bank(user_input: BankInput = Body(...), api_key: APIKey = Depends(get_api_key)):
    try:
        input_dict = user_input.dict(by_alias=True)
        df_input = pd.DataFrame([input_dict], columns=features_order)
        
        if df_input.isna().any().any():
            raise ValueError("Missing columns or NaN values detected")
            
        df_scaled = scaler.transform(df_input)
        cluster_label = kmeans_model.predict(df_scaled)[0]
        recommended_bank = cluster_to_bank.get(cluster_label, "Unknown Bank")
        
        return {"recommended_bank": recommended_bank}

    except Exception as e:
        print(f"Prediction error: {str(e)}")  # Keep error logging
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    # Use production settings when deployed
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
