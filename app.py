import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from core.data_processor import GlucoseProcessor
from core.llm_coach import LLMCoachingAssistant

app = FastAPI(title="CGM Spike Predictor")
processor = GlucoseProcessor()
coach = LLMCoachingAssistant()

try:
    artifact = joblib.load('models/spike_rf.joblib')
    model = artifact['model']
    features_list = artifact['features']
    print("Model loaded successfully.")
except FileNotFoundError:
    print("Error: 'models/spike_rf.joblib' not found. Please run train_model.py first.")

class CGMPoint(BaseModel):
    user_id: str
    timestamp: str 
    glucose: float
    carbs: float = 0.0
    meal_type: str = "N/A"

class PredictionRequest(BaseModel):
    user_id: str
    recent_data: List[CGMPoint]

@app.post("/predict")
async def predict(request: PredictionRequest):
    # Convert incoming data to DataFrame
    raw_input = [p.dict() for p in request.recent_data]
    df = pd.DataFrame(raw_input)
    
    # 1. Validation: Duration check for synthetic data
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp')
    duration = (df['timestamp'].max() - df['timestamp'].min()).total_seconds() / 60
    
    if duration < 60:
        raise HTTPException(
            status_code=400, 
            detail=f"Insufficient history. Data covers {duration} mins. Need at least 60 mins."
        )

    # 2. Feature Engineering
    processed_df = processor.engineer_features(df, is_training=False)
    current_state = processed_df.tail(1)
    
    # 3. Science: Risk Score
    risk_score = model.predict_proba(current_state[features_list])[0][1]
    
    # 4. Extract context for the LLM
    latest_meal = df['meal_type'].iloc[-1] 
    # If the latest is N/A, look back to find the most recent meal in this window
    if latest_meal == "N/A":
        meals_in_window = df[df['meal_type'] != "N/A"]
        latest_meal = meals_in_window['meal_type'].iloc[-1] if not meals_in_window.empty else "N/A"

    # 5. Coaching: Updated call with 5 arguments
    explanation = await coach.get_explanation(
        risk_score=risk_score,
        glc=current_state['glucose'].iloc[0],
        vel=current_state['slope_15'].iloc[0],
        cob=current_state['cob_2h'].iloc[0],
        meal_type=latest_meal  # <--- THIS WAS MISSING
    )
    
    return {
        "user_id": request.user_id,
        "will_spike": bool(risk_score > 0.5),
        "risk_score": round(float(risk_score), 2),
        "explanation": explanation
    }
    
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)