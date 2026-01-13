# CGM Spike Predictor & Coaching Assistant (Technical Assignment)

### What is a CGM?

A Continuous Glucose Monitor (CGM) is a wearable sensor that measures interstitial glucose levels at frequent intervals throughout the day. CGMs enable detailed analysis of how meals, activity, and other factors influence glucose trends over time.

### Goal of This Assignment

The objective of this assignment is to build an MVP that:

- Predicts whether a user will experience a glucose spike in the next **2 hours**
- Provides a **clear, structured explanation** for the prediction that can be used by coaches and the product

---

## 2. System Overview

This solution implements an end‑to‑end pipeline that converts CGM time‑series data into both a spike risk prediction and an interpretable explanation.

### High‑Level Architecture

1. **Data Processing Layer (`core/data_processor.py`)**

   - Aligns CGM readings and meal events to a fixed **5‑minute grid**
   - Handles missing or irregular timestamps using linear interpolation

2. **Spike Prediction Model (`models/spike_rf.joblib`)**

   - Binary classification model predicting whether glucose will exceed **180 mg/dL** within the next 2 hours
   - Implemented using a **Random Forest classifier**
   - Trained on the **OhioT1DM CGM dataset**

3. **Explanation / Coaching Layer (`core/llm_coach.py`)**

   - Generates a structured, human‑readable explanation for the predicted risk
   - Connects glucose trends with recent meal timing and carbohydrate intake
   - Uses Kimi K2 (Llama‑3) via Groq

4. **API Layer (`app.py`)**

   - FastAPI service exposing a single prediction endpoint
   - Stateless design for simplicity and ease of testing

5. **Training Pipeline (`train_model.py`)**
  - The training script handles the full machine learning lifecycle. 
  - It parses clinical XML, performs feature engineering, handles class imbalance, and evaluates performance using a time-aware split.

---

## 3. Modeling Approach

### Problem Formulation

- **Task**: Binary classification
- **Target**: Whether glucose will exceed **180 mg/dL** within the next 2 hours

### Features Used

- **Glucose Velocity** (`slope_15`, `slope_60`):

  - Rate of glucose change over short and long windows

- **Carbs on Board (`cob_2h`)**:

  - Rolling 2‑hour sum of carbohydrate intake

- **Recent Glucose Levels**:

  - Stabilizes predictions and provides context

### Evaluation

- Time‑aware train/validation split to avoid temporal leakage
- Model performance: **AUC = 0.9234**

---

## 4. Design Choices & Trade‑offs

- **Random Forest over Deep Learning**

  - Easier to interpret and validate feature importance
  - Suitable for an MVP with limited iteration time

- **Stateless API**

  - Each request includes the last ~60 minutes of data
  - Avoids database or streaming complexity

- **Fixed Spike Threshold (180 mg/dL)**

  - Clinically common benchmark
  - Easily configurable for future personalization

- **Fixed 5‑Minute Resolution**

  - Ensures consistent feature shapes across devices
  - Minor smoothing accepted for improved stability

---

## 5. API Specification

### Endpoint

`POST /predict`

### Input

```json
{
  "user_id": "USR_001",
  "recent_data": [
    {
      "user_id": "USR_001",
      "timestamp": "2023-10-01 12:00:00",
      "glucose": 110,
      "carbs": 50,
      "meal_type": "Lunch (Brown Rice)"
    },
    {
      "user_id": "USR_001",
      "timestamp": "2023-10-01 12:05:00",
      "glucose": 112,
      "carbs": 0,
      "meal_type": "N/A"
    },
    {
      "user_id": "USR_001",
      "timestamp": "2023-10-01 12:10:00",
      "glucose": 114,
      "carbs": 0,
      "meal_type": "N/A"
    },
    {
      "user_id": "USR_001",
      "timestamp": "2023-10-01 12:15:00",
      "glucose": 117,
      "carbs": 0,
      "meal_type": "N/A"
    },
    {
      "user_id": "USR_001",
      "timestamp": "2023-10-01 12:20:00",
      "glucose": 120,
      "carbs": 0,
      "meal_type": "N/A"
    },
    {
      "user_id": "USR_001",
      "timestamp": "2023-10-01 12:25:00",
      "glucose": 124,
      "carbs": 0,
      "meal_type": "N/A"
    },
    {
      "user_id": "USR_001",
      "timestamp": "2023-10-01 12:30:00",
      "glucose": 128,
      "carbs": 0,
      "meal_type": "N/A"
    },
    {
      "user_id": "USR_001",
      "timestamp": "2023-10-01 12:35:00",
      "glucose": 133,
      "carbs": 0,
      "meal_type": "N/A"
    },
    {
      "user_id": "USR_001",
      "timestamp": "2023-10-01 12:40:00",
      "glucose": 138,
      "carbs": 0,
      "meal_type": "N/A"
    },
    {
      "user_id": "USR_001",
      "timestamp": "2023-10-01 12:45:00",
      "glucose": 144,
      "carbs": 0,
      "meal_type": "N/A"
    },
    {
      "user_id": "USR_001",
      "timestamp": "2023-10-01 12:50:00",
      "glucose": 150,
      "carbs": 0,
      "meal_type": "N/A"
    },
    {
      "user_id": "USR_001",
      "timestamp": "2023-10-01 12:55:00",
      "glucose": 157,
      "carbs": 0,
      "meal_type": "N/A"
    },
    {
      "user_id": "USR_001",
      "timestamp": "2023-10-01 13:00:00",
      "glucose": 165,
      "carbs": 0,
      "meal_type": "N/A"
    }
  ]
}
```

### Output

```json
{
  "user_id": "USR_001",
  "will_spike": false,
  "risk_score": 0.48,
  "explanation": "Lunch’s 50 g carbs from brown rice are being released slowly, so the modest rise velocity of 1.4 mg/dL/min is expected to keep you near today’s 165 mg/dL without a sharp spike. Take a 10-minute stroll now to nudge glucose downward and next time add chicken or tofu to blunt the peak further."
}
```

---

## 6. Setup & Execution

### Requirements

- Python 3.11+
- Groq API key

### Installation

```bash
git clone <repo-link>
cd CGM-Spike-Predictor
pip install fastapi uvicorn pandas numpy scikit-learn joblib openai python-dotenv
```

### Environment Variables

```bash
echo "GROQ_API_KEY=your_key_here" > .env
```

### Train Model

```bash
python train_model.py
```

### Run API

```bash
python app.py
```

---

## 7. Testing

Sample scenarios covered:

- High‑carbohydrate meal followed by rising glucose → spike predicted
- No meal and stable glucose → no spike
- Falling glucose (e.g., post‑exercise) → no spike

### Sample `curl` requests

- **Scenario**: A heavy carbohydrate breakfast.
- **Expected Result**: `will_spike: true`. High risk score due to sustained rising momentum and high carbs.

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "user_id": "USR_001",
       "recent_data": [
         {"user_id": "USR_001", "timestamp": "2023-10-01 08:00:00", "glucose": 100, "carbs": 65, "meal_type": "Aaloo Paratha"},
         {"user_id": "USR_001", "timestamp": "2023-10-01 08:05:00", "glucose": 105, "carbs": 0, "meal_type": "N/A"},
         {"user_id": "USR_001", "timestamp": "2023-10-01 08:10:00", "glucose": 112, "carbs": 0, "meal_type": "N/A"},
         {"user_id": "USR_001", "timestamp": "2023-10-01 08:15:00", "glucose": 125, "carbs": 0, "meal_type": "N/A"},
         {"user_id": "USR_001", "timestamp": "2023-10-01 08:20:00", "glucose": 138, "carbs": 0, "meal_type": "N/A"},
         {"user_id": "USR_001", "timestamp": "2023-10-01 08:25:00", "glucose": 152, "carbs": 0, "meal_type": "N/A"},
         {"user_id": "USR_001", "timestamp": "2023-10-01 08:30:00", "glucose": 165, "carbs": 0, "meal_type": "N/A"},
         {"user_id": "USR_001", "timestamp": "2023-10-01 08:35:00", "glucose": 180, "carbs": 0, "meal_type": "N/A"},
         {"user_id": "USR_001", "timestamp": "2023-10-01 08:40:00", "glucose": 195, "carbs": 0, "meal_type": "N/A"},
         {"user_id": "USR_001", "timestamp": "2023-10-01 08:45:00", "glucose": 210, "carbs": 0, "meal_type": "N/A"},
         {"user_id": "USR_001", "timestamp": "2023-10-01 08:50:00", "glucose": 225, "carbs": 0, "meal_type": "N/A"},
         {"user_id": "USR_001", "timestamp": "2023-10-01 08:55:00", "glucose": 235, "carbs": 0, "meal_type": "N/A"},
         {"user_id": "USR_001", "timestamp": "2023-10-01 09:00:00", "glucose": 245, "carbs": 0, "meal_type": "N/A"}
       ]
     }'

```

- **Scenario**: Minimal activity, no food intake.
- **Expected Result**: `will_spike: false`. Low risk score despite slight fluctuations.

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "user_id": "USR_001",
       "recent_data": [
         {"user_id": "USR_001", "timestamp": "2023-10-01 10:00:00", "glucose": 98, "carbs": 0, "meal_type": "N/A"},
         {"user_id": "USR_001", "timestamp": "2023-10-01 10:05:00", "glucose": 97, "carbs": 0, "meal_type": "N/A"},
         {"user_id": "USR_001", "timestamp": "2023-10-01 10:10:00", "glucose": 99, "carbs": 0, "meal_type": "N/A"},
         {"user_id": "USR_001", "timestamp": "2023-10-01 10:15:00", "glucose": 98, "carbs": 0, "meal_type": "N/A"},
         {"user_id": "USR_001", "timestamp": "2023-10-01 10:20:00", "glucose": 97, "carbs": 0, "meal_type": "N/A"},
         {"user_id": "USR_001", "timestamp": "2023-10-01 10:25:00", "glucose": 98, "carbs": 0, "meal_type": "N/A"},
         {"user_id": "USR_001", "timestamp": "2023-10-01 10:30:00", "glucose": 99, "carbs": 0, "meal_type": "N/A"},
         {"user_id": "USR_001", "timestamp": "2023-10-01 10:35:00", "glucose": 98, "carbs": 0, "meal_type": "N/A"},
         {"user_id": "USR_001", "timestamp": "2023-10-01 10:40:00", "glucose": 97, "carbs": 0, "meal_type": "N/A"},
         {"user_id": "USR_001", "timestamp": "2023-10-01 10:45:00", "glucose": 98, "carbs": 0, "meal_type": "N/A"},
         {"user_id": "USR_001", "timestamp": "2023-10-01 10:50:00", "glucose": 99, "carbs": 0, "meal_type": "N/A"},
         {"user_id": "USR_001", "timestamp": "2023-10-01 10:55:00", "glucose": 98, "carbs": 0, "meal_type": "N/A"},
         {"user_id": "USR_001", "timestamp": "2023-10-01 11:00:00", "glucose": 97, "carbs": 0, "meal_type": "N/A"}
       ]
     }'

```

- **Scenario**: Glucose is falling rapidly (e.g., post-exercise).
- **Expected Result**: `will_spike: false`. This proves the model understands direction—even if levels are high (e.g., **160 mg/dL**), a negative velocity correctly predicts stability.

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "user_id": "USR_001",
       "recent_data": [
         {"user_id": "USR_001", "timestamp": "2023-10-01 17:00:00", "glucose": 160, "carbs": 0, "meal_type": "N/A"},
         {"user_id": "USR_001", "timestamp": "2023-10-01 17:05:00", "glucose": 155, "carbs": 0, "meal_type": "N/A"},
         {"user_id": "USR_001", "timestamp": "2023-10-01 17:10:00", "glucose": 148, "carbs": 0, "meal_type": "N/A"},
         {"user_id": "USR_001", "timestamp": "2023-10-01 17:15:00", "glucose": 140, "carbs": 0, "meal_type": "N/A"},
         {"user_id": "USR_001", "timestamp": "2023-10-01 17:20:00", "glucose": 132, "carbs": 0, "meal_type": "N/A"},
         {"user_id": "USR_001", "timestamp": "2023-10-01 17:25:00", "glucose": 125, "carbs": 0, "meal_type": "N/A"},
         {"user_id": "USR_001", "timestamp": "2023-10-01 17:30:00", "glucose": 118, "carbs": 0, "meal_type": "N/A"},
         {"user_id": "USR_001", "timestamp": "2023-10-01 17:35:00", "glucose": 112, "carbs": 0, "meal_type": "N/A"},
         {"user_id": "USR_001", "timestamp": "2023-10-01 17:40:00", "glucose": 107, "carbs": 0, "meal_type": "N/A"},
         {"user_id": "USR_001", "timestamp": "2023-10-01 17:45:00", "glucose": 102, "carbs": 0, "meal_type": "N/A"},
         {"user_id": "USR_001", "timestamp": "2023-10-01 17:50:00", "glucose": 98, "carbs": 0, "meal_type": "N/A"},
         {"user_id": "USR_001", "timestamp": "2023-10-01 17:55:00", "glucose": 95, "carbs": 0, "meal_type": "N/A"},
         {"user_id": "USR_001", "timestamp": "2023-10-01 18:00:00", "glucose": 92, "carbs": 0, "meal_type": "N/A"}
       ]
     }'

```
