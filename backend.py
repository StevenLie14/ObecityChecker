from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import uvicorn
import cloudpickle
from model import ObecityPredictionModel

with open('model_with_cloud.pkl', 'rb') as f:
    model = cloudpickle.load(f)
model = ObecityPredictionModel.load_model('model_without_cloud.pkl')

    
app = FastAPI(title="Obesity Prediction API")


class ObesityInput(BaseModel):
    Gender: str
    Age: int
    Height: float
    Weight: float
    family_history_with_overweight: str
    FAVC: str
    FCVC: float
    NCP: float
    CAEC: str
    CH2O: float
    SCC: str
    FAF: float
    TUE: float
    CALC: str


@app.post("/predict")
def predict_obesity(input_data: ObesityInput = ObesityInput(
        Gender="Male",
        Age=22,
        Height=1.75,
        Weight=85.0,
        family_history_with_overweight="yes",
        FAVC="yes",
        FCVC=2.5,
        NCP=3.0,
        CAEC="Sometimes",
        CH2O=2.0,
        SCC="no",
        FAF=1.5,
        TUE=1.0,
        CALC="Sometimes",
    )):
    try:
        input_df = pd.DataFrame([input_data.dict()])
        
        preds, labels = model.predict(input_df)
        return {"prediction": str(labels[0])}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.get("/")
def read_root():
    return {"message": "Obesity Prediction API is running!"}


if __name__ == "__main__":
    uvicorn.run("backend:app", host="0.0.0.0", port=8000, reload=True)
