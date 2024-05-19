from fastapi import FastAPI
from pydantic import BaseModel
import xgboost as xgb
import numpy as np
import pickle
import time

from prometheus_client import Counter, Gauge, Summary, start_http_server
from prometheus_client import disable_created_metrics

# disable _created metric.
disable_created_metrics()


# Define Prometheus metrics
REQUEST_DURATION = Summary('api_timing', 'Request duration in seconds')
api_usage_counter = Counter("api_usage", "API usage counter", ['endpoint','client_ip'])
api_runtime_gauge = Gauge("api_runtime", "API runtime gauge", ['endpoint', 'client_ip'])
api_tltime_gauge = Gauge("api_tltime", "API T/L time gauge", ['endpoint', 'client_ip'])



# Load the pretrained XGBoost model
with open("xgboost_model.pkl", "rb") as f:
    model = pickle.load(f)

# Define the request body using Pydantic
class ObesityPredictionRequest(BaseModel):
    Age: int
    Gender: str
    Height: float
    Weight: float
    CALC: str
    FAVC: str
    FCVC: int
    NCP: int
    SCC: str
    SMOKE: str
    CH2O: int
    family_history_with_overweight: str
    FAF: int
    TUE: int
    CAEC: str
    MTRANS: str
    

# Initialize FastAPI
app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Welcome to the Obesity Prediction API"}

def preprocess_input(data: ObesityPredictionRequest):
    # Map categorical features to numerical values
    gender = 1 if data.Gender.lower() == "male" else 0
    family_history = 1 if data.family_history_with_overweight.lower() == "yes" else 0
    favc = 1 if data.FAVC.lower() == "yes" else 0
    smoke = 1 if data.SMOKE.lower() == "yes" else 0
    scc = 1 if data.SCC.lower() == "yes" else 0
    
    calc_mapping = {"no": 0, "sometimes": 1, "frequently": 2, "always": 3}
    calc = calc_mapping.get(data.CALC.lower(), 0)
    
    caec_mapping = {"no": 0, "sometimes": 1, "frequently": 2, "always": 3}
    caec = caec_mapping.get(data.CAEC.lower(), 0)
    
    mtrans_mapping = {"automobile": 0, "motorbike": 1, "bike": 2, "public_transportation": 3, "walking": 4}
    mtrans = mtrans_mapping.get(data.MTRANS.lower(), 0)

    BMI=round((data.Weight/(data.Height**2)),2)
    # Prepare the input features for the model
    input_features = np.array([
        data.Age,
        gender,
        data.Height,
        data.Weight,
        calc,
        favc,
        data.FCVC,
        data.NCP,
        scc,
        smoke,
        data.CH2O,
        family_history,
        data.FAF,
        data.TUE,
        caec,
        mtrans,
        BMI
    ])
    
    return input_features.reshape(1, -1)



@REQUEST_DURATION.time()
@app.post("/predict")
def predict_obesity_level(request: ObesityPredictionRequest):

    start_time = time.time()

    # Preprocess the input data
    input_features = preprocess_input(request)
    
    # Predict using the XGBoost model
    prediction = model.predict(input_features)
    
    # Assuming the prediction is a continuous value and we need to classify it
    predicted_class = int(np.round(prediction[0]))


    end_time = time.time()
    runtime = end_time - start_time

    tltime = runtime
    
    # Increment usage counter for the client IP
    api_usage_counter.labels(endpoint="/predict",client_ip=request.client.host).inc()

    # Set runtime and T/L time gauges
    api_runtime_gauge.labels(endpoint="/predict", client_ip=request.client.host).set(runtime)
    api_tltime_gauge.labels(endpoint="/predict", client_ip=request.client.host).set(tltime)

    
    return {"predicted_obesity_level": predicted_class}

if __name__ == "__main__":
    # start the exporter metrics service
    start_http_server(18000)

    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)



# from fastapi import FastAPI 
# from pydantic import BaseModel
# import xgboost as xgb
# import numpy as np
# import pickle

# # Load the pretrained XGBoost model
# with open("ProjectZipped\\Project\\xgboost_model.pkl", "rb") as f:
#     model = pickle.load(f)


# # Define the request body using Pydantic
# class ObesityPredictionRequest(BaseModel):
#     Age: int
#     Gender: str
#     Height: float
#     Weight: float
#     CALC: str
#     FAVC: str
#     FCVC: int
#     NCP: int
#     SCC: str
#     SMOKE: str
#     CH2O: int
#     family_history_with_overweight: str
#     FAF: int
#     TUE: int
#     CAEC: str
#     MTRANS: str

#     # Add other features as required

# # Initialize FastAPI
# app = FastAPI()

# @app.get("/")
# def read_root():
#     return {"message": "Welcome to the Obesity Prediction API"}

# def preprocess_input(data: ObesityPredictionRequest):
#     # Map categorical features to numerical values
#     gender = 1 if data.Gender.lower() == "male" else 0
#     family_history = 1 if data.family_history_with_overweight.lower() == "yes" else 0
#     favc = 1 if data.FAVC.lower() == "yes" else 0
#     smoke = 1 if data.SMOKE.lower() == "yes" else 0
#     scc = 1 if data.SCC.lower() == "yes" else 0
    
#     calc_mapping = {"no": 0, "sometimes": 1, "frequently": 2, "always": 3}
#     calc = calc_mapping.get(data.CALC.lower(), 0)
    
#     caec_mapping = {"no": 0, "sometimes": 1, "frequently": 2, "always": 3}
#     caec = caec_mapping.get(data.CAEC.lower(), 0)
    
#     mtrans_mapping = {"automobile": 0, "motorbike": 1, "bike": 2, "public_transportation": 3, "walking": 4}
#     mtrans = mtrans_mapping.get(data.MTRANS.lower(), 0)

#     # Prepare the input features for the model
#     input_features = np.array([
#         data.Age,
#         gender,
#         data.Height,
#         data.Weight,
#         calc,
#         favc,
#         data.FCVC,
#         data.NCP,
#         scc,
#         smoke,
#         data.CH2O,
#         family_history,
#         data.FAF,
#         data.TUE,
#         caec,
#         mtrans
#     ])
    
#     return input_features.reshape(1, -1)

# # @app.get("/predict/")
# # def read_item(Age: int, Gender: str, Height: float, Weight: float,CALC: str,FAVC: str,FCVC: int,NCP: int,SCC: str,SMOKE: str,CH2O: int,family_history_with_overweight: str,FAF: int,TUE: int,CAEC: str,MTRANS: str):
# #     return np.array([ Age, Gender, Height,  Weight, CALC, FAVC, FCVC, NCP,SCC,SMOKE, CH2O, family_history_with_overweight ,FAF, TUE,CAEC,MTRANS])

# @app.post("/predict")
# def predict_obesity_level(request: ObesityPredictionRequest):
#     # Preprocess the input data
#     input_features = preprocess_input(request)
    
#     # # Predict using the XGBoost model
#     # dmatrix = xgb.DMatrix(input_features)
#     # prediction = model.predict(dmatrix)
    
#     # # Assuming the prediction is a continuous value and we need to classify it
#     # predicted_class = int(np.round(prediction[0]))
    
#     return {"predicted_obesity_level": input_features}

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)

