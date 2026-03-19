
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import numpy as np
from PIL import Image
import io

app = FastAPI(
    title="Handwriting Origin Classifier API",
    description="CNN model classifying English handwriting by nationality",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

COUNTRIES  = ["Indian","American","Chinese","Japanese"]
FLAGS      = ["🇮🇳","🇺🇸","🇨🇳","🇯🇵"]

interpreter = tf.lite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()
input_details  = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def run_prediction(image):
    img = image.convert("L").resize((128,128))
    arr = np.array(img, dtype=np.float32)/255.0
    arr = arr.reshape(1,128,128,1)
    interpreter.set_tensor(input_details[0]["index"], arr)
    interpreter.invoke()
    return interpreter.get_tensor(output_details[0]["index"])[0]

@app.get("/")
def home():
    return {
        "message" : "Handwriting Origin Classifier API",
        "version" : "1.0.0",
        "author"  : "Harshal Mathur",
        "accuracy": "70.63%",
        "endpoints": {
            "GET  /"        : "API info",
            "GET  /health"  : "Health check",
            "GET  /model"   : "Model details",
            "POST /predict" : "Predict nationality"
        }
    }

@app.get("/health")
def health():
    return {"status":"healthy","model":"loaded"}

@app.get("/model")
def model_info():
    return {
        "architecture": "3-layer CNN",
        "filters"     : [32,64,64],
        "parameters"  : "2.15M",
        "input_size"  : "128x128 grayscale",
        "classes"     : COUNTRIES,
        "accuracy"    : "70.63%",
        "model_size"  : "2.1MB TFLite"
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents   = await file.read()
    image      = Image.open(io.BytesIO(contents))
    preds      = run_prediction(image)
    pred_class = int(np.argmax(preds))
    confidence = float(preds[pred_class])
    return {
        "predicted_country": COUNTRIES[pred_class],
        "flag"             : FLAGS[pred_class],
        "confidence"       : round(confidence*100, 2),
        "all_scores"       : {
            COUNTRIES[i]: round(float(preds[i])*100, 2)
            for i in range(4)
        }
    }
