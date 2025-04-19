from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from model import PredictWebInput, PredictExcel
import uvicorn
import pandas as pd
import io
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class InputData(BaseModel):
    input_data: dict

@app.post("/predict_input")
def predict_input(data: InputData):
    predictor = PredictWebInput(data.input_data)
    result_df = predictor.make_prediction()
    result = result_df.to_dict(orient='records')[0]  # Pobieramy pierwszy (i jedyny) rekord
    return result

@app.post("/predict_excel")
def upload_excel(file: UploadFile = File(...)):
    # Sprawdzenie, czy przesłany plik jest typu Excel
    if file.filename.endswith(('.xls', '.xlsx')):
        contents = file.file.read()
        input_path = 'model/predictions/input/input.xlsx'
        with open(input_path, 'wb') as f:
            f.write(contents)
        
        # Przetworzenie pliku za pomocą PredictExcel
        predictor = PredictExcel()
        result_df = predictor.make_prediction()

        # Zapisanie wyników do pliku Excela
        output = io.BytesIO()
        result_df.to_excel(output, index=False)
        output.seek(0)

        return FileResponse(output, media_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet', filename='output.xlsx')
    else:
        return {"error": "Invalid file type. Please upload an Excel file."}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)