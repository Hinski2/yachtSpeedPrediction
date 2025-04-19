import model

predictor = model.PredictExcel()
result_df = predictor.make_prediction()
print(result_df)