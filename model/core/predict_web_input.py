from .predict import Predict
import pandas as pd

class PredictWebInput(Predict):
    def __init__(self, input_data):
        super().__init__()
        self.input_data = input_data

    def make_prediction(self):
        df = pd.DataFrame([self.input_data])

        # clean, transform and prepare user input
        X = self.prepare_data(df)

        # make prediction
        Y = self.predict(X)

        # transfrom list of prediction into df
        ans_df = self.prediction_to_df(Y)
        
        # round data 
        ans_df = ans_df.apply(lambda x: round(x, 2))
        return ans_df
