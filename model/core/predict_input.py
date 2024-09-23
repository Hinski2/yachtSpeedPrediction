from .predict import Predict 
import pandas as pd 

class PredictInput(Predict):
    def __init__(self):
        super().__init__()
    
    def collect_use_input(self):
        """
        collects user input and returnes it as df
        """
        input_names =  self.indf.columns.to_list()
        
        input_data = {} 
        for input_name in input_names:
            input_data[input_name] = input(f"enter {input_name}:")
         
        return pd.DataFrame([input_data])
    
    def make_prediction(self):
        # get user input
        df = self.collect_use_input()
        
        # clean, transform and prepare user input
        X = self.prepare_data(df)
        
        # make prediction
        Y = self.predict(X)
        
        # transfrom list of prediction into df
        ans_df = self.prediction_to_df(Y)
        return ans_df