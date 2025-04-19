from .predict import Predict
import pandas as pd 
import os

class PredictExcel(Predict):
    def __init__(self):
        super().__init__()
    
    def get_user_input(self):
        """
        get excel file from model/predictions/input/input.xlsx and return it as df
        """
        if not os.path.exists('model/predictions/input/input.xlsx'):
            print("file input.xlsx doesn't exist, you have to add excel file to folder model/predictions/input")
            quit(2)
            
        df = pd.read_excel("model/predictions/input/input.xlsx")
        df.columns = df.columns.str.lower().str.strip()
        
        input_names = self.indf.columns.to_list()
        ans = pd.DataFrame()
        
        for name in input_names:
            cleaned_name = name.lower().strip()
            if not cleaned_name in df:
                print(f"error: column name {name} doesn't exist in excel file, maybe you misspell the name")
                quit(3)
            ans[name] = df[cleaned_name]
        
        return ans 
    
    def make_prediction(self):
        """
        make predictions based on excel file in model/predictions/input/input.xlsx
        and return theam as df and save theam in predictions folder
        """
        # get user input
        df = self.get_user_input()
        
        # clean, transform and prepare user input
        X = self.prepare_data(df)
        
        # make prediction
        Y = self.predict(X)
        
        # transfrom list of prediction into df
        ans_df = self.prediction_to_df(Y)
        ans_df.to_excel("model/predictions/output/output.xlsx") #save df in predictions
        
        return ans_df