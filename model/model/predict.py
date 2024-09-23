import torch  
import os 
import numpy as np 
import pandas as pd 
import re 
from typing import List
from .best_nn_settings.model_class_0 import ModelClass0 
from .best_nn_settings.model_class_1 import ModelClass1

class Predict:
    def __init__(self):
        if os.path.isfile('data/clean/dataLimits.xlsx'):
            self.indf = pd.read_excel('data/clean/scaledin.xlsx').drop('Unnamed: 0', axis=1)
            self.outdf = pd.read_excel('data/clean/scaledout.xlsx').drop('Unnamed: 0', axis=1)
            
        else:
            print("file dataLimits doesn't exists at first you have to run dataAnalys notebook")
            
        #hyperparemeters
        self.device = 'cuda' if torch.cuda.is_available else 'cpu'
        self.IN_FEATURES = len(self.indf.columns)
        self.HIDDEN_UNITS1 = 128
        self.HIDDEN_UNITS2 = 64
        self.OUT_FEATURES = 1
        
        #create models
        self.models = {} 
        filenames = self.outdf.columns.to_list()
        torch.manual_seed(42)
        
        for file in filenames[:-8]:
            self.models[file] = ModelClass0(input_shape=self.IN_FEATURES,
                                    hidden_units1=self.HIDDEN_UNITS1,
                                    hidden_units2=self.HIDDEN_UNITS2,
                                    output_shape=self.OUT_FEATURES).to(self.device)
            self.models[file].load_state_dict(torch.load('model/model/models/' + file))
            self.models[file].eval()
            self.models[file] = self.models[file].to(self.device)
        
        for file in filenames[-8:]:
            self.models[file] = ModelClass1(input_shape=self.IN_FEATURES,
                                    hidden_units1=self.HIDDEN_UNITS1,
                                    hidden_units2=self.HIDDEN_UNITS2,
                                    output_shape=self.OUT_FEATURES).to(self.device)
            self.models[file].load_state_dict(torch.load('model/model/models/' + file))
            self.models[file].eval()
            self.models[file] = self.models[file].to(self.device)
        
    def prepare_data(self, df: pd.DataFrame):
        """
            prepare dataframe for model prediction 
            Args:
                df (pd.DataFrame): input dataframe
            Returns:
                list of tensors ready to make predictions
        """ 
        
        def doRegex(x):
            """
            removes all non-numeric characters (exept .) from the string
            """
            return re.sub(r'[^\d.]+', '', x)

        def scale_item(l, r, x):
            """
            Scales value 'x' to a range between 0 and 1
            Args:
                l: The lower bound of the range
                r: The upper bound of the range
                x: The value to be scaled
            """
            return (x - l) / (r - l)
        
        def scale_dataFrame(df: pd.DataFrame):
            """
            Scales the whole dataframe to range between 0 and 1
            """
            for row in range(len(df)):
                for column in df.columns.tolist():
                    l, r = self.indf[column] 
                    df.loc[row, column] = np.float32(scale_item(l, r, df.loc[row, column]))
            return df
        
        # changing IMS division
        int_of_div = {'Cruiser/Racer': 2, 
                'Performance': 3, 
                'Sportboat': 1}
        
        for row in range(len(df)):
            if df.loc[row, 'IMS Division'] in int_of_div:
                df.loc[row, 'IMS Division'] = int_of_div[df.loc[row, 'IMS Division']]
            else:
                df.loc[row, 'IMS Division'] = 0
            
        # change data
        try:
            df['Series date'] = pd.to_datetime(df['Series date'], format='%m/%Y').astype(np.int64).astype(np.float32)
        except:
            df['Series date'] = df['Series date'].astype(np.float32)
            
        # remove trdundant characters
        df[self.indf.columns.tolist()] = df[self.indf.columns.tolist()].astype(str)
        df[[column for column in self.indf.columns.tolist() if column != 'Series date']] = df[[column for column in self.indf.columns.tolist() if column != 'Series date']].map(doRegex)
        
        # change to float
        df[self.indf.columns.tolist()] = df[self.indf.columns.tolist()].astype(np.float32)
        
        # scaling data
        df = scale_dataFrame(df)
    
        # convert to tensor 
        return [torch.tensor(row, dtype=torch.float32) for row in df.to_numpy()]
    
    def predict(self, tensorList: List[torch.tensor]):
        """
        makes prediction for list of tensors (former columns of df)
        Args:
            tensorList (List of torch.tensor): output of prepare data function
        Returns:
            list of dictionaries
        """
        def make_pred(model, X: torch.tensor):
            """
            makes prediction with one model (out of 96)
            Args:
                model: name of model (you can find theam in self.outdf.columns.to_list())
                X (torch.tensor): input
            Returns:
                predicted float value
            """
            X = X.to(self.device)
            
            self.models[model].eval()
            with torch.inference_mode():
                pred = self.models[model](X).squeeze()
            return pred.item()
        
        #making predictions for list of tensors 
        ans = []
        for X in tensorList:
            y = {}
            for column in self.outdf.columns.tolist():
                y[column] = make_pred(column, X)
            ans.append(y)
        return ans
    
    def prediction_to_df(self, Y):
        """
        converts list of dictionraies (returned by functinoo predict) to dataFrame
        Args:
            Y (list of dict of {column, predicted value}
        Return:
            dataframe
        """ 
        
        df = pd.DataFrame()
        for y in Y: 
            df = pd.concat([df, pd.DataFrame([y])])
        return df 