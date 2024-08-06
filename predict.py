import torch
from torch import nn 
import os 
import numpy as np
import pandas as pd 
import re
import sys

# load data limist 

if os.path.isfile('data/clean/dataLimits.xlsx'):
    indf = pd.read_excel('data/clean/scaledin.xlsx').drop('Unnamed: 0', axis=1)
    outdf = pd.read_excel('data/clean/scaledout.xlsx').drop('Unnamed: 0', axis=1)
    
else:
    print("file dataLimits doesn't exists at first you have to run dataAnalys notebook")

#hyperparameters
device = 'cuda' if torch.cuda.is_available else 'cpu'
IN_FEATURES = len(indf.columns)
HIDDEN_UNITS1 = 128
HIDDEN_UNITS2 = 64
OUT_FEATURES = 1

# model classes 
class ModelClass0(nn.Module):
    def __init__(self, 
                 input_shape: int, 
                 hidden_units1: int, 
                 hidden_units2: int, 
                 output_shape: int): 
        super().__init__()
        self.layer_stack = nn.Sequential(
            nn.Linear(in_features=input_shape, out_features= hidden_units2),
            nn.Linear(in_features=hidden_units2, out_features=hidden_units2),
            nn.Linear(in_features=hidden_units2, out_features=hidden_units1),
            nn.Linear(in_features=hidden_units1, out_features=output_shape)
        )
        
    def forward(self, x):
        return self.layer_stack(x) 

class ModelClass1(nn.Module):
    def __init__(self, 
                 input_shape: int, 
                 hidden_units1: int, 
                 hidden_units2: int, 
                 output_shape: int): 
        super().__init__()
        self.layer_stack = nn.Sequential(
            nn.Linear(in_features=input_shape, out_features= hidden_units1),
            nn.ReLU(),
            nn.Linear(in_features=hidden_units1, out_features=hidden_units2),
            nn.ReLU(),
            nn.Linear(in_features=hidden_units2, out_features=hidden_units2),
            nn.ReLU(),
            nn.Linear(in_features=hidden_units2, out_features=hidden_units1),
            nn.ReLU(),
            nn.Linear(in_features=hidden_units1, out_features=output_shape), 
            nn.ReLU(),
        )
        
    def forward(self, x):
        return self.layer_stack(x) 

#loading models 
models = {}
filenames = outdf.columns.to_list()
torch.manual_seed(42)

for file in filenames[:-8]:
    models[file] = ModelClass0(input_shape=IN_FEATURES,
                               hidden_units1=HIDDEN_UNITS1,
                               hidden_units2=HIDDEN_UNITS2,
                               output_shape=OUT_FEATURES).to(device)
    models[file].load_state_dict(torch.load('models/' + file))
    models[file].eval()
    models[file] = models[file].to(device)
    
for file in filenames[-8:]:
    models[file] = ModelClass1(input_shape=IN_FEATURES,
                               hidden_units1=HIDDEN_UNITS1,
                               hidden_units2=HIDDEN_UNITS2,
                               output_shape=OUT_FEATURES).to(device)
    models[file].load_state_dict(torch.load('models/' + file))
    models[file].eval()
    models[file] = models[file].to(device)
    
# generation predictions
def scale_item(l, r, x):
    return (x - l) / (r - l)

def scale_dataFrame(df: pd.DataFrame):
    for column in df.columns.tolist():
        l, r = indf[column] 
        df.loc[0, column] = np.float32(scale_item(l, r, df.loc[0, column]))
    return df

def make_pred(model, X: torch.tensor):
    X = X.to(device)
    
    models[model].eval()
    with torch.inference_mode():
        pred = models[model](X).squeeze()
    return pred.item()

def doRegex(x):
    return re.sub(r'[^\d.]+', '', x)

def clean_data(df: pd.DataFrame):
	# changing IMS division
    int_of_div = {'Cruiser/Racer': 2, 
              'Performance': 3, 
              'Sportboat': 1}

    if df.loc[0, 'IMS Division'] in int_of_div:
        df.loc[0, 'IMS Division'] = int_of_div[df.loc[0, 'IMS Division']]

    else:
        df.loc[0, 'IMS Division'] = 0
        
    # change data
    df['Series date'] = pd.to_datetime(df['Series date'], format='%m/%Y').astype(np.int64).astype(np.float32)
        
    # removing reductant characters
    df[indf.columns.tolist()] = df[indf.columns.tolist()].astype(str)
    df[[column for column in indf.columns.tolist() if column != 'Series date']] = df[[column for column in indf.columns.tolist() if column != 'Series date']].map(doRegex)
    
    # changing to float
    df[indf.columns.tolist()] = df[indf.columns.tolist()].astype(np.float32)
    
    # scaling data
    df = scale_dataFrame(df)
    
    # convert to tensor 
    return torch.tensor(df.values, dtype=torch.float32)
        

# get data from user
df = pd.DataFrame()
if(len(sys.argv) == 1):
    input_data = {}
    
    # get data
    for column in indf.columns.tolist():
        input_data[column] = input(f'input {column} >')
        
    df = pd.DataFrame([input_data])
    
elif(len(sys.argv) == 3):
        df = pd.read_excel(f'predictions/input/{str(sys.argv[2])}')

else:
    print("inapropriate noumber of arguments")
    exit(42)
    
# where i schould save results 
path = input('how do you want to name file with results? >')
path = 'results/' + path + '.xlsx'

# clean data
X = clean_data(df)

# make predictions
output_data = {}
for column in outdf.columns.tolist():
    output_data[column] = make_pred(column, X)
    
y = pd.DataFrame([output_data])

# save predictions
if not os.path.exists('results'):
    os.mkdir('results')
y.to_excel(path)