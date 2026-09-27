# Import Data Mainpulation Libraries
import numpy as np 
import pandas as pd 

def data_loader():
    df = pd.read_csv(r'C:\ai-impact-jobs-layoff-prediction_model\data\ai-impact-jobs-layoff-risk-dataset.csv')
    
    return df