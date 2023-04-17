# PACKAGES IMPORT
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import pickle
import os.path
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build

import xgboost as xgb
from sklearn.svm import SVR
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import StackingRegressor

from sklearn.metrics import mean_squared_error as mse
from sklearn.model_selection import GridSearchCV

from datetime import date
from dateutil.relativedelta import relativedelta

import multiprocessing
import time

# Google sheet api connection check
def gsheet_api_check(SCOPES):
    creds = None
    if os.path.exists('token.pickle'):
        with open('token.pickle', 'rb') as token:
            creds = pickle.load(token)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                'credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        with open('token.pickle', 'wb') as token:
            pickle.dump(creds, token)
    return creds

# Google sheet data pull
def pull_sheet_data(SCOPES,SPREADSHEET_ID,DATA_TO_PULL):
    creds = gsheet_api_check(SCOPES)
    service = build('sheets', 'v4', credentials=creds)
    sheet = service.spreadsheets()
    result = sheet.values().get(
        spreadsheetId=SPREADSHEET_ID,
        range=DATA_TO_PULL).execute()
    values = result.get('values', [])
    
    if not values:
        print('No data found.')
    else:
        rows = sheet.values().get(spreadsheetId=SPREADSHEET_ID,
                                  range=DATA_TO_PULL).execute()
        data = rows.get('values')
        print("COMPLETE: Data copied")
        return data
    
def Feature_engineering():
    SCOPES = ['https://www.googleapis.com/auth/spreadsheets']
    SPREADSHEET_ID = '1MH-ufKNr2U_OY-L9p5rknxinAb225jbcp55irHUCdoE'
    data = pull_sheet_data(SCOPES,SPREADSHEET_ID, 'g_block')
    g = pd.DataFrame(data[1:], columns=data[0])
    
    
    # for blockchain backend easyness raw data saving
    g.to_csv('raw_g_block.csv', index = False , mode='w+')
    g.to_json('raw_g_block.json', orient="values" )

    current_g = g[pd.to_datetime(g['DateTime']).dt.date == pd.to_datetime(g.iloc[-1]['DateTime']).date()]

    current_g.to_csv('raw_today_g_block.csv', index = False , mode='w+')
    current_g.to_json('raw_today_g_block.json', orient="values" )

    g.drop('cputemp' ,axis= 1 , inplace = True)
    g[[ 'Vplus', 'Qv', 'Vminus']] = g[[ 'Vplus', 'Qv', 'Vminus']].astype(float)
    g['DateTime'] = pd.to_datetime( g['DateTime'])
    g['weekday'] = g['DateTime'].apply(lambda datetime : datetime.isocalendar().weekday)
    g['month'] = g['DateTime'].apply(lambda datetime : datetime.month)
    g['year'] = g['DateTime'].apply(lambda datetime : datetime.year)
    g['day'] = g['DateTime'].apply(lambda datetime : datetime.day)
    g['time'] = g['DateTime'].apply(lambda datetime : datetime.time())
    g['Date'] = g['DateTime'].dt.date

    g.drop(['Qv' , 'Vminus' , 'DateTime'] , axis = 1 , inplace = True )
    
    max_g = g.groupby(['year','month' ,'day'] , group_keys = True).max().reset_index()
    max_g.set_index('Date' , inplace = True)
    min_g = g.groupby(['year','month' ,'day'] , group_keys = True).min().reset_index()
    min_g.set_index('Date' , inplace = True)
    girls_hostel = min_g.copy()[['year','month' ,'day' , 'weekday']]
    girls_hostel['total_usage'] = max_g['Vplus']-min_g['Vplus']
    girls_hostel.fillna(girls_hostel.mean() , inplace = True)
    
    csv_data = girls_hostel.reset_index().to_csv('girls.csv', index = False , mode='w+')
    json_data = girls_hostel.reset_index().to_json('girls.json', orient="values" )
    
    return girls_hostel

def Build_model_1():
    while True:    
        girls_hostel = Feature_engineering()
        # Machine Learning
        x_train = girls_hostel[['day','weekday' , 'month']]
        y_train = girls_hostel['total_usage']

        DM_train = xgb.DMatrix(data=x_train, label=y_train)
        grid_param = {"learning_rate": [0.01, 0.001, 0.1],
                    "n_estimators": [100, 150, 200 , 250 , 300 , 350, 400],
                    "alpha": [0.1,0.75 , 0.5, 1],
                    "max_depth": [2, 3, 4 , 6, 9 , 11]}

        grid_mse = GridSearchCV(estimator= xgb.XGBRegressor(), param_grid=grid_param,
                            scoring="neg_mean_squared_error",
                            cv=4, verbose=1)
        grid_mse.fit(x_train, y_train)
        print("Best parameters found: ", grid_mse.best_params_)
        print("Lowest RMSE found: ", np.sqrt(np.abs(grid_mse.best_score_)))

        xgb_model = xgb.XGBRegressor(objective ='reg:squarederror', colsample_bytree = 1, **grid_mse.best_params_)
        xgb_model.fit(x_train, y_train)
        with open('girls_model.pkl', 'w+b') as saved_model:
            pickle.dump(xgb_model, saved_model)

# This model need 3-4 hours for training  based on i7 processor and 16 gb ram. 
# so choose trained model wisely
def Build_model_2():
    while True:    
        
        girls_hostel = Feature_engineering()
        # Machine Learning
        x_train = girls_hostel[['day','weekday' , 'month']]
        y_train = girls_hostel['total_usage']

        estimators = [
            ('XGB', xgb.XGBRegressor()),
            ('svr', SVR()),
            ('forest', RandomForestRegressor()),
            ('LR' ,Ridge(alpha=1.0))
        ]
        reg = StackingRegressor(
            estimators=estimators,
            final_estimator=RandomForestRegressor()
        )
        grid_param = {
                    "XGB__learning_rate": [0.01, 0.001, 0.1],
                    "XGB__n_estimators": [100, 150, 200 , 250 , 300 ],
                    "XGB__alpha": [0.1,0.75 , 0.5, 1],
                    "XGB__max_depth": [2, 3, 4 , 6, 9 ],
                    
                    'svr__C': [0.1, 1, 10, 100 ], 
                    'svr__gamma': [1, 0.1, 0.01, 0.001 ,'scale', 'auto'],
                    'svr__kernel': ['linear', 'rbf', 'sigmoid'],

                    'forest__max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],
                    'forest__n_estimators': [100 ,200,300 , 400, 600, ],
                    
                    'LR__alpha': [0.1 , 0.5 , 1.0 , 1.5]
                    }
        grid = GridSearchCV(reg, grid_param, refit = True, verbose = 3 , scoring = "neg_mean_squared_error" )
        
        # fitting the model for grid search
        grid.fit(x_train, y_train)
        print("Best parameters found: ", grid.best_params_)
        print("Lowest RMSE found: ", np.sqrt(np.abs(grid.best_score_)))

        with open('girls_model.pkl', 'w+b') as saved_model:
            pickle.dump(grid, saved_model)


def Predict():
    while True:
        # Future data parameter creation
        predictions = pd.DataFrame(pd.date_range(date.today(), (date.today() + relativedelta(months=1)),freq='d'), columns=['Date'])
        predictions['day'] = predictions['Date'].dt.day
        predictions['weekday'] = predictions['Date'].dt.weekday
        predictions['month'] = predictions['Date'].dt.month
        # predictions = predictions.set_index('Date')

        with open('girls_model.pkl', 'rb') as model:
            load_model = pickle.load(model)
        predictions['total_usage'] = load_model.predict(predictions.drop('Date' , axis=1))
        predictions['Date'] = predictions['Date'].dt.date.astype(object)
        csv_data = predictions.to_csv('girls_future.csv', index = False , mode='w+')
        json_data = predictions.to_json('girls_future.json', orient="values"  )
        print('\nCSV String:\n', csv_data)
        time.sleep(3600)


def main():
    Feature_engineering()
    # Build_model_1()
    # Predict()
    
    # creating processes
    # p1 = multiprocessing.Process(target=Predict())
    # p2 = multiprocessing.Process(target=Build_model_1())

    # p1.start()
    # p2.start()

    # p1.join()
    # p2.join()

    

if __name__ == "__main__":
    main()

