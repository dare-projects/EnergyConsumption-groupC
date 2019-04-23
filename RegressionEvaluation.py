#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 09:40:40 2019

@author: morettini-simone
"""
from sklearn.metrics import mean_squared_error, r2_score, matthews_corrcoef
import pandas
import xlrd
import pandas as pd

from sklearn.svm import SVC
from sklearn.svm import SVR

from sklearn import linear_model
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression

import pickle


#Evaluate a regression
#df: the dataframe
#columnNameX: array of name of the column for the indipendent variables
#columnNameY: name of the column for the dipendent variables
#trainPerc: value from 0 to 1 about how much data use for training
#typeRegression: 'linear' or 'not_linear' or random_forest
#model: it return the model obtained
def regression(df, columnNameX, columnNameY, trainPerc, typeRegression):
    msk = np.random.rand(len(df)) < trainPerc
    train = df[msk]
    test = df[~msk]
    
    #prepare data for training
    X=[]
    for name in columnNameX:
        X.append(list(train[name]))
    y=list(train[columnNameY])
    #prepare data for testing
    X_test=[]
    for name in columnNameX:
        X_test.append(list(test[name]))
    y_test=list(test[columnNameY])
    
    if(trainPerc==1):
        X_test=X
        y_test=y
    
    if(typeRegression=="linear"):
        return doLinearRegression(X, y, X_test, y_test)
    else:
        if(typeRegression=="not_linear"):
            return doNonLinearRegression(X, y, X_test, y_test)
        else:
            return doRandomForest(X, y, X_test, y_test)
    
def doLinearRegression(X, y, X_test, y_test): 
    #print("LINEAR")
    regr = linear_model.LinearRegression()
    regr.fit(np.array(X).transpose(),y )
        
    # Make predictions using the testing set
    y_pred = regr.predict(np.array(X_test).transpose())
    
    model=regr
    
    result='Coefficients: \n'+ str(regr.coef_ )
    result=result+' Intercept: \n'+ str(regr.intercept_)
    result=result+" MSE: "+ str(mean_squared_error(y_test, y_pred))
    result=result+' R2: '+ str(r2_score(y_test, y_pred))
    return result, model

def doNonLinearRegression(X, y, X_test, y_test):  
    #print("NON LINEAR")
    clf = SVR(gamma='auto')
    refrNotlinear=clf.fit(np.array(X).transpose(), y)

    # Make predictions using the testing set
    y_pred = refrNotlinear.predict(np.array(X_test).transpose())
    model=refrNotlinear
    
    result="MSE: "+ str(mean_squared_error(y_test, y_pred))+' R2: '+ str(r2_score(y_test, y_pred))
    return result, model

def doRandomForest(X, y, X_test, y_test): 
    #print("RANDOM FOREST")
    regr = RandomForestRegressor(max_depth=500, #random_state=0,
                          n_estimators=2000)
    regr.fit(np.array(X).transpose(), y)

    print("Important feature")
    print(regr.feature_importances_)
    
    y_pred =regr.predict(np.array(X_test).transpose())
    model=regr
    
    result="MSE: "+ str(mean_squared_error(y_test, y_pred))+' R2: '+ str(r2_score(y_test, y_pred))
    return result, model

def evaluateModel(df, columnNameX, model):
    X=[]
    for name in columnNameX:
        X.append(list(df[name]))
    return model.predict(np.array(X).transpose())

################################################################
##############RAW VALUE########################################
################################################################    
#LOAD DATA#
df = pandas.read_excel('./EnergySignature/allData_wLight.xls', header = 1)
df["Date"]=pd.to_datetime(df["Date"], format='%Y-%m%-%d %H:%M:%S')
df["HOURS"]=df["Date"].dt.hour
df["DAYS"]=df["Date"].dt.weekday#.map({0: 1, 1: 3, 2: 3, 3: 3, 4: 3, 5: 7, 6: 10})
df["SEASON"]=df["Date"].dt.month #.map({1: 1, 2: 1, 3: 1, 4: 2, 5: 2, 6: 2, 7: 2, 8: 2, 9: 2, 10: 1, 11: 1, 12: 1})
df["DIFF_TEMPERATURE"]=df["Value_InsideTemp"]-df["Value_OutsideTemp"]

################################################################
#################FIRST APPROACH#################################
################################################################ 
indipendentVariables=[
        "DIFF_TEMPERATURE",
        "SEASON",
        "DAYS",
        "HOURS",
        "Value_RoofIrradiance"
        ]
res, model=regression(df, indipendentVariables, "Value_TotalEnergy", 0.75, "random_forest")
print(res +" \n")

################################################################
#################SECOND APPROACH(BEST)##########################
################################################################    
indipendentVariables=[
        "DIFF_TEMPERATURE",
        "SEASON",
        "DAYS",
        "HOURS",
        "Value_RoofIrradiance",
        "Value_Light"
        ]
res, model=regression(df, indipendentVariables, "Value_TotalEnergy", 0.75, "random_forest")
print(res +" \n")
#save the prediction for plotting later
predictedValue=evaluateModel(df, indipendentVariables, model)
pickle.dump( predictedValue, open( "predictedValueWithRandomForest.p", "wb" ) )

################################################################
##############MEANS per day VALUE###############################
################################################################    
#LOAD DATA#
df = pandas.read_csv('./EnergySignature/all_mean_perday.csv', header = 0)

df["Date"]=pd.to_datetime(df["Date"], format='%Y-%m-%d %H:%M:%S')
df["DAYS"]=df["Date"].dt.weekday.map({0: 1, 1: 3, 2: 3, 3: 3, 4: 3, 5: 7, 6: 10})
df["SEASON"]=df["Date"].dt.month #.map({1: 1, 2: 1, 3: 1, 4: 2, 5: 2, 6: 2, 7: 2, 8: 2, 9: 2, 10: 1, 11: 1, 12: 1})
df["DIFF_TEMPERATURE"]=df["Value_InsideTemp"]-df["Value_OutsideTemp"]

indipendentVariables=[
        "DIFF_TEMPERATURE",
        "SEASON",
        "DAYS",
        "Value_RoofIrradiance"
        ]

print("\n")
res, model=regression(df, indipendentVariables, "Value_TotalEnergy", 0.75, "random_forest")
print(res +" \n")

################################################################
##############MEANS SEPARATED(DAY/NIGHT) VALUE##################
################################################################ 

print("\n MEANS SEPARATED(DAY/NIGHT) VALUE")
#LOAD DATA#
df_night = pandas.read_csv('./EnergySignature/mean_nighthours.csv', header = 0)
df_sunHours = pandas.read_csv('./EnergySignature/mean_laboralhours.csv', header = 0)

df_night["Data"]=pd.to_datetime(df_night["Data"], format='%Y-%m-%d %H:%M:%S')
df_sunHours["Data"]=pd.to_datetime(df_sunHours["Data"], format='%Y-%m-%d %H:%M:%S')
msk = df_sunHours["Data"].dt.weekday <5 
df_sunHours_week = df_sunHours[msk]
df_sunHours_weekend = df_sunHours[~msk]

indipendentVariables=[
        "solar_irradiace",
        "inside_temp",
        "outside_temp"
        ]
print("____sunHours weekday____")
res, model=regression(df_sunHours_week, indipendentVariables, "total_energy", 0.75, "random_forest")
print(res +" \n")
print("____night____")
res, model=regression(df_night, indipendentVariables, "total_energy", 0.75, "random_forest")
print(res +" \n")
print("____sunHours weekend____")
res, model=regression(df_sunHours_weekend, indipendentVariables, "total_energy", 0.75, "random_forest")
print(res +" \n")




