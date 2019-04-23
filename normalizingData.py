#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 11:59:50 2019

@author: quau3
"""

import pandas
import xlrd
import pandas as pd
from pandas import DataFrame

import plotly
import plotly.plotly as py
import plotly.graph_objs as go 
import datetime
from datetime import timedelta



import pickle
import csv

plotly.offline.init_notebook_mode(connected=False)
df = pandas.read_excel('./EnergySignature/allData.xls', header =1)
data = DataFrame(df, columns= ['Date', 'Value_RoofIrradiance', 'Value_TotalEnergy','Value_OutsideTemp', 'Value_InsideTemp'])

dataList = data.values.tolist()
#Scripprint(df.Date)
"""
with open('mean_perday_mayagus.csv', mode='w') as csv_file:
    fieldnames = ['Data', 'solar_irradiace', 'total_energy', 'inside_temp', 'outside_temp']
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

    writer.writeheader()
    count=0
    solar=0
    ouT=0
    inT=0
    tEn=0
    d=0
    for v in dataList:
        solar+=v[1]
        tEn+=v[2]
        ouT+=v[3]
        inT+=v[4]
        print(v[1])
        count+=1
        if str(v[0].time()) == "00:00:00":
            #solarMean:s.append(solar/count)
            #outTemp.append(ouT/count)
            #inTemp.append(inT/count)
            #totalEne.append(tEn/count)
            writer.writerow({'Data':v[0],'solar_irradiace':solar/count, 'total_energy': tEn/count, 'inside_temp':inT/count,'outside_temp':ouT/count })
            inT=0
            ouT=0
            tEn=0
            solar=0
            count=0
        
#dayAverag=[solarMeans,outTemp,inTemp,totalEne]

#with open('dayAverage_3rd.pickle', 'wb') as handle:
 #   pickle.dump(dayAverag, handle, protocol=pickle.HIGHEST_PROTOCOL)

#Script to get laboral hours
with open('mean_laboralhours.csv', mode='w') as csv_file:
    fieldnames = ['Data', 'solar_irradiace', 'total_energy', 'inside_temp', 'outside_temp']
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()
    count=0
    solar=0
    ouT=0
    inT=0
    tEn=0
    for i in range(0,len(dataList)):
        if str(dataList[i][0].time())=="06:00:00":
            for j in range(0,14*4+1):
                if str(dataList[i+j][0].time())=="20:15:00":
                    break;
                else:
                    count+=1
                    solar+=dataList[i+j][1]
                    ouT+=dataList[i+j][2]
                    inT+=dataList[i+j][3]
                    tEn+=dataList[i+j][4]
            writer.writerow({'Data':dataList[i][0],'solar_irradiace':solar/count, 'total_energy': tEn/count, 'inside_temp':inT/count,'outside_temp':ouT/count })
            count=0
            solar=0
            ouT=0
            inT=0
            tEn=0
"""            
#Script to get night hours 


with open('mean_nighthours.csv', mode='w') as csv_file:
    fieldnames = ['Data', 'solar_irradiace', 'total_energy', 'inside_temp', 'outside_temp']
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()
    count=0
    solar=0
    ouT=0
    inT=0
    tEn=0
    for i in range(0,len(dataList)):
        if str(dataList[i][0].time())=="20:15:00":
            for j in range(0,10*4+1):
                if str(dataList[i+j][0].time())=="06:00:00":
                    break;
                else:
                    print(dataList[i+j][0])
                    count+=1
                    solar+=dataList[i+j][1]
                    ouT+=dataList[i+j][2]
                    inT+=dataList[i+j][3]
                    tEn+=dataList[i+j][4]
            writer.writerow({'Data':dataList[i][0],'solar_irradiace':solar/count, 'total_energy': tEn/count, 'inside_temp':inT/count,'outside_temp':ouT/count })
            count=0
            solar=0
            ouT=0
            inT=0
            tEn=0             