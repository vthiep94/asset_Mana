#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 28 20:17:05 2020

@author: hiepvu
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import requests
from bs4 import BeautifulSoup 
import requests
import os
import time
import math
import statsmodels.api as sm
import seaborn as sns
from pandas_datareader import data
from statsmodels.sandbox.regression.predstd import wls_prediction_std

sns.set(style="whitegrid")
plt.style.use("Solarize_Light2")

data = pd.read_csv("allRawData.csv", index_col = 0)
extra = pd.read_csv("extra1.csv", index_col = 0)
extra2 = pd.read_csv("extra2.csv", index_col = 0)
extra.index = ["Revenue", "Cash"]
extra2.index = ["InterEx"]
relCols =[]
growth  = pd.read_csv("growth.csv", index_col = 0)
beta20 = pd.read_csv("beta20.csv", index_col = 0)


newD = extra

for col in extra.columns:
    if col not in data.columns:
        newD = newD.drop([col])
        
for col in data.columns:
    if col not in newD.columns:
        newD[col] = [np.nan] * len(newD.index)

for row in newD.index:
    data.loc[row] = newD.loc[row]





def add(newD, data = data):
    for col in extra.columns:
        if col not in data.columns:
            newD = newD.drop([col])
    for col in data.columns:
        if col not in newD.columns:
            newD[col] = [np.nan] * len(newD.index)
    
    if len(newD.columns) != len(data.columns):
        print("Not matched length")
        return
    
    for row in newD.index:
        data.loc[row] = newD.loc[row]
        
    return data

data = add(extra2)
data = add(extra)
data = add(growth)
data = add(beta20)

for col in data.columns:
    data.loc[data[col] == "N/A", col] = np.nan

data1 = data.dropna(axis = 1, how = "all")


test = ["1,290.2B", "8.56% ", "930.15M", np.nan, "1,294.56", 2.4]
test1 = []

def clean(item):
    if type(item) == str:
        item = item.replace(",", "")
        item = item.replace(" ", "")
        if  "B" in item or "b" in item:
            item = item.replace("B", "")
            item = float(item) * 1000000000
        elif  "M" in item or "m" in item:
            item = item.replace("M", "")
            item = float(item) * 1000000
        elif "K" in item or "k" in item:
            item = item.replace("K", "")
            item = item.replace("k", "")
            item = float(item) * 1000
        elif "%" in item:
            item = float(item.replace("%", ""))/100
    return item

for item in test:
    print(clean(item))

for i in range(len(data1)):
    if i not in [11, 12]:
        data1.iloc[i, :] = data1.iloc[i, :].apply(lambda x: clean(x))
        data1.iloc[i, :] = pd.to_numeric(data1.iloc[i,:])




data1 = data1.T


peR = "PE Ratio (TTM)"
beta = "Beta (5Y Monthly)"
poR = "Payout Ratio 4"
roeR = "Return on Equity (ttm)"
levR = "Net Lev"
incR = "IntCo"


data1["Net Lev"] = (data1["EBITDA "] - data1["Cash"])/data1["Total Debt (mrq)"]
data1["EBIT"] = data1["Revenue"] * data1["Operating Margin (ttm)"]
data1["IntCo"] = data1["EBIT"]/data1["InterEx"]
data1["Price"] = data1[peR] * data1["EPS (TTM)"]
data1["NoShare"] = data1["Market Cap"]/data1["Price"]
data1["FirmV"] = (data1["Market Cap"] + data1["Total Debt (mrq)"] - data1["Cash"])
data1["FirmV/EBITDA"] = data1["FirmV"]/data1["EBITDA "]
data1["BV"] = data1["Net Income Avi to Common (ttm)"]/data1["Return on Equity (ttm)"]
corCols = [s for s in data1.columns if s not in ["Industry", "Sector"]]
data1[corCols] = data1[corCols].apply(pd.to_numeric)
data1["Utilities"] = np.where(data1["Sector"] == "Utilities", 1,0)






def regress(cols, dep, data1 = data1, sd = False, log = False, orig = False):
    data1 = data1[cols].dropna(axis = 0, how = "any")
    if sd == True:
        for col in cols:
            std = np.std(data1[col])
            mean = np.mean(data1[col])
            data1 = data1.loc[(data1[col] > mean - std * 3) & 
                              (data1[col] < mean + std * 3)]
    
#    print(len(data1))
    indeps = [x for x in cols if x != dep]
#    sns.pairplot(data1)
    
    if log == True :
        for col in indeps:
            try:
                dataT[col] = np.log(dataT[col])
            except:
                pass
        
    y = data1[dep]
    X = data1[indeps]
    
    if orig == False:
        X = sm.add_constant(X)

    regress = sm.OLS(y.astype(float), X.astype(float))
    result = regress.fit()

    return result


dummies = pd.get_dummies(data1["Sector"])
dummies[peR] = data1[peR]
res1 = regress(cols = list(dummies.columns), dep = peR, data1 = dummies)
print(res1.summary())


dum2 = pd.get_dummies(data1["Industry"])
dum2[peR] = data1[peR]
res2 = regress(cols = list(dum2.columns), dep = peR, data1 = dum2)
print(res2.summary())

sigInd = []
for x in range(len(res2.tvalues)):
    if abs(res2.tvalues[x]) > 2:
        sigInd = sigInd + [res2.tvalues.index[x]]
 
for ind in sigInd:
    data1[ind] = np.where(data1["Industry"] == ind, 1, 0)

dumCols = sigInd + ["Utilities"]

data2 = data1.copy()
for col in data2.columns:
    if col not in dumCols + [peR] + ["Sector", "Industry"]:
        data2[col] = (data2[col] - np.mean(data2[col]))/np.std(data2[col])



secU = data1[[peR, poR, beta, roeR, levR, incR, "Sector"]].dropna(axis = 0, how = "any").groupby(by = "Sector").mean()
secU["count"] = data1["Sector"].value_counts()
plt.show()




fig, ax = plt.subplots()
ax.bar(x = secU.index, height = secU[peR])
plt.xticks(rotation=90)
plt.show()

indU = data1[[peR, poR, beta, roeR, levR, incR, "Industry"]].dropna(axis = 0, how = "any").groupby(by = "Industry").mean()
indU["count"] = data1["Industry"].value_counts()

median = round(np.median(data1[peR].dropna()),2)
mean = round(np.mean(data1[peR].dropna()),2)

fig, ax = plt.subplots(figsize = (20,10))
ax.bar(x = indU.index, height = indU[peR])
textstr = "mean = " + str(mean) + "\n" + "median = " + str(median)

# these are matplotlib.patch.Patch properties

# place a text box in upper left in axes coords
props = dict(boxstyle='round', facecolor='white', alpha=0.5)
ax.text(0.05, 0.95, textstr, transform=ax.transAxes, 
        bbox = props, fontsize= 20, verticalalignment='top')

plt.xticks(rotation=90)
plt.grid(b=None)
plt.show()


def regress(cols, dep, data1 = data1, sd = False, log = False, orig = False, dumCols = dumCols):

    regD = data1[cols].dropna(axis = 0, how = "any")
    if sd == True:
        for col in regD.columns:
            if col not in dumCols:
                std = np.std(regD[col])
                mean = np.mean(regD[col])
                regD = regD.loc[(regD[col] > mean - std * 3) & 
                                  (data1[col] < mean + std * 3)]
        
#    print(len(data1))

#    sns.pairplot(data1)
    indeps = [x for x in regD.columns if x != dep]
    
    if log == True :
        for col in indeps:
            if col not in dumCols:
                regD[col] = np.log(regD[col])
    
    dumCols = [s for s in regD.columns if s in dumCols]
    
    for col in regD[dumCols]:
            if np.sum(regD[col]) == 0:
                regD = regD.drop([col], axis = 1)
    
    
    regD = regD.replace([np.inf, -np.inf], np.nan)
    regD = regD.dropna(axis = 0, how = "any")
    indeps = [x for x in regD.columns if x != dep]
    X = regD[indeps]
    y = regD[dep]

    if orig == False:
        X = sm.add_constant(X)
    
    regress = sm.OLS(y.astype(float), X.astype(float))
    result = regress.fit()

    
    return result


def regPre(cols, dep, data1 = data1, sd = False, log = False, orig = False, dumCols = dumCols):

    regD = data1[cols].dropna(axis = 0, how = "any")
    if sd == True:
        for col in regD.columns:
            if col not in dumCols:
                std = np.std(regD[col])
                mean = np.mean(regD[col])
                regD = regD.loc[(regD[col] > mean - std * 3) & 
                                  (data1[col] < mean + std * 3)]
        
#    print(len(data1))

#    sns.pairplot(data1)
    indeps = [x for x in regD.columns if x != dep]
    
    if log == True :
        for col in indeps:
            if col not in dumCols:
                regD[col] = np.log(regD[col])
    
    dumCols = [s for s in regD.columns if s in dumCols]
    
    for col in regD[dumCols]:
            if np.sum(regD[col]) == 0:
                regD = regD.drop([col], axis = 1)
    
    
    regD = regD.replace([np.inf, -np.inf], np.nan)
    regD = regD.dropna(axis = 0, how = "any")
    indeps = [x for x in regD.columns if x != dep]
    
    return regD

exp = []
for col in data1.columns:
    if col not in ["Industry", "Sector", "NoShare", peR] + dumCols:
        res = regress([col, peR], peR, sd = True, orig = True)
        if abs(res.tvalues[0]) > 2 and res.rsquared > 0.1:
            exp = exp + [col]
            print(res.tvalues[0], ":", col, ":", res.rsquared)


            

roa = "Return on Assets (ttm)"
testCols = [roa, "EPS growth", "beta20", poR, peR] + dumCols
preD = regPre(testCols, peR, log = True)
y = preD[peR]
X = preD[[col for col in preD.columns if col != peR]]
X = sm.add_constant(X)
res = sm.OLS(y.astype(float), X.astype(float)).fit()
print(res.summary())

corrD = regPre([roa, roeR], roeR)

newCol = [roa, "EPS growth", "beta20", poR, peR] + ["Medical Instruments & Supplies", 
         "Scientific & Technical Instruments", "Gambling"]
plt.show()
preD = regPre(newCol, peR, log = True)
sns.pairplot(preD[[roa, "EPS growth", "beta20", poR, peR]])
y = preD[peR]
X = preD[[col for col in preD.columns if col != peR]]
X = sm.add_constant(X)
res = sm.OLS(y.astype(float), X.astype(float)).fit()
print(res.summary())

yPre = res.predict()



def rScore(y,X):
    X = sm.add_constant(X)
    lr = sm.OLS(y, X.values).fit()
    return lr.rsquared

def feature_selection(y, X):
    bestScore = 0
    bestCols = []
    for col in X.columns:
        if rScore(y, X[col]) > bestScore:
            bestScore = rScore(y, X[col])
            bestCol = col
    return bestScore, bestCol

def adjR(y,X):
    X = sm.add_constant(X)
    lr = sm.OLS(y, X.values).fit()
    return lr.rsquared_adj

def recurse(y, bestCols, X):
    X_s = X.drop(bestCols, axis = 1)
    bestScore = 0
    for col in X_s:
        tempCols = bestCols + [col]
        X_temp = X[tempCols]
        if rScore(y, X_temp) > bestScore:
            bestScore = rScore(y, X_temp)
            bestCol = col
    bestCols = bestCols + [bestCol]
    return bestScore, bestCols

def forward_selection(y, X):
    bestScore, bestCol = feature_selection(y, X)
    bestCols = [bestCol]
    tolCols = bestCols
    for i in range(len(X.columns) - 1): 
        bestScore, bestCols = recurse(y, bestCols, X)
        tolCols = tolCols + [bestCols]
    bestScore = 0
    sol = []
    for cols in tolCols:
        if rScore(y, X[cols]) > bestScore + 0.005: 
            sol = cols
            bestScore = adjR(y, X[cols])
#            print(bestScore)
    return bestScore, sol








fig, ax = plt.subplots(figsize = (20,10))
plt.bar(x = indU.index, height = indU[peR])
plt.xticks(rotation=90)
plt.show()




###too much noise

for col in data2.columns:
    if col not in ["Industry", "Sector"]:
        std = np.std(data2[col])
        mean = np.mean(data2[col])
        data2 = data2.loc[(data2[col] > mean - 3 * std) & 
             (data2[col] < mean + 3 * std)]

plt.scatter(data1[beta], data1[peR])

sns.pairplot(data1[corCols].dropna(axis = 0, how = "any"))



corCols = [s for s in data1.columns if s not in ["Industry", "Sector"]]
colD = data1[corCols]
colD = colD.apply(pd.to_numeric)
corr = colD.corr()
sns.heatmap(corr)


corr2 = data2[corCols].apply(pd.to_numeric).corr()
sns.heatmap(corr2)


sd3 = True
log = True
orig = False
dep = peR
cols = [peR, poR]




def regress(cols, dep, data1 = data1, sd = False, log = False, orig = False):
    data1 = data1[cols].dropna(axis = 0, how = "any")
    if sd == True:
        for col in cols:
            std = np.std(data1[col])
            mean = np.mean(data1[col])
            data1 = data1.loc[(data1[col] > mean - std * 3) & 
                              (data1[col] < mean + std * 3)]
    
#    print(len(data1))
    indeps = [x for x in cols if x != dep]
#    sns.pairplot(data1)
    
    if log == True:
        for col in indeps:
            try:
                dataT[col] = np.log(dataT[col])
            except:
                pass
        
    y = data1[dep]
    X = data1[indeps]
    
    if orig == False:
        X = sm.add_constant(X)

    regress = sm.OLS(y.astype(float), X.astype(float))
    result = regress.fit()

    return result



#sns.pairplot(data1[cols].dropna(axis = 0, how = "any"))
testCols = [peR, "Market Cap"] 
res = regress(testCols, dep = peR, sd = True, log = True)
tVal = res.tvalues[1]


checkCols = []
numerCols = [x for x in corCols if x != peR]
for col in numerCols:
    testCols = [peR, col] 
    res = regress(testCols, dep = peR, sd = True, log = True)
    tVal = res.tvalues[1]   
    if tVal > 2:
        checkCols = checkCols + [col]




pld = data1.loc[(data1[peR].isna() != True) & 
                (data1[roeR].isna() != True), [peR, roeR]]
plt.scatter(pld[roeR], pld[peR])

plt.figure(figsize=(10, 5))
plt.show()
plt.hist(data1[peR], bins = 200)
