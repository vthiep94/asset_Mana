#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 28 09:11:18 2020

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

tickers = pd.read_csv("tickers.csv", header = None)[0].to_list()

#
#dax = "https://finance.yahoo.com/quote/%5EGDAXI/components?p=%5EGDAXI"
#mdax = "https://finance.yahoo.com/quote/%5EMDAXI/components?p=%5EMDAXI"
#sdax = "https://finance.yahoo.com/quote/%5ESDAXI/components?p=%5ESDAXI"

def loadPage(site):
    max_retries = 10# some int
    retry_delay = 10# some int
    n = 1
    ready = 0
    while n < max_retries:
      try:
         response = requests.get(site)
         if response.ok:
            ready = 1
            break
      except requests.exceptions.RequestException:
         print("Website not availabe...")
      n += 1
      time.sleep(retry_delay)
    
    if ready != 1:
      print("Problem")
    else:
      print("All good")
    return response


#def get_tickers(index):
#    page = requests.get(index)
#    soup = BeautifulSoup(page.content, 'html.parser')
#    tickers = []
#    main = soup.find_all("tr", class_= "BdT Bdc($seperatorColor) Ta(end) Fz(s)")
#    for x in main:
#        ticker = x.find("a").get_text()
#        tickers = tickers + [ticker]
#    
#    return tickers

#daxTick = get_tickers(dax)
#mdaxTick = get_tickers(mdax)
#sdaxTick = get_tickers(sdax)
#allTicks = daxTick + mdaxTick + sdaxTick

stock = "BOSS.DE"
summary = "https://finance.yahoo.com/quote/" + stock + "?p=" + stock
page1 = requests.get(summary)
soup1 = BeautifulSoup(page1.content, "html.parser")
feats = []
vals = []
main1 = soup1.find("table", class_ = "W(100%) M(0) Bdcl(c)")
main2 = main1.find("tbody")
elements = main2.find_all("tr")
for x in elements[:4]:
    spans = x.find_all("span")      
    feat = spans[0].get_text()
    val = spans[1].get_text()
    vals = vals + [val]
    feats = feats + [feat]

keystats = "https://finance.yahoo.com/quote/" + stock + "/key-statistics?p=" + stock
page2 = requests.get(keystats)
#print(page2)
soup2 = BeautifulSoup(page2.content, "html.parser")
elements1 = soup2.find_all("table", class_ = "W(100%) Bdcl(c)")
feats1 = []
vals1 = []
for x in elements1:
    trs = x.find_all("tr")
    for y in trs:
        tds = y.find_all("td")
        vals1 = vals1 + [tds[1].get_text()]
        feats1 = feats1 + [tds[0].get_text()]

sectors = []
industries = []

url = "https://finance.yahoo.com/quote/" + stock + "/profile?p=" + stock
page3 = loadPage(url)
soup3 = BeautifulSoup(page3.content, "html.parser")
quals = ["Sector", "Industry", "No. Employees"]
vals2 = []
try:
    main = soup3.find("div", class_ = "asset-profile-container")
    main1 = main.find("p", class_ = "D(ib) Va(t)")
    main2 = main1.find_all("span", class_ = "Fw(600)")
    for x in main2: 
        vals2 = vals2 + [(x.get_text())]
except:
    vals2 = [np.nan] * 3
    pass
    
    

allVals = vals + vals1 + vals2
indexDF = feats + feats1 + quals


crits = indexDF[:4] + ["Payout Ratio 4", "Return on Equity (ttm)",
                  "EBITDA ", "Return on Assets (ttm)", 
                  "Total Debt (mrq)", "Operating Margin (ttm)", 
                  "Net Income Avi to Common (ttm)"] + quals



stock = "BOSS.DE"
summary = "https://finance.yahoo.com/quote/" + stock + "?p=" + stock
page1 = loadPage(summary)
soup1 = BeautifulSoup(page1.content, "html.parser")
feats = []
vals = []
try:
    main1 = soup1.find("table", class_ = "W(100%) M(0) Bdcl(c)")
    elements = main1.find_all("tr")
    for x in elements[:4]:
        spans = x.find_all("td")      
        feat = spans[0].get_text()
        val = spans[1].get_text()
        vals = vals + [val]
        feats = feats + [feat]
except:
    vals = [np.nan] * 4
    print(stock, " Element")
    if stock not in lostTicks:
        lostTicks = lostTicks + [stock]
if len(vals) != 4:
    print(stock, " vals problem")



keystats = "https://finance.yahoo.com/quote/" + stock + "/key-statistics?p=" + stock
page2 = loadPage(keystats)
#print(page2)
soup2 = BeautifulSoup(page2.content, "html.parser")
feats1 = []
vals1 = []
try:
    test1 = soup2.find("div", class_ = "Mstart(a) Mend(a)")
    elements1 = test1.find_all("table")
    for x in elements1:
        trs = x.find_all("tr")
        for y in trs:
            tds = y.find_all("td")
            if tds != []:
                if tds[0].get_text() in crits:
                    vals1 = vals1 + [tds[1].get_text()]
                    feats1 = feats1 + [tds[0].get_text()]

except:
    vals1 = [np.nan] * (len(crits) - len(vals) - len(quals))
    print(stock, " Element 2")
    if stock not in lostTicks:
        lostTicks = lostTicks + [stock]
if len(vals1) != 7:
    print(stock, " ", len(vals1))
        
url = "https://finance.yahoo.com/quote/" + stock + "/profile?p=" + stock
page3 = loadPage(url)
soup3 = BeautifulSoup(page3.content, "html.parser")
quals = ["Sector", "Industry", "No. Employees"]
vals2 = []
try:
    main = soup3.find("div", class_ = "asset-profile-container")
    main1 = main.find("p", class_ = "D(ib) Va(t)")
    main2 = main1.find_all("span", class_ = "Fw(600)")
    for x in main2: 
        vals2 = vals2 + [(x.get_text())]

except:
    vals2 = [np.nan] * 3

if len(vals2) != 3:
    print(stock, " vals2 problem")

allVals = vals + vals1 + vals2
indexDF1 = feats + feats1 + quals



def getData(tickers, crits = crits, quals = quals):
    dicts = {}
    lostTicks = []
    for stock in tickers:
        summary = "https://finance.yahoo.com/quote/" + stock + "?p=" + stock
        page1 = loadPage(summary)
        soup1 = BeautifulSoup(page1.content, "html.parser")
        feats = []
        vals = []
        try:
            main1 = soup1.find("table", class_ = "W(100%) M(0) Bdcl(c)")
            elements = main1.find_all("tr")
            for x in elements[:4]:
                spans = x.find_all("td")      
                feat = spans[0].get_text()
                val = spans[1].get_text()
                vals = vals + [val]
        except:
            vals = [np.nan] * 4
            print(stock, " Element")
            if stock not in lostTicks:
                lostTicks = lostTicks + [stock]
        if len(vals) != 4:
            print(stock, " vals problem")
    
        flag = True
        i = 0
        while (i < 10):
            keystats = "https://finance.yahoo.com/quote/" + stock + "/key-statistics?p=" + stock
            page2 = loadPage(keystats)
            #print(page2)
            soup2 = BeautifulSoup(page2.content, "html.parser")
            feats1 = []
            vals1 = []
            try:
                test1 = soup2.find("div", class_ = "Mstart(a) Mend(a)")
                elements1 = test1.find_all("table")
                for x in elements1:
                    trs = x.find_all("tr")
                    for y in trs:
                        tds = y.find_all("td")
                        if tds != []:
                            if tds[0].get_text() in crits and tds[0].get_text() != "Beta (5Y Monthly)":
                                vals1 = vals1 + [tds[1].get_text()]
                                feats1 = feats1 + [tds[0].get_text()]
    
            except:
                vals1 = [np.nan] * (len(crits) - len(vals) - len(quals))
                print(stock, " Element 2")
                if stock not in lostTicks:
                    lostTicks = lostTicks + [stock]
            if len(vals1) != 7:
                flag = True
            else:
                flag = False
                break
            i = i + 1
        
        if len(vals1) != 7:
            print("fuck this", stock)
        
        
        
        url = "https://finance.yahoo.com/quote/" + stock + "/profile?p=" + stock
        page3 = loadPage(url)
        soup3 = BeautifulSoup(page3.content, "html.parser")
        quals = ["Sector", "Industry", "No. Employees"]
        vals2 = []
        try:
            main = soup3.find("div", class_ = "asset-profile-container")
            main1 = main.find("p", class_ = "D(ib) Va(t)")
            main2 = main1.find_all("span", class_ = "Fw(600)")
            for x in main2: 
                vals2 = vals2 + [(x.get_text())]
        
        except:
            vals2 = [np.nan] * 3
        
        if len(vals2) != 3:
            print(stock, " vals2 problem")
    
        
                
        allVals = vals + vals1 + vals2
        dicts[stock] = allVals      
    
    
    
    return dicts, lostTicks

data, lostTicks = getData(tickers)
data1, losTicks1 = getData(lostTicks)


dataDF = pd.DataFrame(data)
data1DF = pd.DataFrame(data1)
for col in data1DF.columns:
    if col in dataDF.columns:
        dataDF = dataDF.drop(col, axis = 1)
        dataDF[col] = data1DF[col]

dataDF.index = indexDF1
alldata1 = pd.DataFrame(dataDF)
alldata1.to_csv("allRawData.csv")

"""Scraping sector and industry data"""
stock = "BOSS.DE"
summary = "https://finance.yahoo.com/quote/" + stock + "?p=" + stock
page1 = requests.get(summary)
soup1 = BeautifulSoup(page1.content, "html.parser")
feats = []
vals = []
main1 = soup1.find("table", class_ = "W(100%) M(0) Bdcl(c)")
main2 = main1.find("tbody")
elements = main2.find_all("tr")
for x in elements[:4]:
    spans = x.find_all("span")      
    feat = spans[0].get_text()
    val = spans[1].get_text()
    vals = vals + [val]
    feats = feats + [feat]

keystats = "https://finance.yahoo.com/quote/" + stock + "/key-statistics?p=" + stock
page2 = requests.get(keystats)
#print(page2)
soup2 = BeautifulSoup(page2.content, "html.parser")
elements1 = soup2.find_all("table", class_ = "W(100%) Bdcl(c)")
feats1 = []
vals1 = []
for x in elements1:
    trs = x.find_all("tr")
    for y in trs:
        tds = y.find_all("td")
        vals1 = vals1 + [tds[1].get_text()]
        feats1 = feats1 + [tds[0].get_text()]

sectors = []
industries = []

url = "https://finance.yahoo.com/quote/" + stock + "/profile?p=" + stock
page3 = loadPage(url)
soup3 = BeautifulSoup(page3.content, "html.parser")
quals = ["Sector", "Industry", "No. Employees"]
vals2 = []
try:
    main = soup3.find("div", class_ = "asset-profile-container")
    main1 = main.find("p", class_ = "D(ib) Va(t)")
    main2 = main1.find_all("span", class_ = "Fw(600)")
    for x in main2: 
        vals2 = vals2 + [(x.get_text())]
except:
    vals2 = [np.nan] * 3
    pass
    
    

allVals = vals + vals1 + vals2
indexDF = feats + feats1 + quals


crits = ["Total Cash (mrq)", "Revenue (ttm)"]

dicts = {}
tickers = tickers
lostTicks = []
for stock in tickers:
    
    keystats = "https://finance.yahoo.com/quote/" + stock + "/key-statistics?p=" + stock
    page2 = loadPage(keystats)
    #print(page2)
    soup2 = BeautifulSoup(page2.content, "html.parser")
    feats1 = []
    vals1 = []
    try:
        test1 = soup2.find("div", class_ = "Mstart(a) Mend(a)")
        elements1 = test1.find_all("table")
        for x in elements1:
            trs = x.find_all("tr")
            for y in trs:
                tds = y.find_all("td")
                if tds != []:
                    if tds[0].get_text() in crits:
                        vals1 = vals1 + [tds[1].get_text()]
                        feats1 = feats1 + [tds[0].get_text()]
    
    except:
        vals1 = [np.nan] * len(crits)
        print(stock, " Element 2")
        if stock not in lostTicks:
            lostTicks = lostTicks + [stock]
    dicts[stock] = vals1


extra = pd.DataFrame(dicts)
extra.to_csv("extra1.csv")


"""Scraping other financial data"""

crits = ["Interest Expense"]
dicts = {}
for stock in tickers:
    fin = "https://finance.yahoo.com/quote/" + stock + "/financials?p=" + stock
    page1 = loadPage(fin)
    soup1 = BeautifulSoup(page1.content, "html.parser")
    
    try:
        main1 = soup1.find("div", id = "Col1-1-Financials-Proxy")
        div1 = main1.find("div", class_ = "D(tbrg)")
        divs = div1.find_all("div", class_ = "D(tbr) fi-row Bgc($hoverBgColor):h")
        
        vals = []
        feats = []
        for div in divs:
            spans = div.find_all("span")
            feat = spans[0].get_text()
            if feat in crits:
                vals = vals + [spans[1].get_text()]
    except:
        vals = [np.nan] * len(crits)
    dicts[stock] = vals

for key in dicts.keys():
    if len(dicts[key]) == 0:
        dicts[key] = [np.nan] * len(crits)
inter = pd.DataFrame(dicts)
inter.to_csv("extra2.csv")


sDivs = divs[0].find_all("span")
for x in sDivs:
    print(x.get_text())
        

main2 = main1.find("tbody")
elements = main2.find_all("tr")
for x in elements[:4]:
    spans = x.find_all("span")      
    feat = spans[0].get_text()
    val = spans[1].get_text()
    vals = vals + [val]
    feats = feats + [feat]

keystats = "https://finance.yahoo.com/quote/" + stock + "/key-statistics?p=" + stock
page2 = requests.get(keystats)
#print(page2)
soup2 = BeautifulSoup(page2.content, "html.parser")
elements1 = soup2.find_all("table", class_ = "W(100%) Bdcl(c)")
feats1 = []
vals1 = []
for x in elements1:
    trs = x.find_all("tr")
    for y in trs:
        tds = y.find_all("td")
        vals1 = vals1 + [tds[1].get_text()]
        feats1 = feats1 + [tds[0].get_text()]

sectors = []
industries = []

url = "https://finance.yahoo.com/quote/" + stock + "/profile?p=" + stock
page3 = loadPage(url)
soup3 = BeautifulSoup(page3.content, "html.parser")
quals = ["Sector", "Industry", "No. Employees"]
vals2 = []
try:
    main = soup3.find("div", class_ = "asset-profile-container")
    main1 = main.find("p", class_ = "D(ib) Va(t)")
    main2 = main1.find_all("span", class_ = "Fw(600)")
    for x in main2: 
        vals2 = vals2 + [(x.get_text())]
except:
    vals2 = [np.nan] * 3
    pass
    
    
allVals = vals + vals1 + vals2
indexDF = feats + feats1 + quals



dicts = {}
tickers = tickers
lostTicks = []
for stock in tickers:
    
    fin = "https://finance.yahoo.com/quote/" + stock + "/financials=" + stock
    page2 = loadPage(fin)
    #print(page2)
    soup2 = BeautifulSoup(page2.content, "html.parser")
    feats1 = []
    vals1 = []
    try:
        test1 = soup2.find("div", class_ = "Mstart(a) Mend(a)")
        elements1 = test1.find_all("table")
        for x in elements1:
            trs = x.find_all("tr")
            for y in trs:
                tds = y.find_all("td")
                if tds != []:
                    if tds[0].get_text() in crits:
                        vals1 = vals1 + [tds[1].get_text()]
                        feats1 = feats1 + [tds[0].get_text()]
    
    except:
        vals1 = [np.nan] * len(crits)
        print(stock, " Element 2")
        if stock not in lostTicks:
            lostTicks = lostTicks + [stock]
    dicts[stock] = vals1


extra = pd.DataFrame(dicts)
extra.to_csv("extra1.csv")



vals = []
dicts = {}
for stock in tickers:
    try:
        anl = "https://finance.yahoo.com/quote/" + stock + "/analysis?=" + stock
        feat = "Next 5 Years (per annum)"
        page = loadPage(anl)
        soup = BeautifulSoup(page.content, "html.parser")
        main = soup.find("div", id = "Col1-0-AnalystLeafPage-Proxy")
    #    print(main.prettify())
        tables = main.find_all("table")
        tbody = tables[-1].find("tbody")
        trs = tbody.find_all("tr", class_ = "BdT Bdc($seperatorColor)")
        for tr in trs:
            tds = tr.find_all("td")
            if tds[0].get_text() == feat:
                print(tds[0].get_text())
                val = tds[1].get_text()
                print(val)
                vals = vals + [val]
                dicts[stock] = val
    except:
        pass

growth = pd.DataFrame(dicts, index = ["EPS growth"])
growth.to_csv("growth.csv")

#trs = tar.find_all("tr")


