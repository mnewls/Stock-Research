import ta_screen_0_9 as ta_long
import time
import pandas as pd
import csv

import bs4 as bs
import pickle
import requests

df = pd.DataFrame()
#this_list = ta_long.TA_screening('BRK-B')

'''stonk = 'MMM'
this_list = ta_long.TA_screening(stonk)

df[stonk] = pd.Series(this_list)'''

#df.to_csv('tickers_and_indicators.csv')

def save_sp500_tickers():
    resp = requests.get('http://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    soup = bs.BeautifulSoup(resp.text, 'lxml')
    table = soup.find('table', {'class': 'wikitable sortable'})
    tickers = []
    for row in table.findAll('tr')[1:]:
        ticker = row.findAll('td')[0].text

        ticker = ticker.replace('\n','')
        ticker = ticker.replace('.','-')

        tickers.append(ticker)
        
    with open("sp500tickers.pickle","wb") as f:
        pickle.dump(tickers,f)
        
    return tickers

def save_dow_tickers():
    resp = requests.get('https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average')
    soup = bs.BeautifulSoup(resp.text, 'lxml')
    table = soup.find('table', {'class': 'wikitable sortable'})
    tickers = []
    for row in table.findAll('tr')[1:]:
        ticker = row.findAll('td')[1].text

        ticker = ticker.replace('\n','')

        tickers.append(ticker)
        
    with open("DJIA.pickle","wb") as f:
        pickle.dump(tickers,f)
        
    return tickers

def save_nasdaq_tickers():
    resp = requests.get('https://en.wikipedia.org/wiki/NASDAQ-100#Components')
    soup = bs.BeautifulSoup(resp.text, 'lxml')
    table = soup.find('table', {'id': 'constituents'})
    tickers = []
    for row in table.findAll('tr')[1:]:
        ticker = row.findAll('td')[1].text

        ticker = ticker.replace('\n','')

        tickers.append(ticker)
        
    with open("NDAQ.pickle","wb") as f:
        pickle.dump(tickers,f)
        
    return tickers

stocklist_SP = save_sp500_tickers()
print(stocklist_SP)

stocklist_DOW = save_dow_tickers()
print(stocklist_DOW)

stocklist_NDAQ = save_nasdaq_tickers()
print(stocklist_NDAQ)

tic = time.perf_counter()

for stock in stocklist_SP:

    this_list = ta_long.TA_screening(stock)

    df[stock] = pd.Series(this_list)

for stock in stocklist_DOW:

    this_list = ta_long.TA_screening(stock)

    df[stock] = pd.Series(this_list)

for stock in stocklist_NDAQ:
    this_list = ta_long.TA_screening(stock)

    df[stock] = pd.Series(this_list)

df.to_csv('tickers_and_indicators.csv')

toc = time.perf_counter()

tic_toc = (toc - tic) / 60

print(f"completed Pred in {tic_toc:0.4f} min")