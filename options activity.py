from pandas_datareader import data as pdr
from yahoo_fin import stock_info as si

import yfinance as yf
import pandas as pd
import requests
import datetime
import time
import talib

from selenium.webdriver.support.ui import Select

from selenium import webdriver
from bs4 import BeautifulSoup
import pyautogui

driver = webdriver.Chrome(executable_path=r'C:\Users\Michael\Desktop\Python\Automate Application\chromedriver.exe')

# Grabbing first list for unusual options activity
# first list ranked by Open INT

driver.get("https://www.barchart.com/options/unusual-activity/stocks?orderBy=openInterest&orderDir=desc")

page_source = driver.page_source
soup = BeautifulSoup(page_source, "html5lib")
time.sleep(5)

#select_OI = driver.find_element_by_xpath("/html/body/main/div/div[2]/div[2]/div/div[2]/div/div/div/div[7]/div/div[2]/div/div/ng-transclude/table/thead/tr/th[12]")
#driver.execute_script("arguments[0].click();", select_OI)

time.sleep(5)
ticker_list = []

for i in range(100):
    i+=1

    path = "/html/body/main/div/div[2]/div[2]/div/div[2]/div/div/div/div[7]/div/div[2]/div/div/ng-transclude/table/tbody/tr[" + str(i) + "]/td[1]/div/span[2]/a"

    ticker = driver.find_element_by_xpath(path).text[:]
    ticker_list.append(ticker)

## testing if this works

#print(*ticker_list)

# repeating for second chart from same site www.barchart.com

driver.get("https://www.barchart.com/options/iv-rank-percentile/stocks?viewName=main&orderBy=optionsIVRank1y&orderDir=desc")

page_source = driver.page_source
soup = BeautifulSoup(page_source, "html5lib")
time.sleep(5)

#select_OI = driver.find_element_by_xpath("/html/body/main/div/div[2]/div[2]/div/div[2]/div/div/div/div[7]/div/div[2]/div/div/ng-transclude/table/thead/tr/th[12]")
#driver.execute_script("arguments[0].click();", select_OI)

time.sleep(5)
ticker_list_2 = []

for i in range(100):
    i+=1
    #/html/body/main/div/div[2]/div[2]/div/div[2]/div/div/div/div[5]/div/div[2]/div/div/ng-transclude/table/tbody/tr[1]/td[1]/div/span[2]/a
    #/html/body/main/div/div[2]/div[2]/div/div[2]/div/div/div/div[5]/div/div[2]/div/div/ng-transclude/table/tbody/tr[2]/td[1]/div/span[2]/a
    path = "/html/body/main/div/div[2]/div[2]/div/div[2]/div/div/div/div[5]/div/div[2]/div/div/ng-transclude/table/tbody/tr[" + str(i) + "]/td[1]/div/span[2]/a"

    ticker = driver.find_element_by_xpath(path).text[:]
    ticker_list_2.append(ticker)

fin_list = set(ticker_list) & set(ticker_list_2)

print(fin_list)