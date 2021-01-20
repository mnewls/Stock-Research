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

# Grabbing first list from WSB
def unbiased_stock_scrape():

    driver.get("http://unbiastock.com/reddit.php")

    #page_source = driver.page_source
    #soup = BeautifulSoup(page_source, "html5lib")
    time.sleep(5)

    select_source = Select(driver.find_element_by_xpath("/html/body/div/div/div/div/div[2]/div/form/select[1]"))
    select_source.select_by_visible_text("Wall Street Bets")

    select_time = Select(driver.find_element_by_xpath("/html/body/div/div/div/div/div[2]/div/form/select[2]"))
    select_time.select_by_visible_text("Last 7 Days")

    click_but = driver.find_element_by_xpath("/html/body/div/div/div/div/div[2]/div/form/input")
    click_but.click()

    time.sleep(5)
    ticker_list = []

    for i in range(25):
        i+=1

        path = "/html/body/div/div/div/div/div[2]/div/div[2]/div/table/tbody/tr[" + str(i) + "]/th[1]"

        ticker = driver.find_element_by_xpath(path).text[:]
        ticker_list.append(ticker)

    ## Resetting for Round 2

    driver.get("http://unbiastock.com/reddit.php")

    #page_source = driver.page_source
    time.sleep(5)


    select_source = Select(driver.find_element_by_xpath("/html/body/div/div/div/div/div[2]/div/form/select[1]"))
    select_source.select_by_visible_text("Day Trading")

    select_time = Select(driver.find_element_by_xpath("/html/body/div/div/div/div/div[2]/div/form/select[2]"))
    select_time.select_by_visible_text("Last 7 Days")

    click_but = driver.find_element_by_xpath("/html/body/div/div/div/div/div[2]/div/form/input")
    click_but.click()

    time.sleep(5)
    ticker_list_2 = []

    for i in range(25):
        i+=1

        path = "/html/body/div/div/div/div/div[2]/div/div[2]/div/table/tbody/tr[" + str(i) + "]/th[1]"

        ticker = driver.find_element_by_xpath(path).text[:]
        ticker_list_2.append(ticker)

    # list 3

    driver.get("http://unbiastock.com/reddit.php")

    #page_source = driver.page_source
    time.sleep(5)


    select_source = Select(driver.find_element_by_xpath("/html/body/div/div/div/div/div[2]/div/form/select[1]"))
    select_source.select_by_visible_text("Stock Market")

    select_time = Select(driver.find_element_by_xpath("/html/body/div/div/div/div/div[2]/div/form/select[2]"))
    select_time.select_by_visible_text("Last 7 Days")

    click_but = driver.find_element_by_xpath("/html/body/div/div/div/div/div[2]/div/form/input")
    click_but.click()

    time.sleep(5)
    ticker_list_3 = []

    for i in range(25):
        i+=1

        path = "/html/body/div/div/div/div/div[2]/div/div[2]/div/table/tbody/tr[" + str(i) + "]/th[1]"

        ticker = driver.find_element_by_xpath(path).text[:]
        ticker_list_3.append(ticker)

    # list 4

    driver.get("http://unbiastock.com/reddit.php")

    #page_source = driver.page_source
    #soup = BeautifulSoup(page_source, "html5lib")
    time.sleep(5)


    select_source = Select(driver.find_element_by_xpath("/html/body/div/div/div/div/div[2]/div/form/select[1]"))
    select_source.select_by_visible_text("Stocks")

    select_time = Select(driver.find_element_by_xpath("/html/body/div/div/div/div/div[2]/div/form/select[2]"))
    select_time.select_by_visible_text("Last 7 Days")

    click_but = driver.find_element_by_xpath("/html/body/div/div/div/div/div[2]/div/form/input")
    click_but.click()

    time.sleep(5)
    ticker_list_4 = []

    for i in range(25):
        i+=1

        path = "/html/body/div/div/div/div/div[2]/div/div[2]/div/table/tbody/tr[" + str(i) + "]/th[1]"

        ticker = driver.find_element_by_xpath(path).text[:]
        ticker_list_4.append(ticker)

    #list 5
    driver.get("http://unbiastock.com/reddit.php")

    #page_source = driver.page_source
    time.sleep(5)


    select_source = Select(driver.find_element_by_xpath("/html/body/div/div/div/div/div[2]/div/form/select[1]"))
    select_source.select_by_visible_text("Investing")

    select_time = Select(driver.find_element_by_xpath("/html/body/div/div/div/div/div[2]/div/form/select[2]"))
    select_time.select_by_visible_text("Last 7 Days")

    click_but = driver.find_element_by_xpath("/html/body/div/div/div/div/div[2]/div/form/input")
    click_but.click()

    time.sleep(5)
    ticker_list_5 = []

    for i in range(25):
        i+=1

        path = "/html/body/div/div/div/div/div[2]/div/div[2]/div/table/tbody/tr[" + str(i) + "]/th[1]"

        ticker = driver.find_element_by_xpath(path).text[:]
        ticker_list_5.append(ticker)

    fin_list = set(ticker_list) & set(ticker_list_2) & set(ticker_list_3) & set(ticker_list_4) & set(ticker_list_5)

    return fin_list

def bar_chart_stock_scrape():

    driver.get("https://www.barchart.com/options/unusual-activity/stocks?orderBy=openInterest&orderDir=desc")

    #page_source = driver.page_source
    time.sleep(5)

    #select_OI = driver.find_element_by_xpath("/html/body/main/div/div[2]/div[2]/div/div[2]/div/div/div/div[7]/div/div[2]/div/div/ng-transclude/table/thead/tr/th[12]")
    #driver.execute_script("arguments[0].click();", select_OI)

    time.sleep(5)
    ticker_list_6 = []

    for i in range(100):
        i+=1

        path = "/html/body/main/div/div[2]/div[2]/div/div[2]/div/div/div/div[7]/div/div[2]/div/div/ng-transclude/table/tbody/tr[" + str(i) + "]/td[1]/div/span[2]/a"

        ticker = driver.find_element_by_xpath(path).text[:]
        ticker_list_6.append(ticker)

    ## testing if this works

    #print(*ticker_list)

    # repeating for second chart from same site www.barchart.com

    driver.get("https://www.barchart.com/options/iv-rank-percentile/stocks?viewName=main&orderBy=optionsIVRank1y&orderDir=desc")

    #page_source = driver.page_source
    time.sleep(5)

    #select_OI = driver.find_element_by_xpath("/html/body/main/div/div[2]/div[2]/div/div[2]/div/div/div/div[7]/div/div[2]/div/div/ng-transclude/table/thead/tr/th[12]")
    #driver.execute_script("arguments[0].click();", select_OI)

    time.sleep(5)
    ticker_list_7 = []

    for i in range(100):
        i+=1
        #/html/body/main/div/div[2]/div[2]/div/div[2]/div/div/div/div[5]/div/div[2]/div/div/ng-transclude/table/tbody/tr[1]/td[1]/div/span[2]/a
        #/html/body/main/div/div[2]/div[2]/div/div[2]/div/div/div/div[5]/div/div[2]/div/div/ng-transclude/table/tbody/tr[2]/td[1]/div/span[2]/a
        path = "/html/body/main/div/div[2]/div[2]/div/div[2]/div/div/div/div[5]/div/div[2]/div/div/ng-transclude/table/tbody/tr[" + str(i) + "]/td[1]/div/span[2]/a"

        ticker = driver.find_element_by_xpath(path).text[:]
        ticker_list_7.append(ticker)

    fin_list_2 = set(ticker_list_6) & set(ticker_list_7)

    return fin_list_2

unbiased_ticker_list = unbiased_stock_scrape()
bar_chart_ticker_list = bar_chart_stock_scrape()

'''print(ticker_list)
print(ticker_list_2)
print(ticker_list_3)
print(ticker_list_4)
print(ticker_list_5)'''

fin_list_comb = set(unbiased_ticker_list) & set(bar_chart_ticker_list)

print(unbiased_ticker_list)
print(bar_chart_ticker_list)

#print(fin_list_comb)
#print(soup.prettify())

yf.pdr_override()

final = []
index = []
n = -1

for stock in unbiased_ticker_list:
    n += 1
    time.sleep(1)
    
    print ("\npulling {} with index {}".format(stock, n))

    # RS_Rating 
    start_date = datetime.datetime.now() - datetime.timedelta(days=59)
    end_date = datetime.date.today()
    
    df = pdr.get_data_yahoo(stock, start=start_date, end=end_date, interval = "30m")

    df.index = df.index.tz_localize(None)

    print(df.head(5))

    file_name = str(stock) + ".csv"

    df.to_csv(file_name)