import sched, time
from bs4 import BeautifulSoup
import time, random, os, csv, platform
import logging
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import pandas as pd
from datetime import datetime, timedelta

options = webdriver.ChromeOptions()
options.add_argument("--start-maximized")

test_date = datetime.now() + timedelta(days=7)

date_to_string = str(test_date.strftime("%d"))

string_strip = date_to_string.lstrip("0")

#print(string_strip)

driver = webdriver.Chrome(executable_path=r'C:\Users\Michael\Desktop\Python\Stonks\earnings scrape\chromedriver.exe', chrome_options=options)

start_string = 'https://www.earningswhispers.com/calendar?sb=c&d=' + string_strip + '&t=all'

driver.get(start_string)

time.sleep(5)

page_source = driver.page_source
soup = BeautifulSoup(page_source, "html5lib")

ticker_list = soup.find_all("div", {"class": "ticker"})

rev_list = soup.find_all("div", {"class": "revgrowthprint"})

all_tickers = [ticker.text for ticker in ticker_list]

#print(*rev_list)

cleaned_list = []

for rev in rev_list:
    rev_text = rev.text.rstrip("%")
    rev_text.strip("-")
    rev_text.rstrip(" ")

    print(rev_text)
    try:
        float_conv = float(rev_text)
        cleaned_list.append(float_conv)
    except ValueError:
        print('err')
        


earnings_df = pd.DataFrame(zip(all_tickers, cleaned_list), columns = ['Ticker', 'Earnings %'])

positive_filter = earnings_df[earnings_df['Earnings %'] > 0]

#print(positive_filter)

import smtplib
from email.message import EmailMessage
from string import Template
from pathlib import Path 

email = EmailMessage()
email['from'] = 'Stonk Bot'
email['to'] = ['newlin.michael@gmail.com']

#subject_text = date_to_string

email['subject'] = 'Earnings Whisper for: ' + date_to_string

email.set_content(positive_filter.to_string(index = False, col_space = 10, justify = 'left'))

with smtplib.SMTP(host='smtp.gmail.com', port=587) as smtp:
    smtp.ehlo()
    smtp.starttls()
    smtp.login('newlin.michael@gmail.com', 'ugpvuwacvbipfgwt')
    smtp.send_message(email)
    #print('all good boss!')

driver.close()
