from pandas_datareader import data as pdr
from yahoo_fin import stock_info as si
import yfinance as yf
import pandas as pd
import datetime
import time
import talib
from talib import *
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from xgboost import XGBClassifier
import xgboost as xgb
import math
import numpy as np
from sklearn.decomposition import PCA
import pickle

import os
os.environ["PATH"] += os.pathsep + r'C:\Program Files\Graphviz\bin'

yf.pdr_override()

def TA_screening(stock):

    #print(stocklist)

    #tic = time.perf_counter()
    index = []
    #start_date = datetime.datetime.now() - datetime.timedelta(days=59)

    #end_date = datetime.datetime.now()

    #df = pdr.get_data_yahoo(stock, start=start_date, end=end_date, interval = "2m", prepost=True)

    df = pdr.get_data_yahoo(stock, period = "max", interval = "1d", prepost = True)

    #df.index = df.index.tz_localize(None)

   #print(df.size)

    '''#2 min ticker
    # 30 intervals = 1 hour
    # 195 intervals = trading day''' # < old
                                                # there are more intervals that we can use / change

    #1 interval = 1 day

    really_fast = 5
    fast = 15
    slow = 30

    # these are the overlap studies

    def add_indicators():
        
        upper_band, mid_band, lower_band = BBANDS(df['Adj Close'],timeperiod=really_fast, nbdevup=2, nbdevdn=2, matype=0)
        d_ema = DEMA(df['Adj Close'], timeperiod=really_fast)
        E_M_A = EMA(df['Adj Close'], timeperiod=fast)
        ht_trend = HT_TRENDLINE(df['Adj Close'])
        kama = KAMA(df['Adj Close'], timeperiod=fast)
        ma = MA(df['Adj Close'], timeperiod=fast, matype=0)
        #mama, fama = MAMA(df['Adj Close'], fastlimit=really_fast, slowlimit=slow) < this gave me issues?
        #mavp = MAVP(df['Adj Close'])
        mid = MIDPOINT(df['Adj Close'], timeperiod=fast)
        mid_price = MIDPRICE(df['High'], df['Low'], timeperiod=fast)
        sar = SAR(df['High'], df['Low'], acceleration=.02, maximum=.2)
        sarext = SAREXT(df['High'], df['Low'], startvalue=0, offsetonreverse=0, accelerationinitlong=.02, accelerationlong=.02, accelerationmaxlong=.2, accelerationinitshort=.02, accelerationshort=.02, accelerationmaxshort=.2)
        sma = SMA(df['Adj Close'], timeperiod=slow)
        tema = TEMA(df['Adj Close'], timeperiod=slow)
        trima = TRIMA(df['Adj Close'], timeperiod=slow)
        wma = WMA(df['Adj Close'], timeperiod=slow)

        #this is some of the beginning stuff

        O_B_V = OBV(df['Adj Close'], df['Volume'])
        A_D_O_S_C = ADOSC(df['High'], df['Low'], df['Adj Close'], df['Volume'], fastperiod=fast, slowperiod=slow)
        O_G_chaikin = AD(df['High'], df['Low'], df['Adj Close'], df['Volume'])
        HT_DCper = HT_DCPERIOD(df['Adj Close'])
        HT_DCphase = HT_DCPHASE(df['Adj Close'])
        inphase, quad = HT_PHASOR(df['Adj Close'])
        r_sin, leadsin = HT_SINE(df['Adj Close'])

        #volatility
        atr = ATR(df['High'], df['Low'], df['Adj Close'], timeperiod=slow)
        natr = NATR(df['High'], df['Low'], df['Adj Close'], timeperiod=slow)
        t_range = TRANGE(df['High'], df['Low'], df['Adj Close'])

        #below here are momentum ind
        
        adx = ADX(df['High'], df['Low'], df['Adj Close'], timeperiod=fast)
        adxr = ADXR(df['High'], df['Low'], df['Adj Close'], timeperiod=fast)
        apo = APO(df['Adj Close'], fastperiod=really_fast, slowperiod=fast, matype=0)
        aroon_d, aroon_u = AROON(df['High'], df['Low'], timeperiod=fast)
        aroon_osc = AROONOSC(df['High'], df['Low'], timeperiod=fast)
        bop = BOP(df['Open'], df['High'], df['Low'], df['Adj Close'])
        cci = CCI(df['High'], df['Low'], df['Adj Close'], timeperiod=fast)
        cmo = CMO(df['Adj Close'], timeperiod=fast)
        dx = DX(df['High'], df['Low'], df['Adj Close'], timeperiod=fast)
        macd, macdsig, macdhist = MACD(df['Adj Close'], fastperiod=fast, slowperiod=slow, signalperiod=really_fast)
        macdex, macdexsig, macdexhist = MACDEXT(df['Adj Close'], fastperiod=fast, fastmatype=0, slowperiod=slow, slowmatype=0, signalperiod=really_fast, signalmatype=0)
        macdfixd, macdfixdsig, macdfixdhist = MACDFIX(df['Adj Close'], signalperiod=really_fast)
        # more momo's

        mfi = MFI(df['High'], df['Low'], df['Adj Close'],df['Volume'],timeperiod=fast)
        min_di = MINUS_DI(df['High'], df['Low'], df['Adj Close'], timeperiod=fast)
        min_dm = MINUS_DM(df['High'], df['Low'], timeperiod=fast)
        momo = MOM(df['Adj Close'], timeperiod=really_fast)
        plus_di = PLUS_DI(df['High'], df['Low'], df['Adj Close'], timeperiod=fast)
        plus_dm = PLUS_DM(df['High'], df['Low'], timeperiod=fast)
        ppo = PPO(df['Adj Close'], fastperiod=really_fast, slowperiod=fast, matype=0)
        roc = ROC(df['Adj Close'], timeperiod=fast)
        rocp = ROCP(df['Adj Close'], timeperiod=fast)
        rocr = ROCR(df['Adj Close'], timeperiod=fast)
        rocr_hund = ROCR100(df['Adj Close'], timeperiod = fast)
        rsi_fastk, rsi_fastd = STOCHRSI(df['Adj Close'], timeperiod=fast, fastk_period=slow, fastd_period=really_fast, fastd_matype=0)
        trix = TRIX(df['Adj Close'], timeperiod=slow)
        ult_osc = ULTOSC(df['High'], df['Low'], df['Adj Close'], timeperiod1=really_fast, timeperiod2=fast, timeperiod3=slow)


        #old some of the first added
        R_S_I = RSI(df['Adj Close'], timeperiod=slow)
        slowk, slowd = STOCH(df['High'], df['Low'], df['Adj Close'], fastk_period=fast, slowk_period=slow, slowk_matype=0, slowd_period=slow, slowd_matype=0)
        fastk, fastd = STOCHF(df['High'], df['Low'], df['Adj Close'], fastk_period=fast, fastd_period=really_fast, fastd_matype=0)

        real = WILLR(df['High'], df['Low'], df['Adj Close'], timeperiod=slow)

        # below are the TA indicators

        two_crows = CDL2CROWS(df['Open'], df['High'], df['Low'], df['Adj Close'])
        three_crows = CDL3BLACKCROWS(df['Open'], df['High'], df['Low'], df['Adj Close'])
        three_inside = CDL3INSIDE(df['Open'], df['High'], df['Low'], df['Adj Close'])
        three_line = CDL3LINESTRIKE(df['Open'], df['High'], df['Low'], df['Adj Close'])
        three_out = CDL3OUTSIDE(df['Open'], df['High'], df['Low'], df['Adj Close'])
        three_stars = CDL3STARSINSOUTH(df['Open'], df['High'], df['Low'], df['Adj Close'])
        three_soldier = CDL3WHITESOLDIERS(df['Open'], df['High'], df['Low'], df['Adj Close'])
        baby = CDLABANDONEDBABY(df['Open'], df['High'], df['Low'], df['Adj Close'], penetration=0)
        adv = CDLADVANCEBLOCK(df['Open'], df['High'], df['Low'], df['Adj Close'])
        belt_hold = CDLBELTHOLD(df['Open'], df['High'], df['Low'], df['Adj Close'])
        breakaway = CDLBREAKAWAY(df['Open'], df['High'], df['Low'], df['Adj Close'])
        closingmara = CDLCLOSINGMARUBOZU(df['Open'], df['High'], df['Low'], df['Adj Close'])
        baby_swallow = CDLCONCEALBABYSWALL(df['Open'], df['High'], df['Low'], df['Adj Close'])

        #more TA

        counter = CDLCOUNTERATTACK(df['Open'], df['High'], df['Low'], df['Adj Close'])
        dark_cloud = CDLDARKCLOUDCOVER(df['Open'], df['High'], df['Low'], df['Adj Close'], penetration=0)
        doji = CDLDOJI(df['Open'], df['High'], df['Low'], df['Adj Close'])
        doji_star = CDLDOJISTAR(df['Open'], df['High'], df['Low'], df['Adj Close'])
        dragon_doji = CDLDRAGONFLYDOJI(df['Open'], df['High'], df['Low'], df['Adj Close'])
        engulf = CDLENGULFING(df['Open'], df['High'], df['Low'], df['Adj Close'])
        evening_star = CDLEVENINGSTAR(df['Open'], df['High'], df['Low'], df['Adj Close'])
        gapside = CDLGAPSIDESIDEWHITE(df['Open'], df['High'], df['Low'], df['Adj Close'])
        gravestone = CDLGRAVESTONEDOJI(df['Open'], df['High'], df['Low'], df['Adj Close'])
        hammer = CDLHAMMER(df['Open'], df['High'], df['Low'], df['Adj Close'])
        hang_man = CDLHANGINGMAN(df['Open'], df['High'], df['Low'], df['Adj Close'])
        harami = CDLHARAMI(df['Open'], df['High'], df['Low'], df['Adj Close'])
        harami_cross = CDLHARAMICROSS(df['Open'], df['High'], df['Low'], df['Adj Close'])

        #more TA

        high_wave = CDLHIGHWAVE(df['Open'], df['High'], df['Low'], df['Adj Close'])
        hikkake = CDLHIKKAKE(df['Open'], df['High'], df['Low'], df['Adj Close'])
        hikkake_mod = CDLHIKKAKEMOD(df['Open'], df['High'], df['Low'], df['Adj Close'])
        pidgeon = CDLHOMINGPIGEON(df['Open'], df['High'], df['Low'], df['Adj Close'])
        id_three_crows = CDLIDENTICAL3CROWS(df['Open'], df['High'], df['Low'], df['Adj Close'])
        in_neck = CDLINNECK(df['Open'], df['High'], df['Low'], df['Adj Close'])
        inv_hammer = CDLINVERTEDHAMMER(df['Open'], df['High'], df['Low'], df['Adj Close'])
        kicking = CDLKICKING(df['Open'], df['High'], df['Low'], df['Adj Close'])
        kicking_len = CDLKICKINGBYLENGTH(df['Open'], df['High'], df['Low'], df['Adj Close'])
        ladder_bot = CDLLADDERBOTTOM(df['Open'], df['High'], df['Low'], df['Adj Close'])
        doji_long = CDLLONGLEGGEDDOJI(df['Open'], df['High'], df['Low'], df['Adj Close'])
        long_line = CDLLONGLINE(df['Open'], df['High'], df['Low'], df['Adj Close'])
        marabozu = CDLMARUBOZU(df['Open'], df['High'], df['Low'], df['Adj Close'])

        #more TA

        match_glow = CDLMATCHINGLOW(df['Open'], df['High'], df['Low'], df['Adj Close'])
        mat_hold = CDLMATHOLD(df['Open'], df['High'], df['Low'], df['Adj Close'], penetration=0)
        morning_doji = CDLMORNINGDOJISTAR(df['Open'], df['High'], df['Low'], df['Adj Close'], penetration=0)
        morning_star = CDLMORNINGSTAR(df['Open'], df['High'], df['Low'], df['Adj Close'], penetration=0)
        on_neck = CDLONNECK(df['Open'], df['High'], df['Low'], df['Adj Close'])
        pierce = CDLPIERCING(df['Open'], df['High'], df['Low'], df['Adj Close'])
        rickshaw = CDLRICKSHAWMAN(df['Open'], df['High'], df['Low'], df['Adj Close'])
        rise_fall = CDLRISEFALL3METHODS(df['Open'], df['High'], df['Low'], df['Adj Close'])
        sep_line = CDLSEPARATINGLINES(df['Open'], df['High'], df['Low'], df['Adj Close'])
        shooting_star = CDLSHOOTINGSTAR(df['Open'], df['High'], df['Low'], df['Adj Close'])
        sl_candle = CDLSHORTLINE(df['Open'], df['High'], df['Low'], df['Adj Close'])
        spin_top = CDLSPINNINGTOP(df['Open'], df['High'], df['Low'], df['Adj Close'])
        stalled = CDLSTALLEDPATTERN(df['Open'], df['High'], df['Low'], df['Adj Close'])

        #more TA

        stick_sand = CDLSTICKSANDWICH(df['Open'], df['High'], df['Low'], df['Adj Close'])
        takuri = CDLTAKURI(df['Open'], df['High'], df['Low'], df['Adj Close'])
        tasuki_gap = CDLTASUKIGAP(df['Open'], df['High'], df['Low'], df['Adj Close'])
        thrust = CDLTHRUSTING(df['Open'], df['High'], df['Low'], df['Adj Close'])
        tristar = CDLTRISTAR(df['Open'], df['High'], df['Low'], df['Adj Close'])
        three_river = CDLUNIQUE3RIVER(df['Open'], df['High'], df['Low'], df['Adj Close'])
        ud_two_gap = CDLUPSIDEGAP2CROWS(df['Open'], df['High'], df['Low'], df['Adj Close'])
        down_three_gap = CDLXSIDEGAP3METHODS(df['Open'], df['High'], df['Low'], df['Adj Close'])

        #76 vars

        #are_all_zero = (test_TA == 0).all()
        #true if all values are 0
        #false if contain a non 0'''

        df.drop(['Close'], axis =1, inplace = True)

        df['upper_band'] = upper_band
        df['lower_band'] = lower_band
        df['mid_band'] = mid_band
        df['d_ema'] = d_ema
        df['ht_trend'] = ht_trend
        df['kama'] = kama
        df['ma'] = ma
        #df['mama'] = mama
        df['mid'] = mid
        df['mid_price'] = mid_price

        df['sar'] = sar
        df['sarext'] = sarext
        df['sma'] = sma
        df['tema'] = tema
        df['trima'] = trima
        df['wma'] = wma
        #df['fama'] = fama

        df['EMA'] = E_M_A
        df['SlowK'] = slowk
        df['SlowD'] = slowd
        df['R_S_I'] = R_S_I
        df['FastK'] = fastk
        df['FastD'] = fastd
        df['WilliamsR'] = real

        df['atr'] = atr
        df['natr'] = natr
        df['t_range'] = t_range


        #df['na_tr'] = natr

        df['OBV'] = O_B_V
        df['ADOSC'] = A_D_O_S_C
        df['ogchaikin'] = O_G_chaikin
        df['HTDCperiod'] = HT_DCper
        df['HTDCphase'] = HT_DCphase
        df['inphase'] = inphase
        df['quad'] = quad
        df['rsin'] = r_sin
        df['leadsin'] = leadsin

        df['mfi'] = mfi
        df['min_di'] = min_di
        df['min_dm'] = min_dm
        df['momo'] = momo
        df['plus_di'] = plus_di
        df['plus_dm'] = plus_dm
        df['ppo'] = ppo
        df['roc'] = roc
        df['rocp'] = rocp

        df['rocr'] = rocr
        df['rocr_hund'] = rocr_hund
        df['rsi_fastk'] = rsi_fastk
        df['rsi_fastd'] = rsi_fastd
        df['trix'] = trix
        df['ult_osc'] = ult_osc

        df['adx'] = adx
        df['adxr'] = adxr
        df['apo'] = apo
        df['aroon_d'] = aroon_d
        df['aroon_u'] = aroon_u
        df['aroon_osc'] = aroon_osc
        df['bop'] = bop
        df['cci'] = cci
        df['cmo'] = cmo

        df['dx'] = dx
        df['macd'] = macd
        df['macdsig'] = macdsig
        df['macdhist'] = macdhist
        df['macdex'] = macdex
        df['macdexsig'] = macdexsig
        df['macdexhist'] = macdexhist
        df['macdfixd'] = macdfixd
        df['macdfixdsig'] = macdfixdsig
        df['macdfixdhist'] = macdfixdhist
        
        df['two_crows'] = two_crows
        df['three_crows'] = three_crows
        df['three_inside'] = three_inside
        df['three_line'] = three_line
        df['three_out'] = three_out
        df['three_stars'] = three_stars
        df['three_soldier'] = three_soldier
        df['baby'] = baby
        df['adv'] = adv
        df['belt_hold'] = belt_hold
        df['breakaway'] = breakaway
        df['closingmara'] = closingmara
        df['baby_swallow'] = belt_hold

        df['counter'] = counter
        df['dark_cloud'] = dark_cloud
        df['doji'] = doji
        df['doji_star'] = doji_star
        df['dragon_doji'] = dragon_doji
        df['engulf'] = engulf
        df['evening_star'] = evening_star
        df['gapside'] = gapside
        df['gravestone'] = gravestone
        df['hammer'] = hammer
        df['hang_man'] = hang_man
        df['harami'] = harami
        df['harami_cross'] = harami_cross

        df['high_wave'] = high_wave
        df['hikkake'] = hikkake
        df['hikkake_mod'] = hikkake_mod
        df['pidgeon'] = pidgeon
        df['id_three_crows'] = id_three_crows
        df['in_neck'] = in_neck
        df['inv_hammer'] = inv_hammer
        df['kicking'] = kicking
        df['kicking_len'] = kicking_len
        df['ladder_bot'] = ladder_bot
        df['doji_long'] = doji_long
        df['long_line'] = long_line
        df['marabozu'] = marabozu
                                                        # this is  a comment
        df['match_glow'] = match_glow
        df['mat_hold'] = mat_hold
        df['morning_doji'] = morning_doji
        df['morning_star'] = morning_star
        df['on_neck'] = on_neck
        df['pierce'] = pierce
        df['rickshaw'] = rickshaw
        df['rise_fall'] = rise_fall
        df['sep_line'] = sep_line
        df['shooting_star'] = shooting_star
        df['sl_candle'] = sl_candle
        df['spin_top'] = spin_top
        df['stalled'] = stalled

        df['stick_sand'] = stick_sand
        df['takuri'] = takuri
        df['tasuki_gap'] = tasuki_gap
        df['thrust'] = thrust
        df['tristar'] = tristar
        df['three_river'] = three_river
        df['ud_two_gap'] = ud_two_gap
        df['down_three_gap'] = down_three_gap

    add_indicators()
    # Convert Date column to datetime
    df.reset_index(level=0, inplace=True)

    # Change all column headings to be lower case, and remove spacing
    df.columns = [str(x).lower().replace(' ', '_') for x in df.columns]

    # Get difference between high and low of each day
    df['range_hl'] = df['high'] - df['low']
    df.drop(['high', 'low'], axis=1, inplace=True)
    # Get difference between open and close of each day
    df['range_oc'] = df['open'] - df['adj_close']
    df.drop(['open'], axis=1, inplace=True)
    # Add a column 'order_day' to indicate the order of the rows by date
    df['order_day'] = [x for x in list(range(len(df)))]
    # merging_keys
    merging_keys = ['order_day']

    #define shift range
    # 2 min intervals - 30 = 1hr

    #N = 15
    def add_lag(num_lag_cols, this_df):
        lag_cols = ['ema', 'slowk','slowd','r_s_i','fastk','fastd','williamsr','volume','range_hl','range_oc','adj_close', 'obv', 'adosc', 'ogchaikin', 'htdcperiod','htdcphase',
                    'inphase','quad','rsin','leadsin', 'two_crows', 'three_crows', 'three_inside', 'three_line', 'three_out', 'three_stars', 'three_soldier', 'baby', 'adv', 'belt_hold',
                    'breakaway', 'closingmara', 'baby_swallow', 'counter','dark_cloud','doji','doji_star','dragon_doji','engulf','evening_star','gapside','gravestone','hammer',
                    'hang_man','harami','harami_cross','high_wave','hikkake','hikkake_mod','pidgeon','id_three_crows','in_neck','inv_hammer','kicking','kicking_len','ladder_bot',
                    'doji_long','long_line','marabozu', 'match_glow','mat_hold','morning_doji','morning_star','on_neck','pierce','rickshaw','rise_fall','sep_line','shooting_star',
                    'sl_candle','spin_top','stalled','stick_sand','takuri','tasuki_gap','thrust','tristar','three_river','ud_two_gap','down_three_gap', 'upper_band','lower_band',
                    'mid_band','d_ema','ht_trend','kama','ma','mid','mid_price','sar','sarext','sma','tema','trima','wma','adx','adxr','apo','aroon_d','aroon_u','aroon_osc',
                    'bop','cci','cmo','dx','macd','macdsig','macdhist','macdex','macdexsig','macdexhist','macdfixd','macdfixdsig','macdfixdhist','mfi','min_di','min_dm',
                    'momo','plus_di','plus_dm','ppo','roc','rocp','rocr','rocr_hund','rsi_fastk','rsi_fastd','trix','ult_osc', 'atr','natr','t_range'
                    ]

        shift_range = [x+1 for x in range(num_lag_cols)]

        for shift in shift_range:
            train_shift = this_df[merging_keys + lag_cols].copy()
            
            # E.g. order_day of 0 becomes 1, for shift = 1.
            # So when this is merged with order_day of 1 in df, this will represent lag of 1.
            train_shift['order_day'] = train_shift['order_day'] + shift
            
            foo = lambda x: '{}_lag_{}'.format(x, shift) if x in lag_cols else x
            train_shift = train_shift.rename(columns=foo)

            this_df = pd.merge(this_df, train_shift, on=merging_keys, how='left') #.fillna(0)
            
        del train_shift

        return this_df

    num_interval_lag = 15

    df = add_lag(num_interval_lag, df)


    #print(df.columns.values)
                                                # other ways to render the NAN values exist

    """
    Data is labeled as per the logic in research paper
    Label code : BUY => 1, SELL => 0, HOLD => 2
    params :
        df => Dataframe with data
        col_name => name of column which should be used to determine strategy
    returns : numpy array with integer codes for labels with
                size = total-(window_size)+1
    """
    total_rows = len(df)
    
    #df['labels'] = labels

    

    #print(df.head(30))
    #print(df.shape)
    def add_labels(df):
        window_size = 10
                                                            #sets the window range for trades
                                                            #30 intervals = 1 hr
        list_labels = [None] * (window_size-1)

        for row_counter in range(total_rows):
            if row_counter >= window_size - 1:
                window_begin = row_counter - (window_size - 1)
                window_end = row_counter
                window_middle = (window_begin + window_end) / 2

                min_ = np.inf
                min_index = -1
                max_ = -np.inf
                max_index = -1
                for i in range(window_begin, window_end + 1):
                    price = df.iloc[i]['adj_close']
                    if price < min_:
                        min_ = price
                        min_index = i
                    if price > max_:
                        max_ = price
                        max_index = i

                #print('trying')

                if max_index == window_middle:
                    #df.at[window_middle, 'labels'] = 0
                    list_labels.append(0)
                elif min_index == window_middle:
                    #df.at[window_middle, 'labels'] = 1
                    list_labels.append(1)
                else:
                    #df.at[window_middle, 'labels'] = 2
                    list_labels.append(2)
        
        return list_labels

    list_labels = add_labels(df)

    #print(len(list_labels))

    df['labels'] = list_labels

    df.drop(['adj_close'], axis=1, inplace=True)

    df.fillna(0, inplace=True)
    ##print(df.tail(30))
    ##are_all_zero = (df['labels'] == 0).all()
    ##print(are_all_zero)

    #BUY => 0.5, SELL => 0, HOLD => 1

    #defining test and train len
    #print(df.shape)

    num_test = int(.05*len(df))
    num_train = len(df) - num_test

    #print(num_test)

    #print(num_train)

    # Split into train, cv, and test
    train = df[:num_train]
    test = df[num_train:]

    #print(train.columns.values)

    def add_scale(num_interval_lag):

        cols_to_scale = ['ema', 'slowk','slowd','r_s_i','fastk','fastd','williamsr','volume','range_hl','range_oc', 'obv', 'adosc', 'ogchaikin', 'htdcperiod','htdcphase',
                        'inphase','quad','rsin','leadsin', 'two_crows', 'three_crows', 'three_inside', 'three_line', 'three_out', 'three_stars', 'three_soldier', 'baby', 'adv', 'belt_hold',
                        'breakaway', 'closingmara', 'baby_swallow', 'counter','dark_cloud','doji','doji_star','dragon_doji','engulf','evening_star','gapside','gravestone','hammer',
                        'hang_man','harami','harami_cross','high_wave','hikkake','hikkake_mod','pidgeon','id_three_crows','in_neck','inv_hammer','kicking','kicking_len','ladder_bot',
                        'doji_long','long_line','marabozu', 'match_glow','mat_hold','morning_doji','morning_star','on_neck','pierce','rickshaw','rise_fall','sep_line','shooting_star',
                        'sl_candle','spin_top','stalled','stick_sand','takuri','tasuki_gap','thrust','tristar','three_river','ud_two_gap','down_three_gap', 'upper_band','lower_band',
                        'mid_band','d_ema','ht_trend','kama','ma','mid','mid_price','sar','sarext','sma','tema','trima','wma','adx','adxr','apo','aroon_d','aroon_u','aroon_osc',
                        'bop','cci','cmo','dx','macd','macdsig','macdhist','macdex','macdexsig','macdexhist','macdfixd','macdfixdsig','macdfixdhist','mfi','min_di','min_dm',
                        'momo','plus_di','plus_dm','ppo','roc','rocp','rocr','rocr_hund','rsi_fastk','rsi_fastd','trix','ult_osc', 'atr','natr','t_range', 'labels'
                        ]

        for i in range(1,num_interval_lag+1):
            cols_to_scale.append("ema_lag_"+str(i))
            cols_to_scale.append("slowk_lag_"+str(i))
            cols_to_scale.append("slowd_lag_"+str(i))
            cols_to_scale.append("r_s_i_lag_"+str(i))
            cols_to_scale.append("fastk_lag_"+str(i))
            cols_to_scale.append("fastd_lag_"+str(i))
            cols_to_scale.append("williamsr_lag_"+str(i))
            cols_to_scale.append("volume_lag_"+str(i))
            cols_to_scale.append("range_hl_lag_"+str(i))
            cols_to_scale.append("range_oc_lag_"+str(i))
            cols_to_scale.append("adj_close_lag_"+str(i))

            cols_to_scale.append("upper_band_lag_"+str(i))
            cols_to_scale.append("lower_band_lag_"+str(i))
            cols_to_scale.append("mid_band_lag_"+str(i))
            cols_to_scale.append("d_ema_lag_"+str(i))
            cols_to_scale.append("ht_trend_lag_"+str(i))
            cols_to_scale.append("kama_lag_"+str(i))
            cols_to_scale.append("ma_lag_"+str(i))
            cols_to_scale.append("mid_lag_"+str(i))
            cols_to_scale.append("mid_price_lag_"+str(i))
            cols_to_scale.append("sar_lag_"+str(i))
            cols_to_scale.append("sarext_lag_"+str(i))
            cols_to_scale.append("sma_lag_"+str(i))
            cols_to_scale.append("tema_lag_"+str(i))
            cols_to_scale.append("trima_lag_"+str(i))
            cols_to_scale.append("wma_lag_"+str(i))

            cols_to_scale.append("atr_lag_"+str(i))
            cols_to_scale.append("natr_lag_"+str(i))
            cols_to_scale.append("t_range_lag_"+str(i))

            #momentum indicator lag cols

            cols_to_scale.append("adx_lag_"+str(i))
            cols_to_scale.append("adxr_lag_"+str(i))
            cols_to_scale.append("apo_lag_"+str(i))
            cols_to_scale.append("aroon_d_lag_"+str(i))
            cols_to_scale.append("aroon_u_lag_"+str(i))
            cols_to_scale.append("aroon_osc_lag_"+str(i))
            cols_to_scale.append("bop_lag_"+str(i))
            cols_to_scale.append("cci_lag_"+str(i))
            cols_to_scale.append("cmo_lag_"+str(i))
            cols_to_scale.append("dx_lag_"+str(i))
            cols_to_scale.append("macd_lag_"+str(i))
            cols_to_scale.append("macdsig_lag_"+str(i))
            cols_to_scale.append("macdhist_lag_"+str(i))
            cols_to_scale.append("macdex_lag_"+str(i))

            cols_to_scale.append("mfi_lag_"+str(i))
            cols_to_scale.append("min_di_lag_"+str(i))
            cols_to_scale.append("min_dm_lag_"+str(i))
            cols_to_scale.append("momo_lag_"+str(i))
            cols_to_scale.append("plus_di_lag_"+str(i))
            cols_to_scale.append("plus_dm_lag_"+str(i))
            cols_to_scale.append("ppo_lag_"+str(i))
            cols_to_scale.append("roc_lag_"+str(i))
            cols_to_scale.append("rocp_lag_"+str(i))
            cols_to_scale.append("rocr_lag_"+str(i))
            cols_to_scale.append("rocr_hund_lag_"+str(i))
            cols_to_scale.append("rsi_fastk_lag_"+str(i))
            cols_to_scale.append("rsi_fastd_lag_"+str(i))
            cols_to_scale.append("trix_lag_"+str(i))
            cols_to_scale.append("ult_osc_lag_"+str(i))

            cols_to_scale.append("macdexsig_lag_"+str(i))
            cols_to_scale.append("macdexhist_lag_"+str(i))
            cols_to_scale.append("macdfixd_lag_"+str(i))
            cols_to_scale.append("macdfixdsig_lag_"+str(i))
            cols_to_scale.append("macdfixdhist_lag_"+str(i))


            #cols_to_scale.append("mama_lag_"+str(i))
            #cols_to_scale.append("NATR_lag_"+str(i))
            
            cols_to_scale.append("obv_lag_" +str(i))
            cols_to_scale.append("adosc_lag_"+str(i))
            cols_to_scale.append("ogchaikin_lag_"+str(i))
            cols_to_scale.append("htdcperiod_lag_"+str(i))
            cols_to_scale.append("htdcphase_lag_"+str(i))
            cols_to_scale.append("inphase_lag_"+str(i))
            cols_to_scale.append("quad_lag_"+str(i))
            cols_to_scale.append("rsin_lag_"+str(i))
            cols_to_scale.append("leadsin_lag_"+str(i))
            #cols_to_scale.append("fama_lag_"+str(i))

            cols_to_scale.append("two_crows_lag_" +str(i))
            cols_to_scale.append("three_crows_lag_"+str(i))
            cols_to_scale.append("three_inside_lag_"+str(i))
            cols_to_scale.append("three_line_lag_"+str(i))
            cols_to_scale.append("three_out_lag_"+str(i))
            cols_to_scale.append("three_stars_lag_"+str(i))
            cols_to_scale.append("three_soldier_lag_"+str(i))
            cols_to_scale.append("baby_lag_"+str(i))
            cols_to_scale.append("adv_lag_"+str(i))
            cols_to_scale.append("belt_hold_lag_"+str(i))
            cols_to_scale.append("breakaway_lag_"+str(i))
            cols_to_scale.append("closingmara_lag_"+str(i))
            cols_to_scale.append("baby_swallow_lag_"+str(i))

            cols_to_scale.append("counter_lag_" +str(i))
            cols_to_scale.append("dark_cloud_lag_"+str(i))
            cols_to_scale.append("doji_lag_"+str(i))
            cols_to_scale.append("doji_star_lag_"+str(i))
            cols_to_scale.append("dragon_doji_lag_"+str(i))
            cols_to_scale.append("engulf_lag_"+str(i))
            cols_to_scale.append("evening_star_lag_"+str(i))
            cols_to_scale.append("gapside_lag_"+str(i))
            cols_to_scale.append("gravestone_lag_"+str(i))
            cols_to_scale.append("hammer_lag_"+str(i))
            cols_to_scale.append("hang_man_lag_"+str(i))
            cols_to_scale.append("harami_lag_"+str(i))
            cols_to_scale.append("harami_cross_lag_"+str(i))

            cols_to_scale.append("high_wave_lag_" +str(i))
            cols_to_scale.append("hikkake_lag_"+str(i))
            cols_to_scale.append("hikkake_mod_lag_"+str(i))
            cols_to_scale.append("pidgeon_lag_"+str(i))
            cols_to_scale.append("id_three_crows_lag_"+str(i))
            cols_to_scale.append("in_neck_lag_"+str(i))
            cols_to_scale.append("inv_hammer_lag_"+str(i))
            cols_to_scale.append("kicking_lag_"+str(i))
            cols_to_scale.append("kicking_len_lag_"+str(i))
            cols_to_scale.append("ladder_bot_lag_"+str(i))
            cols_to_scale.append("doji_long_lag_"+str(i))
            cols_to_scale.append("long_line_lag_"+str(i))
            cols_to_scale.append("marabozu_lag_"+str(i))

            cols_to_scale.append("match_glow_lag_" +str(i))
            cols_to_scale.append("mat_hold_lag_"+str(i))
            cols_to_scale.append("morning_doji_lag_"+str(i))
            cols_to_scale.append("morning_star_lag_"+str(i))
            cols_to_scale.append("on_neck_lag_"+str(i))
            cols_to_scale.append("pierce_lag_"+str(i))
            cols_to_scale.append("rickshaw_lag_"+str(i))
            cols_to_scale.append("rise_fall_lag_"+str(i))
            cols_to_scale.append("sep_line_lag_"+str(i))
            cols_to_scale.append("shooting_star_lag_"+str(i))
            cols_to_scale.append("sl_candle_lag_"+str(i))
            cols_to_scale.append("spin_top_lag_"+str(i))
            cols_to_scale.append("stalled_lag_"+str(i))

            cols_to_scale.append("stick_sand_lag_"+str(i))
            cols_to_scale.append("takuri_lag_"+str(i))
            cols_to_scale.append("tasuki_gap_lag_"+str(i))
            cols_to_scale.append("thrust_lag_"+str(i))
            cols_to_scale.append("tristar_lag_"+str(i))
            cols_to_scale.append("three_river_lag_"+str(i))
            cols_to_scale.append("ud_two_gap_lag_"+str(i))
            cols_to_scale.append("down_three_gap_lag_"+str(i))

        return cols_to_scale

    cols_to_scale = add_scale(num_interval_lag)
    #print(train.columns.tolist())

    # Do scaling for train set
    # Here we only scale the train dataset, and not the entire dataset to prevent information leak
    scaler = StandardScaler()

    scaler.fit(train[cols_to_scale])
    train_scaled = scaler.transform(train[cols_to_scale])

    # Convert the numpy array back into pandas dataframe

    #print(cols_to_scale)

    train_scaled = pd.DataFrame(train_scaled, columns=cols_to_scale)

    train_scaled = train_scaled[30:]

    print(train_scaled['labels'])

    #duplicate_columns = train_scaled.columns[train_scaled.columns.duplicated()]

    #print(duplicate_columns)

    #file_name = stock + "output.csv"

    #df.to_csv(file_name)
                                                                    # this may be a good place to save to a .csv file and export the data to matlab
                                                                    #  / do diagnostic visualizations

    #print(train_scaled.columns.tolist())
    #train_scaled['datetime'] = train.reset_index()['datetime']
    #print("train_scaled.shape = " + str(train_scaled.shape))
    #print(train_scaled.head(5))

                                                                    #this line is needed for the PCA


    scaler_2 = StandardScaler()
    scaler_2.fit(test[cols_to_scale])
    test_scaled = scaler_2.transform(test[cols_to_scale])

    # Convert the numpy array back into pandas dataframe

    test_scaled = pd.DataFrame(test_scaled, columns=cols_to_scale)

    features = []
    target = "labels"
    features = cols_to_scale
    #features.remove(target)

    # Split into X and y
    X_train_scaled = train_scaled[features]
    y_train_scaled = train_scaled[target]

    X_test_scaled = test_scaled[features]
    y_test_scaled = test_scaled[target]
    
                                               ## PCA testing needs to be done here to see what should / should not be included.

    #print(X_sample_scaled.columns.tolist())
    #print(type(X_train_scaled))
    #testing = X_train_scaled.to_numpy()
    #print(np.isnan(testing.any()))
    #print(np.isfinite(testing))
    #pca = PCA(n_components = 80).fit(X_train_scaled)
    #print(pca.explained_variance_ratio_)
    #print(pca.singular_values_)
    #X_train_scaled, y_train_scaled, X_sample_scaled, y_sample_scaled = preprocessing_data(stock = 'BB')

                                                ## these values can be adjusted to customize the model

    #rand = np.random.randint(low=1,high = 999)

    model = XGBRegressor(n_estimators=200,
                        use_label_encoder = False,
                        max_depth=20,
                        learning_rate=0.1,
                        objective = 'multi:softmax',
                        min_child_weight=1,
                        subsample=1,
                        colsample_bytree=1,
                        colsample_bylevel=1,
                        num_class=3,
                        gamma=0.1)

    # Train the regressor

    model.fit(X_train_scaled, y_train_scaled)
    #xgb.plot_importance(model)
    #feat_list = xgb.plot_importance(model).get_yticklabels()[::-1]

    #print(feat_list)

    #xgb.plot_tree(model)
    '''
    #path_out = r'C:\\Users\\Michael\\Desktop\\Python\Stonks\\YF & modeling\\TestSP500Out\\'

    #feat_save_name = path_out + stock + "features"
    #tree_save_name = path_out + stock + "tree"

    #xgb.plot_importance(model).figure.savefig(feat_save_name, dpi=600)
    #xgb.plot_tree(model).figure.savefig(tree_save_name, dpi=600)

    #feature_important = model.get_booster().get_score(importance_type='weight')
    #keys = list(feature_important.keys())
    #values = list(feature_important.values())

    #print(keys)
    #print(values)

    #data = pd.DataFrame(data=values, index=keys, columns=["score"]).sort_values(by = "score", ascending=False)

    #print(type(data))

    this_stock_feat_list = data.index.tolist()
    '''
    #this_stock_feat_frame = pd.DataFrame(this_stock_feat_list, columns = list(stock))
    
    #data.plot(kind='barh')

    #plt.show()


    #doing predictions on model
    #print(X_train_scaled)
    test_pred = model.predict(X_test_scaled)
    #train_pred = model.predict(X_train_scaled)
    #insert back into test_scaled array
    test_scaled['labels'] = test_pred

    #train_scaled['labels'] = train_pred

    ''' there is some consideration to be made if we can grab the top 20-30 most influential features from xgboost and use them to train a different model type'''
    ''' there is also consideration to be made about exporting these models'''


    # this methodology works for saving a trained model
    #pickle.dump(model, open("test.model", "wb"))

    #unscaling
    pred_unscaled = scaler_2.inverse_transform(test_scaled)
    plt.figure()
    #plotting

    #plt.plot(train_scaled.index, train[target])
    #plt.plot(train_scaled.index, train_scaled[target])


    plt.plot(test_scaled.index, test[target])
    plt.plot(test_scaled.index, test_scaled[target])

    
    plt.legend(('True', 'est'), loc='upper left')

    plt.title(str(stock))

    plt.xlabel("Intervals")

    plt.ylabel('BUY => 0.5, SELL => 0, HOLD => 1')

    #stonk_path_out =  path_out + stock

    #plt.savefig(stonk_path_out)
    
    #test_true_num = test[target].iloc[-1]
    #test_pred_num = pred_unscaled[-1, 10]

    #is_going_up = test_pred_num > test_true_num
    #print(stock)
    #print(is_going_up)

    #return this_stock_feat_list

    #change intervals back to date-time


    '''toc = time.perf_counter()

    tic_toc = (toc - tic) / 60

    print(f"completed Pred in {tic_toc:0.4f} min")'''

    plt.show()