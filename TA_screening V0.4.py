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
import math
import numpy as np

from sklearn.decomposition import PCA

yf.pdr_override()

time.sleep(1)
###


stock = 'TXN'


###
tic = time.perf_counter()
index = []
start_date = datetime.datetime.now() - datetime.timedelta(days=59)
#end_date = datetime.date.today()

end_date = datetime.datetime.now()

#print(start_date)
#print(end_date)

df = pdr.get_data_yahoo(stock, start=start_date, end=end_date, interval = "2m")

df.index = df.index.tz_localize(None)

#print(df.columns.tolist())

#2 min ticker
# 30 intervals = 1 hour
# 195 intervals = trading day

really_fast = 90
fast = 120
slow = 190

E_M_A = EMA(df['Adj Close'], timeperiod=fast)
slowk, slowd = STOCH(df['High'], df['Low'], df['Adj Close'], fastk_period=fast, slowk_period=slow, slowk_matype=0, slowd_period=slow, slowd_matype=0)
fastk, fastd = STOCHF(df['High'], df['Low'], df['Adj Close'], fastk_period=fast, fastd_period=really_fast, fastd_matype=0)
real = WILLR(df['High'], df['Low'], df['Adj Close'], timeperiod=slow)
R_S_I = RSI(df['Adj Close'], timeperiod=slow)

#add below here

O_B_V = OBV(df['Adj Close'], df['Volume'])
A_D_O_S_C = ADOSC(df['High'], df['Low'], df['Adj Close'], df['Volume'], fastperiod=fast, slowperiod=slow)
O_G_chaikin = AD(df['High'], df['Low'], df['Adj Close'], df['Volume'])
HT_DCper = HT_DCPERIOD(df['Adj Close'])
HT_DCphase = HT_DCPHASE(df['Adj Close'])
inphase, quad = HT_PHASOR(df['Adj Close'])
r_sin, leadsin = HT_SINE(df['Adj Close'])
# these would be new -- is it worth??

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

#next round

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

# next next

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

# next 

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

# next next next next

stick_sand = CDLSTICKSANDWICH(df['Open'], df['High'], df['Low'], df['Adj Close'])
takuri = CDLTAKURI(df['Open'], df['High'], df['Low'], df['Adj Close'])
tasuki_gap = CDLTASUKIGAP(df['Open'], df['High'], df['Low'], df['Adj Close'])
thrust = CDLTHRUSTING(df['Open'], df['High'], df['Low'], df['Adj Close'])
tristar = CDLTRISTAR(df['Open'], df['High'], df['Low'], df['Adj Close'])
three_river = CDLUNIQUE3RIVER(df['Open'], df['High'], df['Low'], df['Adj Close'])
ud_two_gap = CDLUPSIDEGAP2CROWS(df['Open'], df['High'], df['Low'], df['Adj Close'])
down_three_gap = CDLXSIDEGAP3METHODS(df['Open'], df['High'], df['Low'], df['Adj Close'])

#76 vars

#print(type(test_TA))

#are_all_zero = (test_TA == 0).all()
#true if all values are 0
#false if contain a non 0'''

#print(are_all_zero)
#print(E_M_A.tail(5))

df.drop(['Close'], axis =1, inplace = True)
df['EMA'] = E_M_A
df['SlowK'] = slowk
df['SlowD'] = slowd
df['R_S_I'] = R_S_I
df['FastK'] = fastk
df['FastD'] = fastd
df['WilliamsR'] = real

df['OBV'] = O_B_V
df['ADOSC'] = A_D_O_S_C
df['ogchaikin'] = O_G_chaikin
df['HTDCperiod'] = HT_DCper
df['HTDCphase'] = HT_DCphase
df['inphase'] = inphase
df['quad'] = quad
df['rsin'] = r_sin
df['leadsin'] = leadsin

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

# Convert Date column to datetime
df.reset_index(level=0, inplace=True)
# Change all column headings to be lower case, and remove spacing
df.columns = [str(x).lower().replace(' ', '_') for x in df.columns]

# Get month of each sample
#df['month'] = df['datetime'].dt.month
#file_name = str(stock) + ".csv"
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

N = 30

#
# List of columns that we will use to create lags



lag_cols = ['ema', 'slowk','slowd','r_s_i','fastk','fastd','williamsr','volume','range_hl','range_oc','adj_close', 'obv', 'adosc', 'ogchaikin', 'htdcperiod','htdcphase',
            'inphase','quad','rsin','leadsin', 'two_crows', 'three_crows', 'three_inside', 'three_line', 'three_out', 'three_stars', 'three_soldier', 'baby', 'adv', 'belt_hold',
            'breakaway', 'closingmara', 'baby_swallow', 'counter','dark_cloud','doji','doji_star','dragon_doji','engulf','evening_star','gapside','gravestone','hammer',
            'hang_man','harami','harami_cross','high_wave','hikkake','hikkake_mod','pidgeon','id_three_crows','in_neck','inv_hammer','kicking','kicking_len','ladder_bot',
            'doji_long','long_line','marabozu', 'match_glow','mat_hold','morning_doji','morning_star','on_neck','pierce','rickshaw','rise_fall','sep_line','shooting_star',
            'sl_candle','spin_top','stalled','stick_sand','takuri','tasuki_gap','thrust','tristar','three_river','ud_two_gap','down_three_gap'
            ]

shift_range = [x+1 for x in range(N)]

for shift in shift_range:
    train_shift = df[merging_keys + lag_cols].copy()
    
    # E.g. order_day of 0 becomes 1, for shift = 1.
    # So when this is merged with order_day of 1 in df, this will represent lag of 1.
    train_shift['order_day'] = train_shift['order_day'] + shift
    
    foo = lambda x: '{}_lag_{}'.format(x, shift) if x in lag_cols else x
    train_shift = train_shift.rename(columns=foo)

    df = pd.merge(df, train_shift, on=merging_keys, how='left') #.fillna(0)
    
del train_shift

# Remove the first N rows which contain NaNs
#df = df[80:]

df.fillna(0)



#print(df.head(5))

#print(df.shape)

# Get sizes of each of the datasets
num_test = int(.3*len(df))
num_train = len(df) - num_test
'''print("num_train = " + str(num_train))
print("num_cv = " + str(num_cv))
print("num_test = " + str(num_test))'''

# Split into train, cv, and test
train = df[:num_train]
test = df[num_train:]
'''print("train.shape = " + str(train.shape))
print("cv.shape = " + str(cv.shape))
print("train_cv.shape = " + str(train_cv.shape))
print("test.shape = " + str(test.shape))'''

#print(df.columns.tolist())

cols_to_scale = ['ema', 'slowk','slowd','r_s_i','fastk','fastd','williamsr','volume','range_hl','range_oc','adj_close', 'obv', 'adosc', 'ogchaikin', 'htdcperiod','htdcphase',
                'inphase','quad','rsin','leadsin', 'two_crows', 'three_crows', 'three_inside', 'three_line', 'three_out', 'three_stars', 'three_soldier', 'baby', 'adv', 'belt_hold',
                'breakaway', 'closingmara', 'baby_swallow', 'counter','dark_cloud','doji','doji_star','dragon_doji','engulf','evening_star','gapside','gravestone','hammer',
                'hang_man','harami','harami_cross','high_wave','hikkake','hikkake_mod','pidgeon','id_three_crows','in_neck','inv_hammer','kicking','kicking_len','ladder_bot',
                'doji_long','long_line','marabozu', 'match_glow','mat_hold','morning_doji','morning_star','on_neck','pierce','rickshaw','rise_fall','sep_line','shooting_star',
                'sl_candle','spin_top','stalled','stick_sand','takuri','tasuki_gap','thrust','tristar','three_river','ud_two_gap','down_three_gap'
                ]

for i in range(1,N+1):
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
    
    cols_to_scale.append("obv_lag_" +str(i))
    cols_to_scale.append("adosc_lag_"+str(i))
    cols_to_scale.append("ogchaikin_lag_"+str(i))
    cols_to_scale.append("htdcperiod_lag_"+str(i))
    cols_to_scale.append("htdcphase_lag_"+str(i))
    cols_to_scale.append("inphase_lag_"+str(i))
    cols_to_scale.append("quad_lag_"+str(i))
    cols_to_scale.append("rsin_lag_"+str(i))
    cols_to_scale.append("leadsin_lag_"+str(i))

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

#print(train.columns.tolist())

# Do scaling for train set
# Here we only scale the train dataset, and not the entire dataset to prevent information leak
scaler = StandardScaler()
scaler.fit(train[cols_to_scale])
train_scaled = scaler.transform(train[cols_to_scale])
'''print("scaler.mean_ = " + str(scaler.mean_))
print("scaler.var_ = " + str(scaler.var_))
print("train_scaled.shape = " + str(train_scaled.shape))'''

# Convert the numpy array back into pandas dataframe

train_scaled = pd.DataFrame(train_scaled, columns=cols_to_scale)

                                                                #df.to_csv(file_name)
                                                                # this may be a good place to save to a .csv file and export the data to matlab
                                                                #  / do diagnostic visualizations

#print(train_scaled.columns.tolist())
#train_scaled['datetime'] = train.reset_index()['datetime']
#print("train_scaled.shape = " + str(train_scaled.shape))
#print(train_scaled.head(5))

                            #this line is needed for the PCA

#train_scaled = train_scaled[100:]

scaler_2 = StandardScaler()
scaler_2.fit(test[cols_to_scale])
test_scaled = scaler_2.transform(test[cols_to_scale])

'''print("scaler.mean_ = " + str(scaler.mean_))
print("scaler.var_ = " + str(scaler.var_))
print("train_scaled.shape = " + str(train_scaled.shape))'''

#print(test_scaled.shape)
#print(cols_to_scale)
# Convert the numpy array back into pandas dataframe

test_scaled = pd.DataFrame(test_scaled, columns=cols_to_scale)

#test_scaled[['datetime', 'month']] = test.reset_index()[['datetime', 'month']]
#print(test_scaled.head(5))

features = []
target = "adj_close"
features = cols_to_scale
features.remove(target)

# Split into X and y
X_train_scaled = train_scaled[features]
y_train_scaled = train_scaled[target]

X_sample_scaled = test_scaled[features]
y_sample_scaled = test_scaled[target]

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

'''print("X_train_scaled.shape = " + str(X_train_scaled.shape))
print("y_train.shape = " + str(y_train_scaled.shape))

print("X_sample_scaled.shape = " + str(X_sample_scaled.shape))
print("y_sample_scaled.shape = " + str(y_sample_scaled.shape))'''

                                ## these values can be adjusted to customize the model

model = XGBRegressor(seed=100,
                     n_estimators=200,
                     max_depth=20,
                     learning_rate=0.1,
                     min_child_weight=1,
                     subsample=1,
                     colsample_bytree=1,
                     colsample_bylevel=1,
                     gamma=0.1)

# Train the regressor

model.fit(X_train_scaled, y_train_scaled)

# Do prediction on train set
#est_scaled = model.predict(X_train_scaled)
#est = est_scaled * math.sqrt(scaler.var_[0]) + scaler.mean_[0]
#print(est_scaled)

'''plt.plot(X_train_scaled.index, y_train_scaled)
plt.plot(X_train_scaled.index, est_scaled)'''

#train_pred = model.predict(X_train_scaled)

test_pred = model.predict(X_sample_scaled)
test_scaled['adj_close'] = test_pred
#train_scaled['adj_close'] = train_pred
#re_scaling = test_scaled.drop(['datetime', 'month'], axis = 1)

re_scaling = test_scaled
#re_scaling_2 = train_scaled
#print(test_scaled.columns.tolist())
#print(type(test_scaled))
#print(test_scaled[cols_to_scale].shape)
#print(test_scaled.head(5))
#print(X_sample_scaled.columns.tolist())

test_pred_unscaled = scaler_2.inverse_transform(re_scaling)
#train_pred_unscaled = scaler.inverse_transform(re_scaling_2)
#print(type(train_pred_unscaled))
#print(type(train_scaled.index))
#pred_comb = np.concatenate((train_pred_unscaled, test_pred_unscaled))
#plt.plot(df.index, df['adj_close'])
#plt.plot(df.index, test_pred_unscaled[:,10])

plt.plot(test_scaled.index, test[target])
plt.plot(test_scaled.index, test_pred_unscaled[:, 10])

plt.legend(('True', 'est'), loc='upper left')

plt.title(str(stock))

plt.show()

toc = time.perf_counter()

print(f"completed Pred in {toc - tic:0.4f} seconds")

'''plt.plot(, df['adj_close'])
plt.xlabel('Date')
plt.ylabel('Adj Close')

plt.show()'''