import pandas as pd
import sklearn as sk
from sklearn import preprocessing

data_frame = pd.read_csv("TSLA.csv")

data_frame.drop(['Close'], axis=1, inplace=True)

data_frame.drop(['Datetime'], axis=1, inplace=True)

col_names = list(data_frame.columns)

col_names[4],col_names[3] = col_names[3],col_names[4]

data_frame = data_frame[col_names]

num_test = int(len(data_frame)*.35)
num_train = len(data_frame) - num_test

train = data_frame[:num_train]
test = data_frame[num_train:]

scaler = preprocessing.StandardScaler().fit(train)

train_scaled = scaler.transform(train)

scaler = preprocessing.StandardScaler().fit(test)

test_scaled = scaler.transform(test)

print(pd.DataFrame(train_scaled, columns=col_names).head(5))

