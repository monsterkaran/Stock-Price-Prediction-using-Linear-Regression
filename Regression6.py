import pickle

import pandas as pd
import quandl, math, datetime
import numpy as np
from sklearn import preprocessing, model_selection, svm
from sklearn.linear_model import LinearRegression
import matplotlib.dates as mdates

import matplotlib.pyplot as plt
from matplotlib import style

style.use('ggplot')

GOOGL = r'C:\Users\Karan Raj\Downloads\archive\GOOGL.csv'
df = pd.read_csv(GOOGL)

# print(df.head())

df = df[['Open','High','Low','Close','Volume']]
df['HL_PCT'] = (df['High'] - df['Low']) / df['Low'] * 100
df['PCT_change'] = (df['Close'] - df['Open']) / df['Open'] * 100

df = df[['Close','HL_PCT','PCT_change','Volume']]

forecast_col = 'Close'
df.fillna(-99999, inplace=True)

forecast_out = int(math.ceil(0.01*len(df)))
# print(forecast_out)

df['label'] = df[forecast_col].shift(-forecast_out)


X = np.array(df.drop(['label'],axis=1))
X = preprocessing.scale(X)
X = X[:-forecast_out]
X_lately = X[-forecast_out:]


# X = X[:-forecast_out+1]
# The reason why I was doing this shift initially was because
# we wouldn't actually have labels But we dropped those labels here
# or those rows here? So we didnt need to do what we were doing there, okay?

df.dropna(inplace=True)
y = np.array(df['label'])

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)

# I can comment from line no. 54-60 after executing this one time even after that
# I can execute this after commenting and get the same result. As pickle is storing it

# clf = LinearRegression()
# clf = LinearRegression(n_jobs=-1)  # -1 means running as many as possible by our CPUs
# shifting algorithm to show how easy it is to switch between algorithms
# clf = svm.SVR(kernel='poly')

# clf.fit(X_train, y_train)
# with open('linearregression.pickle','wb') as f:
#     pickle.dump(clf, f)

pickle_in = open('linearregression.pickle','rb')
clf = pickle.load(pickle_in)


accuracy = clf.score(X_test, y_test)

# print(accuracy)

forecast_set = clf.predict(X_lately)
print(forecast_set, accuracy, forecast_out)
df['Forecast'] = np.nan

if not isinstance(df.index, pd.DatetimeIndex):
    df.index = pd.to_datetime(df.index)

last_date = df.iloc[-1].name  # -1 means this is the very last date and we will get names of that
last_unix = last_date.timestamp()
one_day = 86400
one_year = (one_day * 365)
one_month = one_year/12
next_unix = last_unix + one_day

for i in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += one_day
    df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)] + [i]


df['Close'].plot()
df['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()