import matplotlib.pyplot
import pandas as pd
import quandl,math
import numpy as np
from sklearn import preprocessing, model_selection, svm
from sklearn.linear_model import LinearRegression

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
print(forecast_out)

df['label'] = df[forecast_col].shift(-forecast_out)
df.dropna(inplace=True)

X = np.array(df.drop(['label'],axis=1))
y = np.array(df['label'])

X = preprocessing.scale(X)


# X = X[:-forecast_out+1]
# The reason why I was doing this shift initially was because
# we wouldn't actually have labels But we dropped those labels here
# or those rows here? So we didnt need to do what we were doing there, okay?


y = np.array(df['label'])

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)

# clf = LinearRegression()
clf = LinearRegression(n_jobs=-1)  # -1 means running as many as possible by our CPUs
# shifting algorithm to show how easy it is to switch between algorithms
# clf = svm.SVR(kernel='poly')

clf.fit(X_train, y_train)
accuracy = clf.score(X_test, y_test)

print(accuracy)
