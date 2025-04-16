# import yfinance as yf
# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.svm import SVC
# from sklearn.metrics import accuracy_score

# data = yf.download("EURUSD=X", period="6mo", interval="1d")
# data.dropna(inplace=True)

# data['Return'] = data['Close'].pct_change()
# data['Target'] = np.where(data['Close'].shift(-1) > data['Close'], 1, 0)
# data.dropna(inplace=True)

# X = data[['Return']]
# y = data['Target']


# X_train, X_test = X[:-5], X[-5:]
# y_train, y_test = y[:-5], y[-5:]

# model = SVC(kernel='rbf')
# model.fit(X_train, y_train)

# y_pred = model.predict(X_test)

# print("Predicted movements (1=Up, 0=Down):", list(y_pred))
# print("Actual movements   (1=Up, 0=Down):", list(y_test.values))
# print("Accuracy on recent data: {:.2f}%".format(accuracy_score(y_test, y_pred) * 100))
# After predictions

print("hello world..!")

