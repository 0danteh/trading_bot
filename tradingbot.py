import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
import tensorflow as tf
from keras.models import Model
from keras.layers import Dense, Dropout, LSTM, Input, Activation
from keras import optimizers
from sklearn.metrics import mean_squared_error
import seaborn as sns


if __name__ == "__main__":
    start_date = datetime(2010, 1, 1)
    end_date = datetime(2020, 1, 1)
    ticker_symbol = "AAPL"
    train_split = 0.7
    data_set_points = 21

stock_df = yf.download(tickers=ticker_symbol, start=start_date, end=end_date)
new_df = stock_df[['Adj Close']].copy()

def prepare_train_test_split(new_df, data_set_points, train_split):
    new_df.reset_index(inplace=True)
    new_df.drop(0, inplace=True)
    split_index = int(len(new_df) * train_split)
    train_data = new_df[:split_index]
    test_data = new_df[split_index:].reset_index(drop=True)
    train_diff = train_data['Adj Close'].diff().dropna().values
    test_diff = test_data['Adj Close'].diff().dropna().values
    X_train = np.array([train_diff[i : i + data_set_points] for i in range(len(train_diff) - data_set_points)])
    y_train = np.array([train_diff[i + data_set_points] for i in range(len(train_diff) - data_set_points)])
    y_valid = train_data['Adj Close'].tail(len(y_train) // 10).values
    y_valid = y_valid.reshape(-1, 1)
    X_test = np.array([test_diff[i : i + data_set_points] for i in range(len(test_diff) - data_set_points)])
    y_test = test_data['Adj Close'].shift(-data_set_points).dropna().values
    return X_train, y_train, X_test, y_test, test_data

def create_lstm_model(X_train, y_train, data_set_points):
    tf.random.set_seed(20)
    np.random.seed(10)
    lstm_input = Input(shape=(data_set_points, 1), name='lstm_input')
    inputs = LSTM(21, name='lstm_0', return_sequences=True)(lstm_input)
    inputs = Dropout(0.1, name='dropout_0')(inputs)
    inputs = LSTM(32, name='lstm_1')(inputs)
    inputs = Dropout(0.05, name='dropout_1')(inputs)
    inputs = Dense(32, name='dense_0')(inputs)
    inputs = Dense(1, name='dense_1')(inputs)
    output = Activation('linear', name='output')(inputs)
    model = Model(inputs=lstm_input, outputs=output)
    adam = optimizers.Adam(lr=0.002)
    model.compile(optimizer=adam, loss='mse')
    model.fit(x=X_train, y=y_train, batch_size=15, epochs=25, shuffle=True, validation_split=0.1)
    return model

X_train, y_train, X_test, y_test, test_data = prepare_train_test_split(new_df, data_set_points, train_split)
model = create_lstm_model(X_train, y_train, data_set_points)
y_pred = model.predict(X_test)
y_pred = y_pred.flatten()
actual1 = np.array([test_data['Adj Close'][i + data_set_points] for i in range(len(test_data) - data_set_points)])
actual2 = actual1[:-1]
data = np.add(actual2, y_pred)
plt.gcf().set_size_inches(12, 8, forward=True)
plt.title('Plot of real price and predicted price against number of days for test set')
plt.xlabel('Number of days')
plt.ylabel('Adjusted Close Price($)')
plt.plot(actual1[1:], label='Actual Price')
plt.plot(data, label='Predicted Price')
plt.legend(['Actual Price', 'Predicted Price'])
plt.show()

error = actual1[1:] - data
plt.hist(error, bins=25)
plt.xlabel('Prediction Error ($)')
plt.title('Histogram of prediction errors')
plt.ylabel('Frequency')
plt.show()

percentage_tolerance = 0.10
diff = np.abs(actual1[1:] - data)
within_tolerance = np.where((diff / actual1[1:]) <= percentage_tolerance, 1, 0)
accuracy = np.sum(within_tolerance) / len(within_tolerance) * 100
print(f"The Forecast is Accurate by {accuracy}%")
print(f"The Mean Squared Error (MSE) is:",mean_squared_error(actual1[1:], data, squared = False))
