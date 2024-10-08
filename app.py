import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# Load historical metals price data (e.g., Copper)
# Assume a CSV file with columns: Date, Price, Volume, etc.
data = pd.read_csv('historical_metal_prices.csv')

# Ensure the Date column is treated as a datetime object
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

# Focus on price data for simplicity
prices = data['Price'].values

# Feature Scaling (Normalization) for better performance in neural networks
scaler = MinMaxScaler(feature_range=(0, 1))
prices_scaled = scaler.fit_transform(prices.reshape(-1, 1))

# Create a function to prepare data for LSTM, converting the time series into a supervised learning problem
def prepare_data(series, n_steps):
    X, y = [], []
    for i in range(len(series)):
        # Find the end of this pattern
        end_ix = i + n_steps
        if end_ix > len(series)-1:
            break
        # Input and output patterns
        seq_x, seq_y = series[i:end_ix], series[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)

# Set the time window (n_steps) to predict future prices
n_steps = 60
X, y = prepare_data(prices_scaled, n_steps)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Reshape input to be 3D [samples, time steps, features] for LSTM
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))


# Build the LSTM Model
model = Sequential()

# Add LSTM layers
model.add(LSTM(units=50, return_sequences=True, input_shape=(n_steps, 1)))
model.add(Dropout(0.2))  # Regularization layer to prevent overfitting

model.add(LSTM(units=50, return_sequences=False))
model.add(Dropout(0.2))

# Add output layer
model.add(Dense(units=1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))
3. Model Testing and Evaluation
python
Copy code
# Evaluate the model on test data
test_loss = model.evaluate(X_test, y_test)

# Predict prices using the test set
predicted_prices = model.predict(X_test)

# Inverse transform predictions and actual values to get the original scale
predicted_prices = scaler.inverse_transform(predicted_prices)
actual_prices = scaler.inverse_transform(y_test.reshape(-1, 1))

# Visualize the prediction vs actual results
import matplotlib.pyplot as plt

plt.plot(actual_prices, color='blue', label='Actual Prices')
plt.plot(predicted_prices, color='red', label='Predicted Prices')
plt.title('Metal Price Prediction vs Actual')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()

def load_new_data(new_data_path):
    new_data = pd.read_csv(new_data_path)
    new_prices = new_data['Price'].values
    new_prices_scaled = scaler.transform(new_prices.reshape(-1, 1))
    
    # Prepare the new data in the required format
    X_new, _ = prepare_data(new_prices_scaled, n_steps)
    X_new = X_new.reshape((X_new.shape[0], X_new.shape[1], 1))
    
    # Predict using the pre-trained model
    predicted_new_prices = model.predict(X_new)
    predicted_new_prices = scaler.inverse_transform(predicted_new_prices)
    
    return predicted_new_prices

# Example of using real-time data for prediction
predictions = load_new_data('real_time_metal_prices.csv')
print("Predicted Prices for the New Data: ", predictions)

