import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import base64
from flask import Flask, request, Response, jsonify
from functools import wraps
from datetime import datetime

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping, ReduceLROnPlateau

app = Flask(__name__)
CSV_FILE_PATH = 'data.csv' 
# CSV_FILE_PATH = 'BTC-data.csv'
latest_signal = "0"
API_KEYS = {'global': 'iucap'}
EPOCHS = 60

def decode_passkey(encoded_key):    
    decoded_key = base64.b64decode(encoded_key).decode('utf-8')
    return decoded_key

def authenticate_request(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        api_key = request.headers.get('X-API-KEY')

        if not api_key:
            return jsonify({"error": "Missing API Key"}), 400
        
        decoded_key = decode_passkey(api_key)
        
        if decoded_key != API_KEYS['global']:
            return jsonify({"error": "Invalid API key..."}), 403
        
        return f(*args, **kwargs)
    return wrapper

def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path)
    
    df['Date'] = pd.to_datetime(df['Date'], format='ISO8601', errors='coerce')

    for col in ['Close', 'Open', 'High', 'Low']:
        df[col] = df[col].astype(str).str.replace(',', '').astype(float)
   
    df.fillna(method='ffill', inplace=True)

    def calculate_rsi(data, period=14):
        delta = data['Close'].diff()
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)

        avg_gain = pd.Series(gain).ewm(span=period, adjust=False).mean()
        avg_loss = pd.Series(loss).ewm(span=period, adjust=False).mean()

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    df['RSI'] = calculate_rsi(df)
    df.dropna(inplace=True)

    data = df[['Close', 'RSI', 'Volume']].values
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    return df, scaled_data, scaler
# Function to prepare the dataset for LSTM
def create_dataset(data, look_back=60):
    X, y = [], []
    for i in range(look_back, len(data)):
        X.append(data[i-look_back:i])
        y.append(data[i, 0])  # Predicting the 'Price' column
    return np.array(X), np.array(y)
# Function to build and train the model
def build_and_train_model(X_train, y_train, X_test, y_test):
    model = Sequential([
        LSTM(units=64, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
        Dropout(0.2),
        LSTM(64, return_sequences=True, kernel_regularizer=l2(0.01)),
        Dropout(0.2),
        LSTM(32, return_sequences=False, kernel_regularizer=l2(0.01)),
        Dropout(0.2),
        Dense(units=1)
    ])
    # model.compile(optimizer='RMSprop', loss='mean_absolute_error')
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_absolute_error')

    checkpoint_dir = "training_checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)

    checkpoint_callback = ModelCheckpoint(
        filepath=os.path.join(checkpoint_dir, "ckpt_{epoch}.keras"),
        save_weights_only=False        
    )

    # Reduce learning rate if it plateaus
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,  
        patience=5,  
        min_lr=1e-6,  
        verbose=1  
    )

    early_stopping_callback = EarlyStopping(
        monitor='val_loss',
        patience=(15), 
        restore_best_weights=True,  # Restore best weights from the epoch with the lowest validation loss
        verbose=1
    )

    history = model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=32,
        validation_data=(X_test, y_test),
        callbacks=[checkpoint_callback, early_stopping_callback, reduce_lr]
    )

    return model
# Function to generate trading signals
def generate_signals(model, X_test, y_test, scaler):
    predictions = model.predict(X_test)
    predictions = scaler.inverse_transform(np.concatenate((predictions.reshape(-1, 1), X_test[:, -1, 1:]), axis=1))[:, 0]
    y_test_actual = scaler.inverse_transform(np.concatenate((y_test.reshape(-1, 1), X_test[:, -1, 1:]), axis=1))[:, 0]

    signals = np.where(predictions > y_test_actual, 1, -1)
    return signals, predictions, y_test_actual
# Function to process data and generate trading signal
def process_and_generate_signal():

    global latest_signal

    file_path = 'data.csv'  
    df, scaled_data, scaler = load_and_preprocess_data(file_path)
    X, y = create_dataset(scaled_data)
    
    # Split into train and test
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    # Build and train model
    model = build_and_train_model(X_train, y_train, X_test, y_test)
    
    # Generate signal
    signals, predictions, actual_values = generate_signals(model, X_test, y_test, scaler)
    
    # Return the most recent signal (you can change this logic as needed)
    recent_signal = signals[-1]
    prediction = predictions[-1]
    actual_value = actual_values[-1]
    latest_signal = recent_signal

    print(f"Updated latest_signal: {latest_signal}")  # Log the update

# endpoints
@app.route("/authenticate", methods=["POST"])
@authenticate_request
def authenticate():
    return jsonify({"message": "Authentication Success"}), 200

@app.route('/signal', methods=['GET'])
@authenticate_request
def get_signal():
    global latest_signal 
    sent_signal = latest_signal
    latest_signal = 0  

    return str(sent_signal), 200

@app.route('/update', methods=['POST'])
@authenticate_request
def update():
    global latest_signal
    try:        
        data = request.form['text']
        data = data.replace('\x00', '')  
        data_parts = data.split(';')
                
        symbol = data_parts[0]
        open_price = float(data_parts[1])
        high = float(data_parts[2])
        low = float(data_parts[3])
        close = float(data_parts[4])
        volume = float(data_parts[5])
        bid = float(data_parts[6])
        ask = float(data_parts[7])
        
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        # Save to csv
        with open(CSV_FILE_PATH, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([symbol, open_price, high, low, close, volume, bid, ask, timestamp])
        # Run and train model 
        process_and_generate_signal()

        return "Data received and saved successfully.", 200
    except Exception as e:
        return f"Error: {str(e)}", 500

if __name__ == "__main__":
    # process_and_generate_signal()
    app.run(host='0.0.0.0', port=5000, debug=True)
    
