import tkinter as tk
from tkinter import messagebox
import joblib
import requests
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Phase 2: Data Collection and Preprocessing

def fetch_weather_data(api_key, city="London", days=30):
    """
    Fetch historical weather data from OpenWeatherMap API.
    """
    url = f"http://api.openweathermap.org/data/2.5/forecast/daily?q={city}&cnt={days}&appid={api_key}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        
        # Extract relevant data
        weather_data = []
        for day in data['list']:
            weather_data.append({
                'date': day['dt'],
                'temperature': day['temp']['day'] - 273.15,  # Convert Kelvin to Celsius
                'humidity': day['humidity'],
                'pressure': day['pressure']
            })
        
        df = pd.DataFrame(weather_data)
        # Handle missing data
        df = df.fillna(df.mean())
        return df
    except requests.RequestException as e:
        messagebox.showerror("API Error", f"Failed to fetch data: {e}")
        return pd.DataFrame()

# Phase 3: Machine Learning Model Development

def train_and_save_model(weather_df, model_type='random_forest'):
    """
    Train and save a machine learning model.
    """
    if weather_df.empty:
        messagebox.showerror("Error", "No data available for training.")
        return
    
    X = weather_df[['humidity', 'pressure']]  # Features
    y = weather_df['temperature']             # Target

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    if model_type == 'random_forest':
        model = RandomForestRegressor(n_estimators=100, random_state=42)
    elif model_type == 'linear_regression':
        from sklearn.linear_model import LinearRegression
        model = LinearRegression()
    else:
        raise ValueError("Invalid model type. Choose 'random_forest' or 'linear_regression'.")

    model.fit(X_train, y_train)

    # Evaluate model
    y_pred = model.predict(X_test)
    print("Mean Absolute Error:", mean_absolute_error(y_test, y_pred))
    print("Root Mean Squared Error:", mean_squared_error(y_test, y_pred, squared=False))
    print("R2 Score:", r2_score(y_test, y_pred))

    # Save model
    joblib.dump(model, 'weather_model.pkl')

# Phase 4: Desktop App Development

def fetch_real_time_weather(api_key, city):
    """
    Fetch real-time weather data from OpenWeatherMap API.
    """
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        return {
            'temperature': data['main']['temp'] - 273.15,  # Convert Kelvin to Celsius
            'humidity': data['main']['humidity'],
            'pressure': data['main']['pressure']
        }
    except requests.RequestException as e:
        messagebox.showerror("API Error", f"Failed to fetch data: {e}")
        return None

def predict_weather():
    """
    Predict weather based on user input.
    """
    try:
        # Get user input
        city = city_entry.get()
        real_time_data = fetch_real_time_weather(api_key, city)
        if real_time_data:
            humidity = real_time_data['humidity']
            pressure = real_time_data['pressure']
            
            # Predict temperature
            prediction = model.predict([[humidity, pressure]])
            result_label.config(text=f"Predicted Temperature: {prediction[0]:.2f}°C\n"
                                  f"Humidity: {humidity}%\n"
                                  f"Pressure: {pressure} hPa")
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {e}")

# GUI Setup
app = tk.Tk()
app.title("Weather Prediction App")
app.geometry("300x200")

# Input fields
tk.Label(app, text="City").pack()
city_entry = tk.Entry(app)
city_entry.pack()

# Predict button
predict_button = tk.Button(app, text="Predict", command=predict_weather)
predict_button.pack(pady=10)

# Result display
result_label = tk.Label(app, text="")
result_label.pack()

# Load model (if available)
try:
    model = joblib.load('weather_model.pkl')
except FileNotFoundError:
    messagebox.showerror("Error", "Model file not found. Please train the model first.")
    model = None

# Example API key (replace with your own)
api_key = "YOUR_API_KEY"

# Run app
app.mainloop()