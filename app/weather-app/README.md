# Weather Prediction App

This is a simple weather prediction app for Windows. It uses historical weather data to train a machine learning model and predicts the temperature based on real-time weather data fetched from the OpenWeatherMap API.

## Installation

1. Install Python 3.x from [python.org](https://www.python.org/).
2. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/weather-app.git
   ```
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Get your API key from [OpenWeatherMap](https://openweathermap.org/api) and replace `YOUR_API_KEY` in `weather_app.py`.
2. Train the model (optional) by uncommenting the following lines in `weather_app.py`:
   ```python
   api_key = "YOUR_API_KEY"
   weather_df = fetch_weather_data(api_key)
   train_and_save_model(weather_df)
   ```
3. Run the app:
   ```bash
   python weather_app.py
   ```
4. Enter a city name and click "Predict" to see the weather prediction.

## Packaging

To create a standalone `.exe` file, run:
```bash
pyinstaller --onefile --windowed weather_app.py
```

The executable will be located in the `dist` folder.