STOCK PRICE FORECASTING USING LSTM

Project Description

This project aims to predict future stock prices using Long Short-Term Memory (LSTM) networks, a powerful deep learning model known for handling sequential data effectively. The stock market is a dynamic platform for buying and selling company stocks, significantly impacting individual wealth and the broader economy. Each Stock Exchange has its own Stock Index, an average value derived from several stocks, serving as a market barometer to predict movements over time. Efficient stock trend forecasting can minimize financial risks and maximize profits. By preparing historical stock data, designing the model architecture, and tuning its parameters, this project demonstrates the effectiveness of LSTM networks for time series forecasting in the financial domain, highlighting their potential to assist in informed stock market decisions.

The analysis was conducted using Python, utilizing key libraries such as Pandas for data manipulation, Scikit-learn for preprocessing, and TensorFlow/Keras for building and training the LSTM model.

Table of Contents

- [Installation]
- [Usage]
- [Dataset]
- [Model Architecture]
- [Results]
- [Contributing]
- [Contact Information]

Installation

To run this project locally, ensure Python is installed. It's recommended to use a virtual environment. Install the required libraries using `pip` and the provided `requirements.txt` file:

1. Clone this repository to your local machine.
2. Navigate to the project directory in your terminal.
3. (Optional) Create a virtual environment: `python -m venv venv`
4. (Optional) Activate the virtual environment:
   - Windows: `venv\Scripts\activate`
   - macOS and Linux: `source venv/bin/activate`
5. Install the required packages: `pip install -r requirements.txt`

Usage

The project's code is contained within a Jupyter Notebook, guiding you through the entire process from data loading to visualization.

1. Follow the [Installation](#installation) steps and ensure the required libraries are installed.
2. Navigate to the project's main directory in your terminal.
3. Start the Jupyter Notebook server by running: `jupyter notebook`
4. Your web browser will open a new tab displaying the Jupyter Notebook interface.
5. Navigate to the `notebooks` folder and open the `stock_prices_forecasting.ipynb` file.
6. Run the cells sequentially from top to bottom to execute the code and view the results.

Dataset

The dataset, sourced from DHVentech, contains historical stock market data. The dataset file, `stock_data.csv`, is located in the `data/` directory of this repository.

The dataset includes the following columns:

- `Date`: The trading date.
- `Open`: The stock price at the beginning of the trading session.
- `High`: The highest price reached during the trading session.
- `Low`: The lowest price reached during the trading session.
- `Close`: The final price when the market closed (used as the target variable for forecasting).
- `Volume`: The number of shares traded that day.
- `OpenInt`: Open Interest (not relevant for this dataset as values are 0).

Model Architecture

The project utilizes a Long Short-Term Memory (LSTM) network for forecasting. The most effective architecture includes a single LSTM layer followed by a Dense output layer.

The model architecture is defined as follows:

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

model = Sequential([
    LSTM(units=200, return_sequences=False, input_shape=(X_train.shape[1], 1)),
    Dense(units=1)
])

model.compile(optimizer='adam', loss='mean_squared_error')
```

- The LSTM layer has 200 units to capture complex temporal patterns.
- `return_sequences=False` is used as we are predicting a single value (the next day's price).
- The `input_shape` is set to `(n_steps, 1)`, where `n_steps` is the number of past days used in each sequence (30 in this project), and 1 is the number of features (the scaled 'Close' price).
- The final Dense layer with 1 unit provides the single predicted output.
- The model is compiled using the 'adam' optimizer and 'mean_squared_error' as the loss function.

Results

The model was trained on historical data and evaluated on a held-out test set, with performance measured using the Root Mean Squared Error (RMSE).

The final model achieved an RMSE of 0.5366 on the test dataset, indicating good accuracy in predicting stock prices based on historical data.

A visualization comparing actual stock prices and the model's predictions on the test set is included in the Jupyter Notebook for visual assessment of forecasting performance.

Contributing

This is a personal project, but suggestions for improvement are welcome. If you have ideas for enhancements, you can fork the repository and submit a pull request.

Contact Information

For questions or to connect, reach me at: nguyaisharon@gmail.com.