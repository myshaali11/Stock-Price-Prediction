import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import r2_score

# Load each dataset
spcompanies = pd.read_csv("sp500_companies.csv")
spindex = pd.read_csv("sp500_index.csv")
sp500_stocks = pd.read_csv("sp500_stocks.csv")
stocks = pd.read_csv("Stocks.csv")

''' # Print row/column counts
print("sp500_companies:", spcompanies.shape)
print("sp500_index:", spindex.shape)
print("sp500_stocks:", sp500_stocks.shape)
print("Stocks.csv:", stocks.shape)
'''
# Drop rows with missing values
sp500_stocks = sp500_stocks.dropna()

# Select relevant columns
sp500_stocks = sp500_stocks[['Date', 'Symbol', 'Close', 'Volume']]

# Convert 'Date' to datetime format
sp500_stocks['Date'] = pd.to_datetime(sp500_stocks['Date'])

symbol = input("Enter the stock symbol (e.g., NFLX,ORCL): ").strip().upper()
stock_data = sp500_stocks[sp500_stocks['Symbol'] == symbol][['Date', 'Symbol', 'Close', 'Volume']]

if stock_data.empty:
    print(f"Error: No data found for symbol '{symbol}'. Please check the symbol and try again.")
else:
    print(f"Data loaded for symbol '{symbol}'.")

# Sort by date
stock_data = stock_data.sort_values('Date')

# Check for missing values in the filtered dataset
print("Missing values:\n", stock_data.isnull().sum())

# Check the shape of the dataset (rows and columns)
print("Shape of stock_data:", stock_data.shape)
'''
# print(stock_data)

stock_data['Date'] = pd.to_datetime(stock_data['Date'])
'''
# Plot the stock prices
plt.figure(figsize=(10, 5))
plt.plot(stock_data['Date'], stock_data['Close'], label='Close Price', color='blue')
plt.xlabel('Date')
plt.ylabel('Stock Price (Close)')
plt.title('Stock Price Trend')
plt.legend()
plt.show()
'''
stock_data.set_index('Date', inplace=True)

# Use the 'Close' prices for prediction
stock_data = stock_data[['Close']]

# Create features and target
stock_data['Target'] = stock_data['Close'].shift(-1)  # Predict the next day's price
stock_data.dropna(inplace=True)

X = stock_data[['Close']]
y = stock_data['Target']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, predictions)
mae=mean_absolute_error(y_test, predictions)
print(f"Mean Squared Error: {mse}")
print(f"Mean Absolute Error:  {mae}")

plt.figure(figsize=(10, 5))
plt.plot(y_test.index, y_test, label='Actual Prices', color='blue')
plt.plot(y_test.index, predictions, label='Predicted Prices', color='red')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.title('Actual vs. Predicted Stock Prices')
plt.legend()
plt.show()
'''
# Set 'Date' as index
stock_data.set_index('Date', inplace=True)

# Use the 'Close' prices for prediction
stock_data = stock_data[['Close']]

# Create features and target
stock_data['Target'] = stock_data['Close'].shift(-1)  # Predict the next day's price
stock_data.dropna(inplace=True)

# Features (X) and Target (y)
X = stock_data[['Close']]
y = stock_data['Target']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, predictions)
mae = mean_absolute_error(y_test, predictions)
print(f"Mean Squared Error: {mse}")
print(f"Mean Absolute Error: {mae}")
r2 = r2_score(y_test, predictions)
print(f"R-squared: {r2}")

# Plot the actual vs predicted stock prices
plt.figure(figsize=(12, 6))
plt.plot(y_test.index, y_test, label='Actual Prices', color='blue', linewidth=2)
plt.plot(y_test.index, predictions, label='Predicted Prices', color='red', linestyle='--', linewidth=2)
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.title('Actual vs. Predicted Stock Prices')
plt.legend()
plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
plt.tight_layout()
plt.show()