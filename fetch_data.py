import requests
import pandas as pd

def fetch_crypto_data(crypto_pair, start_date):

   #Fetch historical cryptocurrency data from CoinGecko API.

    # Extract cryptocurrency and currency from pair
    crypto, currency = crypto_pair.split('/')
    base_url = f"https://api.coingecko.com/api/v3/coins/{crypto.lower()}/market_chart"
    params = {
        "vs_currency": currency.lower(),
        "days": "365",  # Get data for the last 365 days
        "interval": "daily"
    }

    response = requests.get(base_url, params=params)
    if response.status_code != 200:
        raise Exception("Failed to fetch data: " + response.text)
    
    # Parse the response
    data = response.json()
    prices = data['prices']
    
    # Create a DataFrame
    df = pd.DataFrame(prices, columns=['Date', 'Close'])
    df['Date'] = pd.to_datetime(df['Date'], unit='ms')  # Convert timestamp to datetime
    df['Open'] = df['Close'].shift(1).fillna(df['Close'])
    df['High'] = df['Close'].rolling(2).max().fillna(df['Close'])
    df['Low'] = df['Close'].rolling(2).min().fillna(df['Close'])

    # Filter data based on start_date
    df = df[df['Date'] >= pd.to_datetime(start_date)]
    return df[['Date', 'Open', 'High', 'Low', 'Close']]

def calculate_metrics(data, lookback_days, lookahead_days):
    """
    Calculate trading metrics based on historical crypto data.
    """
    # Calculate metrics for the specified lookback and lookahead periods
    data[f'High_Last_{lookback_days}_Days'] = data['High'].rolling(window=lookback_days).max()
    data[f'Days_Since_High_Last_{lookback_days}_Days'] = (data['Date'] - data['Date'].shift(lookback_days)).dt.days
    data[f'%_Diff_From_High_Last_{lookback_days}_Days'] = ((data['Close'] - data[f'High_Last_{lookback_days}_Days']) / data[f'High_Last_{lookback_days}_Days']) * 100
    
    data[f'Low_Last_{lookback_days}_Days'] = data['Low'].rolling(window=lookback_days).min()
    data[f'Days_Since_Low_Last_{lookback_days}_Days'] = (data['Date'] - data['Date'].shift(lookback_days)).dt.days
    data[f'%_Diff_From_Low_Last_{lookback_days}_Days'] = ((data['Close'] - data[f'Low_Last_{lookback_days}_Days']) / data[f'Low_Last_{lookback_days}_Days']) * 100
    
    data[f'High_Next_{lookahead_days}_Days'] = data['High'].shift(-lookahead_days).rolling(window=lookahead_days).max()
    data[f'%_Diff_From_High_Next_{lookahead_days}_Days'] = ((data[f'High_Next_{lookahead_days}_Days'] - data['Close']) / data['Close']) * 100
    
    data[f'Low_Next_{lookahead_days}_Days'] = data['Low'].shift(-lookahead_days).rolling(window=lookahead_days).min()
    data[f'%_Diff_From_Low_Next_{lookahead_days}_Days'] = ((data[f'Low_Next_{lookahead_days}_Days'] - data['Close']) / data['Close']) * 100

    return data

if __name__ == "__main__":
    # Define the cryptocurrency pair and start date
    crypto_pair = "BinanceCoin/USD"
    start_date = "2023-10-30"  # Ensure this is within the last 365 days

    # Fetch crypto data
    try:
        crypto_data = fetch_crypto_data(crypto_pair, start_date)
        print(crypto_data.head())  # Display the first few rows

        # Process the data using the calculate_metrics function with a 7-day look-back and 5-day look-ahead period
        processed_data = calculate_metrics(crypto_data, 7, 5)

        # Add a 10-day Simple Moving Average (SMA)
        processed_data['SMA_10'] = processed_data['Close'].rolling(window=10).mean()

        # Save the processed data to an Excel file
        processed_data.to_excel("processed_crypto_data.xlsx", index=False)
        
        print("Data fetched and processed successfully.")
        
    except Exception as e:
        print(e)



