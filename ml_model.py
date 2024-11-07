import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# Load the data
data = pd.read_excel('processed_crypto_data.xlsx')

# Check for NaN values and fill them (or drop rows if necessary)
data.ffill(inplace=True)  # Forward fill for time series

# Function to train and evaluate model
def train_model(data, high_target, low_target):
    # Add additional features like lagged values
    data['High_Lag_1'] = data['High'].shift(1)
    data['Low_Lag_1'] = data['Low'].shift(1)
    
    # Remove NaNs generated from shifting
    data.dropna(inplace=True)

    # Define features and target variables
    X = data[['Days_Since_High_Last_7_Days', '%_Diff_From_High_Last_7_Days',
               'Days_Since_Low_Last_7_Days', '%_Diff_From_Low_Last_7_Days', 
               'SMA_10', 'High_Lag_1', 'Low_Lag_1']]  # Include lagged features here
    
    # Targets
    y_high = data[high_target]
    y_low = data[low_target]

    # Check for any NaN values in the data and handle them
    if X.isnull().any().any() or y_high.isnull().any() or y_low.isnull().any():
        print("NaN values found in the features or targets. Dropping them to avoid infinite results.")
        data.dropna(inplace=True)
        X = X.loc[data.index]
        y_high = y_high.loc[data.index]
        y_low = y_low.loc[data.index]

    # Train-test split
    X_train_high, X_test_high, y_train_high, y_test_high = train_test_split(X, y_high, test_size=0.2, random_state=42)
    X_train_low, X_test_low, y_train_low, y_test_low = train_test_split(X, y_low, test_size=0.2, random_state=42)

    # Define a RandomForestRegressor with hyperparameter tuning
    rf = RandomForestRegressor(random_state=42)
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_features': ['sqrt', 'log2', None],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
    }
    
    # Grid search function
    def perform_grid_search(X_train, y_train):
        grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1, verbose=0) #set verbose=0 to directly print the output without printing the grid search regressions
        grid_search.fit(X_train, y_train)
        return grid_search.best_estimator_
    
    # Train and predict for high target
    best_rf_high = perform_grid_search(X_train_high, y_train_high)
    y_pred_high = best_rf_high.predict(X_test_high)
    high_accuracy = best_rf_high.score(X_test_high, y_test_high)
    high_mae = mean_absolute_error(y_test_high, y_pred_high)
    high_mse = mean_squared_error(y_test_high, y_pred_high)

    # Train and predict for low target
    best_rf_low = perform_grid_search(X_train_low, y_train_low)
    y_pred_low = best_rf_low.predict(X_test_low)
    low_accuracy = best_rf_low.score(X_test_low, y_test_low)
    low_mae = mean_absolute_error(y_test_low, y_pred_low)
    low_mse = mean_squared_error(y_test_low, y_pred_low)

    # Check for infinity in predictions and model outputs
    if np.isinf(high_accuracy) or np.isinf(low_accuracy) or np.any(np.isinf(y_pred_high)) or np.any(np.isinf(y_pred_low)):
        print("Infinity detected in model outputs. Please check the data preprocessing steps.")
        return None

    # Print results
    print(f"High target - R^2: {high_accuracy:.2f}, MAE: {high_mae:.2f}, MSE: {high_mse:.2f}")
    print(f"Low target - R^2: {low_accuracy:.2f}, MAE: {low_mae:.2f}, MSE: {low_mse:.2f}")

    # Return predictions and models for further analysis
    return y_pred_high, y_pred_low, best_rf_high, best_rf_low

# Call the train_model function
high_predictions, low_predictions, model_high, model_low = train_model(data, '%_Diff_From_High_Next_5_Days', '%_Diff_From_Low_Next_5_Days')

# Example of outputting the predictions
if high_predictions is not None and low_predictions is not None:
    print("Predicted % Diff From High Next 5 Days:", np.mean(high_predictions))
    print("Predicted % Diff From Low Next 5 Days:", np.mean(low_predictions))