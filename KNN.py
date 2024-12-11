import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# Load the dataset
file_path = "filtered_nyc_data.csv"  # Relative path
try:
    filtered_nyc_data = pd.read_csv(file_path)
    print(filtered_nyc_data.head())  # Display the first few rows
except FileNotFoundError:
    print("The file 'filtered_nyc_data.csv' was not found. Ensure it is in the same directory as this script.")
except PermissionError:
    print("Permission denied. Ensure the file is not open in another program or locked.")
except Exception as e:
    print(f"An error occurred: {e}")

# Preprocessing and Feature Engineering
def preprocess_data(df):
    # Ensure column names are stripped of whitespace
    df.columns = df.columns.str.strip()
    
    # Print columns to verify
    print("Available columns:", list(df.columns))
    
    # Convert Date and Time to datetime
    df['Datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
    
    # Extract time-based features
    df['Hour'] = df['Datetime'].dt.hour
    df['Day_of_Week'] = df['Datetime'].dt.dayofweek
    df['Month'] = df['Datetime'].dt.month
    
    # Calculate total traffic 
    df['Total_Traffic'] = df['Entries']
    
    return df

# Prepare features and target
def prepare_data(df):
    # Select features for prediction
    feature_columns = ['Hour', 'Day_of_Week', 'Month', 'Entries']
    
    # Add categorical columns if they exist
    categorical_columns = ['Line Name', 'Division', 'Unit']
    for col in categorical_columns:
        if col in df.columns:
            # One-hot encode the column
            df = pd.get_dummies(df, columns=[col], prefix='', prefix_sep='')
            feature_columns.extend(df.columns[df.columns.str.startswith(col)])
    
    # Prepare features and target
    X = df[feature_columns]
    y = df['Total_Traffic']
    
    return X, y

# Train KNN Model
def train_knn_model(X, y, n_neighbors=5):
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train KNN Regressor
    knn = KNeighborsRegressor(n_neighbors=n_neighbors)
    knn.fit(X_train_scaled, y_train)
    
    # Predictions
    y_pred = knn.predict(X_test_scaled)
    
    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print("Model Performance:")
    print(f"Mean Squared Error: {mse}")
    print(f"Mean Absolute Error: {mae}")
    print(f"R-squared Score: {r2}")
    
    return knn, scaler, X.columns

# Predict Busiest Times
def predict_busiest_times(df, model, scaler, feature_columns, top_n=10):
    # Prepare features for prediction
    X = df[feature_columns]
    
    # Scale the features
    X_scaled = scaler.transform(X)
    
    # Predict traffic
    predicted_traffic = model.predict(X_scaled)
    
    # Create a DataFrame with predictions
    predictions_df = df.copy()
    predictions_df['Predicted_Traffic'] = predicted_traffic
    
    # Sort and get top busy times
    top_busy_times = predictions_df.nlargest(top_n, 'Predicted_Traffic')
    
    return top_busy_times

# Main Execution
def main():
    # Preprocess the data
    processed_df = preprocess_data(filtered_nyc_data)
    
    # Prepare features and target
    X, y = prepare_data(processed_df)
    
    # Train the KNN model
    knn_model, scaler, feature_columns = train_knn_model(X, y, n_neighbors=10)
    
    # Predict busiest times
    busiest_times = predict_busiest_times(processed_df, knn_model, scaler, feature_columns)
    
    # Print the busiest times
    print("\nPredicted Busiest Times:")
    print(busiest_times[['Datetime', 'Entries', 'Predicted_Traffic']])

# Run the main function
if __name__ == '__main__':
    main()