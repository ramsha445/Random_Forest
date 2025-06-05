# Random_Forest
      import pandas as pd
      df = pd.read_csv("intern_data.csv")
      print(df.shape)           # Get number of rows and columns
      print(df.columns)         # Get column names
      print(df.head())          # View first 5 rows
      print(df.tail())          # View last 5 rows
      print(df.describe())      # Summary statistics
      print(df.info())
      print(df.isnull().sum())
      df.dropna(inplace=True)


# Check for missing values
     print(df.isnull().sum())

# Fill or drop missing values
    df.dropna(inplace=True)

# Normalize or scale if necessary (especially for models like XGBoost it helps)
    from sklearn.preprocessing import StandardScaler

    features = ['task_time', 'feedback_rating', 'attendance_rate']
    X = df[features]
    y = df['score']  # Replace with classification label if needed

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

     from sklearn.ensemble import RandomForestRegressor
     rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)



    from xgboost import XGBRegressor

    xgb = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    xgb.fit(X_train, y_train)

    from sklearn.metrics import mean_squared_error, r2_score

# Predict
    y_pred_rf = rf.predict(X_test)
    y_pred_xgb = xgb.predict(X_test)

# Evaluation
    print("Random Forest R2 Score:", r2_score(y_test, y_pred_rf))
    print("XGBoost R2 Score:", r2_score(y_test, y_pred_xgb))


    import numpy as np
    print("Random Forest RMSE:", np.sqrt(mean_squared_error(y_test, y_pred_rf)))
    print("XGBoost RMSE:", np.sqrt(mean_squared_error(y_test, y_pred_xgb)))

# Add predictions to DataFrame
    df['predicted_score'] = rf.predict(X_scaled)

# Display names with predicted scores
    print(df[['intern_name', 'predicted_score']])

# Define bins and labels (adjust ranges as per your score scale)
    bins = [0, 50, 75, 100]  # e.g., 0-50 = Struggling, 51-75 = Average, 76-100 = Excellent
    labels = ['Struggling', 'Average', 'Excellent']

# Create a new column for performance category based on predicted score
    df['performance_label'] = pd.cut(df['predicted_score'], bins=bins, labels=labels, right=True, include_lowest=True)

# Show intern names, predicted scores, and labels sorted by score
    print(df[['intern_name', 'predicted_score', 'performance_label']].sort_values(by='predicted_score', ascending=False))
