import os
import datetime as dt
import pandas as pd
from mlforecast import MLForecast
from xgboost import XGBRegressor

from window_ops.rolling import rolling_mean, rolling_max, rolling_min
from mlforecast.feature_engineering import transform_exog
from sklearn.metrics import mean_absolute_error

def modelCreation():
    """
    Create XGBoost model based on parameters tuned 
    from notebook "Model Selection and Tuning".

    Parameters
    ----------
    None

    Returns
    -------
    A XGBoost Model
    """
    xgb_model = [XGBRegressor(
        learning_rate = 0.1,
        max_depth = 7,
        min_child_weight = 3,
        random_state = 42
    )]

    model = MLForecast(models=xgb_model,
                       freq='h',
                       lags=[1, 24],
                       lag_transforms={
                           1: [(rolling_mean, 24), (rolling_max, 24), (rolling_min, 24)],
                       },
                       num_threads=6)

    return model

def dataPreProcessing(data):
    """
    Preprocess the data DataFrame: 
    1. Rename the columns.
    2. Add a column named 'unique_id'
    3. Transform the features and add them in data.

    Parameters
    ----------
    A Pandas Dataframe containing cleaned sample data.

    Returns
    -------
    Transformed data based on procedures described above.
    """
    # Rename the columns for mlforecast training purposes.
    data.rename(columns={'Date_Time':'ds', 'Ontario_Demand': 'y'}, inplace=True)
    # Change datatype of column 'ds'.
    data['ds'] = pd.to_datetime(data['ds'])
    # Add a column named 'unique_id' for training in mlforecast.
    data['unique_id'] = 'id_00'
    # Transform features and add lag to data.
    transformed_features = transform_exog(data.drop(columns = ['Weekday']), lags=[1, 24])
    transformed_data = data[['unique_id', 'ds']].merge(transformed_features, on=['unique_id', 'ds'])
    return transformed_data

def main():
    # Get month and day from input.
    month_input = input('Enter month (Jul or Aug): ')
    day_input = input('Enter day of the month: ')

    # Handle exceptions.
    month_input = ''.join(month_input.split())
    if month_input.lower() == 'jul' or month_input.lower() == 'july':
        month = 7
    elif month_input.lower() == 'aug' or month_input.lower() == 'august':
        month = 8
    else:
        return "Please check entered month, must be July or August."
    if int(day_input) < 1 or int(day_input) > 31:
        return "Please enter a number between 1 and 31."
    else:
        day = int(day_input)
    # Open cleaned data.
    script_dir = os.path.dirname(__file__)
    abs_file_path = os.path.join(script_dir, 'Cleaned Dataset.csv')
    data = pd.read_csv(abs_file_path)
    # Transform data.
    transformed_data = dataPreProcessing(data)
    # Create model.
    XGBoost = modelCreation()
    if day == 1:
        if month == 7:
            train_end_date = dt.datetime(year = 2020, month = 6, day = 30, hour = 23)
        elif month == 8:
            train_end_date = dt.datetime(year = 2020, month = 7, day = 31, hour = 23)
    else:
        train_end_date = dt.datetime(year = 2020, month = month, day = day - 1, hour = 23)
    test_end_date = dt.datetime(year = 2020, month = month, day = day, hour = 23)
    
    train = transformed_data[transformed_data['ds'] <= str(train_end_date)]
    test = transformed_data[(transformed_data['ds'] > str(train_end_date)) & (transformed_data['ds'] <= str(test_end_date))]

    XGBoost.preprocess(train, dropna=True)
    XGBoost.fit(train, id_col='unique_id', time_col='ds', target_col='y', static_features=[])
    predictions = XGBoost.predict(24, X_df = test)
    print(f'The MAE is {mean_absolute_error(test["y"], predictions["XGBRegressor"])}')
    for i in range(24):
        print(f'At hour {i+1} of month {month} and day {day}, the predicted demand is {predictions["XGBRegressor"][i]}')
    print(f'The list output is {predictions["XGBRegressor"]}')

if __name__ == '__main__':
    main()
    
