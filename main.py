import os 
import numpy as np
import pandas as pd

from pandas import DataFrame
from bayes_opt import BayesianOptimization
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error

# define global variables
data_dtypes = {
    'fare_amount': 'float32',
    'pickup_datetime': 'float32',
    'pickup_longitude': 'float32',
    'pickup_latitude': 'float32',
    'dropoff_longitude' : 'float32',
    'dropoff_latitude' : 'float32',
    'passenger_count' : 'uint8'
}
# define the coordinates for the major airports in new york
lag_air_lat, lag_air_long = 40.776863, -73.874069
jfk_air_lat, jfk_air_long = 40.641766, -73.780968
ewr_air_lat, ewr_air_long = 40.689491, -74.174538
# define the columns to use for x and y variables
X_label = ['pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude', 
           'passenger_count', 'pickup_datetime_year', 'pickup_datetime_month', 'pickup_datetime_day', 
           'pickup_datetime_weekday', 'pickup_datetime_hour', 'lag_dist_pickup', 'lag_dist_dropoff', 
           'jfk_dist_pickup', 'jfk_dist_dropoff', 'ewr_dist_pickup', 'ewr_dist_dropoff', 
           'taxi_size_five_seater', 'taxi_size_six_seater', 'taxi_size_standard', 'rush_hour_pickup']
y_label = ['fare_amount']
# define the random state for reproducibility
random_state = 12345

def split_datetime(df:DataFrame)-> DataFrame: 
    """Split the datetime objects into their respective columns

    Args:
        df (DataFrame): initial dataframe

    Returns:
        DataFrame: updated dataframe with new columns
    """    
    df['pickup_datetime_year'] = df['pickup_datetime'].dt.year
    df['pickup_datetime_month'] = df['pickup_datetime'].dt.month
    df['pickup_datetime_day'] = df['pickup_datetime'].dt.day
    df['pickup_datetime_weekday'] = df['pickup_datetime'].dt.weekday
    df['pickup_datetime_hour'] = df['pickup_datetime'].dt.hour
    return df

def add_airport(df:DataFrame)-> DataFrame: 
    df['airport_dropoff'] = np.where(df['dropoff_longitude']==lag_air_long, 'lag', 
                                     (np.where(df['dropoff_longitude']==jfk_air_long, 'jfk',
                                               (np.where(df['dropoff_longitude']==ewr_air_long, 'ewr', '')))))
    df['airport_pickup'] = np.where(df['pickup_longitude']==lag_air_long, 'lag', 
                                     (np.where(df['pickup_longitude']==jfk_air_long, 'jfk',
                                               (np.where(df['pickup_longitude']==ewr_air_long, 'ewr', '')))))
    return df

def dist_from_airport(df:DataFrame)-> DataFrame: 
    """Calculate the euclidean distance from coordinates to airports

    Args:
        df (DataFrame): initial dataframe

    Returns:
        DataFrame: updated dataframe with new columns
    """    
    # calculate distance from laguardia
    df['lag_dist_pickup'] = np.sqrt(np.square(df['pickup_longitude'] - lag_air_long) + np.square(df['pickup_latitude'] - lag_air_lat))
    df['lag_dist_dropoff'] = np.sqrt(np.square(df['dropoff_longitude'] - lag_air_long) + np.square(df['dropoff_latitude'] - lag_air_lat))
    # calculate distance from jfk
    df['jfk_dist_pickup'] = np.sqrt(np.square(df['pickup_longitude'] - jfk_air_long) + np.square(df['pickup_latitude'] - jfk_air_lat))
    df['jfk_dist_dropoff'] = np.sqrt(np.square(df['dropoff_longitude'] - jfk_air_long) + np.square(df['dropoff_latitude'] - jfk_air_lat))
    # calculate distance from ewr
    df['ewr_dist_pickup'] = np.sqrt(np.square(df['pickup_longitude'] - ewr_air_long) + np.square(df['pickup_latitude'] - ewr_air_lat))
    df['ewr_dist_dropoff'] = np.sqrt(np.square(df['dropoff_longitude'] - ewr_air_long) + np.square(df['dropoff_latitude'] - ewr_air_lat))
    return df

def taxi_size(df:DataFrame)-> DataFrame: 
    """Classify the taxi size based on the number of passengers

    Args:
        df (DataFrame): initial dataframe

    Returns:
        DataFrame: updated dataframe with new columns
    """    
    df['taxi_size'] = np.where(df['passenger_count']<=4, 'standard',
                               (np.where(df['passenger_count']==5, 'five_seater', 'six_seater')))
    df = pd.get_dummies(df, prefix='taxi_size', columns=['taxi_size'], dtype=int) # perform one-hot encoding
    return df


def rush_hour(df:DataFrame)-> DataFrame: 
    """Classify the taxi trip within the rush hour based on the pickup time

    Args:
        df (DataFrame): initial dataframe

    Returns:
        DataFrame: updated dataframe with new column
    """    
    # using 1 and 0 similar to one-hot encoding
    df['rush_hour_pickup'] = np.where(
        ((((df['pickup_datetime_hour']>=8) & (df['pickup_datetime_hour']<=9)) | 
        ((df['pickup_datetime_hour']>=15) & (df['pickup_datetime_hour']<=19))) &
        (df['pickup_datetime_weekday']<5)), 1, 0)
    return df

def fare_outlier(df:DataFrame)-> DataFrame: 
    """Remove the fare outliers based on IQR

    Args:
        df (DataFrame): intial dataframe

    Returns:
        DataFrame: updated dataframe without outliers
    """    
    q3 = np.quantile(df['fare_amount'], 0.75)
    q1 = np.quantile(df['fare_amount'], 0.25)
    iqr = q3-q1
    lower_bound = q1 - 1.5*iqr
    upper_bound = q3 + 1.5*iqr
    df = df[(df['fare_amount']>lower_bound) & (df['fare_amount']<upper_bound)]
    return df

def objective_function(n_estimators:int, max_depth:int, min_samples_split:int, max_features:float):
    model = RandomForestRegressor(n_estimators=int(n_estimators),
                                  max_depth=int(max_depth),
                                  min_samples_split=int(min_samples_split),
                                  max_features=min(max_features, 0.999),
                                  random_state=random_state)
    return -1.0 * cross_val_score(model, X_train, np.ravel(y_train), cv=3, scoring="neg_mean_squared_error").mean()

def main(train_path:str, test_path:str, output_path:str)-> None:
    # load the training and testing data
    df_train = pd.read_csv(train_path, dtype=data_dtypes, parse_dates=['pickup_datetime'], nrows=10000)
    df_test = pd.read_csv(test_path, dtype=data_dtypes, parse_dates=['pickup_datetime'])

    # remove invalid fares and passenger_counts
    df_train = df_train[df_train['fare_amount']>=0]
    df_train = df_train[df_train['passenger_count']>0]

    # split the datetime into respective components
    df_train = split_datetime(df_train)
    df_test = split_datetime(df_test)

    # calculate euclidean distance from airports 
    df_train = dist_from_airport(df_train)
    df_test = dist_from_airport(df_test)

    # classify the taxi based on the number of passengers
    df_train = taxi_size(df_train)
    df_test = taxi_size(df_test)

    # classify trip based on rush hour timing
    df_train = rush_hour(df_train)
    df_test = rush_hour(df_test)

    # removal of outliers within fare amount
    df_train = fare_outlier(df_train)

    # split training into training and validation set 
    global X_train, X_eval, y_train, y_eval
    X, y = df_train[X_label], df_train[y_label]
    X_train, X_eval, y_train, y_eval = train_test_split(X, y, test_size=0.2, random_state=random_state)

    # instantiate hyper parameter bounds 
    hyper_param_bounds = {
        'n_estimators':(10,300),
        'max_depth': (1,50), 
        'min_samples_split':(2,30),
        'max_features':(0.1, 0.999),
    }

    # use bayesian optimization to find the best set of hyperparameters
    optimizer = BayesianOptimization(f=objective_function, pbounds=hyper_param_bounds, random_state=random_state)
    optimizer.maximize(init_points=10, n_iter=20)
    best_hyperparms = optimizer.max['params']
    
    # train the model with best hyperparmeters
    best_model = RandomForestRegressor(
        n_estimators=int(best_hyperparms['n_estimators']),
        max_depth=int(best_hyperparms['max_depth']),
        min_samples_split=int(best_hyperparms['min_samples_split']),
        max_features=best_hyperparms['max_features'],
        random_state=random_state
    )
    best_model.fit(X_train, np.ravel(y_train))

    # perform prediction on the evaluation set 
    y_pred = best_model.predict(X_eval)
    print(f"Mean Absolute Error of model on eval set: {mean_absolute_error(y_eval, y_pred)}")

    # retrain the model with complete training set before infering on test set
    best_model = RandomForestRegressor(
        n_estimators=int(best_hyperparms['n_estimators']),
        max_depth=int(best_hyperparms['max_depth']),
        min_samples_split=int(best_hyperparms['min_samples_split']),
        max_features=best_hyperparms['max_features'],
        random_state=random_state
    )
    best_model.fit(X, np.ravel(y))
    X_test = df_test[X_label]
    y_test_pred = best_model.predict(X_test)

    # create output file 
    df_output = pd.DataFrame()
    df_output['key'] = df_test['key']
    df_output['fare_amount'] = y_test_pred
    df_output.to_csv(output_path, index=False)
    return

if __name__ == "__main__":
    # cwd = os.getcwd()
    # train_path = os.path.join(cwd, 'data/train.csv') # instantiate the training path
    # test_path = os.path.join(cwd, 'data/test.csv') # instantiate the testing path
    # output_path = os.path.join(cwd, 'data/output_file.csv') # instantiate the output path
    train_path = '/kaggle/input/new-york-city-taxi-fare-prediction/train.csv'
    test_path = '/kaggle/input/new-york-city-taxi-fare-prediction/test.csv'
    output_path = 'submission.csv'
    main(train_path, test_path, output_path)