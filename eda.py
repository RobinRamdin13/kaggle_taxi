import os 
import pandas as pd
import matplotlib.pyplot as plt 

from main import * 
from pandas import DataFrame

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

def rush_hour_exp(df:DataFrame)-> None:
    df_group = df.groupby(by=['pickup_datetime_hour'])['pickup_datetime_hour'].count()
    ax = df_group.plot(kind='bar')
    ax.axhline(y=round(df_group.mean()), color='r', linestyle='--', label='Mean Taxi Count')
    plt.ylabel('Taxi Count')
    plt.title('Taxi Distribution over Time')
    plt.legend()
    plt.grid()
    plt.savefig('plots/rush_hour_exp.jpeg')
    return

def main(train_path:str)-> None:
    df_train = pd.read_csv(train_path, dtype=data_dtypes, parse_dates=['pickup_datetime'], nrows=10000)

    # remove invalid fares and passenger_counts
    df_train = df_train[df_train['fare_amount']>=0]
    df_train = df_train[df_train['passenger_count']>0]

    # feature addition
    df_train = split_datetime(df_train)
    df_train = dist_from_airport(df_train)
    df_train = taxi_size(df_train)
    df_train = fare_outlier(df_train)

    # rush hour exploration
    rush_hour_exp(df_train)
    # print(df_train.columns.tolist())
    return

if __name__ == "__main__":
    cwd = os.getcwd()
    train_path = os.path.join(cwd, 'data/train.csv') # instantiate the training path
    if not os.path.exists('plots'): os.mkdir('plots')
    main(train_path)