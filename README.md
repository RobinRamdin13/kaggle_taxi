# kaggle_taxi

This following code was develop to predict the taxi fare amount within New York. The data originates from the kaggle platform which can be found in [here](https://www.kaggle.com/competitions/new-york-city-taxi-fare-prediction/overview).

In this code we have used Random Forest Regression model and Bayesian Optimization to find the best performing hyperparameters. Two sets of models were trained, the first was evluated on the evaluation set and the second was used to predict on the test set.

___

### Creating New Features 
#### Distance from Airports
There are three major airports within New York City, there coordinates were extracted from the web and used to compute the Euclidean distance between the pickup and dropoff point from the respective airports. This method was better than labelling whether a pick/dropoff was performed at an airport due to uncertainties within the coordinate system. 

This was performed under the hypothesis that trips to/from airports are generally higher priced as compared to the norm. The heatmaps below show the pickup and dropoff in addition to the ariports being marked. 
![pickup_plot](https://github.com/RobinRamdin13/kaggle_taxi/blob/main/plots/plot_pickup_heatmap.html)

#### Rush Hour Timing 
Another factor which can impact the fare amount is time. The rush hour timings were extracted from the data. The plot below shows the number of taxi picked up grouped by their respective hours in the form of a bar chart, the red line signifies the average number of taxi called for every hour. Using this we are able to observe the rush hour timings to be between 0800hrs - 0900hrs, 1100hrs - 1500hrs and 1700hrs - 2300hrs. We categorize these timings as being under rush hour due to the higher than normal number of taxi called, this assumption is based on the fact that rush hour tends to have heavier traffic which can impact the amount paid by a client.
![taxi_distribution](https://github.com/RobinRamdin13/kaggle_taxi/blob/main/plots/rush_hour_exp.jpeg)
___
### Running the Code 
#### Creating Virtual Environment
To create the virtual environment run the following code in your terminal, you can rename `env` to any name you want for your virtual environment.`python -m venv env`.

In the event the virtual environment has not yet been activate, you need to run the following command: `env\Scripts\activate.bat`. This might defer based on which machine you're using, I was using Visual Studio Code on a Windows and the command prompt as terminal. 

#### Install all the dependencies 
After creating the virtual environment, run the following command, this will download all the required libraries required to replicate the code. `python pip install -r requirements.txt`

#### Executing main.py
To run the main file run the following command within your terminal `python main.py`.

___
### Comments and Contribution 
This was my first project for Kaggle, any comment or contribution are welcome.