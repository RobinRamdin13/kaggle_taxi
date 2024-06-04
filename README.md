# kaggle_taxi

This following code was develop to predict the taxi fare amount within New York. The data originates from the kaggle platform which can be found in [here](https://www.kaggle.com/competitions/new-york-city-taxi-fare-prediction/overview).

In this code we have used Random Forest Regression model and Bayesian Optimization to find the best performing hyperparameters. Two sets of models were trained, the first was evluated on the evaluation set and the second was used to predict on the test set. 

### Creating New Features 
#### Distance from Airports
There are three major airports within New York City, there coordinates were extracted from the web and used to compute the Euclidean distance between the pickup and dropoff point from the respective airports. This method was better than labelling whether a pick/dropoff was performed at an airport due to uncertainties within the coordinate system. 

This was performed under the hypothesis that trips to/from airports are generally higher priced as compared to the norm. 

#### Rush Hour Timing 
Another factor which can impact the fare amount is time. Extracting the approximate rush hour timing from the web, we classified whether a trip was started within the rush hour. This assumption is based on the fact that rush hour tends to have heavier traffic which can impact the amount paid by a client. 
`python -m venv env`

`python pip install -r requirements.txt`