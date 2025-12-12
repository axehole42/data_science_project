import yfinance as yf
import pandas as pd
import sklearn as skl
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression


scaler = StandardScaler() # used to demean 
model = LinearRegression()


data = [0,1,2,3,4,5]

scaler.fit(data) #   ed to standardize our data - dont use the information from the test set

x_train, x_test = X[:split_index], X[:split_index]
Y_train, y_test = Y[:split_index], Y[:split_index] #helps with splitting train test
#so its important to scale the correct data - I need to look into that again - scale only the test data? I will look into it
# adding lagged features just like in stata variable.1, variable.2 (lagging by one or two periods)

data["Close_Lag_1"] = data["Close"].shift(1)  #This