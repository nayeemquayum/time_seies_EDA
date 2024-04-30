import numpy as np
import statsmodels.api as sms
from statsmodels.tsa.stattools import adfuller
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
from datetime import datetime
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

import plotly_express as px

#this function checks the adf result and returns whether data is stationary or not
def stationarrity_check(adf_test_result):
    is_stationary=False
    critical_value=adf_test_result[0]
    p_value=adf_test_result[1]
    t_value_5_CI=adf_test_result[4]["5%"]
    if critical_value < t_value_5_CI:
        is_stationary=True
        print("Reject Null Hypothesis- Time series is stationary")
    else:
        is_stationary=False
        print(" Failed to reject Null Hypothesis- Time series is non-stationary")
    return is_stationary


#load the airlines data
airline_df=pd.read_csv('data/airline_passengers.csv',header=0,parse_dates=[0], index_col=0)
print(airline_df.info())

#rename the column Thousands of Passengers to Passengers_in_Thousands
#Rename columns
airline_df.rename(columns={'Thousands of Passengers' : 'Passengers_in_Thousands'},\
                    inplace=True)
#first make data stationary
I_d=0
is_stationary=False
while is_stationary==False :
     I_d=I_d+1
     print(f"Doing integration step {I_d}")
     # print(airline_df.head())
     # print(airline_df.tail())
     if (I_d==1):
        #since this is the first differenciation step we use the original data from the df
        # shift the passanger information row by one position and find difference
        airline_df['difference_data'] = airline_df['Passengers_in_Thousands'] - airline_df['Passengers_in_Thousands'].shift(1)
        #update shifted data information so that in next round it can be used
        airline_df['shifted_data'] = airline_df['Passengers_in_Thousands'].shift(1)
        # drop Nan
        airline_df.dropna(subset=['difference_data'], inplace=True)
     else:
        #since this not the first differenciation step, we use shifted data from previous step
        #shift data from previously shifted data and calculate the difference
        airline_df['difference_data'] = airline_df['shifted_data']-airline_df['shifted_data'].shift(1)
        #update the latest shifted data
        airline_df['shifted_data'] = airline_df['shifted_data'].shift(1)
        #drop Nan
        airline_df.dropna(subset=['difference_data'],inplace=True)
     #run adf test
     adf_result = adfuller(airline_df['difference_data'],autolag='AIC')
     is_stationary= stationarrity_check(adf_result)
#draw a line graph with original data and itegrated data(after making it stationary
fig=px.line(airline_df,y=['Passengers_in_Thousands','difference_data'])
#fig.show()
#For ARIMA (use difference data for doing acg and pacf
fig = plt.figure(figsize=(12,8))
ax1= fig.add_subplot(2,1,1)
plot_pacf(airline_df['difference_data'],ax=ax1)
ax2 = fig.add_subplot(2,1,2)
plot_acf(airline_df['difference_data'],ax=ax2)
plt.show()
#For ARIMA, by observing pacf, we pick AR(p)= 2
AR_p=2
#by oserving acf, we pick MA(q)=
MA_q=4
print(f"We chose I(d)={I_d}, AR(p)={AR_p}, MA(q)={MA_q}")
#split data set to train and test the model
#for training the mode, we'll take 70% of data. We have 144 rows,
# so we are picking 100 rows for training the data
train_data=airline_df.head(100)
print(f"training data shape:{train_data.shape}")
#for testing the model we are taking remaining 40 rows
test_data=airline_df.tail(40)
print(f"test data shape:{test_data.shape}")
#build the ARIMA model
arima_model=ARIMA(train_data['Passengers_in_Thousands'],order=(AR_p,I_d,MA_q))
arima_model_fit = arima_model.fit()
print(arima_model_fit.summary())
#now do the prediction
prediction_start_date=test_data.index[0]
prediction_end_date=test_data.index[-1]
arima_prediction=arima_model_fit.predict(start=prediction_start_date,end=prediction_end_date)
test_data['ARIMA_prediction']=arima_prediction

fig=px.line(test_data,y=['Passengers_in_Thousands','ARIMA_prediction'])
fig.show()

