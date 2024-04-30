import numpy as np
import statsmodels.api as sms
from statsmodels.tsa.stattools import adfuller
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
from datetime import datetime
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
#For SARIMAX we do 1 integration with 12 position shift
airline_df['seasonal_difference'] = airline_df['Passengers_in_Thousands']-airline_df['Passengers_in_Thousands'].shift(12)
airline_df.dropna(subset=['seasonal_difference'],inplace=True)

#draw a line graph with original data and itegrated data(after making it stationary
fig=px.line(airline_df,y=['Passengers_in_Thousands','seasonal_difference'])
#fig.show()
#Check whether data is stationary or not
sarimax_adf_result = adfuller(airline_df['seasonal_difference'],autolag='AIC')
is_sarimax_stationary= stationarrity_check(sarimax_adf_result)
if is_sarimax_stationary:
    print(f"After removing seasonality factor, SARIMAX integrated data is stationary")
else:
    print(f"After removing seasonality factor, SARIMAX integrated data is still non-stationary")
#To calculate AR(p) and MA(q) do acf and pacf plot
fig2 = plt.figure(figsize=(12,8))
ax1= fig2.add_subplot(2,1,1)
plot_pacf(airline_df['seasonal_difference'],ax=ax1)
ax2 = fig2.add_subplot(2,1,2)
plot_acf(airline_df['seasonal_difference'],ax=ax2)
#plt.show()
#For SARIMAX, by observing pacf we pic sarimax_AR(p)=2
sarimax_AR_p=2
# #by oserving acf, we pick sarimax_MA(q)= 5
sarimax_MA_q=5
sarimax_I_d=1
print(f"For SARIMAX, we chose I(d)={sarimax_I_d}, AR(p)={sarimax_AR_p}, MA(q)={sarimax_MA_q}")
#split data set to train and test the model
print(f"SARIMAX data frame shape:{airline_df.shape}")
#for training the mode, we'll take 70% of data. We have 132 rows,
#so we are picking 94 rows for training the data
sarimax_train_data=airline_df.head(94)
sarimax_test_data=airline_df.tail(38)

#Now we do the SARIMAX mode
sarimax_model=SARIMAX(sarimax_train_data['Passengers_in_Thousands'],order=(sarimax_AR_p,sarimax_I_d,sarimax_MA_q),\
                      seasonal_order=(sarimax_AR_p,sarimax_I_d,sarimax_MA_q,12))
sarimax_model_fit = sarimax_model.fit()
#Do the prediction for SARIMAX
prediction_start_date=sarimax_test_data.index[0]
prediction_end_date=sarimax_test_data.index[-1]
sarimax_prediction=sarimax_model_fit.predict(start=prediction_start_date,end=prediction_end_date)
sarimax_test_data['SARIMAX_prediction']=sarimax_prediction
print(f"sarima test restul:{sarimax_test_data.head(50)}")
#draw a line graph with SARIMA predicted data against observed data(from test data set)
SARIMA_fig= px.line(sarimax_test_data,
                   y=['Passengers_in_Thousands','SARIMAX_prediction'])
SARIMA_fig.show()
