import pandas_datareader.data as web
import yfinance as yf
import pandas as pd
import numpy as np
import plotly_express as px
from datetime import datetime
yf.pdr_override()
#get the Tesla ('TSLA') data from yahoo finance
try:
    #TSLA_df = web.DataReader('TSLA', 'yahoo')
    data=web.get_data_yahoo("TSLA",start="2019-04-29",end="2024-04-26")
except Exception as e:
	#handle the exception
	print (f"Exception {e} occured. Data couldn't be loaded.")
else:
    print (f"Data loading was successful.")
print ("###############################Original dataframe################################")
print(data.info())
print(data["Close"].head())
#Calculate simple moving average (SMA) for window size 14 on closing prince
# and add them to a new column called SMA_5_Close
data['SMA_14_Close']=data['Close'].rolling(window=14,min_periods=1).mean()
data['SMA_14_EMA']=data['Close'].ewm(span=14).mean()
data['SMA_14_EMA_Alpha']=data['Close'].ewm(alpha=0.13).mean()
data['SMA_14_EWMA']=data['Close'].ewm(alpha=0.13,adjust=False).mean()
print(f"SMA_14_Alpha:{data.SMA_14_EMA_Alpha}")
#data['SMA_14_EWMA']=data['Close'].ewm(span=14,adjust=False).mean()
#Add a column for CMA
data['Close_CMA']=data['Close'].expanding().mean()
print(data.info())
#draw a plotly_express plot with closing price and 14 days SMA closing price
fig=px.line(data,y=['SMA_14_EMA','SMA_14_EMA_Alpha','SMA_14_EWMA'],title="TSLA Stock Analysis (Closing price VS SMA VS CMA")
#fig.add_scatter(data,y='SMA_14_Close')
fig.show()
