





# In this assignment students have to make ARIMA model over shampoo sales data and
# check the MSE between predicted and actual value.





import numpy as np
import pandas as pd
from pandas import read_csv
from pandas import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

from statsmodels.tsa.arima_model import ARIMA

from sklearn.metrics import mean_squared_error





get_ipython().run_line_magic('matplotlib', 'inline')


#  Load Data




df = pd.read_csv('Shampoo-Sales.csv')


# Understand Dataset and Data




df.head()





df.tail()





df = df.iloc[0:df.shape[0]-1,:]





df.tail()





df['Month'] = '190' + df['Month']
df.head()





df.Month = pd.to_datetime(df.Month,format ='%Y-%m')
df.head()





df.rename(mapper={'Sales of shampoo over a three year period': 'Shampoo Sales'},axis =1,inplace = True)
df.head()





series =  df['Shampoo Sales']





series.head()


# Creating a time series of  Shampoo Sales data with Month as Index




series.index = df.Month
series.head()





series.describe(include ='all')





series.index


# Plotting the time series chart of Shampoo Sales




plt.figure(figsize = (20,10))
plt.title('Shampoo Sales over a 3-year period')
plt.xlabel('Month')
plt.ylabel('Shampoo Sales')
series.plot()


# Analysis of Shampoo Sales Time-Series using HP-Filter




sales_cycle, sales_trend = sm.tsa.filters.hpfilter(series)





sales_cycle.head()





sales_trend.head()





sales_df = pd.DataFrame(data = series.values, index = series.index,columns = ['Sales'])
sales_df.head()





sales_df['Trend'] = sales_trend





sales_df['Cycle'] = sales_cycle





sales_df.head()





sales_df.plot(title ='Shampoo Sales, Sales Trend and Sales Cycle over a 3-year period',figsize = (20,10))





series.index = pd.to_datetime(series.index)


# EWMA( Exponentially Weighted Moving Average)




ewma_df = pd.DataFrame({'Actual Series':series, 'EWMA Series':series.ewm(span = 3).mean()})





ewma_df.plot(figsize = (20,10))


# Analysis of series to use ARIMA Model(p,d,q) 

#  Plotting rolling mean and standard deviation along with actual data
# 




series.rolling(3).mean().plot(label = '3 month rolling mean', figsize = (20,10))
series.rolling(3).std().plot(label = '3 month rolling standard deviation', figsize = (20,10))
series.plot(label = 'Actual data')
plt.legend()


#  Decomposing the series using ETS




from statsmodels.tsa.seasonal import seasonal_decompose





ets_result = seasonal_decompose(series, model = 'additive')





ets_result_fig = ets_result.plot()
ets_result_fig.set_size_inches(20,10)





ets_result_df = pd.DataFrame({'Trend': ets_result.trend, 'Seasonality': ets_result.seasonal,
                              'Residual': ets_result.resid,'Observed':ets_result.observed})





ets_result_df.plot(figsize=(20,10))


# There is trend and seasonality in this time series data

#  Checking the stationarity of the time series data (Using Augmented Dickey - Fuller Unit Root Test 




from statsmodels.tsa.stattools import adfuller





def adf_check(time_series):
    adfuller_result = adfuller(time_series)
    print('Augmented Dickey-Fuller Test:')
    labels = ['ADF Test Statistic','p-value','#Lags Used','Number of Observations Used']

    for value,label in zip(adfuller_result, labels):
        print(label + ": " + str(value))
    return adfuller_result





def find_stationarity_and_difference(time_series):
    
    for i in range(0,10):
        
        if(i==0):
            print('Actual Time Series')
        else:
            print(str(i)+'-Differenced Time Series')
            
        print('-' * 60)

        p_stationarity = adf_check(time_series)[1]   
        
        print("\nStationarity:")
        
        if(p_stationarity <= 0.05):
            print('Data is Stationary')
            break
        else:
            print('Data is Non-Stationary\n')
            time_series = (time_series - time_series.shift(1))
            time_series.dropna(inplace = True)
    return i, time_series





d,stationary_series = find_stationarity_and_difference(series)





print('No.of times differenced = ', d)
stationary_series.plot()





sm.stats.durbin_watson(series)





sm.stats.durbin_watson(stationary_series)


#  Plotting Autocorrelation and Partial Autocorrelation Plots

# Autocorrelation Plot using Pandas




from pandas.tools.plotting import autocorrelation_plot





autocorrelation_plot(series)





from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


# Autocorrelation and Partial Autocorrelation Plot




fig_first_acf = plot_acf(stationary_series)
fig_first_acf.set_size_inches(20,10)

fig_first_pacf = plot_pacf(stationary_series)
fig_first_pacf.set_size_inches(20,10)


# We can see in the Autocorrelation Plot, that there is sharp drop in the correlation after 2 lags.Therefore, we can choose, q = 2

#  We can see in the Partial Autocorrelation Plot, that there is sharp positive drop in the correlation at 4th lag.Therefore, we can choose,AR-k(k=4) for the autoregression model, i.e. we choose p = 4




from statsmodels.tsa.arima_model import ARIMA
model = ARIMA(series,order =(4,1,2)) 
arima_results = model.fit()
print(arima_results.summary())





residuals = pd.DataFrame(arima_results.resid)
residuals.plot(figsize=(10,10))
residuals.plot(kind='kde',figsize=(10,10) )





residuals.describe()





arima_results.aic





predicted_plot = arima_results.plot_predict()


# Forecast using ARIMA 




from sklearn.metrics import mean_squared_error

X = series.values
size = int(len(X) * 0.66)
train, test = X[0:size], X[size:len(X)]
history = [x for x in train]
predictions = list()
for t in range(len(test)):
    model = ARIMA(train, order=(4,1,2))
    model_fit = model.fit(disp=0)
    output = model_fit.forecast()
    yhat = output[0]
    predictions.append(yhat)
    obs = test[t]
    history.append(obs)
    print('predicted=%f, expected=%f' % (yhat, obs))
    
error = mean_squared_error(test, predictions)
print('Test MSE: %.3f' % error)

print ("Model aic score",model_fit.aic, "\nModel bic score", model_fit.bic)

# plot
plt.plot(test,color='blue', label='Actuals')
plt.plot(predictions, color='red', label='Rolling forecast' )
plt.legend()
plt.show()


# There is a problem with our forecast, as our rolling forecast is just a horizontal straight line. Therefore, try various values of p,q and select the model which gives the lowest AIC score

#  Trying to find p,q values (d =1) that give lowest AIC score




ararray = X

p=0
q=0
d=1
pdq=[]
aic=[]

for p in range(6):
    for q in range(4):
            try:
                model = ARIMA(ararray, (p,d,q)).fit()
                x = model.aic
                x1 = (p,d,q)

                print (x1, x)
                aic.append(x)
                pdq.append(x1)
            except:
                pass

keys = pdq
values = aic
d = dict(zip(keys, values))
minaic=min(d, key=d.get)

for i in range(3):
    p=minaic[0]
    d=minaic[1]
    q=minaic[2]
print ("Best Model is :", (p,d,q))
ARIMIAmod = ARIMA(ararray, (p,d,q)).fit()


# The model gives error for the values (4,1,2) and other values of p,d,q chosen by lower aic score. But works for (p,d,q) = (5,1,0)




X = series.values
size = int(len(X) * 0.66)
train, test = X[0:size], X[size:len(X)]
history = [x for x in train]
predictions = list()
for t in range(len(test)):
    model = ARIMA(history, order=(5,1,0))
    model_fit = model.fit(disp=-1)
    output = model_fit.forecast()
    yhat = output[0]
    predictions.append(yhat)
    obs = test[t]
    history.append(obs)
    print('predicted=%f, expected=%f' % (yhat, obs))
    
error = mean_squared_error(test, predictions)
print('Test MSE: %.3f' % error)

# plot
plt.plot(test,color='blue', label='Actuals')
plt.plot(predictions, color='red', label='Rolling forecast' )
plt.legend()
plt.show()





print('The model has the parameters: (p,d,q) = (5,1,0). Its an autoregressive model')

