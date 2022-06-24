# -*- coding: utf-8 -*-
"""
Created on Fri Jun 17 10:08:22 2022

@author: Alexis
"""

#Import du data set et des libs
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import sklearn.metrics
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

#chargement du jeu de donnees + mise en forme
flights = sns.load_dataset('flights')
data = flights.copy()
data['month'] = pd.to_datetime(data.month,format='%b').dt.strftime('%m')
cols=["year","month"]
data['date'] = data[cols].apply(lambda x: '-'.join(x.values.astype(str)), axis="columns")
data.index = pd.to_datetime(data['date'])
data.set_index('date').asfreq('m')
data.drop(['date', 'year', 'month'], axis=1, inplace=True)

#visualisation 1 - 'tendance (moyenne) = croissance' --> non stationnaire
data.plot(y="passengers", c='red', lw=2)

#visualisation 2 - decomposition
from statsmodels.tsa.seasonal import seasonal_decompose
decompose_data = seasonal_decompose(data, model="additive")
decompose_data.plot();

seasonality=decompose_data.seasonal
seasonality.plot(color='green')

#visualisation 3 - tendance 'régression OLS'
from statsmodels.api import OLS
y = data.passengers
X = np.ones((len(y), 2))
X[:,1] = np.arange(0,len(y))
reg = OLS(y,X)
results = reg.fit()
results.params

from statsmodels.graphics.regressionplots import abline_plot
fig = abline_plot(model_results=results)
ax = fig.axes[0]
ax.plot(X[:,1], y, 'b')
ax.margins(.1)

#Vérifier stationnarité série avec ADF + statistiques mobiles
def get_stationarity(timeseries):
    
    # Statistiques mobiles
    rolling_mean = timeseries.rolling(window=12).mean()
    rolling_std = timeseries.rolling(window=12).std()
    
    # tracé statistiques mobiles
    original = plt.plot(timeseries, color='blue', label='Origine')
    mean = plt.plot(rolling_mean, color='red', label='Moyenne Mobile')
    std = plt.plot(rolling_std, color='black', label='Ecart-type Mobile')
    plt.legend(loc='best')
    plt.title('Moyenne et écart-type Mobiles')
    plt.show(block=False)
    
    # Test Dickey–Fuller :
    from statsmodels.tsa.stattools import adfuller    
    result = adfuller(timeseries['passengers'])
    print('Statistiques ADF : {}'.format(result[0]))
    print('p-value : {}'.format(result[1]))
    print('Valeurs Critiques :')
    for key, value in result[4].items():
        print('\t{}: {}'.format(key, value))
        #Est-ce que la statistique ADF est loin des valeurs critiques et la p-value > 0.05%?
  
#est-ce que moyenne mobile et écart-type augmentent avec temps?
get_stationarity(data)


#autocorrélation (lag) = 
#negative autocorelation = plus x.t est élevé, plus x.t+n sera faible:
#plus les passagers voyagent en périodes estivales, moins ils voyageront le reste de l'année
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
plot_acf(data, lags=24) #cycle de 12 mois --> détermine moyenne mobile (MA)
plot_pacf(data, lags=24) #autocorrelation d'ordre 1 --> détermine ordre du modèle (AR)
#autocorrélation partielle = tester la correlation entre n et n+K indépendemment des n<n+k<n+K
       

#TRANSFORMATION SERIE TEMPORELLE EN SERIE STATIONNAIRE, 3 SOLUTIONS!
#Log de Y réduit le taux d'augmentation de la moyenne mobile
df_log = np.log(data)
plt.plot(df_log)
get_stationarity(df_log) #élimine la variabilité croissante        
plot_acf(df_log, lags=12) 
plot_pacf(df_log, lags=12)

#1 - Soustraction moyenne mobile (enlève les pics de saisonnalité)
rolling_mean = df_log.rolling(window=12).mean()
df_log_minus_mean = df_log - rolling_mean
df_log_minus_mean.dropna(inplace=True) 
get_stationarity(df_log_minus_mean)
plot_acf(df_log_minus_mean, lags=12) 
plot_pacf(df_log_minus_mean, lags=12)

#2 - Lisage exponentiel (moyennes pondérées avec un poids plus élevés des observations récentes) )
rolling_mean_exp_decay = df_log.ewm(halflife=12, min_periods=0, adjust=True).mean()
df_log_exp_decay = df_log - rolling_mean_exp_decay
df_log_exp_decay.dropna(inplace=True)
get_stationarity(df_log_exp_decay)
plot_acf(df_log_exp_decay, lags=12) 
plot_pacf(df_log_exp_decay, lags=12)

#3 - Différence (Lag) --> Lag1 meilleur en ACF et PACF
df_log_shift = df_log - df_log.shift(1)
df_log_shift.dropna(inplace=True)
get_stationarity(df_log_shift)
plot_acf(df_log_shift, lags=12) 
plot_pacf(df_log_shift, lags=12)

df_log_shift2 = df_log - df_log.shift(2)
df_log_shift2.dropna(inplace=True)
get_stationarity(df_log_shift2)
plot_acf(df_log_shift2, lags=12) 
plot_pacf(df_log_shift2, lags=12)

df_log_shift3 = df_log - df_log.shift(3)
df_log_shift3.dropna(inplace=True)
get_stationarity(df_log_shift3)
plot_acf(df_log_shift3, lags=12) 
plot_pacf(df_log_shift3, lags=12)

#Régression modèle SARIMA (séries non stationnaires) + Prévisions sur données observées
from statsmodels.tsa.statespace.sarimax import SARIMAX
model=SARIMAX(data.iloc[0:90],order=(2,2,2),seasonal_order=(1, 0, 0, 12))
model_fit=model.fit()
model_fit.summary()

data['predictions'] = model_fit.predict(start=90, end=np.max(data.index)).rename('predictions')

#Forecasting Performance Measure
#Overprediction
data[['passengers','predictions']].plot()

expected = data['passengers']
pred = data['predictions'].dropna()
forecast_errors = [expected[i]-pred[i] for i in range(len(pred))]
print('Forecast Errors: %s' % forecast_errors)

#Stationnarité des résultats
plt.plot(rolling_mean)
plt.plot(model_fit.fittedvalues, color = 'red')

#Régression modèle ARIMA + Prévisions sur nouvelles dates
from statsmodels.tsa.statespace.sarimax import SARIMAX
model=SARIMAX(data['passengers'],order=(2,2,2),seasonal_order=(1, 0, 0, 12))
results=model.fit()

from pandas.tseries.offsets import DateOffset
new_dates=[data.index[-1]+DateOffset(months=x) for x in range(1,144)]
df_pred=pd.DataFrame(index=new_dates,columns =data.columns)
df_pred.head()

df2=pd.concat([data,df_pred])
df2['predictions']=results.predict(start=143,end=287)
df2[['passengers','predictions']].plot()




