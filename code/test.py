
# coding: utf-8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import trend_pf
import statsmodels.api as sm

pd_data = pd.read_csv('UK.csv')
data_len=132
data = np.array(np.log(pd_data['drivers']))
data_sm = sm.tsa.seasonal_decompose(np.array(data), freq=12)
# all_uk_data = data_sm.trend[6:-6]+data_sm.resid[6:-6]
all_uk_data = data[6:-6]-data_sm.seasonal[6:-6]
plt.plot(all_uk_data)
plt.plot(data[6:-6],label='no')
plt.show()
plt.legend()
# print(uk_data.shape)
uk_data = all_uk_data[:data_len]

# I=0
# J=0
# num = 0
# start=np.array([0,7])
# for i in np.arange(start[0],1,0.1):
#     for j in np.arange(start[1],9,0.1):
#         model = trend_pf.ParticleFilter(uk_data,10000,np.array([i,0.0]),j)
#         result = model.simulate()
#         forecast = model.forecast(result,all_uk_data.shape[0]-data_len)['Forecasts_mean']
#         # print(forecast.shape)
#         Err = trend_pf.rmse(all_uk_data[data_len:],(forecast[:,0]+forecast[:,1]).reshape(all_uk_data.shape[0]-data_len,1))
#         if i==start[0] and j==start[1]:
#             err = Err
#             I=i
#             J=j
#             print('\n i:{},j:{},rmse:{}'.format(i,j,err))
#         if err>Err:
#             err = Err
#             print('\ni:{},j:{},rmse:{}'.format(i,j,err))
#             I=i
#             J=j
#         num = num+1
#         print("\r calculating... num={}".format(num), end="")
#

I = 0.1
J = 7.1
model = trend_pf.ParticleFilter(uk_data,10000,np.array([I,0.0]),J)
result = model.simulate()
plt.plot(result['filtered_value'][:,0])
plt.title('level')
plt.show()
plt.plot(result['filtered_value'][:,1])
plt.title('slope')
plt.show()
plt.plot(result['filtered_value'][:,0]+result['filtered_value'][:,1],label='pre')
plt.plot(uk_data,label='obs')
plt.title("predicts")
plt.legend()
plt.show()

forecast = pd.DataFrame(model.forecast(result,50)['Forecasts_mean'])
forecast.index = forecast.index+data_len
plt.plot(result['filtered_value'][:,0]+result['filtered_value'][:,1],label='predict')
plt.plot(forecast.loc[:,0]+forecast.loc[:,1],label='forecast')
plt.plot(all_uk_data,label='obs')
plt.legend()
plt.title('level')
plt.show()
