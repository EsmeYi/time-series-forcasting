from pandas import read_csv
from numpy import concatenate
import numpy as np
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.core import Dense, Activation, Dropout
from matplotlib import pyplot

# 1: 1 2 3 7 8 9 14 16 18
# 2: 4 10 11 12 13 14 15 17 19 20
rmse11 = [3.356,0.741,31.890,4.159,4.035,4.898,0.064,0.015,1.490]
mape11 = [121,110,411,158,101,91,100,100,97]

rmse21 = [3.083,0.596,26.397,3.739,2.539,3.123,0.020,0.010,0.952]
mape21 = [65,103,218,113,80,65,97,99,85]

pyplot.plot(rmse11, color='blue')
pyplot.plot(rmse21, color='red')
#pyplot.plot(mape11, color='blue')
#pyplot.plot(mape21, color='red')
pyplot.show()
#print(sum(rmse11) / len(rmse11))



