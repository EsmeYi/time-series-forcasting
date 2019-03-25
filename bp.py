from math import sqrt
from numpy import concatenate
import numpy as np
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR
from matplotlib import pyplot

# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
	n_vars = 1 if type(data) is list else data.shape[1]
	df = DataFrame(data)
	cols, names = list(), list()
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
		if i == 0:
			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
		else:
			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
	# put it all together
	agg = concat(cols, axis=1)
	agg.columns = names
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg

# load dataset
dataset = read_csv('sensordata.csv', header = 0, index_col=0)
values = dataset.values
# ensure all data is float
values = values.astype('float32')
# normalise features
scaler = MinMaxScaler(feature_range=(0,1))
scaled = scaler.fit_transform(values)
# frame as supervised learning
n_mins = 5
n_features = 24
reframed = series_to_supervised(scaled, n_mins, 1)
# drop columns we don't want to predict
print(reframed.shape)

# split into train and test sets
values = reframed.values
n_train_mins = 4 * 24 * 60
train = values[:n_train_mins, :]
test = values[n_train_mins:, :]
# split into input and outputs
n_obs = n_mins * n_features
train_X, train_y = train[:, :n_obs], train[:, -1]
test_X,  test_y =test[:, :n_obs], test[:,-1]
train_y = train_y.reshape(train_y.shape[0],1)
test_y = test_y.reshape(test_y.shape[0],1)
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

# sigmoid function
def nonlin(x,deriv=False):
	if(deriv==True):
		return x*(1-x)
	return 1/(1+np.exp(-x))

# design bp network
# seed random numbers to make calculation
# deterministic (just a good practice)
np.random.seed(1)

# initialize weights randomly with mean 0
syn0 = 2*np.random.random((120,24)) -1
syn1 = 2*np.random.random((24,1)) - 1
for iter in range(3):
	# forward propagation
	l0 = train_X
	l1 = nonlin(np.dot(l0,syn0))
	l2 = nonlin(np.dot(l1,syn1))
	# how much did we miss?
	l2_error = train_y - l2
	# if(iter % 1000) == 0:
	# 	print("Error:"+str(np.mean(np.abs(l2_error)))
	# multiply how much we missed by the slope of the sigmoid at the values in l1
	l2_delta=l2_error*nonlin(l2,True)
	l1_error=l2_delta.dot(syn1.T)
	l1_delta=l1_error*nonlin(l1,True)
	
	# update weights
	syn1 += l1.T.dot(l2_delta)
	syn0 += l0.T.dot(l1_delta)

predict_y_delta = nonlin(np.dot(test_X,syn0))
predict_y = nonlin(np.dot(predict_y_delta,syn1))
print(predict_y)

# calculate RMSE
rmse = sqrt(mean_squared_error(test_y, predict_y))
print('Test RMSE: %.3f' % rmse)
# calculate MAPE
mape = np.mean(np.abs((test_y - predict_y) / test_y)) * 100
print('Test mape: %d' % mape)

# plot 
# origin_data = test_y
# print("origin data:%s"%origin_data)
# pyplot.plot([x for x in range(1, test_y.shape[0]+1)], test_y, linestyle='-', color='blue', label='actual mode')
pyplot.plot([x1 for x1 in range(1, predict_y.shape[0]+1)], predict_y, linestyle='-', color='red', label='prediction mode')
pyplot.plot([x for x in range(1, test_y.shape[0]+1)], test_y, linestyle='-', color='blue', label='actual mode')
pyplot.legend(loc=1, prop={'size': 12})
pyplot.show()
