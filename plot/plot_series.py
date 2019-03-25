from pandas import read_csv
from pandas import DataFrame
from matplotlib import pyplot
import numpy as np
# load dataset
dataset = read_csv('sensordata.csv', header=0, index_col=0)
values = dataset.values

# specify columns to plot
groups = [0,6,7,9]
#groups = [0,1,2,3,4,5,6,7,8,9,10,11,12]
#groups = [13,14,15,16,17,18,19,20,21,22,23]
i = 1
# plot each column
for group in groups:
	pyplot.subplot(len(groups), 1, i)
	pyplot.plot(values[:, group])
	pyplot.title(dataset.columns[group], y=0.5, loc='right')
	i += 1
pyplot.show()
