from pandas import read_csv
from datetime import datetime

#load data
dataset = read_csv('raw.csv',parse_dates=['Time'], index_col=0)
dataset.index.name = 'date'
dataset = dataset.sort_index()
print(dataset.head(5))
dataset.to_csv('sensordata.csv')
