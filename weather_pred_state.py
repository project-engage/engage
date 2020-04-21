from os import listdir
from os.path import isfile, join
import os
import glob
from tqdm import tqdm
import pandas as pd
from tqdm import tqdm
from os import listdir
from os.path import isfile, join
import numpy as np
from numpy import array
from datetime import datetime, date, time, timedelta
# Import models
# grid search holt winter's exponential smoothing
from math import sqrt
from multiprocessing import cpu_count
from joblib import Parallel
from joblib import delayed
from warnings import catch_warnings
from warnings import filterwarnings
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller
from statsmodels.tools.eval_measures import rmse, aic
import matplotlib.pyplot as plt
from matplotlib import pyplot
from statsmodels.tsa.seasonal import seasonal_decompose

from tensorflow.python.client import device_lib
import tensorflow.compat.v1.keras as tf
import sys
from keras import layers
from keras.layers import Dropout
from keras.layers import Dense
from sklearn.preprocessing import StandardScaler
from keras.callbacks import EarlyStopping
from keras import backend as K
from keras.optimizers import RMSprop, Adam
from keras.layers import Masking
import threading
from sklearn.utils import class_weight
import tensorflow
from tensorflow.keras.optimizers import RMSprop, Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Masking, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.python.framework import ops
from keras.optimizers import RMSprop, Adam
from keras.layers import Masking
from sklearn.utils import class_weight

from sklearn.preprocessing import MinMaxScaler


import ast

config = tensorflow.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tensorflow.compat.v1.Session(config=config)


EXCLUDE_WEATHER_COLUMN = ['lat', 'lon', 'wban_x', 'call', 'elev', 'begin', 'end', 'usaf', 'wban_y', 'name',
'count_temp', 'count_dewp', 'count_slp', 'count_stp', 'count_visib', 'count_wdsp', 'flag_max', 'flag_min', 'flag_prcp', 'fog',
'rain_drizzle','snow_ice_pellets', 'hail','thunder','tornado_funnel_cloud', 'sndp', 'stp', 'visib', 'wdsp', 'slp', 'min', 'max',
'year', 'mo', 'da', 'stn', 'country']

import warnings
warnings.filterwarnings("ignore")


# root mean squared error or rmse
def measure_rmse(actual, predicted):
	return sqrt(mean_squared_error(actual, predicted))

# split a univariate dataset into train/test sets
def train_test_split(data, n_test):
	return data[:-n_test], data[-n_test:]

# split a univariate sequence into samples
def split_sequence(sequence, n_steps):
	X, y = list(), list()
	for i in range(len(sequence)):
		# find the end of this pattern
		end_ix = i + n_steps
		# check if we are beyond the sequence
		if end_ix > len(sequence)-1:
			break
		# gather input and output parts of the pattern
		seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
		X.append(seq_x)
		y.append(seq_y)
	return array(X), array(y)

def root_mean_squared_error(y_true, y_pred):
	return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1)) 

def generate_model(n_steps, n_features, training_data):
	RNN = tf.layers.LSTM
	model = Sequential()
	model.add(RNN(50,  batch_input_shape=(training_data.shape[0], training_data.shape[1], training_data.shape[2])))
	model.add(Dense(1, activation="tanh"))
	model.compile(optimizer='adam', loss='mse')
	return model

if __name__ == '__main__':

	### Read Station data and select only relevant stations

	print("----Load Station----")
	
	data_station = pd.read_csv("station.csv")

	data_station = data_station.loc[data_station['begin']!=data_station['end'],:]
	data_station = data_station.loc[data_station['end']>=20170101,:]
	data_station = data_station.loc[data_station['country']=="US",:]
	data_station = data_station.loc[data_station['lat']!=0.000,:]
	data_station = data_station.loc[data_station['lon']!=0.000,:]
	data_station.dropna(subset=['state'], inplace=True)
	data_station.drop_duplicates(subset='usaf', keep='last', inplace=True)
	data_station["usaf"] = data_station["usaf"].astype('str')
	data_station.reset_index(drop=True, inplace=True)

	print("Number of Station :", len(set(data_station['usaf'])))

	### Load weather data

	data_weather2017 = pd.read_csv("2017weather.csv")
	data_weather2018 = pd.read_csv("2018weather.csv")
	data_weather2019 = pd.read_csv("2019weather.csv")
	data_weather2020 = pd.read_csv("2020weather.csv")

	data_weather = pd.concat([data_weather2017, data_weather2018, data_weather2019, data_weather2020])
	data_weather["stn"] = data_weather["stn"].astype('str')

	### Merge Station data with weather data

	dataset = data_weather.merge(data_station, left_on='stn', right_on='usaf')

	print("Number of Station in Weather :", len(set(dataset['stn'])))

	# Format date

	dataset["mo"] = dataset.mo.map("{:02}".format)
	dataset["da"] = dataset.da.map("{:02}".format)
	dataset['date'] = dataset['year'].astype(str)+dataset['mo'].astype(str)+dataset['da'].astype(str)

	### Exclude unused column

	dataset = dataset[dataset.columns[~dataset.columns.isin(EXCLUDE_WEATHER_COLUMN)]]

	### Replace missing value with NaN for calculation

	dataset.replace(9999.9, np.nan, inplace=True)
	dataset.replace(999.9, np.nan, inplace=True)

	### Export Prelim dataset

	dataset.to_csv('dataset.csv', index=False)

	### Aggregate features

	avg_weather = dataset.groupby(['state', 'date']).mean()
	avg_weather.reset_index(drop=False, inplace=True)

	print(avg_weather.head())

	### Interpolate Missing Value (NaN)

	list_state = list(set(avg_weather['state']))
	interpolate_data = pd.DataFrame()

	state_abs_max_df = pd.DataFrame()

	for state in list_state:
		temp_df = avg_weather.loc[avg_weather['state']==state, :]
		interpolate_feature_list = ['temp', 'dewp', 'mxpsd', 'gust', 'prcp']
		for f in interpolate_feature_list:
			temp_df[f] = temp_df[f].interpolate()
			temp_df[f] = temp_df[f].ffill()
			temp_df[f] = temp_df[f].bfill()
			abs_max_value = abs(temp_df[f].max())
			state_abs_max_df.loc[state, f] = abs_max_value
			temp_df[f] = temp_df[f]/abs_max_value
		
		interpolate_data = interpolate_data.append(temp_df)

	print(state_abs_max_df)
	state_abs_max_df.index.name = 'state'
	state_abs_max_df.to_csv('state_abs_max.csv', index=True)

	### Export Weather Dataset

	interpolate_data.to_csv('weather_dataset.csv', index=True)

	### Model Training and Future Weather Prediction

	pred_actual = pd.DataFrame()
	future_pred_df = pd.DataFrame()
	rsme_score_df = pd.DataFrame()

	# Data Parameter
	n_steps = 365
	n_test = 100
	n_features = 1
	n_future_pred = 60

	# Train model for each state and feature

	for state in list_state:
		state_df = interpolate_data.loc[interpolate_data['state']==state, :]
		state_df = state_df[state_df.columns[~state_df.columns.isin(['state'])]]
		state_df.dropna(axis='columns', inplace=True)
		state_df.set_index('date', inplace=True)

		print('State : ', state)
		print(state_df.describe())

		# List the date that will be used for future prediction

		date_test_list = list(state_df.index.values)[-n_test:]
		first_date_for_future = datetime.strptime(date_test_list[-1], '%Y%m%d')
		date_future_list = [datetime.strftime(d, '%Y%m%d') for d in (first_date_for_future + timedelta(n) for n in range(n_future_pred))]

		# Open the log file for that state		

		file = open(state+"_log.txt","w")
		file.write('State : ' + state + "\n")
		file.write(state_df.describe().to_string() + "\n")

		temp_pred_actual = pd.DataFrame()
		temp_future_pred = pd.DataFrame()
		temp_rsme_score = pd.DataFrame()

		for f in state_df.columns:
			# define dataset
			data = state_df[f]
			train_data = data[:-n_test]
			test_data = data[-n_test:]
			print(data)
			# split into train/test samples
			X, y = split_sequence(data, n_steps)
			train_x = X[:-n_test]
			train_y = y[:-n_test]
			test_x = X[-n_test:]
			test_y = y[-n_test:]

			train_x = train_x.reshape((train_x.shape[0], train_x.shape[1], n_features))
			test_x = test_x.reshape((test_x.shape[0], test_x.shape[1], n_features))

			print(train_x.shape)
			print(test_x.shape)

			# create weather prediction model
			model = generate_model(n_steps, n_features, train_x)
			earlystopping = EarlyStopping(monitor='loss', patience=25, mode='min', restore_best_weights=True)
			callbacks = [earlystopping]

			# train the model
			history = model.fit(train_x, train_y, epochs=500, batch_size=train_x.shape[0], verbose=2, callbacks=callbacks)

			# predict the test dataset
			pred_y = model.predict(test_x, verbose=0)

			temp_pred_actual[f+'_actual'] =[p * state_abs_max_df.loc[state, f] for p in test_y]
			temp_pred_actual[f+'_pred'] = [pred[0] * state_abs_max_df.loc[state, f] for pred in pred_y] 

			# calculate RMSE score for that feature of the state and record to log file
			rmse_score = measure_rmse(test_y, [pred[0] for pred in pred_y])
			file.write("RMSE of " + f + " : " + str(rmse_score) + "\n")
			temp_rsme_score.loc[state, f+'_rsme'] = rmse_score

			# plot the actual vs prediction graph of the feature of that state
			pyplot.title("Prediction/Actual of "+f+" in "+state)
			pyplot.plot([p * state_abs_max_df.loc[state, f] for p in test_y], color='blue', label="prediction" )
			pyplot.plot([pred[0] * state_abs_max_df.loc[state, f] for pred in pred_y], color='red', label="prediction")
			pyplot.savefig(os.path.join(state+"_"+f+'_plot.png'))
			pyplot.clf()

			# Predict the next n days
			print("---Prediction for Future Date---")

			future_x = X[-1:][0]
			future_pred = list()

			for i in range(n_future_pred):
				x_input = array(future_x)
				x_input = x_input.reshape((1, n_steps, n_features))
				y = model.predict(x_input, verbose=0)
				future_pred.append(y[0][0])
				future_x = np.append(future_x[1:], [y])

			temp_future_pred[f+'_pred'] = [p * state_abs_max_df.loc[state, f] for p in future_pred]
			
			# save model
			model.save(state+"_"+f+"_model.h5")

			# reset graph for the next model (to avoid out of memory)
			model.reset_states()
			ops.reset_default_graph()

		# record all result to files
		temp_pred_actual['date'] = date_test_list
		temp_pred_actual['state'] = state
		
		temp_future_pred['state'] = state
		temp_future_pred['date_idx'] = range(n_future_pred)
		temp_future_pred['date'] = date_future_list

		pred_actual = pred_actual.append(temp_pred_actual)
		future_pred_df = future_pred_df.append(temp_future_pred)
		rsme_score_df = rsme_score_df.append(temp_rsme_score)

		print(pred_actual)
		print(future_pred_df)

		file.close()

		pred_actual.to_csv('pred_actual.csv', index=True)
		future_pred_df.to_csv('future_pred.csv', index=True)
		rsme_score_df.to_csv('rsme_score.csv', index=True)