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

pd.options.display.width = 0

EXCLUDE_STATE = ['VT', 'RI']


if __name__ == '__main__':

	INCLUDE_COL = ['date', 'state', 'positive', 'recovered', 'death', 'positiveIncrease', 'deathIncrease']

	data_covid = pd.read_csv("daily_covid_usstates.csv")[INCLUDE_COL].sort_values(by=['date'], ascending=True).reset_index(drop=True)

	data_covid = data_covid.loc[~data_covid['state'].isin(EXCLUDE_STATE), :]
	data_covid = data_covid.loc[data_covid['positive']!=0, :]

	state_list = set(data_covid.state)

	temp_df = pd.DataFrame()
	count_include = 0
	count_remove = 0
	for state in state_list:
		if (not pd.isnull(data_covid.loc[data_covid['state']==state, 'recovered']).all()) or (not (data_covid.loc[data_covid['state']==state, 'recovered'] == 0).all()):
			temp_df = temp_df.append(data_covid.loc[data_covid['state']==state, :].ffill(axis = 0))
			count_include += 1
		else:
			# print(data_covid.loc[data_covid['state']==state, :])
			count_remove += 1
	print('Include State : ', count_include)
	print('Exclude State : ', count_remove)

	data_covid = temp_df

	data_covid['death'].fillna(0, inplace=True)
	data_covid['positive'].fillna(0, inplace=True)
	data_covid['recovered'].fillna(0, inplace=True)

	state_list = set(data_covid.state)
	temp_df = pd.DataFrame()

	for state in state_list:
		temp = data_covid.loc[data_covid['state']==state, :].reset_index(drop=True)
		for idx in range(len(temp)-1):
			if idx == 0:
				ini_recovered = temp.iloc[idx]['recovered']
			else:
				if temp.iloc[idx]['recovered'] > temp.iloc[idx+1]['recovered']:
					left_index = (len(temp)-1)-idx # how many index left we have to check
					# print('left index : ',left_index)
					anchor_index = 0
					anchor_count = 0
					for j in range(1, left_index):
						if temp.iloc[idx+j]['recovered'] > temp.iloc[idx]['recovered']: # if the checking index is larger than current
							anchor_index = idx+j # record larger index
							anchor_count = j # record how many step to the larger index
							break
					final = temp.iloc[idx+j]['recovered'] # get the nearest larger number
					if anchor_count != 0: # if there is next larger number, if not use the last step_increase
						step_increase = int((final - temp.iloc[idx-1]['recovered'])/anchor_count) # calculate step increase linearly
					else:
						step_increase = int(temp.iloc[idx-1]['recovered'] - temp.iloc[idx-2]['recovered'])
					for j in range(0, anchor_count):
						temp.at[idx+j,'recovered'] = temp.iloc[idx-1]['recovered'] + (step_increase * j) # replace the data with the incrase number linearly
					ini_recovered = temp.iloc[idx]['recovered']
				else:
					ini_recovered = temp.iloc[idx]['recovered']
		temp_df = temp_df.append(temp)

	data_covid = temp_df.reset_index(drop=True)

	print(data_covid.head())

	data_covid['recoveredIncrease'] = data_covid.groupby('state')['recovered'].apply(lambda x: x - x.shift(1))
	
	data_covid['positiveIncrease'].fillna(0, inplace=True)
	data_covid['recoveredIncrease'].fillna(0, inplace=True)
	data_covid['deathIncrease'].fillna(0, inplace=True)

	# data_covid.to_csv('prelim.csv', index=False)

	data_population = pd.read_csv("covid_county_population_usafacts.csv")
	data_population = data_population.groupby(['State']).sum().reset_index(drop=False).loc[:,['State','population']]
	data_population.columns = ['state','population']
	dataset = data_covid.merge(data_population, on='state')
	
	dataset['population'].fillna(0, inplace=True)

	dataset['death'] = dataset['death'].astype(int)
	dataset['positive'] = dataset['positive'].astype(int)
	dataset['recovered'] = dataset['recovered'].astype(int)
	dataset['positiveIncrease'] = dataset['positiveIncrease'].astype(int)
	dataset['recoveredIncrease'] = dataset['recoveredIncrease'].astype(int)
	dataset['deathIncrease'] = dataset['deathIncrease'].astype(int)
	dataset['population'] = dataset['population'].astype(int)

	# dataset['Date'] = [datetime.strftime(d, '%Y%m%d') for d in [datetime.strptime(t, '%m/%d/%y') for t in dataset['Date']]]

	dataset.to_csv('pop_dataset.csv', index=False)
	print(dataset.head())

	data_weather_future = pd.read_csv("weather_model/future_pred.csv").loc[:,['date', 'state','dewp_pred', 'gust_pred', 'prcp_pred', 'temp_pred']]
	data_weather_history = pd.read_csv("weather_model/pred_actual.csv").loc[:,['date', 'state', 'dewp_actual', 'gust_actual', 'prcp_actual', 'temp_actual','dewp_pred', 'gust_pred', 'prcp_pred', 'temp_pred']]
	# dataset_weather = data_weather_history.merge(data_weather_future, how='outer', on=['date','state'])
	dataset_weather = pd.concat([data_weather_history, data_weather_future]).loc[:,['date', 'state', 'dewp_actual', 'gust_actual', 'prcp_actual', 'temp_actual','dewp_pred', 'gust_pred', 'prcp_pred', 'temp_pred']]
	
	dataset_weather.dewp_actual.fillna(dataset_weather.dewp_pred, inplace=True)
	dataset_weather.gust_actual.fillna(dataset_weather.gust_pred, inplace=True)
	dataset_weather.prcp_actual.fillna(dataset_weather.prcp_pred, inplace=True)
	dataset_weather.temp_actual.fillna(dataset_weather.temp_pred, inplace=True)
	dataset_weather = dataset_weather.loc[:,['date', 'state', 'dewp_actual', 'gust_actual', 'prcp_actual', 'temp_actual']]

	# print(dataset_weather[dataset_weather.isna().any(axis=1)])
	print(dataset_weather.columns)

	# dataset_weather.to_csv('wea_dataset.csv', index=False)

	dataset = dataset_weather.merge(dataset, on=['date', 'state'])

	dataset.drop_duplicates(subset = ['date', 'state'], keep = 'first', inplace = True)

	print(dataset.head())

	dataset.to_csv('covid_pred_dataset.csv', index=False)

	# Step 1 : Calculate R0

	state_list = set(dataset.state)

	temp_df = pd.DataFrame()
	r0_df = pd.DataFrame()

	for state in state_list:
		temp = dataset.loc[dataset['state']==state, :].sort_values(by=['date'], ascending=True).reset_index(drop=True)
		temp = temp.head(30)
		if not (temp['recoveredIncrease'] == 0).all():
			sum_delta_death = temp['deathIncrease'].sum()
			sum_delta_positive = temp['positiveIncrease'].sum()
			sum_delta_recovered = temp['recoveredIncrease'].sum()
			# print(sum_delta_death , sum_delta_positive , sum_delta_recovered)
			r0 = (sum_delta_positive + sum_delta_recovered + sum_delta_death)/(sum_delta_recovered + sum_delta_death)
			r0_df.loc[state, 'sum_delta_death'] = sum_delta_death
			r0_df.loc[state, 'sum_delta_positive'] = sum_delta_positive
			r0_df.loc[state, 'sum_delta_recovered'] = sum_delta_recovered
			r0_df.loc[state, 'r0'] = r0
			temp_df = temp_df.append(temp)

	r0_df.index.name = 'state'
	print(temp_df)
	print(r0_df)
	r0_df.to_csv('1_r0_score.csv', index=True)

	# Step 2.1 : Calculate Beta, change in recovery based on last day infection
	# Step 2.2 : Calculate Gamma, change in death based on last day infection
	# Step 3 : Calculate Alpha, 

	state_list = r0_df.index.values.tolist()

	abg_df = pd.DataFrame()

	import statsmodels.formula.api as sm

	for state in state_list:
		# get first day that has more than 0 positive as start date
		temp = dataset.loc[dataset['state']==state, ['date','recoveredIncrease', 'deathIncrease', 'positive']].sort_values(by=['date'], ascending=True).reset_index(drop=True)
		temp.positive = temp.positive.shift(1) # shift I(t-1) to match delta(R(t)) and delta(D(t))
		temp.dropna(inplace=True)
		result_temp1 = sm.ols(formula="recoveredIncrease ~ positive", data=temp).fit()
		result_temp2 = sm.ols(formula="deathIncrease ~ positive", data=temp).fit()
		beta = result_temp1.params['positive']
		gamma = result_temp2.params['positive']
		abg_df.loc[state, 'beta'] = beta
		abg_df.loc[state, 'gamma'] = gamma
		r0 = r0_df.loc[state, 'r0']
		alpha = r0 * (beta + gamma)
		abg_df.loc[state, 'alpha'] = alpha
	
	abg_df.index.name = 'state'
	print(abg_df)
	abg_df.to_csv('2_abg_coef.csv', index=True)

	# Step 4: accounting for external effects on the confirmed infected cases
	# Step 4.1 : Get S(t) and calculate H

	h_df = pd.DataFrame()

	for state in state_list:
		temp = dataset.loc[dataset['state']==state, ['state', 'date', 'recoveredIncrease', 'deathIncrease', 'positive', 'population']].sort_values(by=['date'], ascending=True).reset_index(drop=True)
		alpha = abg_df.loc[state, 'alpha']
		for i in range(len(temp)):
			if i == 0:
				temp.at[i,'suspect'] = temp.loc[i,'population']
				temp.at[i,'h'] = temp.loc[i,'suspect'] * temp.loc[i,'positive']
			else:
				temp.at[i,'suspect'] = int(temp.loc[i-1,'suspect'] - ((alpha/temp.loc[i,'population'])*temp.loc[i-1,'suspect']*temp.loc[i-1,'positive']))
				temp.at[i,'h'] = temp.loc[i,'suspect'] * temp.loc[i,'positive']
		h_df = h_df.append(temp)

	h_df.set_index('state', inplace=True)
	print(h_df)
	h_df.to_csv('3_h_score.csv', index=True)

	temp_h = h_df.copy()
	temp_h.reset_index(inplace=True, drop=False)
	temp_h = temp_h.loc[:, ['state','date', 'population', 'h']]

	weather_coef_df = pd.DataFrame()

	# state_list = set(temp_h['state'])

	# Step 4.2 : Get coefficient of weather
	step_42_df = pd.DataFrame()

	for state in state_list:
		# print(state)
		temp = dataset.loc[dataset['state']==state, ['state', 'date', 'positiveIncrease', 'positive', 'dewp_actual', 'gust_actual', 'prcp_actual', 'temp_actual']].sort_values(by=['date'], ascending=True).reset_index(drop=True)
		temp = temp.merge(temp_h, on=['state','date'])

		temp['positive_t-1'] = temp.positive.shift(1)
		temp['h_t-1'] = temp.h.shift(1)
		temp = temp.iloc[1:]

		# print(temp)

		alpha = abg_df.loc[state, 'alpha']
		beta = abg_df.loc[state, 'beta']
		gamma = abg_df.loc[state, 'gamma']

		temp.reset_index(drop=True, inplace=True)

		for i in range(len(temp)):
			variable_cal = temp.loc[i,'positiveIncrease'] - ((alpha/temp.loc[i,'population'])*temp.loc[i,'h_t-1']) - (beta*temp.loc[i,'positive_t-1']) - (gamma*temp.loc[i,'positive_t-1'])
			temp.loc[i,'var_cal'] = variable_cal
		step_42_df = step_42_df.append(temp)

		expected_feature = ['dewp_actual', 'gust_actual', 'prcp_actual', 'temp_actual']
		formula_string = "var_cal ~ "

		for f in expected_feature:
			if pd.isnull(temp.loc[:, f]).all():
				continue
			else:
				formula_string += f + " +"

		formula_string = formula_string[:-2]

		result_temp3 = sm.ols(formula=formula_string, data=temp).fit()

		for f in expected_feature:
			if pd.isnull(temp.loc[:, f]).all():
				continue
			else:
				weather_coef_df.loc[state, f] = result_temp3.params[f]


	step_42_df.to_csv('4_before_calculation.csv', index=False)
	print(step_42_df)
	weather_coef_df.to_csv('4_weather_coef.csv', index=True)
	print(weather_coef_df)
	exit()