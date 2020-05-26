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
import numpy as np


def rename_location(df):
	location_list = pd.read_csv("covid_data/location_match.csv")

	for i in range(len(location_list)):
		# print(location_list.columns)
		df.loc[(df['province_state']==location_list.loc[i , 'province_state_old'])&(df['country_region']==location_list.loc[i , 'country_region_old']), 'province_state'] = location_list.loc[i , 'province_state_new']
		df.loc[(df['province_state']==location_list.loc[i , 'province_state_new'])&(df['country_region']==location_list.loc[i , 'country_region_old']), 'country_region'] = location_list.loc[i , 'country_region_new']
	return df

def aggregate_loc_weather(df):
	# location_list = pd.read_csv("location_aggregate_weather.csv")
	location_list = sorted(list(set(df['country_region'])))
	# print(location_list)
	location_list.remove("United States")
	temp = df.groupby(['date', 'country_region']).mean().reset_index(drop=False)

	for i in range(len(location_list)):
		# loc = location_list.loc[i , 'country_aggregate']
		loc = location_list[i]
		select_temp = temp.loc[temp['country_region']==loc, :]
		select_temp.at[: ,'province_state'] = 'UNK'
		# print(select_temp)
		df = df.loc[df['country_region']!=loc, :]
		# print(len(df))
		df = pd.concat([df, select_temp])
		# print(len(df))

	return df

def aggregate_loc_covid(df):
	# location_list = pd.read_csv("location_aggregate_covid.csv")
	location_list = sorted(list(set(df['country_region'])))
	# print(location_list)
	location_list.remove("United States")
	temp = df.groupby(['date', 'country_region']).sum().reset_index(drop=False)

	for i in range(len(location_list)):
		# loc = location_list.loc[i , 'country_aggregate']
		loc = location_list[i]
		select_temp = temp.loc[temp['country_region']==loc, :]
		select_temp.at[: ,'province_state'] = 'UNK'
		# print(select_temp)
		df = df.loc[df['country_region']!=loc, :]
		# print(len(df))
		df = pd.concat([df, select_temp])
		# print(len(df))

	return df



if __name__ == '__main__':

	INCLUDE_COL = ['date', 'state', 'positive', 'recovered', 'death']
	EXCLUDE_COL = ['latitude', 'longitude', 'location_geom']

	confirmed_covid = pd.read_csv("covid_data/jhu_confirmed_covid.csv")
	confirmed_covid = confirmed_covid.loc[:,~confirmed_covid.columns.isin(EXCLUDE_COL)]
	recovered_covid = pd.read_csv("covid_data/jhu_recovered_covid.csv")
	recovered_covid = recovered_covid.loc[:,~recovered_covid.columns.isin(EXCLUDE_COL)]
	death_covid = pd.read_csv("covid_data/jhu_death_covid.csv")
	death_covid = death_covid.loc[:,~death_covid.columns.isin(EXCLUDE_COL)]

	us_covid = pd.read_csv("covid_data/daily_covid_usstates.csv")[INCLUDE_COL].sort_values(by=['date'], ascending=True).reset_index(drop=True)
	us_covid.rename(columns={"state": "province_state", "positive" : "confirmed"}, inplace=True)
	us_covid['country_region'] = 'United States'
	us_covid.fillna(0, inplace=True)

	data_population = pd.read_csv("covid_data/covid_county_population_usafacts.csv")
	data_population = data_population.groupby(['State']).sum().reset_index(drop=False).loc[:,['State','population']]
	data_population.rename(columns={"State": "province_state"}, inplace=True)
	

	pivot_date_columns = confirmed_covid.columns.tolist()
	pivot_date_columns.remove('province_state')
	pivot_date_columns.remove('country_region')

	confirmed_covid = confirmed_covid.melt(['province_state', 'country_region'], var_name='date', value_name='confirmed')
	confirmed_covid['date'] = [datetime.strftime(d, '%Y%m%d') for d in [datetime.strptime(t, '_%m_%d_%y') for t in confirmed_covid['date']]]

	recovered_covid = recovered_covid.melt(['province_state', 'country_region'], var_name='date', value_name='recovered')
	recovered_covid['date'] = [datetime.strftime(d, '%Y%m%d') for d in [datetime.strptime(t, '_%m_%d_%y') for t in recovered_covid['date']]]

	death_covid = death_covid.melt(['province_state', 'country_region'], var_name='date', value_name='death')
	death_covid['date'] = [datetime.strftime(d, '%Y%m%d') for d in [datetime.strptime(t, '_%m_%d_%y') for t in death_covid['date']]]

	inter_covid = confirmed_covid.merge(recovered_covid, on=['province_state', 'country_region', 'date'], how='inner')
	inter_covid = inter_covid.merge(death_covid, on=['province_state', 'country_region', 'date'], how='inner')

	inter_covid.loc[inter_covid['country_region']=='US','country_region'] = 'United States'
	inter_covid = rename_location(inter_covid)

	inter_covid = aggregate_loc_covid(inter_covid)

	jhu_population = pd.read_csv("covid_data/jhu_countries_with_code.csv")

	jhu_population['countries_and_territories'] = [x.replace("_", " ") for x in jhu_population['countries_and_territories']]
	jhu_population = jhu_population[['countries_and_territories', 'pop_data_2018']]
	jhu_population.rename(columns={"countries_and_territories": "country_region", "pop_data_2018" : 'population'}, inplace=True)
	jhu_population.drop_duplicates(inplace=True, ignore_index=True)

	inter_covid = inter_covid.merge(jhu_population, on=['country_region'], how='inner')

	us_covid = us_covid.merge(data_population, on='province_state')

	# print(inter_covid.loc[inter_covid['country_region']=='US', :][['country_region','confirmed', 'recovered', 'death']].groupby(['country_region']).sum())
	# print(us_covid[['country_region','confirmed', 'recovered', 'death']].groupby(['country_region']).sum())

	inter_covid = pd.concat([inter_covid, us_covid])
	inter_covid['province_state'].fillna('UNK', inplace=True)
	inter_covid.fillna(0, inplace=True)
	inter_covid['date'] = inter_covid['date'].astype(int)
	
	inter_covid['country_region'].map(lambda x: x.rstrip('*'))
	inter_covid['country_region'] = inter_covid['country_region'].str.strip()

	# print(inter_covid.head())

	# inter_covid.to_csv('data_covid.csv')

	first_date_covid = min(inter_covid['date'])

	weather_station_list = pd.read_csv("weather_meta_data/ghcnd_stations.csv")[['id', 'state']]
	weather_station_list['state'] = weather_station_list['state'].str.strip()
	weather_station_list['state'] = weather_station_list['state'].replace('', 'UNK')
	# print(weather_station_list)

	weather_country_list = pd.read_csv("weather_meta_data/ghcnd_countries.csv")
	# print(weather_country_list)
	weather_country_list.rename(columns={"code": "country_region", "name" : "country_name"}, inplace=True)
	# print(weather_country_list)

	weather_file_list = []
	li = []

	year_list = ['2019', '2020']

	# os.chdir("")
	for file in glob.glob("./weather_data/*.csv"):
		if any(ele in file for ele in year_list):
			weather_file_list.append(file)
		
	for file in tqdm(weather_file_list):
		df = pd.read_csv(file, index_col=None, header=0, low_memory=False)
		li.append(df)

	weather_frame = pd.concat(li, axis=0, ignore_index=True)
	weather_frame = weather_frame[['id', 'date', 'element', 'value']]
	weather_frame['date'] = weather_frame['date'].astype(str)
	weather_frame['date'] = weather_frame['date'].str.replace('-', '')
	weather_frame['date'] = weather_frame['date'].astype(int)

	weather_frame = weather_frame.loc[weather_frame['date']>=first_date_covid, :]

	weather_frame = pd.pivot_table(weather_frame, values='value', index=['id', 'date'], columns='element')
	weather_frame = weather_frame[['TAVG']]
	weather_frame.reset_index(drop=False, inplace=True)
	weather_frame['country_region'] = weather_frame['id'].str[:2]
	
	weather_frame = weather_frame.merge(weather_station_list, on=['id'], how='left')
	weather_frame = weather_frame[['date', 'country_region', 'state', 'TAVG']]
	# print(weather_frame)
	# print(weather_frame.columns)
	weather_frame = weather_frame.groupby(['date', 'country_region', 'state']).mean().reset_index(drop=False)

	weather_frame = weather_frame.merge(weather_country_list, on=['country_region'])

	weather_frame = weather_frame[['date', 'country_name', 'state', 'TAVG']]
	weather_frame.rename(columns={"country_name": "country_region", "state" : "province_state"}, inplace=True)
	weather_frame['country_region'] = weather_frame['country_region'].str.strip()

	# print(set(weather_frame['country_region']))

	# print(weather_frame.loc[weather_frame['country_region']=='United States', 'province_state'])

	weather_future_pred = pd.read_csv("weather_output/future_pred.csv")
	weather_future_pred.rename(columns={"state": "province_state", "country" : "country_region", 'TAVG_pred' : 'TAVG', 'PRCP_pred' : 'PRCP'}, inplace=True)

	# print(weather_future_pred)

	weather_frame = pd.concat([weather_frame, weather_future_pred])
	weather_frame['date_idx'].fillna(-1, inplace=True)

	# print(weather_frame)

	weather_frame = rename_location(weather_frame)
	
	weather_frame = aggregate_loc_weather(weather_frame)
	
	# weather_frame.to_csv('weather_frame.csv')

	location_list_weather = sorted(list(set(weather_frame['country_region'] + " : " +  weather_frame['province_state'])))
	location_list_covid = sorted(list(set(inter_covid['country_region'] + " : " +  inter_covid['province_state'])))

	# print(location_list_weather)
	# print(location_list_covid)

	len_weather = len(location_list_weather)
	len_covid = len(location_list_covid)

	# print(len(location_list_covid), len(location_list_weather))

	if len_weather > len_covid:
		# print('pad covid')
		append_list = ([None] * (len_weather - len_covid))
		location_list_covid = location_list_covid + append_list

	else:
		# print('pad weather')
		# print(len_covid - len_weather)
		append_list = ([None] * (len_covid - len_weather))
		# print(append_list)
		location_list_weather = location_list_weather + append_list
		# print(len(location_list_covid), len(location_list_weather))

	
	# location_df = pd.concat([location_list_weather,location_list_covid], ignore_index=True, axis=1)
	location_df = pd.DataFrame()
	location_df = location_df.assign(weather_loc = location_list_weather, covid_loc = location_list_covid)
	# location_df.columns = ['weather_loc', 'covid_loc']
	
	# location_df.to_csv('location_df_rename.csv', index=False)

	# inter_covid.to_csv('data_covid.csv')
	# weather_frame.to_csv('weather_frame.csv')

	interpolate_data = pd.DataFrame()

	import math

	list_country = set(weather_frame['country_region'])
	# print(list_country)

	for country in list_country:
		list_state = set(weather_frame.loc[weather_frame['country_region']==country, 'province_state'])
		temp_df = weather_frame.loc[weather_frame['country_region']==country, :]
		for state in list_state:
			temp_df2 = temp_df.loc[temp_df['province_state']==state, :].sort_values(by=['date'], ascending=True).reset_index(drop=True)
			interpolate_feature_list = ['TAVG']
			for f in interpolate_feature_list:
				if pd.isnull(temp_df2[f]).all():
					temp_df2[f].fillna(0, inplace=True)
				else:
					for i in range(len(temp_df2[f])):
						if i == 0 or i == len(temp_df2[f])-1:
							if math.isnan(temp_df2[f][i]):
								temp_df2[f].iloc[i] = 0
						else:
							previous_v = temp_df2[f].iloc[i-1]
							next_v = temp_df2[f].iloc[i+1]
							if math.isnan(next_v):
								temp_df2[f].iloc[i] = previous_v/2
							else:
								temp_df2[f].iloc[i] = (previous_v+next_v)/2

			# if country == "United States":
				# print(temp_df2)
			
			interpolate_data = interpolate_data.append(temp_df2)

	interpolate_data.fillna(0, inplace=True)
	# print(interpolate_data)

	# interpolate_data.to_csv('interpolate_data.csv')

	weather_frame = interpolate_data.copy()

	dataset = inter_covid.merge(weather_frame, on=['date', 'country_region','province_state'], how='right')

	temp = dataset.groupby(['country_region', 'province_state']).max().reset_index(drop=False)[['country_region', 'province_state','date_idx']]
	temp = temp.loc[temp['date_idx']==179,:]

	list_loc = (temp['country_region'] + " : " + temp['province_state']).tolist()

	# print(list_loc)

	get_only_has_weather_pred = pd.DataFrame()

	list_country = set(dataset['country_region'])

	for country in list_country:
		list_state = set(dataset.loc[dataset['country_region']==country, 'province_state'])
		temp_df = dataset.loc[dataset['country_region']==country, :]
		for state in list_state:
			loc_string = country + " : " + state
			if loc_string in list_loc:
				temp_df2 = temp_df.loc[temp_df['province_state']==state, :].sort_values(by=['date'], ascending=True).reset_index(drop=True)
				get_only_has_weather_pred = get_only_has_weather_pred.append(temp_df2)

	dataset = get_only_has_weather_pred

	# get_only_has_temp = pd.DataFrame()

	# list_country = set(dataset['country_region'])

	# for country in list_country:
	# 	list_state = set(dataset.loc[dataset['country_region']==country, 'province_state'])
	# 	temp_df = dataset.loc[dataset['country_region']==country, :]
	# 	for state in list_state:
	# 		temp_df2 = temp_df.loc[temp_df['province_state']==state, :].sort_values(by=['date'], ascending=True).reset_index(drop=True)
	# 		if not any(temp_df2['TAVG'].isnull()):
	# 			if all(temp_df2['TAVG'] != 0):
	# 				get_only_has_temp = get_only_has_temp.append(temp_df2)

	# dataset = get_only_has_temp

	get_only_has_population = pd.DataFrame()

	list_country = set(dataset['country_region'])

	for country in list_country:
		list_state = set(dataset.loc[dataset['country_region']==country, 'province_state'])
		temp_df = dataset.loc[dataset['country_region']==country, :]
		for state in list_state:
			temp_df2 = temp_df.loc[temp_df['province_state']==state, :].sort_values(by=['date'], ascending=True).reset_index(drop=True)
			population = temp_df2['population']
			# print(country, state)
			# print(population.isnull().sum())
			if not population.isnull().all():
				pop_num = population.dropna().reset_index(drop=True).iloc[0]
				# print(pop_num)
				if pop_num != 0:
					temp_df2.at[: , 'population'] = pop_num
					if temp_df2['population'].isnull().any():
						exit()
					get_only_has_population = get_only_has_population.append(temp_df2)

	dataset = get_only_has_population

	# get_only_has_recovered = pd.DataFrame()

	# list_country = set(dataset['country_region'])

	# for country in list_country:
	# 	list_state = set(dataset.loc[dataset['country_region']==country, 'province_state'])
	# 	temp_df = dataset.loc[dataset['country_region']==country, :]
	# 	for state in list_state:
	# 		temp_df2 = temp_df.loc[temp_df['province_state']==state, :].sort_values(by=['date'], ascending=True).reset_index(drop=True)
	# 		recovered = temp_df2['recovered'].dropna()
	# 		print(recovered)
	# 		print(country, state)
	# 		print(any(i > 0 for i in recovered))
	# 		if any(i > 0 for i in recovered):
	# 			get_only_has_recovered = get_only_has_recovered.append(temp_df2)

	# print(get_only_has_recovered)

	# dataset = get_only_has_recovered

	dataset['location_name'] = (dataset['country_region'] + " : " + dataset['province_state']).tolist()

	# print(dataset)

	dataset.to_csv('simulation_data/dataset_full.csv')
