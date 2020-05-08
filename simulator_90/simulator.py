
import numpy as np
import pandas as pd
from csv import reader
from csv import writer
import scipy
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import timedelta, datetime
import argparse
import sys
import json
import ssl
import urllib.request
import statsmodels.api as sm
import pandasql as psql
from patsy import dmatrices
import os
import errno
from tqdm import tqdm
 
pd.options.mode.chained_assignment = None  # default='warn'
# np.seterr(divide='ignore', invalid='ignore')

def load_data(path,sourcefile):
		df = pd.read_csv(path+"/"+sourcefile)
		df['date'] =  pd.to_datetime(df['date'], format='%Y%m%d').dt.strftime('%m/%d/%y')
		return df

class SIR_ext_framework(object):
	def __init__(self,path,datasource,pop_data, location_name,location_value, start_date, predict_range,predict_start_date):
		self.location_name = location_name
		self.location_value = location_value
		self.start_date = start_date
		self.predict_range = predict_range 
		self.predict_start_date = predict_start_date
		# self.pop_data =load_data(path=path,sourcefile=pop_source)
		self.pop_data = pop_data
		# self.s_0 = 100000  #self.pop_data['population'].values[0]
		self.s_0 = pop_data
		self.i_0 = 2
		self.r_0 = 0
		self.dataset = load_data(path=path,sourcefile=datasource)
		self.dataset = self.dataset.loc[self.dataset['location_name']==self.location_value, :].reset_index(drop=True)

		self.dataset = self.dataset.rename(columns={'PRCP':'prcp','TAVG':'temp'})
		self.dataset['temp'] = self.dataset['temp'] /10
		self.dataset['prcp'] = self.dataset['prcp'] /10

		fnewdate = self.predict_start_date
		fnewdate = datetime.strptime(fnewdate, '%Y%m%d')
		# newdate = datetime.strptime(self.start_date, '%m/%d/%y')
		newdate = datetime.strptime(self.start_date, '%Y%m%d')
		self.dataset["date"] =pd.to_datetime(self.dataset["date"], format= '%m/%d/%y', errors='ignore')
		# self.dataset["date"] =pd.to_datetime(self.dataset["date"], format= '%Y%m%d', errors='ignore')
		self.dataset = self.dataset[(self.dataset["date"]>= (newdate))]

		self.covid_data =self.dataset[['country_region','date','confirmed','recovered','death']]
		self.covid_data = self.covid_data[self.covid_data["date"]< (fnewdate)]
		self.exog_data =self.dataset[['country_region','date','prcp','temp']]
		self.exog_data = self.exog_data[ (self.exog_data["date"]< fnewdate)]
		self.exog_preddata = self.dataset[(self.dataset["date"]>= fnewdate) ]
				
		# self.population = self.pop_data['population'].values[0]
		self.population = self.pop_data
		# print(newdate)
		# print(fnewdate)
		# print(self.dataset["date"])
		self.train_dataset = self.dataset[(self.dataset["date"]>= newdate) & (self.dataset["date"]< fnewdate)]
		# print(self.train_dataset)
		self.sim_dataset =  self.dataset[ (self.dataset["date"]>= fnewdate)]
		# print(self.sim_dataset)

		self.train_dataset['d_temp'] = np.where(self.train_dataset['temp']>=20, 1, 0)
		self.train_dataset['d_prcp'] = np.where(self.train_dataset['prcp']>=np.average(self.train_dataset['prcp'])+np.std(self.train_dataset['prcp']), 1, 0)

		newdate = datetime.strptime("3/20/20", '%m/%d/%y')
		z = pd.to_datetime(self.train_dataset["date"], format= '%m/%d/%y', errors='ignore')
		self.train_dataset['period_group'] = np.where(z>=newdate, 1, 0)
		self.train_dataset['did1'] = self.train_dataset['period_group']*self.train_dataset['d_temp']
		# print(self.predict_range,len(self.sim_dataset['date']))
		self.predict_range = np.min([self.predict_range, len(self.sim_dataset['date'])])
		
		 
	
	def test_covid(self):
		# df = pd.read_csv('data/time_series_19-covid-Confirmed-country.csv')
		# df[self.location_name] = df['Country/Region']
		loc_part = self.location_value.replace(" : ","_")
		loc_part = loc_part.replace(" ","_")

		filename = "./covid_log2/output_"+loc_part+".txt"
		if not os.path.exists(os.path.dirname(filename)):
			try:
				os.makedirs(os.path.dirname(filename))
			except OSError as exc: # Guard against race condition
				if exc.errno != errno.EEXIST:
					raise

		text_file = open(filename, "w")

		data = self.train_dataset[self.train_dataset['confirmed']>0]
		 
		data = data[data['confirmed']>0]
		data = psql.sqldf("""
		select {loc},d_temp,period_group,did1,d_prcp,  
		(ifnull(recovered,0)) as recovered,
		(ifnull(death,0)) as death,
		(ifnull(confirmed,0)) as confirmed from data
		
		""".format(loc=self.location_name))
		# print(np.std(data['confirmed']))
		Y = data['confirmed']
		#Y = (data['confirmed']-np.average(data['confirmed']))/np.std(data['confirmed'])
		X = data[['d_temp','period_group','did1']]
		# X = sm.add_constant(X)
		# model = sm.OLS(Y,X)
		# results = model.fit()
		# print(results.summary())

		formula = """confirmed ~ d_temp  + period_group + did1 """
		response, predictors = dmatrices(formula, data, return_type='dataframe')
		# print(self.location_value)
		# print('----response and predictors----')
		# print(response)
		# print(predictors)
		po_results = sm.GLM(response, predictors, family=sm.families.Poisson()).fit()
		# print(po_results.summary())
		text_file.write(po_results.summary().as_text())
		text_file.write('\n')

		Y = data['recovered']
		Y = (data['recovered']-np.average(data['recovered']))/np.std(data['recovered'])
		# model = sm.OLS(Y,X)
		# results = model.fit()
		# print(results.summary())
		formula = """recovered ~ d_temp  + period_group + did1 """
		response, predictors = dmatrices(formula, data, return_type='dataframe')
		# print(self.location_value)
		# print('----response and predictors----')
		# print(response)
		# print(predictors)
		po_results = sm.GLM(response, predictors, family=sm.families.Poisson()).fit()
		# print(po_results.summary())
		text_file.write(po_results.summary().as_text())
		text_file.write('\n')

		Y = data['death']
		Y = (data['death']-np.average(data['death']))/np.std(data['death'])
		# model = sm.OLS(Y,X)
		# results = model.fit()
		# print(results.summary())

		formula = """death ~ d_temp  + period_group + did1 """
		response, predictors = dmatrices(formula, data, return_type='dataframe')
		# print(self.location_value)
		# print('----response and predictors----')
		# print(response)
		# print(predictors)
		po_results = sm.GLM(response, predictors, family=sm.families.Poisson()).fit() 	
		# print(po_results.summary())
		text_file.write(po_results.summary().as_text())
		text_file.write('\n')

		text_file.close()
		
		

		
	def load_exog(self, location_name):
		#print( self.exog_data)
		df =self.train_dataset[['date', 'prcp','country_region','temp']]
		newdate = datetime.strptime(self.start_date, '%Y%m%d')
		df["date"] =pd.to_datetime(df["date"], format= '%m/%d/%y', errors='ignore')
		# location_name_df = df[(df[location_name] == self.location_value) &   (df["date"]>= (newdate))]
		location_name_df = df[(df["date"]>= (newdate))]
		# print(location_name_df)
		location_name_df=location_name_df.sort_values(by=['date'])
		return location_name_df 

	def load_pred_exog(self, location_name):
		df = self.sim_dataset[['date','date_idx','prcp','country_region','temp']] 
		 
		newdate = datetime.strptime(self.start_date,  '%Y%m%d')
		df["date"] =pd.to_datetime(df["date"], format= '%m/%d/%y', errors='ignore')
		location_name_df = df[(df[location_name] == self.location_value) &   (df["date"]>= (newdate))]
		#print(df)
		
		location_name_df = location_name_df.sort_values(by=['date'])
		return location_name_df 
	
	def load_pop(self, location_name):
		df = self.pop_data
		newdate = datetime.strptime(self.start_date,  '%Y%m%d')
		df["date"] =pd.to_datetime(df["date"], format= '%m/%d/%y', errors='ignore')
		location_name_df = df[(df[location_name] == self.location_value) &   (df["date"]>= (newdate))]
		#print(df)
		 
		location_name_df = location_name_df.sort_values(by=['date'])
		location_name_df=location_name_df.set_index('date')
		location_name_df = location_name_df['population']
		return location_name_df.iloc[0].loc[self.start_date:]

	def load_confirmed(self, location_name):
		df = self.train_dataset
		newdate = datetime.strptime(self.start_date,  '%Y%m%d')
		df["date"] =pd.to_datetime(df["date"], format= '%m/%d/%y', errors='ignore')
		location_name_df = df[(df[location_name] == self.location_value) &   (df["date"]>= (newdate))]
		#print(df)
		location_name_df = location_name_df.sort_values(by=['date'])
		location_name_df = location_name_df.fillna(0)
		location_name_df = pd.DataFrame({'confirmed':location_name_df['confirmed'],'date':location_name_df['date']})
		location_name_df["date"] =pd.to_datetime(location_name_df["date"], format= '%m/%d/%y', errors='ignore')
		location_name_df=location_name_df.set_index('date')
		return location_name_df



	def load_recovered(self, location_name):
		df = self.train_dataset

		newdate = datetime.strptime(self.start_date, '%Y%m%d')
		df["date"] =pd.to_datetime(df["date"], format= '%m/%d/%y', errors='ignore')
		location_name_df = df[(df[location_name] == self.location_value) &   (df["date"]>= (newdate))]
		location_name_df = location_name_df.sort_values(by=['date'])
		location_name_df = location_name_df.fillna(0)
		location_name_df = pd.DataFrame({'recovered':location_name_df['recovered'],'date':location_name_df['date']})
		location_name_df["date"] =pd.to_datetime(location_name_df["date"], format= '%m/%d/%y', errors='ignore')
		location_name_df=location_name_df.set_index('date')
		 
		return location_name_df


	def load_dead(self, location_name):
		df = self.train_dataset
		newdate = datetime.strptime(self.start_date, '%Y%m%d')
		df["date"] =pd.to_datetime(df["date"], format= '%m/%d/%y', errors='ignore')
		location_name_df = df[(df[location_name] == self.location_value) &   (df["date"]>= (newdate))]
		#print(df)
		location_name_df =location_name_df.sort_values(by=['date'])
		location_name_df = location_name_df.fillna(0)
		location_name_df = pd.DataFrame({'death':location_name_df['death'],'date':location_name_df['date']})
		location_name_df["date"] =pd.to_datetime(location_name_df["date"], format= '%m/%d/%y', errors='ignore')
		location_name_df=location_name_df.set_index('date')
		return location_name_df
	

	def extend_index(self, index, new_size):
		#index1 = [datetime.strptime(x,'%m/%d/%Y') for x in index]
		index1 = index
		values = index1.values
		current = index1[-1] #datetime.strptime(index1[-1], '%m/%d/%y')
		values = [pd.to_datetime(str(d)) for d in values]
		while len(values) < new_size:
			current = current + timedelta(days=1)
			values = np.append(values, current) # datetime.strftime(current, '%m/%d/%y'))
		return values

	def predict(self, beta, gamma,exog_coefs, data, recovered, death, location_name, s_0, i_0, r_0):
		new_index = self.extend_index(data.index, self.predict_range)
		new_points = np.zeros(4)
		exog = self.exog_preddata[0:len(new_index)]
		new_points[0] = beta
		new_points[1] = gamma
		new_points[2] = exog_coefs[0]
		new_points[3] =  exog_coefs[1]
		xx = np.zeros(len(new_index))
		for s in range(2):
			i = 0
			for c in ['prcp','temp']:
				xx[i] = xx[i]+exog[c].values[i]*exog_coefs[i]
				i = i + 1
		size = len(new_index)
		def SIR(t, y):
			S = y[0]
			I = y[1]
			R = y[2]    
			return [-beta*S*I, beta*S*I-gamma*I, gamma*I]
		extended_actual = np.concatenate((data.values, [0] * (size - len(data.values))))
		extended_recovered = np.concatenate((recovered.values, [None] * (size - len(recovered.values))))
		extended_death = np.concatenate((death.values, [None] * (size - len(death.values))))
		# print("new_index", new_index)
		# print("extended_actual", extended_actual)
		# print("xx", xx)
		# print("extended_actual", len(extended_actual))
		# print("xx", len(xx))
		# print("extended_recovered", extended_recovered)
		# print("extended_death", extended_death)
		return new_index, extended_actual+xx, extended_recovered, extended_death, solve_ivp(SIR, [0, size], [s_0,i_0,r_0], t_eval=np.arange(0, size, 1))
		# return new_index, extended_actual.append(xx), extended_recovered, extended_death, solve_ivp(SIR, [0, size], [s_0,i_0,r_0], t_eval=np.arange(0, size, 1))

	def train(self):
		recovered = self.load_recovered(self.location_name)['recovered']
		death = self.load_dead(self.location_name)['death']
		data = (self.load_confirmed(self.location_name) )['confirmed']   #- recovered - death)
		new_points =[]
		lb = []
		ub = []
		b = []
		for i in range(4):
		    if i <=1:
		        new_points = new_points +[0.0]
		        lb = lb + [0.00000001]
		        ub = ub + [0.4]
		        b = b + [(0.00000001,0.6)]
		    else:
		        new_points = new_points +[0.0]
		        lb = lb + [-0.50]
		        ub = ub + [0.5]
		        b = b + [(-0.5,0.5)]


		m_bounds = b
		# print('---data===')
		# print(data)
		optimal = minimize(self.loss, new_points, args=(data), method='L-BFGS-B', bounds=m_bounds)
		print(optimal)
			
		new_points = optimal.x
		beta = new_points[0]
		gamma = new_points[1]
		exog_coefs = [x for x in new_points[2:]]
		new_index, extended_actual, extended_recovered, extended_death, prediction = self.predict(beta, gamma,exog_coefs, data, recovered, death, self.location_name, self.s_0, self.i_0, self.r_0)
		df = pd.DataFrame({'Infected data': extended_actual, 'Recovered data': extended_recovered, 'Death data': extended_death, 'Susceptible': prediction.y[0], 'Infected': prediction.y[1], 'Recovered': prediction.y[2]}, index=new_index)
		filename = self.location_value.replace(' : ','')
		df.to_csv(filename+'_report.csv')
		fig, ax = plt.subplots(figsize=(15, 10))
		ax.set_title(self.location_value)
		df.plot(ax=ax)
		print(f"location_name={self.location_value}, beta={beta:.8f}, gamma={gamma:.8f}, r_0:{(beta/gamma):.8f}")
		loc_part = self.location_value.replace(" : ","_")
		loc_part = loc_part.replace(" ","_")
		filename = "output_"+loc_part+".png"
		fig.savefig(filename)


	def loss(self,point,data):
		"""
		RMSE between actual confirmed cases and the estimated infectious people with given beta and gamma.
		"""
		x_data =self.load_exog(self.location_name)[['prcp','temp']]
		# print('--x-data--')
		# print(x_data)
		size = len(data)
		# print(size)
		the_t = np.arange(0, size, 1)
		xx = np.zeros(size)
			
		days = range(len(the_t))
		#d = dict([days,the_t])
			
		new_points = point
		#print(point)
		beta = new_points[0]
		gamma = new_points[1]
		exog_coefs = [x for x in new_points[2:]]
		for s in the_t:
			i = 0
			for c in ['prcp','temp']:
				xx[i] = xx[i]=x_data[c].values[i]*exog_coefs[i]
				i = i + 1

		def SIR(t, y):
			S = y[0]
			I = y[1]
			R = y[2]           
			return [-beta*S*I, beta*S*I-gamma*I, gamma*I]

		solution = solve_ivp(SIR, t_span=[0, size], y0=[self.s_0,self.i_0,self.r_0], t_eval=np.arange(0, size, 1), vectorized=True)
		

		return np.sqrt(np.mean((solution.y[1]+ xx - data)**2))

def simulate_SIR(
	path="./data",
		datasource="dataset_full.csv",
		pop_source="dataset_full.csv",
		location_name="location_name",
		start_date="1/22/20",
	    forecast_horizon=90,
        predict_start_date="4/27/20"):

	locations = sorted(set(pd.read_csv('./data/dataset_full.csv')['location_name'].tolist()))
	populations_df = pd.read_csv('./data/dataset_full.csv')[['location_name','population']]
	dataset = pd.read_csv('./data/dataset_full.csv')

	predict_range = forecast_horizon

	# filename = "./covid_log/running_log.txt"
	# if not os.path.exists(os.path.dirname(filename)):
	# 	try:
	# 		os.makedirs(os.path.dirname(filename))
	# 	except OSError as exc: # Guard against race condition
	# 		if exc.errno != errno.EEXIST:
	# 			raise

	text_file = open('error_location_log.txt', "w+")

	# locations = ["China : UNK"]

	#path,datasource, location_name,location_value, start_date, predict_range,predict_start_date
	for location_value in locations:
		print(location_value)
		# text_file.write(location_value+'\n')
		pop_data = populations_df.loc[populations_df['location_name']==location_value, 'population'].dropna().reset_index().iloc[0]['population']
		
		predict_start_date = str(int(dataset.loc[(dataset['location_name']==location_value), ['date','confirmed']].dropna().sort_values(by=['date'], ascending=False).iloc[7]['date']))
		# predict_start_date = datetime.strptime(predict_start_date, '%y%m%d')
		# predict_start_date = pd.to_datetime(predict_start_date, format= '%Y%m%d', errors='ignore')
		print(predict_start_date)
		
		start_date = str(dataset.loc[(dataset['location_name']==location_value)&(dataset['confirmed']>=1), 'date'].iloc[0])
		# start_date = pd.to_datetime(start_date, format= '%y%m%d', errors='ignore')
		print(start_date)
		print(pop_data)
		SIR = SIR_ext_framework(path=path,
		datasource=datasource,
		pop_data=pop_data,
		location_name=location_name,
		location_value=location_value,
		start_date=start_date,
		predict_range=predict_range,
        predict_start_date=predict_start_date)
		try:
			SIR.test_covid()
			SIR.train()
		except ValueError as err:
			text_file.write(location_value+ ':' + str(err) +'\n')

	text_file.close()
		
if __name__ == '__main__':
	simulate_SIR()
