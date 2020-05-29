
from matplotlib.dates import AutoDateFormatter, AutoDateLocator
import numpy as np
import pandas as pd
from scipy.integrate import odeint
from csv import reader
from csv import writer
import scipy
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from datetime import timedelta, datetime
import argparse
import sys
import os
import json
import ssl
import urllib.request
import statsmodels.api as sm

import pandasql as psql

from patsy import dmatrices


def load_data(path,sourcefile):
        df = pd.read_csv(path+"/"+sourcefile)
        #df['date'] =  pd.to_datetime(df['date'], format='%Y%m%d').dt.strftime('%m/%d/%y')
        return df



def test_covid(dataset,location_name):
     
    data = dataset[dataset['confirmed']>0]
    data = psql.sqldf("""
    select {loc},d_temp,period_group,did1,
    avg(ifnull(recovered,0)) as recovered,
    avg(ifnull(death,0)) as death,
    avg(ifnull(confirmed,0)) as positive from data
    group by 1,2,3,4,5
    """.format(loc=location_name))
    
    Y = data['positive']
    Y = (data['positive']-np.average(data['positive']))/np.std(data['positive'])
    X = data[['d_temp','period_group','did1']]
    

    formula = """positive ~ d_temp + period_group + did1 """
    response, predictors = dmatrices(formula, data, return_type='dataframe')
    po_results = sm.GLM(response, predictors, family=sm.families.Poisson()).fit()
    print(po_results.summary())

    Y = data['recovered']
    Y = (data['recovered']-np.average(data['recovered']))/np.std(data['recovered'])
    formula = """recovered ~ d_temp + period_group + did1 """
    response, predictors = dmatrices(formula, data, return_type='dataframe')
    po_results = sm.GLM(response, predictors, family=sm.families.Poisson()).fit()
    print(po_results.summary())

    Y = data['death']
    Y = (data['death']-np.average(data['death']))/np.std(data['death'])
    
    formula = """death ~ d_temp + period_group + did1 """
    response, predictors = dmatrices(formula, data, return_type='dataframe')
    po_results = sm.GLM(response, predictors, family=sm.families.Poisson()).fit()
    print(po_results.summary())


    
def test_gov_covid(dataset,location_name):
     
    data = dataset[dataset['positive']>0]        
    Y = data['positive']
    Y = (data['positive']-np.average(data['positive']))/np.std(data['positive'])
    X = data[['d_temp','period_group','did1']]    
    formula = """positive ~ d_temp  + period_group+CS+ED+GP+NEBC+OTH+SAH + did1 """
    response, predictors = dmatrices(formula, data, return_type='dataframe')
    po_results = sm.GLM(response, predictors, family=sm.families.Poisson()).fit()
    print(po_results.summary())
    Y = data['recovered']
    Y = (data['recovered']-np.average(data['recovered']))/np.std(data['recovered'])    
    formula = """recovered ~ d_temp  + period_group+CS+ED+GP+NEBC+OTH+SAH + did1 """
    response, predictors = dmatrices(formula, data, return_type='dataframe')
    po_results = sm.GLM(response, predictors, family=sm.families.Poisson()).fit()
    print(po_results.summary())
    Y = data['death']
    Y = (data['death']-np.average(data['death']))/np.std(data['death'])    
    formula = """death ~ d_temp  + period_group+CS+ED+GP+NEBC+OTH+SAH + did1 """
    response, predictors = dmatrices(formula, data, return_type='dataframe')
    po_results = sm.GLM(response, predictors, family=sm.families.Poisson()).fit()
    print(po_results.summary()) 



def solve_one(state,df,cfr,alpha,beta,c_x,c_names,N,horizon):  
    def mysysfunc(h,s): 
        z1 = 0.0
        z2 = 0.0  
        tt = int(s) 
        if tt>horizon and horizon<=len(df['dateval']):
            tt=horizon-1
            z1 = 0.0
            z2 = 0.0
        try:      
            for i in range(len(c_x)):
            
                    z1 = z1 + df[c_names[i]].values[tt]*c_x[i]
                     
        except:
            z1 = z1
             
        dsdt = - (alpha/N) * h[0]*h[1]
        didt = (alpha/N)*h[0]*h[1] - beta*h[1]  -z1
        drdt = z1+beta*h[1]  
          
        dhdt = [dsdt,didt,drdt]
        return dhdt

    t = [i for i in range(horizon)]
    # ############SETUP AND RUN THE SIMULATION=============
    #try:
    h0 = [df['population'].values[0],df['confirmed'].values[0],df['removed'].values[0]]
     
    y = odeint(mysysfunc,h0,t) # integrate
    y = np.where(y<0,0,y)
     

    return y
     
    

def runDynamicSimulator(data1,coefsdfR,sir_names,xnamesr,horizon1):
    	###DEFINE SIMULATION CONSTANTS#######################################################
	# horizon = len(date_list)  # days . - forecast days
    # data1 starts from end of historical data date minus 7 days (from training)
    data = data1.copy()
    data = data.fillna(0)
    
    coefsR = coefsdfR.copy()
     
    coefsR = coefsR.fillna(0.0)
     
    
    its = 0
    for state in data['state'].drop_duplicates():
        
        if 1==1:
        #try:
            
            df = data[data['state']==state]
            print(df[0:10])
            if len(df['dateval'])>0:
                df = df.sort_values(by=['dateval'])
                df = df.loc[df['confirmed'].ne(0).idxmax():]
                df['pred_confirmed'] = 0.0
                df['pred_susceptible'] = 0.0
                df['pred_removed'] = 0.0
                #df = df.reset_index()
                dd = df[df['confirmed']>0]
                N = np.max(df['population'])
                print("Simulating state= "+state)
                horizon = len(df['dateval'])  
                
                cfr = coefsR[coefsR['state']==state] 
                alpha = df['alpha'].values[0] 
                beta =  coefsR['lag_confirmed'].values[0]
                horizon_days = [i for i in range(horizon)]
                x1 = ['Intercept','gov_action','TAVG']
                c_xr = np.zeros(len(x1))
                
                df['pred_susceptible']= N #df['susceptible']
                for t in horizon_days:
                    if t==0:
                        df['pred_confirmed'].values[t] =df['confirmed'].values[t]
                        df['pred_susceptible'].values[t] = N #df['susceptible'].values[t]
                        df['pred_removed'].values[t] = df['removed'].values[t]
                        
                    if t>0 and t<horizon:
                        z1 = 0.0
                        for c in range(len(x1)):
                            z1 = z1 + cfr[x1[c]].values[0]*df[x1[c]].values[t]
                        df['pred_confirmed'].values[t] = df['pred_confirmed'].values[t-1]+(alpha)*df['pred_susceptible'].values[t-1]*df['pred_confirmed'].values[t-1] - beta*df['pred_confirmed'].values[t-1] - z1
                        
                        df['pred_removed'].values[t] = df['pred_removed'].values[t-1]+beta*df['pred_confirmed'].values[t-1] + z1
                        
                        df['pred_susceptible'].values[t] = df['pred_susceptible'].values[t-1] - (alpha)*df['pred_susceptible'].values[t-1]*df['pred_confirmed'].values[t-1]
                        
                
            if its ==0:
                out_df = df
            else:
                out_df = out_df.append(df)    
            
            
            # # plot results
            plt.figure(1)
            plt.plot(out_df['dateval'],out_df['susceptible'],'b-')
            plt.plot(out_df['dateval'],out_df['confirmed'],'r--')
            plt.plot(out_df['dateval'],out_df['removed'],'g--')
            
            plt.xticks(rotation=45)
            plt.xlabel('Time')
            plt.ylabel('Populations')
            plt.legend(['Suceptibes','Confirmed','Removed'])
            plt.tight_layout()
            plt.savefig(os.path.join('/Users/aokossi/Documents/public service resarch/output/covid_plot_'+state+'.png'))
            plt.clf()
            its = its +1

        else:
        #except:
            print("skipped state for lack of government data "+state)

    return out_df
        


def runSimulator(data1,coefsdfR,sir_names,xnamesr,horizon1,date, print_graph):
    	###DEFINE SIMULATION CONSTANTS#######################################################
	# horizon = len(date_list)  # days . - forecast days
    # data1 starts from end of historical data date minus 7 days (from training)
    # print(data1.columns)
    if date > 0:
        data1.loc[data1.date>date, 'gov_action'] = 0
    data = data1.copy()
    data = data.fillna(0)
    
    coefsR = coefsdfR.copy()
     
    coefsR = coefsR.fillna(0.0)
     
    df = pd.DataFrame()
    out_df = pd.DataFrame()
    its = 0
    for state in coefsR['state'].drop_duplicates():
        if 1==1:
        #try:
            cr = coefsR[coefsR['state']==state]
            df = data[data['state']==cr['state'].values[0]] 
            if len(df)>0 and len(cr)>0:
            #if len(df['dateval'])>0:    
                df = df.sort_values(by=['dateval'])
                df = df.reset_index()
                df = df.loc[df['confirmed'].ne(0).idxmax():]
                
                dd = df[df['confirmed']>0]
                N = np.max(df['population'])
                print("Simulating state= "+state)
                horizon = len(df['dateval']) # np.where(len(df['dateval'])<horizon1,len(df['dateval']),horizon1)
                
                cfr = cr 
                alpha = df['alpha'].values[0]  
                #t = np.linspace(0, horizon)  #[0:horizon] # times to report solution . - time in day sequence
                horizon_days = [i for i in range(horizon)]
                x1 = ['Intercept','gov_action','TAVG']
                c_xr = np.zeros(len(x1))
                for c in range(len(x1)):
                    c_xr[c] = cfr[x1[c]].values[0]
                c_xd = np.zeros(len(x1))

                # print(cfr)
                # print(c_xr)
                # print(c_xd)

                y = solve_one(state=state,df=df,cfr=cfr,alpha=alpha,beta=cfr['lag_confirmed'].values[0],c_x = c_xr,c_names = x1,N=N,horizon=horizon)
                
                print("solution for "+state)
                df['pred_susceptible'] = y[0:horizon,0]
                df['pred_confirmed'] = y[0:horizon,1]
                df['pred_removed'] = y[0:horizon,2]

                if print_graph == True:
                    # plot results
                    plt.figure(1)
                    plt.figure(figsize=(15,10))
                    xtick_locator = AutoDateLocator()
                    xtick_formatter = AutoDateFormatter(xtick_locator)

                    date_list = pd.to_datetime(df['dateval'])
                    
                    ax = plt.axes()
                    ax.xaxis.set_major_locator(xtick_locator)
                    ax.xaxis.set_major_formatter(xtick_formatter)
                    plt.plot(date_list,y[:horizon,0],'b-')
                    plt.plot(date_list,y[:horizon,1],'r--')
                    plt.plot(date_list,y[:horizon,2],'g--')
                    plt.xlabel('Time')
                    plt.ylabel('Populations')
                    plt.legend(['Suceptibes','Confirmed','removed'])
                    plt.title('Prediction at '+state)
                    plt.savefig(os.path.join('covid_plot/covid_plot_'+state+'_'+str(date)+'.png'))
                    plt.clf()
                    plt.close()
                out_df = out_df.append(df,ignore_index=True )   
            
            
            
            its = its +1

        else:
        #except:
            print("skipped state for lack of government data "+state)

    return out_df



def causal_simulation(path,start_date,f_start_date,datafile="dataset_full.csv",govpolicyfile="gov_dates_mandates.csv", print_graph=True):
    data = pd.read_csv(path+"/"+datafile)
    
    start_dt = datetime.strptime(start_date, '%m/%d/%y').strftime('%Y-%m-%d')
    print(start_dt)
      
    dateval = pd.date_range(start_dt, periods=horizon+180).tolist()
    dates = pd.DataFrame({'dateval': dateval})
    dates['dateval'] = dates['dateval'].apply(lambda x: datetime.strptime(str(x),'%Y-%m-%d %H:%M:%S').strftime('%Y-%m-%d') )
    data['dateval'] = data['date'].apply(lambda x: datetime.strptime(str(x),'%Y%m%d' ).strftime('%Y-%m-%d') )
     
    data['state'] = data['province_state']
   
    zx = 0
         
    
    data = psql.sqldf("""
    select province_state , country_region as country, date, confirmed,
       recovered, death, population, TAVG/10 as TAVG,
     a1.location_name, a1.dateval, country_region ||'-'||state as state,
    case when TAVG<=0 then 1 else 0 end as is_freezing,
    case when TAVG>0 and TAVG/10<20 then 1 else 0 end as is_cold, 
    case when TAVG>=20 and TAVG/10<35 then 1 else 0 end as is_warm,
    case when TAVG>=35 then 1 else 0 end as is_hot,
    case when TAVG>=20 then 1 else 0 end as temp_th,
     
      case when julianday(a1.dateval)>julianday('2020-03-20') then 1 else 0 end
       as gov_action
    from data a1   
    """).drop_duplicates()

   
    data['Intercept'] = 1.0
    data = data[data['dateval']>=start_dt]
    data['holdout'] = np.where(data['dateval']>=datetime.strptime(f_start_date, '%m/%d/%y').strftime('%Y-%m-%d'),1,0)
    
    data_save = data.copy()  
    # data smoothing to correct irregular data issues: like dropped cumulative values
    data1 = pd.DataFrame()
     
    z = 0

    # data['recovered'] = data['recovered']/100+0.0
    # data['confirmed'] = data['confirmed']/100+0.0
    # data['death'] = data['death']/100+0.0
    # data['population'] = data['population']/100+0.0
    # data['TAVG'] = data['TAVG']/100+0.0
    
    for state in data['state'].drop_duplicates():
            dat = data[(data['state']==state) ].sort_values(by=['dateval'])
            if len(dat['dateval'])>1:
                dat = dat.fillna(0)
                dat = dat.loc[dat['confirmed'].ne(0.0).idxmax():]
                rho_data = dat[dat['confirmed']>0].sort_values(by=['dateval'])
                rho_data = rho_data[0:30]
                
                zz1 = 0.0
                zz2 = 0.0
                zz3 = 0.0
                for s in range(len(rho_data['dateval'])):
                    if (s>0):
                        if rho_data['confirmed'].values[s]-rho_data['confirmed'].values[s-1]>0:
                            zz1 = zz1 + rho_data['confirmed'].values[s]-rho_data['confirmed'].values[s-1]
                        if rho_data['recovered'].values[s]-rho_data['recovered'].values[s-1]>0:
                            zz2 = zz2 + rho_data['recovered'].values[s]-rho_data['recovered'].values[s-1]
                        if rho_data['death'].values[s]-rho_data['death'].values[s-1]>0:
                            zz3  = zz3 + rho_data['death'].values[s]-rho_data['death'].values[s-1]
                rho = 0.0
                if (zz2+zz3) >0.0:
                    rho = (zz1+zz2+zz3)/(zz2+zz3) 
                print("R_0 for "+state +" : "+str(rho))
                dat['lag_confirmed'] = 0.0
                dat['lag_recovered'] = 0.0
                dat['lag_death'] = 0.0
                dat['lag_removed'] = 0.0

                dat['d_recovered'] = 0.0
                dat['d_death'] = 0.0
                dat['d_removed'] = 0.0
                  
                N = dat['population'].values[0]
                dd = dat[dat['confirmed']>1]
                
                
                dat['R_0'] = rho
                z1 = 0
                tt = 1
                dat['removed'] = dat['death'] + dat['recovered'] 
                
                for t in range(len(dat)):
                        
                    
                    if t>0 and t<=len(dat):
                        dat['lag_confirmed'].values[t] = dat['confirmed'].values[t-1]
                        dat['lag_removed'].values[t] = dat['removed'].values[t-1]
                        dat['lag_recovered'].values[t] = dat['recovered'].values[t-1]
                        dat['lag_death'].values[t] = dat['death'].values[t-1] 
                        dat['d_recovered'].values[t] =  dat['recovered'].values[t] - dat['recovered'].values[t-1] 
                        dat['d_death'].values[t] = dat['death'].values[t]-dat['death'].values[t-1]
                        dat['d_removed'].values[t] = dat['removed'].values[t]-dat['removed'].values[t-1]

                               
                # if z==0:
                #     data1 = dat
                # else:    
                data1 = data1.append(dat,ignore_index=True)
                z = z +1
                         

    data = data1.copy()
    rb = np.mean(data[data['R_0']>0]['R_0'])
    data['R_0'] = np.where(data['R_0']==0,rb,data['R_0'])
    data = data.fillna(0)   
    data.to_csv(path+"/input_data.csv")
    data_train = data[(data['removed']>0) & (data['holdout']==0)][['Intercept','state','TAVG','gov_action','is_freezing','is_cold','is_warm','is_hot','lag_confirmed','lag_death','lag_recovered','d_death','d_recovered','d_removed','removed']]
      
    endog =data_train['d_removed']
    exog = data_train[[ 'Intercept','gov_action','TAVG','lag_confirmed']]
    model = sm.MixedLM(endog, exog, exog_re=exog[[ 'Intercept','lag_confirmed']],  groups=data_train["state"])
    po_results = model.fit()
    print(po_results.summary())
     
 
    k = []
    v1 = []
    v2 = []
    v3 = []
    d = po_results.random_effects
    
    for i in d:
        my_str = ''.join((ch if ch in '0123456789.-' else ' ') for ch in str(d[i]))
        listOfNumbers = [float(i) for i in my_str.split()]
        v1 = v1 +[str(d[i]).split(" ")[0] ]
        l = str(d[i]).split(" ")
        if str(listOfNumbers[0]).strip()=='':
            v2 = v2 +[0.00 ]
            v3 = v3 + [0.0]
        else:    
            v2 = v2 +[listOfNumbers[0] ]
            v3 = v3 +[listOfNumbers[1] ]
        
        k = k + [i]
    r_combined = pd.DataFrame({'state':k,'coef_name':v1,'coef_value':v2,'re_lag_confirmed':v3})
    
    r_combined['fe_Intercept'] = po_results.fe_params['Intercept']
    r_combined['Intercept'] = r_combined['fe_Intercept']+r_combined['coef_value']
   
    r_combined['lag_confirmed'] = po_results.fe_params['lag_confirmed'] +r_combined["re_lag_confirmed"]
    r_combined['gov_action'] = po_results.fe_params['gov_action']
    r_combined['TAVG'] = po_results.fe_params['TAVG']
    
    r_combined.fillna(0.0)
    r_combined.to_csv(path+"/recover_coefs.csv")
    
    mean_beta = np.mean(r_combined[r_combined['lag_confirmed']>0]['lag_confirmed'])
    r_combined['lag_confirmed'] = np.where(r_combined['lag_confirmed']<0,mean_beta,r_combined['lag_confirmed'])
    
    tti = 0
     
    states = data['state'].drop_duplicates()
    data2 = data.copy()
    data3 = pd.DataFrame()
    for s in states:
        rc = r_combined[r_combined['state']==s]
        #print(rc)
        dat = data2[data2['state']==s]
         
        if len(dat)>0 and len(rc)>0:
            print(s)
            dat = dat.sort_values(by=['dateval'])
            #dat = dat.reset_index()
            beta = rc['lag_confirmed'].values[0]
            N = dat['population'].values[0]
            dat['susceptible'] = np.where(dat['holdout']==0, N+0.0,0.0)
            
            alpha = dat['R_0'].values[0]*beta
            dat['alpha'] = alpha

            
            if tti==0:
               data3 = dat
            else:
                data3 = data3.append(dat,ignore_index=True )
            tti = tti +1     

    
    
    print(data3['state'].drop_duplicates())
     
    # data3.to_csv(path+"/before_sim_data_test.csv")
    #runDynamicSimulator 
    #runSimulator(data1,coefsdfR,sir_names,xnamesr,horizon1)
    date_start_sim = 20200510

    sim_data_output_after = runSimulator(data1=data3[data3['holdout']==1],
    coefsdfR=r_combined,
    sir_names=['susceptible','confirmed','death','removed'],
    xnamesr=['Intercept','gov_action','TAVG','lag_confirmed'],
    horizon1=60, date=date_start_sim, print_graph)
    sim_data_output_after.to_csv("simulator_output/simulations_adjust_at_"+str(date_start_sim)+".csv")

    sim_data_output_before = runSimulator(data1=data3[data3['holdout']==1],
    coefsdfR=r_combined,
    sir_names=['susceptible','confirmed','death','removed'],
    xnamesr=['Intercept','gov_action','TAVG','lag_confirmed'],
    horizon1=60, date=0, print_graph)
    sim_data_output_before.to_csv("simulator_output/simulations_adjust_at_"+str(date_start_sim)+".csv")

    sim_data_compare = sim_data_output_after.merge(sim_data_output_before, on=['index', 'province_state', 'country','date','dateval','location_name'], suffixes=('_after', '_before'))
    sim_data_compare['diff_susceptible'] = sim_data_compare['pred_susceptible_after'] - sim_data_compare['pred_susceptible_before']
    sim_data_compare['diff_confirmed'] = sim_data_compare['pred_confirmed_after'] - sim_data_compare['pred_confirmed_before']
    sim_data_compare['diff_removed'] = sim_data_compare['pred_removed_after'] - sim_data_compare['pred_removed_before']

    sim_data_compare = sim_data_compare.loc[:, ['province_state', 'country','date','dateval','location_name','pred_susceptible_after', 'pred_confirmed_after', 'pred_removed_after', 'pred_susceptible_before', 'pred_confirmed_before', 'pred_removed_before', 'diff_susceptible', 'diff_confirmed', 'diff_removed']]
    sim_data_compare.to_csv("simulator_output/simulations_compare"+str(date_start_sim)+".csv")

    if print_graph == True:
        for location in sim_data_compare['location_name'].drop_duplicates():
            dat = sim_data_compare[(sim_data_compare['location_name']==location)].sort_values(by=['date'])
            # plot results
            plt.figure(1)
            plt.figure(figsize=(15,10))
            xtick_locator = AutoDateLocator()
            xtick_formatter = AutoDateFormatter(xtick_locator)
            date_list = pd.to_datetime(dat['dateval'])
            ax = plt.axes()
            ax.xaxis.set_major_locator(xtick_locator)
            ax.xaxis.set_major_formatter(xtick_formatter)
            plt.plot(date_list,dat['diff_susceptible'],'b-')
            plt.plot(date_list,dat['diff_confirmed'],'r--')
            plt.plot(date_list,dat['diff_removed'],'g--')
            plt.xlabel('Time')
            plt.ylabel('Populations')
            plt.title('Compare Before/After Gov. Intervention Adjust at '+''.join(e for e in location if e.isalnum())+' : '+str(date_start_sim))
            plt.legend(['Diff Suceptibes','Diff Confirmed','Diff Removed'])
            plt.savefig(os.path.join('covid_plot/covid_plot_compare_'+''.join(e for e in location if e.isalnum())+'_'+str(date_start_sim)+'.png'))
            plt.clf()
            plt.close() 


if __name__=="__main__":

    path="simulation_data"
    datasource="dataset_full.csv"
    pop_source="pop_dataset.csv"
    location_name="country_region"
    start_date="2/22/20"
    horizon = 30

    causal_simulation(path=path,start_date=start_date,f_start_date="4/20/20", datafile="dataset_full.csv",govpolicyfile="gov_dates_mandates.csv", print_graph=True)

    exit(0)

    # locations_csv=['Italy','Korea, South','United States']
    # df = load_data(path=path,sourcefile=datasource)
    # df['dateval'] =  pd.to_datetime(df['date'], format='%Y%m%d').dt.strftime('%m/%d/%y')
    # df['state'] = df['province_state']
    
    
    # df = df.rename(columns={'PRCP':'prcp','TAVG':'temp'})
    # df['temp'] = df['temp'] /10
    # df['prcp'] = df['prcp'] /10
    # dataset = df.rename(columns={'PRCP':'prcp','TAVG':'temp'})          
    # dataset['d_temp'] = np.where(dataset['temp']>=20, 1, 0)
    # dataset['d_prcp'] = np.where(dataset['prcp']>=np.average(dataset['prcp'])+np.std(dataset['prcp']), 1, 0)
    # newdate = datetime.strptime("3/20/20", '%m/%d/%y')     
    # dataset['period_group'] = np.where(dataset['date'].astype(int)>=20200320, 1, 0)
    # dataset['did1'] = dataset['period_group']*dataset['d_temp']
    # test_covid(dataset=dataset,location_name=location_name)
    #====================================================================================

    # datasource="dataset_full.csv"
    # df = load_data(path=path,sourcefile=datasource)
    # df['dateval'] =  pd.to_datetime(df['date'], format='%Y%m%d').dt.strftime('%m/%d/%y')
    # df['state'] = df['province_state']
    # g_df = load_data(path=path,sourcefile="agg_gov_data.csv")[['state','CS','ED','GP','NEBC','OTH','SAH']]
    # df = pd.merge(df,g_df,on=['state'],how="left").fillna(0)

    #df = df.rename(columns={'PRCP':'prcp','TAVG':'temp'})

    # df['temp1'] = df['TAVG']
    # df['prcp1'] = df ['PRCP']     
    # df['temp1'] = df['temp1'] /10
    # df['prcp1'] = df['prcp1'] /10   
    #dataset = df.rename(columns={'PRCP':'prcp','TAVG':'temp'})          
    # dataset = df.copy()
    # dataset['d_temp'] = np.where(dataset['temp1']>=20, 1, 0)
    # dataset['d_prcp'] = np.where(dataset['prcp1']>=np.average(dataset['prcp1'])+np.std(dataset['prcp1']), 1, 0)
    # newdate = datetime.strptime("3/20/20", '%m/%d/%y')     
    # dataset['period_group'] = np.where(dataset['date'].astype(int)>=20200320, 1, 0)
    # dataset['did1'] = dataset['period_group']*dataset['d_temp']
    # print(dataset.columns)

    # data = psql.sqldf("""
    # select {loc},d_temp,period_group,did1,d_prcp,
    # CS,ED,GP,NEBC,OTH,SAH,
    # avg(ifnull(recovered,0)) as recovered,
    # avg(ifnull(death,0)) as death,
    # avg(ifnull(confirmed,0)) as positive from dataset
    # group by 1,2,3,4,5,6,7,8,9,10
    # """.format(loc=location_name))
    # test_gov_covid(dataset=data,location_name=location_name)

