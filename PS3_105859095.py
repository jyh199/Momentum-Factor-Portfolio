import os
import datetime as dt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.tseries.offsets import MonthEnd

# Question 1

# read csv files
os.chdir("D:/MY/UCLA/Academic/2022 spring/QAM/data/")
crsp_raw = pd.read_csv('crsp_raw.csv')
dlret_raw = pd.read_csv('dlret_raw.csv')

# Change variable types
crsp_raw[['permno','permco']] = crsp_raw[['permno','permco']].astype(int)
crsp_raw['date'] = pd.to_datetime(crsp_raw['date'], format='%Y-%m-%d', errors='ignore')
crsp_raw['date'] = pd.DataFrame(crsp_raw['date'].values.astype('datetime64[ns]')) + MonthEnd(0)
crsp_raw = crsp_raw.sort_values(by=['permno', 'date']).reset_index(drop=True)

dlret_raw['permno'] = dlret_raw['permno'].astype(int)
dlret_raw['date'] = pd.to_datetime(dlret_raw['dlstdt'], format='%Y-%m-%d', errors='ignore')
dlret_raw['date'] = pd.DataFrame(dlret_raw['date'].values.astype('datetime64[ns]')) + MonthEnd(0)
dlret_raw = dlret_raw.sort_values(by=['permno', 'date']).reset_index(drop=True)

# Merging with delisting returns
crsp = crsp_raw.merge(dlret_raw, how='outer', on=['date', 'permno'])
crsp['ret'] = np.where(crsp['ret'].notna() & crsp['dlret'].notna(), 
                        (1+crsp['ret'])*(1+crsp['dlret'])-1, crsp['ret'])
crsp['ret'] = np.where(crsp['ret'].isna() & crsp['dlret'].notna(), crsp['dlret'], crsp['ret'])

# Create lagged market value
crsp['me'] = crsp['prc'].abs() * crsp['shrout'] * 1e-3 # million
crsp = crsp[['permno','date','ret','me','exchcd']]
lag_me = crsp[['permno','date','me']]
# add 25 days to the date of 'me' (why 25: take Feb into account)
date_lag1 = crsp['date'] + dt.timedelta(days=25) 
# change the date to the last day of the month and add in the dataframe
lag_me['date'] = pd.DataFrame((date_lag1).values.astype('datetime64[ns]')) + MonthEnd(0)
lag_me = lag_me.rename(columns = {'me':'lag_Mkt_Cap'})
# merge lagged market value with return
crsp_stock = lag_me.merge(crsp[['permno','date','ret','exchcd']], 
                          how='inner', on=['date', 'permno'])
crsp_stock = crsp_stock.rename(columns = {'ret':'Ret'})

# drop na in ret(t) and me(t-1)
crsp_stock = crsp_stock[(crsp_stock['lag_Mkt_Cap'].notna()) | 
                        (crsp_stock['Ret'].notna())].reset_index(drop=True)

# Cumulative returns
crsp_stock['log_ret'] = np.log(crsp_stock['Ret'] + 1)
crsp_stock['Ranking_Ret'] = crsp_stock.groupby('permno')['log_ret'].rolling(11).sum().values
crsp_stock['Ranking_Ret'] = crsp_stock.groupby('permno')['Ranking_Ret'].shift(2)
crsp_stock = crsp_stock[crsp_stock['Ranking_Ret'].notna()].reset_index(drop=True)

# output
crsp_stock_mom = crsp_stock.copy()
crsp_stock_mom['Year'] = crsp_stock['date'].dt.year
crsp_stock_mom['Month'] = crsp_stock['date'].dt.month
crsp_stock_mom = crsp_stock_mom[['Year','Month', 'permno',
                                 'exchcd','lag_Mkt_Cap','Ret','Ranking_Ret']]
crsp_stock_mom = crsp_stock_mom[(crsp_stock_mom.Year < 2022) &
                                (crsp_stock_mom.Year > 1926)]

# Question 2

# Daniel and Moskowitz deciles
stock_dec = crsp_stock_mom.copy()
stock_dec['DM_decile'] = stock_dec.groupby(['Year','Month'])['Ranking_Ret'].\
                         apply(lambda x: pd.qcut(x, 10, labels = np.arange(1,11)))

# Kenneth R. French deciles
def nyse_brkpts(x, K):
    exc = x['exchcd']
    ret = x['Ranking_Ret']
    loc_nyse = (exc == 1) # the locations of nyse stocks 
    breakpoints = pd.qcut(ret[loc_nyse], K, retbins=True, labels=False)[1]
    breakpoints[0] = -np.inf
    breakpoints[K] = np.inf
    return pd.cut(ret, bins=breakpoints, labels=False) + 1

stock_dec['KRF_decile'] = np.nan
for y in range(1927,2022):
    for m in range(1,13):
        t = stock_dec[(stock_dec.Year == y) & (stock_dec.Month == m)]
        decile = nyse_brkpts(t, 10)
        stock_dec.loc[t.index, 'KRF_decile'] = decile

# output
crsp_stock_dec = stock_dec[['Year','Month', 'permno', 'lag_Mkt_Cap',
                            'Ret', 'DM_decile', 'KRF_decile']]

# Question 3

# load and clean FF3 data
FF_mkt = pd.read_csv('FF3.csv')
FF_mkt = FF_mkt.reset_index()
FF_mkt['date'] = pd.to_datetime(FF_mkt['Date'], format='%Y-%m-%d', errors='ignore')
FF_mkt['date'] = pd.DataFrame(FF_mkt['date'].values.astype('datetime64[ns]')) + MonthEnd(0)
FF_mkt['Year'] = FF_mkt['date'].dt.year
FF_mkt['Month'] = FF_mkt['date'].dt.month

# Daniel and Moskowitz vwret
vwret = crsp_stock_dec.copy()
vwret['sum_lme_dm'] = vwret.groupby(['Year','Month','DM_decile'])['lag_Mkt_Cap'].transform('sum')
vwret['single_vwret_dm'] = vwret['lag_Mkt_Cap'] / vwret['sum_lme_dm'] * vwret['Ret']
vwret_dm = vwret.groupby(['Year','Month','DM_decile'])['single_vwret_dm'].sum()

# Kenneth R. French vwret
vwret['sum_lme_krf'] = vwret.groupby(['Year','Month','KRF_decile'])['lag_Mkt_Cap'].transform('sum')
vwret['single_vwret_krf'] = vwret['lag_Mkt_Cap'] / vwret['sum_lme_krf'] * vwret['Ret']
vwret_krf = vwret.groupby(['Year','Month','KRF_decile'])['single_vwret_krf'].sum()

# output
crsp_mom_ret = vwret_dm.reset_index()
crsp_mom_ret.columns = 'Year', 'Month', 'decile', 'DM_ret'
vwret_krf = vwret_krf.reset_index()
vwret_krf.columns = 'Year', 'Month', 'decile', 'KRF_ret'
crsp_mom_ret = crsp_mom_ret.merge(vwret_krf, how='inner', on=['Year', 'Month', 'decile'])
crsp_mom_ret = crsp_mom_ret.merge(FF_mkt[['Year','Month','RF']], 
                                  how='inner', on=['Year', 'Month'])

# Question 4

# 192701-201303
mom_ret_q4 = crsp_mom_ret[crsp_mom_ret.Year < 2014]
mom_ret_q4 = mom_ret_q4.drop(mom_ret_q4[(mom_ret_q4.Year==2013) & (mom_ret_q4.Month>3)].index)

# excess returns and log returns
mom_ret_q4['DM_exret'] = mom_ret_q4['DM_ret'] - mom_ret_q4['RF']
mom_ret_q4['DM_logret'] = np.log(mom_ret_q4['DM_ret'] + 1)

Q4 = pd.DataFrame(columns = ['Ave(r-rf)','sigma','SR','sk(m)'])

# decile portfolios

# mean annualized excess ret (%)
exret_mean = mom_ret_q4.groupby('decile')['DM_exret'].mean() * 1200
Q4['Ave(r-rf)'] = exret_mean

# annualized volatility (%)
exret_vol = mom_ret_q4.groupby('decile')['DM_exret'].std() * np.sqrt(12) * 100
Q4['sigma'] = exret_vol

# sharpe ratios
Q4['SR'] = Q4['Ave(r-rf)'] / Q4['sigma']

# skewness
skew = mom_ret_q4.groupby('decile')['DM_logret'].skew()
Q4['sk(m)'] = skew

Q4 = Q4.T

# WML
mom_ret_1 = mom_ret_q4[mom_ret_q4.decile == 1][['Year','Month','DM_ret', 'RF']]
mom_ret_10 = mom_ret_q4[mom_ret_q4.decile == 10][['Year','Month','DM_ret']]
mom_wml = mom_ret_1.merge(mom_ret_10, how='inner', on=['Year','Month'])
mom_wml['wml_ret'] = mom_wml['DM_ret_y'] - mom_wml['DM_ret_x']

wml_mean = mom_wml['wml_ret'].mean() * 1200
wml_std = mom_wml['wml_ret'].std() * np.sqrt(12) * 100
wml_sr = wml_mean / wml_std
wml_skew = np.log(mom_wml['wml_ret'] + mom_wml['RF'] + 1).skew()

Q4['WML'] = np.nan
Q4.iloc[0,10] = wml_mean
Q4.iloc[1,10] = wml_std
Q4.iloc[2,10] = wml_sr
Q4.iloc[3,10] = wml_skew

Q4.columns = 'Decile 1', 'Decile 2', 'Decile 3',\
             'Decile 4', 'Decile 5', 'Decile 6' ,\
             'Decile 7', 'Decile 8', 'Decile 9',\
             'Decile 10', 'WML'
             
# Question 5

# load and clean DM returns (192701-201612)
DM_data = pd.read_fwf('m_m_pt_tot.txt', header=None)
DM_data.columns = 'date','decile','web_ret','ave_me','num_firm'
DM_data['date'] = pd.to_datetime(DM_data['date'].astype('str'), 
                                 format='%Y-%m-%d', errors='ignore')
DM_data['Year'] = DM_data['date'].dt.year
DM_data['Month'] = DM_data['date'].dt.month
DM_data = DM_data[['Year','Month','decile','web_ret']]

# Merging with my data
DM_data = DM_data.merge(crsp_mom_ret[['Year','Month','decile','DM_ret','RF']], 
                        how='inner', on=['Year','Month','decile'])

# DM correlations
corr_DM = DM_data.groupby('decile')[['web_ret','DM_ret']].corr()
corr_DM = corr_DM.reset_index().drop_duplicates(subset=['decile'],keep='last')\
          [['decile','web_ret']].reset_index(drop=True)
corr_DM = corr_DM.rename(columns = {'web_ret':'DM_correlation'})

# load and clean KRF returns (192701-202203)
KRF_data = pd.read_csv('10_Portfolios_Prior_12_2.csv')
KRF_data['date'] = pd.to_datetime(KRF_data['date'].astype('str'), 
                                  format='%Y%m', errors='ignore')
KRF_data['date'] = pd.DataFrame(KRF_data['date'].values.astype('datetime64[ns]')) + MonthEnd(0)
KRF_data = KRF_data.merge(FF_mkt[['date','RF']], how = 'inner', on =['date'])
KRF_data = KRF_data[KRF_data['date'].dt.year < 2022]

# KRF correlations
corr = corr_DM.copy()
corr['KRF_correlation'] = np.nan

for i in range(1,11):
    my_krf = crsp_mom_ret[crsp_mom_ret.decile == i]['KRF_ret'].reset_index(drop=True)
    web_krf = KRF_data.iloc[:,i] / 100
    corr.iloc[i-1,2] = np.corrcoef(web_krf, my_krf)[0,1]

corr = corr.T

# WML
corr['WML'] = np.nan
KRF_wml = (KRF_data['Hi PRIOR'] - KRF_data['Lo PRIOR']).values / 100
KRF_mywml = (crsp_mom_ret[crsp_mom_ret.decile == 10]['KRF_ret'].values - 
             crsp_mom_ret[crsp_mom_ret.decile == 1]['KRF_ret'].values)
corr.loc['KRF_correlation','WML'] = np.corrcoef(KRF_wml, KRF_mywml)[0,1]

DM_wml = (DM_data[DM_data.decile == 10]['web_ret'].values - 
          DM_data[DM_data.decile == 1]['web_ret'].values)
DM_mywml = (DM_data[DM_data.decile == 10]['DM_ret'].values - 
            DM_data[DM_data.decile == 1]['DM_ret'].values)
corr.loc['DM_correlation','WML'] = np.corrcoef(DM_wml, DM_mywml)[0,1]

# output
corr = corr.drop('decile')
corr.columns = 'Decile 1', 'Decile 2', 'Decile 3',\
               'Decile 4', 'Decile 5', 'Decile 6' ,\
               'Decile 7', 'Decile 8', 'Decile 9',\
               'Decile 10', 'WML'

# Question 6

# plot the figure of returns after 2011
mom_ret_2012 = crsp_mom_ret[crsp_mom_ret.Year > 2011]
date_1221 = KRF_data[KRF_data['date'].dt.year > 2011]['date']

# DM
plt.figure()
for i in range(1,11):
    decile = mom_ret_2012[mom_ret_2012.decile == i]
    decile_logret = np.log(decile['DM_ret'] + 1)
    decile_cumret = decile_logret.cumsum()
    plt.plot(date_1221, decile_cumret, label=str(i))
plt.legend()
plt.show()

# KRF
plt.figure()
for i in range(1,11):
    decile = mom_ret_2012[mom_ret_2012.decile == i]
    decile_logret = np.log(decile['KRF_ret'] + 1)
    decile_cumret = decile_logret.cumsum()
    plt.plot(date_1221, decile_cumret, label=str(i))
plt.legend()
plt.show()

# WML
plt.figure()
wml_ret_dm = (mom_ret_2012[mom_ret_2012.decile == 10]['DM_ret'].values -
              mom_ret_2012[mom_ret_2012.decile == 1]['DM_ret'].values)
wml_ret_krf = (mom_ret_2012[mom_ret_2012.decile == 10]['KRF_ret'].values -
               mom_ret_2012[mom_ret_2012.decile == 1]['KRF_ret'].values)
plt.plot(date_1221, np.log(wml_ret_dm+1).cumsum(), label='DM WML')
plt.plot(date_1221, np.log(wml_ret_krf+1).cumsum(), label='KRF WML')
plt.legend()
plt.show()












