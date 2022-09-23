# -*- coding: utf-8 -*-
"""
Created on Thu Sep 15 14:43:15 2022

@author: lenovo
"""

import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt
import seaborn as sb
import pickle
import scipy
from statsmodels.stats.weightstats import ztest
from sklearn.linear_model import LinearRegression
pd.options.mode.chained_assignment = None

currency_pairs = [
    "BTCUSDT",
    "ETHUSDT",
    "LTCUSDT",
    "DOGEUSDT",
    "NEOUSDT",
    "BNBUSDT",
    "XRPUSDT",
    "LINKUSDT",
    "EOSUSDT",
    "TRXUSDT",
    "ETCUSDT",
    "XLMUSDT",
    "ZECUSDT",
    "ADAUSDT",
    "QTUMUSDT",
    "DASHUSDT",
    "XMRUSDT",
    "BTTUSDT"]
period = "1h"
fw = open(f"D:/QuantToolkit/python/factor/crypto_hfs_{period}.pkl", "rb")
df_collection = pickle.load(fw)
fw.close()



hold_s = [1, 6, 12, 24, 48, 72, 96, 144]
lookback_s = [1, 6, 12, 24, 48, 72, 96, 144]
IC_mat = pd.DataFrame(np.zeros([len(hold_s), len(lookback_s)]))
IC_mat.columns = hold_s
IC_mat.index = lookback_s
z_statistic_mat = IC_mat.copy()
p_mat = IC_mat.copy()
significance_mat = IC_mat.copy()
significance_thresh = 1e-4 # p value should be smaller than this threshold
if period == "1d":
    time_delta = datetime.timedelta(days=1)
elif period == "1h":
    time_delta = datetime.timedelta(hours=1)
factor_list = ['skewness_1', 'kurtosis_1', 'downside_volatility_1', 'price_volume_correlation_1', 'trend_strength_1']
#%% Determine latest start & earlist end
for i, pair in enumerate(currency_pairs):
    df = df_collection[pair]
    if i==0:
        l_start = df.index[0]
        e_end = df.index[-1]
    l_start = max(df.loc[df['is_tradable']].index[0], l_start)
    e_end = min(df.loc[df['is_tradable']].index[-1], e_end)
clipped_df_collection = dict()
for i, pair in enumerate(currency_pairs):
    df = df_collection[pair]
    clipped_df = df.loc[(df.index>=l_start) & (df.index<=e_end)]
    clipped_df_collection[pair] = clipped_df

# perform winsorisation on original factor values since some factors contains 2nd, 3rd and even 4th moment of intraday returns
for i, pair in enumerate(currency_pairs):
    df = clipped_df_collection[pair]
    for f in factor_list:
        df[f] = scipy.stats.mstats.winsorize(df[f], limits = [0.01, 0.01])
    clipped_df_collection[pair] = df


ICmat_dict = dict()
z_statistic_mat_dict = dict()
p_mat_dict = dict()
significance_mat_dict = dict()
for f in factor_list:
    ICmat_dict[f] = IC_mat.copy()
    z_statistic_mat_dict[f] = z_statistic_mat.copy()
    p_mat_dict[f] = p_mat.copy()
    significance_mat_dict[f] = significance_mat.copy()
#%% symmetric orthogonalisation on cross section
for hold in hold_s:
    for lookback in lookback_s: 
        print(f"now in hold {hold}, lookback {lookback}\n")
        rankICs_dict = dict()
        for f in factor_list:
            rankICs_dict[f] = []
        for day in clipped_df_collection["BTCUSDT"].index[lookback-1:-hold]:
            print(f"\r {day}", end="", flush=True)
            if clipped_df_collection["BTCUSDT"]['is_tradable'][day] == False:
                continue
            factor_values = []
            return_values = []
            Fnk = np.zeros([len(currency_pairs), len(factor_list)])
            for n, pair in enumerate(currency_pairs):
                for k, f in enumerate(factor_list):
                    Fnk[n, k] = clipped_df_collection[pair][f][day - time_delta*(lookback-1) : day].mean()
            # standardize cross sectional factor values
            for k, f in enumerate(factor_list):
                Fnk[:, k] = (Fnk[:, k] - Fnk[:, k].mean())/Fnk[:, k].std()
            # calculate transition matrix
            M = (Fnk.shape[0]-1)* np.cov(Fnk.T.astype(float))   # matrix M
            D,U = np.linalg.eig(M)  # 获取特征值和特征向量
            U = np.mat(U)           # 转换为np中的矩阵
            d = np.mat(np.diag(D**(-0.5)))  # 对特征根元素开(-0.5)指数
            S = U * d * U.T         # 获取过渡矩阵S
            factors_orthogonal_mat = np.mat(Fnk) * S   # 获取对称正交矩阵
            factors_orthogonal= pd.DataFrame(factors_orthogonal_mat,columns = factor_list,index=currency_pairs)   # 矩阵转为dataframe
            # calculate rank IC at this day
            # use factors_orthogonal_mat to meet any factor combination rankings
            for pair in currency_pairs:
                df = clipped_df_collection[pair]
                return_values.append(np.array(df["return"][day + time_delta : day + time_delta * hold]).cumsum()[-1])
            
            return_values = np.array(return_values)
            for f in factor_list:
                factor_values = np.array(factors_orthogonal[f])
                IC = np.corrcoef(np.argsort(factor_values), np.argsort(return_values))[0,1]
                rankICs_dict[f].append(IC)
        for f in factor_list:
            z_stats, p_value = ztest(np.nan_to_num(np.array(rankICs_dict[f])))
            z_statistic_mat_dict[f][hold][lookback] = z_stats
            p_mat_dict[f][hold][lookback] = p_value
            significance_mat_dict[f][hold][lookback] = 1 if p_value < significance_thresh else 0
            ICmat_dict[f][hold][lookback] = np.nan_to_num(np.array(rankICs_dict[f])).mean()
#%% select a factor to plot
factor = 'trend_strength_1'
z_statistic_mat = z_statistic_mat_dict[factor]
p_mat = p_mat_dict[factor]
significance_mat = significance_mat_dict[factor]
#%% plot heatmap of z-statistics
ax1 = plt.axes()
ax1.clear()
sb.heatmap(np.array(z_statistic_mat).astype("float32"), annot=True, xticklabels=hold_s, yticklabels=lookback_s, ax = ax1)
ax1.set_title(factor.strip("_1") + " factor z_statistics")
#%% plot heatmap of p_values
ax1 = plt.axes()
ax1.clear()
sb.heatmap(np.array(p_mat).astype("float32"), robust=True, annot=True, xticklabels=hold_s, yticklabels=lookback_s, ax = ax1)
ax1.set_title(factor.strip("_1") + " factor p_values")
#%% plot heatmap of significance
ax1 = plt.axes()
ax1.clear()
sb.heatmap(np.array(significance_mat).astype("float32"), annot=True, xticklabels=hold_s, yticklabels=lookback_s, ax = ax1)
ax1.set_title(factor.strip("_1") + " factor significance (p threshold=1e-4)")
