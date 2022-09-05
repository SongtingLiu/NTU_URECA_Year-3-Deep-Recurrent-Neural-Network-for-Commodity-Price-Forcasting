# -*- coding: utf-8 -*-
"""
Created on Sun Aug 14 12:48:12 2022

@author: lenovo
"""

import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt
import seaborn as sb
import pickle
from statsmodels.stats.weightstats import ztest
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
# calculate rank IC day by day based on available currency pairs
# select factor and holding period
factor = "trend_strength_1"
# hold_s = [1, 6, 12, 24, 48, 72, 96, 144]
# lookback_s = [1, 6, 12, 24, 48, 72, 96, 144]
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
#%% calculate rank ICs of multi periods
for hold in hold_s:
    for lookback in lookback_s: 
        rankICs = []
        print(f"now in hold {hold}, lookback {lookback}\n")
        for day in df_collection["BTCUSDT"].index[lookback-1:-hold]:
            print(f"\r {day}", end="", flush=True)
            factor_values = []
            return_values = []
            for pair in currency_pairs:
                df = df_collection[pair]
                if df["is_tradable"][day]:
                    factor_values.append(np.array(df[factor][day - time_delta*(lookback-1) : day]).mean().item())
                    return_values.append(np.array(df["return"][day + time_delta : day + time_delta * hold]).cumsum()[-1].item())
            # calculate rank IC at this day
            factor_values = np.array(factor_values)
            return_values = np.array(return_values)
            IC = np.corrcoef(np.argsort(factor_values), np.argsort(return_values))[0,1]
            rankICs.append(IC)
        z_stats, p_value = ztest(np.nan_to_num(np.array(rankICs)))
        z_statistic_mat[hold][lookback] = z_stats
        p_mat[hold][lookback] = p_value
        significance_mat[hold][lookback] = 1 if p_value < significance_thresh else 0
        IC_mat[hold][lookback] = np.nan_to_num(np.array(rankICs)).mean()

#%% cumulative IC plot
factor = "downside_volatility_1"
hold = 24
lookback = 24
rankICs = []
print(f"now in hold {hold}, lookback {lookback}\n")
for day in df_collection["BTCUSDT"].index[lookback-1:-hold]:
    print(f"\r {day}", end="", flush=True)
    factor_values = []
    return_values = []
    for pair in currency_pairs:
        df = df_collection[pair]
        if df["is_tradable"][day]:
            factor_values.append(np.array(df[factor][day - time_delta*(lookback-1) : day]).mean().item())
            return_values.append(np.array(df["return"][day + time_delta : day + time_delta * hold]).cumsum()[-1].item())
    # calculate rank IC at this day
    factor_values = np.array(factor_values)
    return_values = np.array(return_values)
    IC = np.corrcoef(np.argsort(factor_values), np.argsort(return_values))[0,1]
    rankICs.append(IC)

rankICs = np.array(rankICs)
rankICs = rankICs[np.isnan(rankICs)==0]
plt.plot(rankICs.cumsum())
plt.show()    

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
