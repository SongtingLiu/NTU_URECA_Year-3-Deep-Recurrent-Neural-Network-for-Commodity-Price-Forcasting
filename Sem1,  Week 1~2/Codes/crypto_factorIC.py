# -*- coding: utf-8 -*-
"""
Created on Sun Aug 14 12:48:12 2022

@author: lenovo
"""

import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt
import pickle
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
fw = open(f"crypto_hfs_{period}.pkl", "rb")
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
if period == "1d":
    time_delta = datetime.timedelta(days=1)
elif period == "1h":
    time_delta = datetime.timedelta(hours=1)
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
        IC_mat[hold][lookback] = np.nan_to_num(np.array(rankICs)).mean()

#%%
factor = "price_volume_correlation_1"
hold = 1
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
