# -*- coding: utf-8 -*-
"""
Created on Thu Aug 11 16:32:29 2022

@author: lenovo
"""

import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt
import pickle
from scipy import stats
from multiprocessing.dummy import Pool as ThreadPool
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

def load_crypto_minute(**kwargs):
    # load minute cryptocurrency data
    directory = "" # Please fill in downloaded data directory here
    exchange = kwargs.get("exchange","Binance")
    currency_pair = kwargs.get("currency_pair","BTCUSDT")
    frequency = "minute"
    year = kwargs.get("frequency",["2020","2021","2022"])
    dfs = []
    for y in year:
        path = directory + exchange + "_" + currency_pair + "_" + y + "_" + frequency + ".csv"
        df = pd.read_csv(path, header=1)
        df = pd.DataFrame(df.iloc[::-1]).reset_index(drop=True)
        dfs.append(df)
    
    # concat minute data from different years into one dataframe
    minute_df = pd.concat([df for df in dfs], axis=0)
    minute_df.head()
    
    # convert time from str to timestamp
    minute_df.date = pd.to_datetime(minute_df.date, format = "%Y/%m/%d %H:%M")
    minute_df = minute_df.rename(columns={"date":"time"})
    minute_df = minute_df.drop(columns="unix")
    minute_df = minute_df.set_index(minute_df.time).drop(columns="time")
    minute_df.columns = ["symbol", "open", "high", "low", "close", "Volume", "Volume USDT", "tradecount"]
    
    return minute_df

def fill_missing(minute_df, **kwargs):
    # fill in missing data
    # determine starting time and ending time
    
    start_time = kwargs.get("start_time", minute_df.index.min())
    end_time = kwargs.get("end_time", minute_df.index.max())
    timedelta_1min = datetime.timedelta(minutes=1)
    # reindex time index
    minute_timestamps = pd.date_range(start=start_time, end=end_time, freq='min')
    filled_minute_df = minute_df.reindex(minute_timestamps, fill_value=0)
    # fill missing close price with the latest value
    missing_times = filled_minute_df.index[(filled_minute_df['close']==0)]
    for time in missing_times:
        if time == minute_timestamps[0]:
            filled_minute_df['close'][time] = np.array(filled_minute_df['close'])[np.array(filled_minute_df['close']).nonzero()[0][0]]
            filled_minute_df['symbol'][time] = np.array(filled_minute_df['symbol'])[np.array(filled_minute_df['symbol']).nonzero()[0][0]]
            continue
        filled_minute_df['close'][time] = filled_minute_df['close'][time-timedelta_1min]
        filled_minute_df['symbol'][time] = filled_minute_df['symbol'][time-timedelta_1min]
        
    return filled_minute_df

def compress_daily(minute_df, start_time:str, end_time:str):
    # calculate daily return based on minutely data
    daily_timestamps = pd.date_range(start=start_time, end=end_time, freq="D")
    daily_factors = pd.DataFrame(np.zeros_like(daily_timestamps).astype("float32"))
    daily_factors.index = daily_timestamps
    # calculate daily return and fill in the dataframe as the first column
    daily_factors.columns = ['return']
    timedelta_1d = datetime.timedelta(days=1)
    for time in daily_factors.index:
        intraday_price = np.array(minute_df[(minute_df.index>=time) & (minute_df.index<(time+timedelta_1d))]['close'])
        daily_return = np.log(intraday_price[-1]/intraday_price[0])
        daily_factors["return"][time] = daily_return
    
    # add one column indicate whether the asset is tradable 
    daily_factors["is_tradable"] = daily_factors["return"]!=0
    return daily_factors

def compress_hourly(minute_df, start_time:str, end_time:str):
    # calculate daily return based on minutely data
    hourly_timestamps = pd.date_range(start=start_time, end=end_time, freq="H")
    hourly_factors = pd.DataFrame(np.zeros_like(hourly_timestamps).astype("float32"))
    hourly_factors.index = hourly_timestamps
    # calculate daily return and fill in the dataframe as the first column
    hourly_factors.columns = ['return']
    timedelta_1h = datetime.timedelta(hours=1)
    for time in hourly_factors.index:
        print(f"\r {time}...", end="", flush=True)
        intraday_price = np.array(minute_df[(minute_df.index>=time) & (minute_df.index<(time+timedelta_1h))]['close'])
        hourly_return = np.log(intraday_price[-1]/intraday_price[0])
        hourly_factors["return"][time] = hourly_return
    
    # add one column indicate whether the asset is tradable 
    hourly_factors["is_tradable"] = hourly_factors["return"]!=0
    return hourly_factors

def add_skewness_factor(factors, minute_df, period):
    # calculate daily high frequency skewness factor
    factors["skewness_1"] = np.zeros_like(factors["return"])
    if period == "1d":
        time_delta = datetime.timedelta(days=1)
    elif period == "1h":
        time_delta = datetime.timedelta(hours=1)
    for time in factors.index:
        print(f"\r {time}...", end="", flush=True)
        intraday_rt = np.array(minute_df[(minute_df.index>=time) & (minute_df.index<(time+time_delta))]['close'].pct_change(1).fillna(value=0))
        skewness = stats.skew(intraday_rt)
        factors.skewness_1[time] = skewness
    return factors

def add_kurtosis_factor(factors, minute_df, period):
    # calculate daily high frequency skewness factor
    factors["kurtosis_1"] = np.zeros_like(factors["return"])
    if period == "1d":
        time_delta = datetime.timedelta(days=1)
    elif period == "1h":
        time_delta = datetime.timedelta(hours=1)
    for time in factors.index:
        print(f"\r {time}...", end="", flush=True)
        intraday_rt = np.array(minute_df[(minute_df.index>=time) & (minute_df.index<(time+time_delta))]['close'].pct_change(1).fillna(value=0))
        if len(intraday_rt) == 0:
            continue
        kurtosis = stats.kurtosis(intraday_rt)
        factors.kurtosis_1[time] = kurtosis
    return factors


def add_downside_volatility_factor(factors, minute_df, period):
    # calculate daily downside volatility factor
    factors["downside_volatility_1"] = np.zeros_like(factors["return"])
    if period == "1d":
        time_delta = datetime.timedelta(days=1)
    elif period == "1h":
        time_delta = datetime.timedelta(hours=1)
    for time in factors.index:
        print(f"\r {time}...", end="", flush=True)
        intraday_rt = np.array(minute_df[(minute_df.index>=time) & (minute_df.index<(time+time_delta))]['close'].pct_change(1).fillna(value=0))
        downside_volatility = ((intraday_rt**2)*(intraday_rt<0)).sum()/((intraday_rt**2).sum()+1e-4)
        factors.downside_volatility_1[time] = downside_volatility
    return factors

def add_trend_strength_factor(factors, minute_df, period):
    # calculate daily downside volatility factor
    factors["trend_strength_1"] = np.zeros_like(factors["return"])
    if period == "1d":
        time_delta = datetime.timedelta(days=1)
    elif period == "1h":
        time_delta = datetime.timedelta(hours=1)
    for time in factors.index:
        print(f"\r {time}...", end="", flush=True)
        intraday_rt = np.array(minute_df[(minute_df.index>=time) & (minute_df.index<(time+time_delta))]['close'].pct_change(1).fillna(value=0))
        if len(intraday_rt) == 0:
            continue
        trend_strength = intraday_rt.sum() / abs(intraday_rt).sum()
        factors.trend_strength_1[time] = trend_strength
    factors = factors.fillna(value=0)
    return factors

def add_price_volume_corr_factor(factors, minute_df, period):
    # calculate daily price-volume correlation
    factors["price_volume_correlation_1"] = np.zeros_like(factors["return"])
    if period == "1d":
        time_delta = datetime.timedelta(days=1)
    elif period == "1h":
        time_delta = datetime.timedelta(hours=1)
    for time in factors.index:
        print(f"\r {time}...", end="", flush=True)
        intraday_price = np.array(minute_df[(minute_df.index>=time) & (minute_df.index<(time+time_delta))]['close'])
        intraday_volume = np.array(minute_df[(minute_df.index>=time) & (minute_df.index<(time+time_delta))]['Volume'])
        price_volume_correlation = np.corrcoef(intraday_price, intraday_volume/(intraday_volume.sum()))[0,1]
        factors.price_volume_correlation_1[time] = price_volume_correlation
    factors = factors.fillna(value=0)
    return factors


def run_one_pair(symbol, period):
    data_args = {
        "exchange": "Binance",
        "currency_pair": symbol,
        "period": period,
        "start_time": "2020/01/01",
        "end_time": "2022/08/01",
        }
    minute_df = load_crypto_minute(**data_args)
    minute_df = fill_missing(minute_df, **data_args)
    if period == "1d":
        factors = compress_daily(minute_df, start_time=data_args["start_time"], end_time=data_args["end_time"])
    elif period == "1h":
        factors = compress_hourly(minute_df, start_time=data_args["start_time"], end_time=data_args["end_time"])
    factors = add_skewness_factor(factors, minute_df, period)
    factors = add_kurtosis_factor(factors, minute_df, period)
    factors = add_downside_volatility_factor(factors, minute_df, period)
    factors = add_trend_strength_factor(factors, minute_df, period)
    factors = add_price_volume_corr_factor(factors, minute_df, period)
    return factors

def main():
    df_collection = {} # dict of daily factors of different symbols
    period = "1h"
    for symbol in currency_pairs:
        print(f"extracting factors of {symbol}\n")
        df_collection[symbol] = run_one_pair(symbol, period)
    fw = open(f"crypto_hfs_{period}.pkl", "wb")
    pickle.dump(df_collection, fw)
    fw.close()

if __name__ == "__main__":
    main()
