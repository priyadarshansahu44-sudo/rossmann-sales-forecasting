import pandas as pd
import numpy as np

def build_features(df):
    df = df.copy()
    
    # 1. Calendar features
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Day'] = df['Date'].dt.day
    df['DayOfYear'] = df['Date'].dt.dayofyear
    df['WeekOfYear'] = df['Date'].dt.isocalendar().week.astype(int)
    
    # 2. Cyclical encodings
    df['Month_sin'] = np.sin(2 * np.pi * df['Month'] / 12.0)
    df['Month_cos'] = np.cos(2 * np.pi * df['Month'] / 12.0)
    df['DayOfWeek_sin'] = np.sin(2 * np.pi * df['DayOfWeek'] / 7.0)
    df['DayOfWeek_cos'] = np.cos(2 * np.pi * df['DayOfWeek'] / 7.0)
    
    # Sort before computing window functions
    df = df.sort_values(by=['Store', 'Date']).reset_index(drop=True)
    
    # 3. Lags & Rolling statistics
    grouped_sales = df.groupby('Store')['Sales']
    df['Sales_Lag1'] = grouped_sales.shift(1)
    df['Sales_Lag7'] = grouped_sales.shift(7)
    df['Sales_Lag14'] = grouped_sales.shift(14)
    
    shifted_sales = df.groupby('Store')['Sales'].shift(1)
    df['Sales_RollMean7'] = shifted_sales.groupby(df['Store']).transform(lambda x: x.rolling(window=7, min_periods=1).mean())
    df['Sales_RollStd7'] = shifted_sales.groupby(df['Store']).transform(lambda x: x.rolling(window=7, min_periods=1).std()).fillna(0)
    df['Sales_RollMean14'] = shifted_sales.groupby(df['Store']).transform(lambda x: x.rolling(window=14, min_periods=1).mean())
    df['Sales_RollStd14'] = shifted_sales.groupby(df['Store']).transform(lambda x: x.rolling(window=14, min_periods=1).std()).fillna(0)
    
    # Drop NaNs to keep only clean rows
    df = df.dropna(subset=['Sales_Lag14', 'Sales_RollMean14']).reset_index(drop=True)
    return df
