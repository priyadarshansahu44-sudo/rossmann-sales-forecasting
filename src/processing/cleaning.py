import pandas as pd
import numpy as np

def load_and_clean_data(train_path, store_path):
    # Load datasets
    train_df = pd.read_csv(train_path, low_memory=False)
    store_df = pd.read_csv(store_path)
    
    # Parse Dates
    train_df['Date'] = pd.to_datetime(train_df['Date'])
    
    # Merge
    merged_df = pd.merge(train_df, store_df, on='Store', how='left')
    
    # Handle store closures and zero-sales records
    cleaned_df = merged_df[(merged_df['Open'] != 0) & (merged_df['Sales'] > 0)].copy()
    
    # Sort chronologically
    cleaned_df = cleaned_df.sort_values(by='Date').reset_index(drop=True)
    
    return cleaned_df
