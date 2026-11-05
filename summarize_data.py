import pandas as pd
import numpy as np

def summarize(df_path):
    print(f"Loading {df_path}...")
    df = pd.read_csv(df_path)
    
    total_rows = len(df)
    fraud_rows = int(df['isFraud'].sum())
    fraud_rate = (fraud_rows / total_rows) * 100
    
    ring_mask = df['is_injected_ring'] == True
    injected_txn_count = int(ring_mask.sum())
    base_fraud_count = int(fraud_rows - injected_txn_count)
    n_rings = df.loc[ring_mask, 'ring_id'].nunique() if injected_txn_count > 0 else 0
    
    print("\n" + "=" * 55)
    print(f"  DATASET SUMMARY: {df_path}")
    print("=" * 55)
    print(f"Total Transactions:        {total_rows:,}")
    print(f"Total Columns:             {df.shape[1]}")
    print(f"Legitimate Transactions:   {(total_rows - fraud_rows):,} ({100 - fraud_rate:.3f}%)")
    print(f"Total Fraud Transactions:  {fraud_rows:,} ({fraud_rate:.3f}%)")
    print(f"  |-- Base Fraud (Real):   {base_fraud_count:,}")
    print(f"  +-- Injected Ring Fraud: {injected_txn_count:,}")
    print(f"Total Injected Rings:      {n_rings}")
    
    if n_rings > 0:
        ring_sizes = df.loc[ring_mask].groupby('ring_id').size()
        print(f"Ring Size (Min / Max):     {ring_sizes.min()} / {ring_sizes.max()}")
        print(f"Ring Size (Mean / Median): {ring_sizes.mean():.2f} / {ring_sizes.median():.1f}")
        
    has_ring_id = 'ring_id' in df.columns
    has_injected = 'is_injected_ring' in df.columns
    print(f"Column 'ring_id' present:         {has_ring_id} (Nulls: {df['ring_id'].isna().sum() if has_ring_id else 'N/A'})")
    print(f"Column 'is_injected_ring' present:{has_injected} (Nulls: {df['is_injected_ring'].isna().sum() if has_injected else 'N/A'})")
    print(f"Time span (TransactionDT): {df['TransactionDT'].min():,} to {df['TransactionDT'].max():,}")
    print("=" * 55 + "\n")

if __name__ == "__main__":
    import sys
    files_to_check = sys.argv[1:] if len(sys.argv) > 1 else ["augmented_with_rings.csv", "augmented_with_rings_80k.csv"]
    for f in files_to_check:
        try:
            summarize(f)
        except Exception as e:
            print(f"Could not summarize {f}: {e}")

