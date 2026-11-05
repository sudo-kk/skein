import pandas as pd
import subprocess
import sys

def create_80k_sample():
    print("Creating 80k chronological sample of train_transaction.csv...")
    df_tx = pd.read_csv("train_transaction.csv", nrows=80000)
    df_tx.to_csv("train_transaction_80k.csv", index=False)
    print(f"Saved train_transaction_80k.csv with {len(df_tx):,} rows.")
    
    print("Injecting 40 rings into 80k sample...")
    cmd = [
        sys.executable,
        "inject_rings.py",
        "--transaction", "train_transaction_80k.csv",
        "--identity", "train_identity.csv",
        "--out", "augmented_with_rings_80k.csv",
        "--n-rings", "40",
        "--seed", "42"
    ]
    subprocess.run(cmd, check=True)
    print("Successfully created augmented_with_rings_80k.csv!")

if __name__ == "__main__":
    create_80k_sample()
