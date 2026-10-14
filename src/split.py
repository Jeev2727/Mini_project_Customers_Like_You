"""Time based train/test split and the held-out purchases.

Owner: Manavkrishna Manoj.
"""

import pandas as pd

KEYS = ["Customer ID", "StockCode"]


def time_split(df, months=2):
    """Last `months` months are the test period.

    Returns (train, test, held_out, summary). held_out is the unique (customer,
    product) pairs from the test period that the customer never bought in training.
    """
    cutoff = (df["InvoiceDate"].max() - pd.DateOffset(months=months)).normalize()

    train = df[df["InvoiceDate"] < cutoff].copy()
    test = df[df["InvoiceDate"] >= cutoff].copy()

    assert train["InvoiceDate"].max() < test["InvoiceDate"].min()

    # pairs from the test period that the customer never bought before
    train_pairs = train[KEYS].drop_duplicates()
    test_pairs = test[KEYS].drop_duplicates()
    merged = test_pairs.merge(train_pairs, on=KEYS, how="left", indicator=True)
    held_out = merged.loc[merged["_merge"] == "left_only", KEYS].reset_index(drop=True)

    train_customers = set(train["Customer ID"])
    test_customers = set(test["Customer ID"])

    summary = {
        "test_months": months,
        "cutoff_date": str(cutoff.date()),
        "train_rows": len(train),
        "test_rows": len(test),
        "train_share_%": round(len(train) / len(df) * 100, 2),
        "train_start": str(train["InvoiceDate"].min()),
        "train_end": str(train["InvoiceDate"].max()),
        "test_start": str(test["InvoiceDate"].min()),
        "test_end": str(test["InvoiceDate"].max()),
        "train_customers": len(train_customers),
        "test_customers": len(test_customers),
        "customers_in_both": len(train_customers & test_customers),
        "customers_only_in_test": len(test_customers - train_customers),
        "test_pairs": len(test_pairs),
        "held_out_pairs": len(held_out),
        "held_out_customers": held_out["Customer ID"].nunique(),
    }
    return train, test, held_out, summary