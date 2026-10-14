"""Rebuilds everything from the raw Excel file with one command: python src/build_matrix.py

It loads the raw data, cleans it, splits by time, builds the sparse matrix and writes
the reports. Nothing here is random, so every run gives the same result.

Owner: Manavkrishna Manoj.
"""

import time
from pathlib import Path

import pandas as pd
import scipy.sparse as sp

from clean import clean_raw
from matrix import build_matrix, index_tables, matrix_report, restrict_eval_pairs
from split import time_split

ROOT = Path(__file__).resolve().parents[1]
RAW_FILE = ROOT / "data" / "raw" / "online_retail_II.xlsx"
INTERIM = ROOT / "data" / "interim"
PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "outputs" / "results"

TEST_MONTHS = 2
MIN_CUSTOMERS_PER_PRODUCT = 10


def load_raw():
    """Read both sheets once and cache them, since read_excel takes minutes."""
    cache = INTERIM / "raw_combined.pkl"
    if cache.exists():
        return pd.read_pickle(cache)
    sheets = pd.read_excel(RAW_FILE, sheet_name=None)
    df = pd.concat(sheets.values(), ignore_index=True)
    df.to_pickle(cache)
    return df


def save_report(values, path):
    """Write a dict out as a two column csv (metric, value)."""
    pd.DataFrame({"metric": list(values), "value": list(values.values())}).to_csv(
        path, index=False
    )


def main():
    start = time.time()
    for folder in (INTERIM, PROCESSED, RESULTS):
        folder.mkdir(parents=True, exist_ok=True)

    raw = load_raw()
    print(f"Raw rows: {len(raw):,}")
    clean, cleaning_log, removed_codes = clean_raw(raw)
    clean.to_pickle(INTERIM / "clean_transactions.pkl")
    cleaning_log.to_csv(RESULTS / "cleaning_log.csv", index=False)
    removed_codes.head(50).to_csv(RESULTS / "removed_codes.csv", header=["rows"])

    train, test, held_out, split_info = time_split(clean, TEST_MONTHS)
    train.to_pickle(PROCESSED / "train_transactions.pkl")
    test.to_pickle(PROCESSED / "test_transactions.pkl")
    print(f"\nTrain rows: {len(train):,}  Test rows: {len(test):,}")

    A, customers, products, info = build_matrix(train, MIN_CUSTOMERS_PER_PRODUCT)
    sp.save_npz(PROCESSED / "customer_product_train.npz", A)

    customer_index, product_index = index_tables(train, customers, products)
    customer_index.to_csv(PROCESSED / "customer_index.csv", index=False)
    product_index.to_csv(PROCESSED / "product_index.csv", index=False)

    # evaluation pairs as indices, no IDs
    eval_pairs, eval_info = restrict_eval_pairs(held_out, customers, products)
    eval_pairs.to_csv(PROCESSED / "eval_pairs.csv", index=False)

    # reports hold counts only
    report = matrix_report(A, info)
    save_report(report, RESULTS / "matrix_report.csv")
    save_report({**split_info, **eval_info}, RESULTS / "split_report.csv")

    print("\nMATRIX REPORT")
    for name, value in report.items():
        print(f"  {name}: {value}")
    print("\nEVALUATION PAIRS")
    for name, value in eval_info.items():
        print(f"  {name}: {value}")
    print(f"\nDone in {time.time() - start:.1f} seconds")


if __name__ == "__main__":
    main()