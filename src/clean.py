"""Cleaning rules for the invoice lines. Each rule writes down how many rows it dropped.

Owner: Manavkrishna Manoj.
"""

import pandas as pd

# some copies of the file name these columns differently
COLUMN_ALIASES = {
    "InvoiceNo": "Invoice",
    "UnitPrice": "Price",
    "CustomerID": "Customer ID",
}

# 5 digits plus optional letters (like 85123A), or a DCGS gift set. DCGS rows have no
# customer ID anyway, so they never get as far as the matrix.
PRODUCT_CODE_PATTERN = r"\d{5}[A-Z]*|DCGS[A-Z0-9]*"

# real products that don't match the pattern (none so far)
PRODUCT_CODE_EXCEPTIONS = set()


def clean_raw(df):
    """Clean the raw lines. Returns (clean_df, cleaning_log, removed_codes)."""
    df = df.rename(columns=COLUMN_ALIASES).copy()

    df["Invoice"] = df["Invoice"].astype(str)
    # 15056bl and 15056BL are the same thing
    df["StockCode"] = df["StockCode"].astype(str).str.upper().str.strip()
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

    log = [
        {
            "step": 0,
            "rule": "Raw dataset",
            "rows_removed": 0,
            "rows_remaining": len(df),
        }
    ]

    def apply_rule(df, rule, remove_mask):
        """Drop the flagged rows and add a line to the log."""
        removed = int(remove_mask.sum())
        df = df.loc[~remove_mask].copy()
        log.append(
            {
                "step": len(log),
                "rule": rule,
                "rows_removed": removed,
                "rows_remaining": len(df),
            }
        )
        print(f"{len(log) - 1}. {rule}: removed {removed:,}, left {len(df):,}")
        return df

    # 1. cancelled invoices start with a C
    df = apply_rule(
        df,
        "Remove cancellation invoices",
        df["Invoice"].str.upper().str.startswith("C"),
    )

    # 2. anything still <= 0 is an adjustment
    df = apply_rule(df, "Remove quantity <= 0", df["Quantity"] <= 0)

    # 3. zero or negative price: samples or bad debt adjustments
    df = apply_rule(df, "Remove price <= 0", df["Price"] <= 0)

    # 4. real products only (POST, M, D, DOT, GIFT_* etc. fail the pattern)
    is_product = df["StockCode"].str.fullmatch(PRODUCT_CODE_PATTERN)
    is_product = is_product | df["StockCode"].isin(PRODUCT_CODE_EXCEPTIONS)
    removed_codes = df.loc[~is_product, "StockCode"].value_counts()
    df = apply_rule(df, "Remove non-product codes", ~is_product)

    # 5. no customer ID means no row in the matrix
    df = apply_rule(df, "Remove missing Customer ID", df["Customer ID"].isna())

    # 6. exact duplicates look like double entries
    df = apply_rule(df, "Remove exact duplicate lines", df.duplicated(keep="first"))

    # it was a float because of the missing values
    df["Customer ID"] = df["Customer ID"].astype(int)
    df["Description"] = df["Description"].fillna("").astype(str).str.strip()

    return df.reset_index(drop=True), pd.DataFrame(log), removed_codes