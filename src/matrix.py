"""Builds the sparse customer by product matrix, plus the index tables and the matrix report.

Owner: Manavkrishna Manoj.
"""

import numpy as np
import pandas as pd
import scipy.sparse as sp

VALUE_RULES = {
    "log": "log1p(total quantity per customer and product)",
    "count": "total quantity per customer and product (raw)",
    "binary": "1 if the customer ever bought the product",
}


def build_matrix(train, min_customers=10, value="log"):
    """Build the matrix from the training rows only.

    value is what a cell holds: "log" (what we use), "count" or "binary" (the last two
    are only for notebook 06). Returns (A, customers, products, info).
    """
    if value not in VALUE_RULES:
        raise ValueError(f"value must be one of {list(VALUE_RULES)}")
    # keep products with at least min_customers buyers
    customers_before = train["Customer ID"].nunique()
    counts = train.groupby("StockCode")["Customer ID"].nunique()
    keep = counts[counts >= min_customers].index
    filtered = train[train["StockCode"].isin(keep)]

    # one cell per customer and product, total quantity over all training invoices
    grouped = filtered.groupby(["Customer ID", "StockCode"], as_index=False)[
        "Quantity"
    ].sum()
    assert (grouped["Quantity"] > 0).all()  # cleaning should already guarantee this

    quantity = grouped["Quantity"].astype(float)
    if value == "log":
        grouped["value"] = np.log1p(quantity)
    elif value == "count":
        grouped["value"] = quantity
    else:
        grouped["value"] = 1.0

    customers = np.sort(grouped["Customer ID"].unique())
    products = np.sort(grouped["StockCode"].unique())
    rows = pd.Index(customers).get_indexer(grouped["Customer ID"])
    cols = pd.Index(products).get_indexer(grouped["StockCode"])

    A = sp.csr_matrix(
        (grouped["value"].to_numpy(dtype=np.float64), (rows, cols)),
        shape=(len(customers), len(products)),
    )

    info = {
        "min_customers_per_product": min_customers,
        "purchase_value": VALUE_RULES[value],
        "products_before_filter": len(counts),
        "products_kept": len(products),
        "products_removed": len(counts) - len(products),
        "customers_before_filter": customers_before,
        "customers_lost_to_filter": customers_before - len(customers),
    }
    return A, customers, products, info


def matrix_report(A, info):
    """Shape, non-zeros, sparsity and memory, dense vs sparse."""
    n_customers, n_products = A.shape
    total_cells = n_customers * n_products
    dense_mb = total_cells * 8 / 1024**2
    sparse_mb = (A.data.nbytes + A.indices.nbytes + A.indptr.nbytes) / 1024**2

    report = dict(info)
    report.update(
        {
            "customers": n_customers,
            "products": n_products,
            "non_zeros": A.nnz,
            "sparsity_%": round((1 - A.nnz / total_cells) * 100, 4),
            "dense_memory_MB": round(dense_mb, 2),
            "sparse_memory_MB": round(sparse_mb, 2),
            "dense_to_sparse_ratio": round(dense_mb / sparse_mb, 1),
        }
    )
    return report


def index_tables(train, customers, products):
    """Index maps. Products get a description and customers a country, so we can name
    the groups in notebook 05. Both come from the training rows.
    """

    def most_common(frame, key, column):
        counts = frame.groupby([key, column]).size().reset_index(name="n")
        counts = counts.sort_values("n", ascending=False).drop_duplicates(key)
        return counts.set_index(key)[column]

    description = most_common(train, "StockCode", "Description")
    country = most_common(train, "Customer ID", "Country")

    product_index = pd.DataFrame(
        {
            "product_idx": range(len(products)),
            "StockCode": products,
            "Description": description.reindex(products).to_numpy(),
        }
    )
    customer_index = pd.DataFrame(
        {
            "customer_idx": range(len(customers)),
            "Customer ID": customers,
            "Country": country.reindex(customers).to_numpy(),
        }
    )
    return customer_index, product_index


def restrict_eval_pairs(held_out, customers, products):
    """Keep the held-out pairs where the customer is a matrix row and the product is a
    matrix column. Returns index pairs (no IDs) and a count of what was dropped.
    """
    c_idx = pd.Index(customers).get_indexer(held_out["Customer ID"])
    p_idx = pd.Index(products).get_indexer(held_out["StockCode"])
    ok = (c_idx >= 0) & (p_idx >= 0)

    eval_pairs = pd.DataFrame({"customer_idx": c_idx[ok], "product_idx": p_idx[ok]})
    report = {
        "held_out_pairs": len(held_out),
        "dropped_customer_not_in_matrix": int((c_idx < 0).sum()),
        "dropped_product_not_in_matrix": int(((c_idx >= 0) & (p_idx < 0)).sum()),
        "eval_pairs_kept": int(ok.sum()),
        "eval_customers": int(eval_pairs["customer_idx"].nunique()),
    }
    return eval_pairs, report