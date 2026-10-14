"""SVD and PCA recommenders, the best seller baseline and the evaluation helpers.

Owner: Jeevan Mandala.
"""

import time
import tracemalloc
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.linalg import norm as sparse_norm
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.preprocessing import normalize

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
RESULTS_DIR = ROOT / "outputs" / "results"
SEED = 42  # fixed so results repeat (R6)
TOP_N = 10


def load_processed():
    """Matrix, index tables and evaluation pairs written by build_matrix.py."""
    A = sp.load_npz(PROCESSED / "customer_product_train.npz").tocsr()
    customer_index = pd.read_csv(PROCESSED / "customer_index.csv")
    product_index = pd.read_csv(PROCESSED / "product_index.csv")
    eval_pairs = pd.read_csv(PROCESSED / "eval_pairs.csv")
    return A, customer_index, product_index, eval_pairs


def fit_svd(A, k, seed=SEED):
    """Truncated SVD on the sparse matrix, no centring."""
    return TruncatedSVD(n_components=k, random_state=seed).fit(A)


def product_vectors(svd, sigma_power=0.0):
    """Product vectors scaled to length 1, so a dot product is a cosine.
    sigma_power weights each component by its singular value first (0 means plain V).
    """
    V = svd.components_.T * (svd.singular_values_**sigma_power)
    return normalize(V)


def relative_error(A, svd):
    """||A - A_k|| / ||A|| (Frobenius). The rows of V are orthonormal, so
    ||A - A_k||^2 = ||A||^2 - ||A V^T||^2, which means we never have to build A_k.
    """
    total = sparse_norm(A)
    projected = svd.transform(A)  # A V^T, the customer factors
    error_sq = total**2 - np.linalg.norm(projected) ** 2
    return float(np.sqrt(max(error_sq, 0.0)) / total)


def factor_size_mb(A, k):
    """Storage for the customer and product factors as float64."""
    return (A.shape[0] + A.shape[1]) * k * 8 / 1024**2


def top_rows(scores, n=TOP_N):
    """Indices of the n largest scores in every row, best first."""
    top = np.argpartition(-scores, n, axis=1)[:, :n]
    order = np.argsort(-np.take_along_axis(scores, top, axis=1), axis=1)
    return np.take_along_axis(top, order, axis=1)


def similar_products_all(P, n=TOP_N, batch=1000):
    """The n most similar products (cosine) for every product."""
    out = np.empty((P.shape[0], n), dtype=np.int32)
    for start in range(0, P.shape[0], batch):
        stop = min(start + batch, P.shape[0])
        scores = P[start:stop] @ P.T
        scores[np.arange(stop - start), np.arange(start, stop)] = -np.inf  # not itself
        out[start:stop] = top_rows(scores, n)
    return out


def recommend_for_customers(A, P, rows, n=TOP_N, batch=1000):
    """Cosine scoring: similarity to what the customer bought, weighted by purchase
    value. Returns the top n products they haven't bought.
    """
    out = np.empty((len(rows), n), dtype=np.int32)
    for start in range(0, len(rows), batch):
        chunk = rows[start : start + batch]
        bought = A[chunk]
        profile = bought @ P  # customer's taste in the k-dimensional space
        scores = profile @ P.T  # similarity of that taste to every product
        scores[bought.toarray() > 0] = -np.inf  # don't recommend what they already own
        out[start : start + batch] = top_rows(scores, n)
    return out


def best_seller_baseline(A, rows, n=TOP_N, batch=1000):
    """The n best selling training products the customer hasn't bought.
    Best seller means bought by the most different customers.
    """
    popularity = np.asarray((A > 0).sum(axis=0)).ravel()
    order = np.argsort(-popularity, kind="stable")  # ties keep product order
    rank_score = np.empty(len(popularity))
    rank_score[order] = np.arange(len(popularity), 0, -1)  # best seller = highest

    out = np.empty((len(rows), n), dtype=np.int32)
    for start in range(0, len(rows), batch):
        chunk = rows[start : start + batch]
        scores = np.tile(rank_score, (len(chunk), 1))
        scores[A[chunk].toarray() > 0] = -np.inf
        out[start : start + batch] = top_rows(scores, n)
    return out


def hits_at_n(recs, rows, eval_pairs, shape):
    """True for each customer who has at least one held-out product in their list."""
    held_out = sp.csr_matrix(
        (
            np.ones(len(eval_pairs), dtype=np.int8),
            (eval_pairs["customer_idx"], eval_pairs["product_idx"]),
        ),
        shape=shape,
    )
    n = recs.shape[1]
    found = held_out[np.repeat(rows, n), recs.ravel()]
    return np.asarray(found).reshape(len(rows), n).sum(axis=1) > 0


def bootstrap_ci(hits, reps=1000, seed=SEED):
    """95% interval for the hit rate, resampling customers."""
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, len(hits), size=(reps, len(hits)))
    rates = hits[picks].mean(axis=1)
    return float(np.percentile(rates, 2.5)), float(np.percentile(rates, 97.5))


def paired_bootstrap_diff(hits_a, hits_b, reps=1000, seed=SEED):
    """Hit rate of A minus hit rate of B, with a 95% interval. Both are scored on
    the same customers, so the same customers are resampled for both.
    """
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, len(hits_a), size=(reps, len(hits_a)))
    diffs = hits_a[picks].mean(axis=1) - hits_b[picks].mean(axis=1)
    return (
        float(hits_a.mean() - hits_b.mean()),
        float(np.percentile(diffs, 2.5)),
        float(np.percentile(diffs, 97.5)),
    )


def nightly_job(A, k, seed=SEED, mode="cosine"):
    """Times the nightly job: fit, similar products for every product, top 10 for
    every customer. mode is "cosine" or "reconstruct" (the final model).
    """
    start = time.perf_counter()
    svd = fit_svd(A, k, seed)
    fit_seconds = time.perf_counter() - start
    P = product_vectors(svd)

    start = time.perf_counter()
    similar_products_all(P)
    similar_seconds = time.perf_counter() - start

    start = time.perf_counter()
    everyone = np.arange(A.shape[0])
    if mode == "reconstruct":
        recommend_blend(A, svd, everyone, mode="reconstruct", alpha=1.0)
    else:
        recommend_for_customers(A, P, everyone)
    customers_seconds = time.perf_counter() - start

    return {
        "k": k,
        "mode": mode,
        "fit_seconds": round(fit_seconds, 2),
        "similar_products_seconds": round(similar_seconds, 2),
        "all_customers_seconds": round(customers_seconds, 2),
        "total_seconds": round(fit_seconds + similar_seconds + customers_seconds, 2),
    }


def timed_with_peak_memory(function, *args, **kwargs):
    """Run a function and return (result, seconds, peak memory in MB)."""
    tracemalloc.start()
    start = time.perf_counter()
    result = function(*args, **kwargs)
    seconds = time.perf_counter() - start
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return result, seconds, peak / 1024**2


def popularity_scores(A):
    """Standardised log of how many customers bought each product."""
    popularity = np.log1p(np.asarray((A > 0).sum(axis=0)).ravel())
    scores = (popularity - popularity.mean()) / popularity.std()
    # ties are ordered by index like the baseline, so alpha = 0 matches it exactly
    return scores - 1e-9 * np.arange(len(scores))


def model_scores(bought, svd, mode="cosine", sigma_power=0.0):
    """Score every product for a block of customers (the model is a fitted SVD or PCA).

    cosine: similarity of length-1 product vectors to what the customer bought,
      which scales popularity away.
    reconstruct: the customer's row of the rank-k matrix. For PCA the row is centred,
      projected, and the column means are added back, so popular products keep
      their weight.
    """
    if mode == "cosine":
        P = product_vectors(svd, sigma_power)
    elif mode == "reconstruct":
        P = svd.components_.T
    else:
        raise ValueError(f"unknown mode: {mode}")
    return (bought @ P) @ P.T


def recommend_blend(
    A, svd, rows, mode="cosine", alpha=1.0, n=TOP_N, batch=1000, sigma_power=0.0
):
    """Top n unseen products by alpha * model score + (1 - alpha) * popularity.
    The model score is standardised per customer. alpha = 1 is the pure model and
    alpha = 0 is popularity only.
    """
    popularity = popularity_scores(A)
    out = np.empty((len(rows), n), dtype=np.int32)
    for start in range(0, len(rows), batch):
        chunk = rows[start : start + batch]
        bought = A[chunk]
        scores = model_scores(bought, svd, mode, sigma_power)
        spread = scores.std(axis=1, keepdims=True) + 1e-12
        scores = (scores - scores.mean(axis=1, keepdims=True)) / spread
        scores = alpha * scores + (1 - alpha) * popularity
        scores[bought.toarray() > 0] = -np.inf  # don't recommend what they already own
        out[start : start + batch] = top_rows(scores, n)
    return out


def fit_pca(A, k, seed=SEED):
    """PCA on the sparse matrix. The arpack solver centres implicitly, so the matrix
    stays sparse (needs scikit-learn 1.4 or later).
    """
    return PCA(n_components=k, svd_solver="arpack", random_state=seed).fit(A)


def pca_scores_of(pca, X):
    """Centred customer factors (X - mean) V^T for a sparse block X, without
    turning it dense.
    """
    V = pca.components_
    return np.asarray(X @ V.T) - pca.mean_ @ V.T


def pca_relative_error(A, pca):
    """Relative error on the centred matrix A_c, since that's what PCA compresses.
    ||A_c||^2 = ||A||^2 - n ||mean||^2.
    """
    n = A.shape[0]
    total_sq = sparse_norm(A) ** 2 - n * float(np.sum(pca.mean_**2))
    projected = pca_scores_of(pca, A)
    error_sq = total_sq - np.linalg.norm(projected) ** 2
    return float(np.sqrt(max(error_sq, 0.0) / total_sq))


def recommend_pca(A, pca, rows, mode="reconstruct", n=TOP_N, batch=1000):
    """Top n unseen products from a PCA model, scored the same two ways as SVD.
    reconstruct adds each product's mean back, cosine uses length-1 product vectors.
    """
    V = pca.components_
    P = normalize(V.T)
    out = np.empty((len(rows), n), dtype=np.int32)
    for start in range(0, len(rows), batch):
        chunk = rows[start : start + batch]
        bought = A[chunk]
        if mode == "reconstruct":
            scores = pca_scores_of(pca, bought) @ V + pca.mean_
        elif mode == "cosine":
            scores = (bought @ P) @ P.T
        else:
            raise ValueError(f"unknown mode: {mode}")
        scores[bought.toarray() > 0] = -np.inf  # don't recommend what they already own
        out[start : start + batch] = top_rows(scores, n)
    return out