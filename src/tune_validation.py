"""Optional check on a validation period (never the test set).

Run after build_matrix.py: python src/tune_validation.py

The last 2 months of the training rows are held back, as in notebook 02. The
script tests whether weighting components by singular value (sigma_power) or
blending in popularity (alpha) beats the final model (reconstruct, alpha = 1,
k from notebook 02). Results go to outputs/results/svd_tuning_validation.csv.
The final model does not change because of it.

Owner: Jeevan Mandala.
"""

import itertools

import numpy as np
import pandas as pd

from matrix import build_matrix, restrict_eval_pairs
from recommend import (
    PROCESSED,
    RESULTS_DIR,
    best_seller_baseline,
    bootstrap_ci,
    fit_svd,
    hits_at_n,
    recommend_blend,
)
from split import time_split

# The first run (k up to 100) showed that sigma_power = 1.0 hurts and alpha makes no
# difference in reconstruct mode, and that the best k sat at the edge of the grid.
# This run uses a smaller grid and goes up to k = 200.
KS = [20, 50, 100, 150, 200]
SIGMA_POWERS = [0.0, 0.5]  # only used by mode = "cosine"
MODES = ["cosine", "reconstruct"]
ALPHAS = [1.0, 0.8, 0.5, 0.3]  # 1 = pure model, 0 = best sellers only


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    train = pd.read_pickle(PROCESSED / "train_transactions.pkl")
    inner_train, _, val_held_out, _ = time_split(train, months=2)
    A, customers, products, _ = build_matrix(inner_train, 10)
    pairs, _ = restrict_eval_pairs(val_held_out, customers, products)
    rows = np.sort(pairs["customer_idx"].unique())

    baseline_hits = hits_at_n(best_seller_baseline(A, rows), rows, pairs, A.shape)
    results = [
        {
            "method": "best-seller baseline",
            "k": None,
            "mode": None,
            "sigma_power": None,
            "alpha": None,
            "hit_rate": baseline_hits.mean(),
            "ci_low": bootstrap_ci(baseline_hits)[0],
            "ci_high": bootstrap_ci(baseline_hits)[1],
        }
    ]

    for k in KS:
        svd = fit_svd(A, k)
        for mode, sigma_power, alpha in itertools.product(MODES, SIGMA_POWERS, ALPHAS):
            if mode == "reconstruct" and sigma_power != 0.0:
                continue  # sigma_power does not apply to this mode
            recs = recommend_blend(
                A, svd, rows, mode=mode, alpha=alpha, sigma_power=sigma_power
            )
            hits = hits_at_n(recs, rows, pairs, A.shape)
            low, high = bootstrap_ci(hits)
            results.append(
                {
                    "method": "svd blend",
                    "k": k,
                    "mode": mode,
                    "sigma_power": sigma_power,
                    "alpha": alpha,
                    "hit_rate": hits.mean(),
                    "ci_low": low,
                    "ci_high": high,
                }
            )
            print(f"k={k} {mode} sigma={sigma_power} alpha={alpha}: {hits.mean():.4f}")

    table = pd.DataFrame(results).sort_values("hit_rate", ascending=False).round(4)
    out = RESULTS_DIR / "svd_tuning_validation.csv"
    table.to_csv(out, index=False)
    print("\nTop 10 on validation:")
    print(table.head(10).to_string(index=False))
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()