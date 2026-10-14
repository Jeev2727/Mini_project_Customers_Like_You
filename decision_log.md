# Decision log

Each entry has the stage, who made the call, what we decided, why, and what else we looked at.

1. Setup (Manavkrishna). We cache the raw data as a pickle in `data/interim`. read_excel takes minutes, pyarrow failed on Python 3.14 and pickle needs nothing extra. Other options were Parquet or reading the Excel every time.

2. ML1, before fitting (Jeevan). Main method is truncated SVD, with PCA later at the same k. The matrix is sparse and non-negative, so SVD needs no centring and gives both sets of factors. The alternative was PCA.

3. DE2 (Manavkrishna). Cleaning rules run in this order: cancellations, quantity <= 0, price <= 0, non-product codes, missing Customer ID, exact duplicates. The log counts are sequential, so a row that breaks two rules is counted once, and duplicates come last so only valid rows are counted. Separate counts per rule are in notebook 01.

4. DE2 (Manavkrishna). We clean before splitting. Every rule looks at one row, so nothing from the test period can reach training. The alternative was to split first.

5. DE2 (Manavkrishna). A product code is 5 digits plus letters, or DCGS. PADS and SP1002 stay out. This catches vouchers, fees, adjustments and tests. DCGS rows have no customer ID so they change nothing, and SP1002 has 3 rows and would fail the 10-customer filter anyway. The alternative was a fixed list of codes.

6. DE2 (Manavkrishna). We upper-case StockCode, store Customer ID as an integer and drop exact duplicates (26,055 in the sequential cleaning log, 34,335 on the raw data). Case variants are one product, the ID was a float because of missing values, and duplicates look like double entries.

7. DE3 (Manavkrishna). The test period is the last 2 months, with the cutoff on 9 October 2011. Held-out purchases are the (customer, product) pairs the customer never bought in training. This follows the brief. The window is the pre-Christmas peak, which is a risk for the memo. A random split was the alternative.

8. DE4 (Manavkrishna). We keep products with at least 10 training customers, because rare items give noisy factors. It's computed on the training rows only. We could have used no filter.

9. DE4 (Manavkrishna). A matrix cell is log1p of the total quantity, because a few wholesale lines are huge. We compared raw quantity and a 1/0 flag in notebook 06.

10. ML4 (Manavkrishna). We only evaluate held-out pairs whose customer and product are in the matrix, since the rest can't be recommended. The dropped counts are in `split_report.csv`. The alternative was to use all pairs.

11. DE6 (Manavkrishna). One command, `python src/build_matrix.py`. The brief asks for it, and it creates `data/interim` and `data/processed` itself. The alternative was separate scripts.

12. Privacy (Manavkrishna). Files with customer IDs stay in `data/`, and `outputs/` and `report/` only hold counts, names and indices, because customer IDs must not leave the team.

13. ML2 (Jeevan). k is chosen on a validation period cut from the training rows. Using the test set would let it influence the model, and the error curve has no elbow. The alternative was the k with the best test hit rate.

14. ML2 (Jeevan). We take the smallest k within one point of the best validation hit rate, which gives k = 100. Smaller is cheaper, one point is noise, and k = 150 and 200 are no better. k = 50 is 1.6 points lower.

15. ML2 (Jeevan). We accept factors (6.99 MB) that are larger than the sparse matrix (4.76 MB). The saving is against the dense matrix, 22 times, and the memo says so. A k below 68 would be smaller than the sparse matrix but has a lower hit rate.

16. ML3 (Jeevan). We keep the cosine scoring from the brief, report it, and use it for the similar-products lists, since it's the specification.

17. ML3 (Jeevan). The final customer scoring is the customer's row of the rank-k matrix (reconstruct), and owned products are never recommended. Cosine scales popularity away and loses to best sellers. We chose this on the validation period and tested it once. The alternative was cosine only.

18. ML4 (Jeevan). The baseline is the 10 products with the most distinct training buyers that the customer hasn't bought. This matches the brief. Ranking by quantity was the alternative.

19. ML4 (Jeevan). We use a 95% bootstrap interval, plus a paired bootstrap of the difference to the baseline. Both methods are scored on the same customers, so the paired interval is the fairer test. The alternative was a point estimate.

20. ML5 (Jeevan). PCA runs on the sparse matrix with arpack, with the same k and the same two scorings. We report the results but don't use them to change the method, which was fixed in ML1. PCA was no better (0.385 against 0.383). The alternative was to pick the winner on the test set.

21. ML6 (Jeevan). We use 2 SVD components, standardised, then k-means with 3 to 6 groups chosen by silhouette, which gives 3. This follows the brief, and component 1 follows total purchase value. More dimensions would be a next step.

22. ML6 (Jeevan). We describe groups by the ratio of group share to other customers' share, plus the most bought products and countries. The large group would otherwise look like the average. The alternative was most bought products only.

23. ML7 (Jeevan). We keep log1p(quantity). Raw quantity is clearly worse (0.231 against 0.390). The 1/0 flag scored higher on validation (0.418, paired difference +0.028, 95% interval 0.005 to 0.051), so it's probably a real gain. We kept log because it was fixed first, k and the scoring were chosen with it, and the test set has already been looked at once, so switching now would mean a second look at it. The 1/0 flag is a next step.

24. J2 (both). The memo says the hit rate is for personal top-10 lists and that the product-page box was only checked by eye. The evaluation is per customer, while the box uses similar products.