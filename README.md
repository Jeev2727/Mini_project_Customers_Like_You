# Customers Like You: PCA or SVD on Online Retail II

This is our mini-project for the GiftLane brief. We shrink the customer by product purchase table so a nightly job can give "customers who bought this also bought" suggestions that beat the best sellers, and we draw the main customer groups for marketing. It's a pair project: Manavkrishna Manoj is the Data Engineer and Jeevan Mandala is the ML Engineer.

## Who did what

Manavkrishna Manoj did DE1 to DE6. That's `src/clean.py`, `src/split.py`, `src/matrix.py`, `src/build_matrix.py` and notebook 01, decision log rows 1 and 3 to 12, the data, limits and risk parts of the memo, and slides 2 and 3 plus the risks half of slide 9. He also reviewed the ML results for data issues and checked that a fresh clone rebuilds the matrix with one command.

Jeevan Mandala did ML1 to ML7. That's `src/recommend.py`, `src/tune_validation.py` and notebooks 02 to 06, decision log rows 2 and 13 to 24, the method, results, size and speed and customer group parts of the memo, and slides 4 to 8 plus the next steps half of slide 9.

We did J1 (targets), J2 (memo), J3 (presentation, with slide 1 together) and decision log row 25 together. Each of us commits our own work under our own name, and `contribution_log.md` lists every task with its owner, date and evidence.

## Targets

We agreed these at the start and added the results once we had the numbers.

- Factors at least 20 times smaller than the dense matrix. Result: 6.99 MB against 153.86 MB, so 22 times (but bigger than the 4.76 MB sparse matrix).
- Nightly job under 60 seconds. Result: 0.88 seconds.
- Hit rate at 10 above the best seller baseline on purchases the model never saw. Result: 0.383 against 0.270, a difference of +0.113 (95% interval 0.086 to 0.140).
- A customer map with 3 to 6 groups we can name. Result: 3 groups, named from their products and countries.

## Data

UCI Online Retail II (Chen 2012, CC BY 4.0): https://archive.ics.uci.edu/dataset/502/online+retail+ii

Put `online_retail_II.xlsx` in `data/raw/`. The data isn't committed to Git. Both sheets together have 1,067,371 rows (1 December 2009 to 9 December 2011).

## How to run

We used Python 3.14.2.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python src/build_matrix.py
```

That last command takes the raw Excel file all the way to the sparse matrix and the reports. It takes about a minute the first time because of the Excel read. After that, run notebooks 01 to 06 in order, each one top to bottom. Notebook 02 writes `outputs/results/chosen_k.txt`, which notebooks 03 to 06 read.

`python src/tune_validation.py` is an optional extra check on the validation period. Seeds are fixed (`SEED = 42` in `src/recommend.py`).

## What's in the repo

`src/build_matrix.py` is the one command. It uses `clean.py` for the cleaning rules and log, `split.py` for the time split and held-out purchases, and `matrix.py` for the sparse matrix, index tables, matrix report and evaluation pairs. `recommend.py` has the SVD and PCA recommenders, the baseline, the evaluation and the timing, and `tune_validation.py` is the optional check.

The notebooks are 01 profile and cleaning, 02 choosing k, 03 recommendations and evaluation, 04 PCA against SVD, 05 customer map and 06 purchase value sensitivity. Result tables and figures go in `outputs/` (counts only, no customer IDs), and the memo and slides are in `report/`. `decision_log.md` has our decisions with reasons and owners.

Files that contain customer IDs stay in `data/`, which Git ignores.

## Limits

The results cover one test period, the last two months before Christmas. Lines without a customer ID (22.8% of the raw data) can't be used. Customers with no history and products with fewer than 10 buyers get no suggestions, which was about 3 in 10 of the held-out purchases. The hit rate measures personal top-10 lists. We only checked the product-page box itself (similar products) by eye.