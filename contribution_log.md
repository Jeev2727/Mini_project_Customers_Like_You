# Contribution log

This is a pair project: Manavkrishna Manoj is the Data Engineer and Jeevan Mandala is the ML Engineer. The roles are in the README.

How we fill this in:

- The dates and hashes come from the repository history (`git log --date=short --format="%ad  %h  %an  %s"`) and we don't edit them.
- The first version of the ML work was built locally before the repository was created, and its first commit (6678900) says so.
- Manavkrishna's commits appear under his GitHub username, Extirio. Jeevan's appear under his GitHub username, Jeev2727.
- Evidence is a commit hash or a file. A task only goes to Done once the owner's commit exists.

| Date | Task | Owner | Evidence | Status |
|------|------|-------|----------|--------|
| 2026-10-13 | J1 success targets | Both (committed by Jeevan) | `README.md`, 342cd2c | Done |
| 2026-10-13 | DE1 data profile | Manavkrishna Manoj | `notebooks/01_profile_and_cleaning.ipynb` (6a09058), `outputs/results/data_profile.csv` (ff4de75) | Done |
| 2026-10-13 | DE2 cleaning and cleaning log | Manavkrishna Manoj | `src/clean.py` (816f978), `outputs/results/cleaning_log.csv`, `removed_codes.csv` (ff4de75) | Done |
| 2026-10-13 | DE3 time split and held-out purchases | Manavkrishna Manoj | `src/split.py` (816f978), `outputs/results/split_report.csv` (ff4de75) | Done |
| 2026-10-13 | DE4 sparse matrix and index maps | Manavkrishna Manoj | `src/matrix.py` (816f978), `data/processed/` (not in Git) | Done |
| 2026-10-13 | DE5 matrix report | Manavkrishna Manoj | `outputs/results/matrix_report.csv` (ff4de75) | Done |
| 2026-10-13 | DE6 one-command rebuild | Manavkrishna Manoj | `src/build_matrix.py` (816f978) | Done |
| | DE review of ML results for data issues | Manavkrishna Manoj | notes in `decision_log.md` or a commit message | To do |
| | DE fresh clone check | Manavkrishna Manoj | `README.md` (how to run), `requirements.txt` | To do |
| 2026-10-13 | ML1 method decision before fitting | Jeevan Mandala | `decision_log.md` (entry 2, committed in 5fd02e0), `notebooks/02_choose_k.ipynb` (6678900) | Done |
| 2026-10-13 | ML2 choosing k | Jeevan Mandala | `notebooks/02_choose_k.ipynb`, `outputs/figures/svd_k_selection.png`, `svd_singular_values.png` (6678900) | Done |
| 2026-10-13 | ML3 recommendation function | Jeevan Mandala | `src/recommend.py`, `notebooks/03_recommend_and_evaluate.ipynb` (6678900) | Done |
| 2026-10-13 | ML4 evaluation and timing | Jeevan Mandala | `outputs/results/hit_rate_results.csv`, `nightly_job_timing.csv` (6678900), `paired_differences.csv` (24a438a) | Done |
| 2026-10-13 | ML5 PCA vs SVD table | Jeevan Mandala | `notebooks/04_pca_vs_svd.ipynb`, `outputs/results/pca_vs_svd.csv` (6678900), `pca_vs_svd_hit_rates.csv` (24a438a) | Done |
| 2026-10-13 | ML6 customer map | Jeevan Mandala | `notebooks/05_customer_map.ipynb`, `outputs/figures/customer_map.png`, `outputs/results/customer_segments.csv` (6678900) | Done |
| 2026-10-13 | ML7 purchase value sensitivity (stretch) | Jeevan Mandala | `notebooks/06_purchase_value_sensitivity.ipynb` (6678900), `purchase_value_paired.csv`, `purchase_value_sensitivity.csv` (24a438a) | Done |
| 2026-10-13 | J2 memo, data and risk sections | Manavkrishna Manoj | `report/memo.pdf` (2d217ef, the whole memo was committed by Jeevan) | Done |
| 2026-10-13 | J2 memo, method, results and groups sections | Jeevan Mandala | `report/memo.pdf` (2d217ef) | Done |
| 2026-10-13 | J3 slides 2 and 3, and the risks half of slide 9 | Manavkrishna Manoj | `report/slides.pptx` (a3b1221, the whole deck was committed by Jeevan) | Done |
| 2026-10-13 | J3 slides 4 to 8, and the next steps half of slide 9 | Jeevan Mandala | `report/slides.pptx` (a3b1221) | Done |
| 2026-10-13 | J3 slide 1 | Both | `report/slides.pptx` (a3b1221) | Done |
| 2026-10-13 | Decision log, rows 1 and 3 to 12 | Manavkrishna Manoj | `decision_log.md` (5fd02e0) | Done |
| 2026-10-13 | Decision log, rows 2 and 13 to 24, and row 25 together | Jeevan Mandala | `decision_log.md` (5fd02e0, the whole file was committed by Manavkrishna) | Done |
| | Individual reflection (half a page) | Manavkrishna Manoj | submitted separately | To do |
| | Individual reflection (half a page) | Jeevan Mandala | submitted separately | To do |

Each of us submitted the peer evaluation privately.

