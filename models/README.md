# Model bundle

Everything the API serves. It is built offline, committed to Git (about 0.9 MB), and loaded once at API startup by `ml/model_loader.py`. Nothing is retrained per request or during deployment.

| File | Produced by | Used by |
|---|---|---|
| `xgboost.joblib`, `xgboost_metadata.json` | `python -m ml.train_xgboost` | live XGBoost forecasts (`/api/v1/forecasts`) |
| `xgboost_residual_quantiles.json` | `python -m ml.evaluate` | XGBoost 80% intervals |
| `prophet_forecasts.csv`, `prophet_metadata.json` | `python -m ml.train_prophet` | precomputed Prophet forecasts |
| `kmeans.joblib`, `segments.csv`, `segment_profiles.csv`, `segment_centers_standardized.csv`, `k_selection.csv` | `python -m ml.train_kmeans` | `/api/v1/segments` |
| `metrics/horizon_*.csv`, `metrics/interval_coverage.csv`, `metrics/shap_*.json` | `python -m ml.evaluate` | `/api/v1/model-insights`, forecast accuracy |
| `metrics/one_step_scorecard.csv` | `scripts/train_all.py` (copied by `ml.evaluate`) | `/api/v1/model-insights` |
| `manifest.json` | `ml.bundle.write_manifest()` | integrity check in `scripts/bootstrap_demo.py` |

## Rebuild

From the repository root, with the full environment (`pip install -r requirements.txt`) and generated data (`python scripts/bootstrap.py --skip-db`):

```bash
python -m ml.train_xgboost     # tune on 3 folds, refit on all history (~10 s)
python -m ml.train_kmeans      # ~2 s
python -m ml.train_prophet     # 18 series, ~15 s
python -m ml.evaluate          # 1-3 month backtest, interval calibration, SHAP (~2.5 min)
python -c "from ml.bundle import write_manifest; write_manifest()"
python scripts/bootstrap_demo.py   # verify hashes and that the bundle loads
```

## Version rules

- `scikit-learn` and `xgboost` must match the versions recorded in `xgboost_metadata.json` (pinned in `backend/requirements.txt`). The loader raises `BundleError` on a mismatch, and the API then reports "not ready" instead of serving predictions.
- On Linux the backend installs `xgboost-cpu` (same version) to avoid the large GPU dependency; the loader treats it as `xgboost`.
- `model_version` strings include the data equation version and a hash of the model file, so the API response identifies the exact artifact.
