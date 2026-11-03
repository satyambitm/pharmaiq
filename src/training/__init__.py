"""Model training on an ingested user dataset (run by the offline worker, never inside API requests).

Kept import-light: the API reads TRAINING_VERSION without loading XGBoost or Prophet.
"""

TRAINING_VERSION = "byod-train-1"
