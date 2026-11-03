"""Process queued training jobs for uploaded datasets. Never runs inside the API.

    python -m ml.byod_worker --once        # take at most one job, then exit
    python -m ml.byod_worker               # keep polling (every 15 s by default)

Connects to DATASET_STORE_URL, falling back to DATABASE_URL. Run it wherever the training dependencies are
installed (the root requirements.txt): a laptop pointed at the Render database's external URL, or a
dedicated worker service. Jobs survive restarts because the queue lives in the database; a job whose worker
stops heart-beating is requeued (up to 3 attempts).
"""

from __future__ import annotations

import argparse
import logging
import os
import socket
import threading
import time
from datetime import timedelta

from sqlalchemy import create_engine

from backend.app.core.config import get_settings
from backend.app.core.logging import configure_logging
from backend.app.db.datasets import DatasetStore
from backend.app.db.training import TrainingStore, fingerprint
from src.training import TRAINING_VERSION

logger = logging.getLogger("ml.byod_worker")
HEARTBEAT_SECONDS = 30
STALE_AFTER = timedelta(minutes=10)
MAX_ATTEMPTS = 3


def summary(results: dict) -> dict:
    fc = results["forecasting"]
    metrics = fc["evaluation"]["metrics"]
    return {
        "recommended_model": fc.get("recommended_model"),
        "holdout_months": fc["evaluation"].get("holdout_months"),
        "metrics": {m: metrics[m] for m in ("xgboost", "naive_last_month", "seasonal_naive") if m in metrics},
        "prophet": results["prophet"].get("metrics") if results["prophet"]["status"] == "ok" else None,
        "segments_k": results["segmentation"].get("k"),
        "shap": results["explainability"]["status"],
        "data": results["data"],
    }


class _Heartbeat:
    def __init__(self, store: TrainingStore, job_id: str) -> None:
        self.store, self.job_id, self.stop = store, job_id, threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self.stop.wait(HEARTBEAT_SECONDS):
            try:
                self.store.heartbeat(self.job_id)
            except Exception:  # a missed heartbeat only risks a requeue
                logger.warning("heartbeat_failed job=%s", self.job_id)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.stop.set()
        self.thread.join(timeout=5)


def process(job: dict, datasets: DatasetStore, training: TrainingStore, with_prophet: bool) -> str | None:
    from src.training.run import train_dataset  # heavy imports only when there is work

    owner, dataset_id = job["owner_id"], job["dataset_id"]
    meta = datasets.get(owner, dataset_id, with_report=True)
    if meta is None or meta["status"] != "ready":
        training.fail(job["job_id"], "The dataset no longer exists or is not ingested.")
        return None
    if fingerprint(meta["file_sha256"], meta["schema_mapping"], TRAINING_VERSION) != job["fingerprint"]:
        training.fail(job["job_id"], "The dataset or training code changed after this job was queued; request training again.")
        return None
    try:
        with _Heartbeat(training, job["job_id"]):
            out = train_dataset(datasets.rows(owner, dataset_id), meta["quality_report"] or {}, with_prophet=with_prophet)
        version_id = training.complete(job, out.results, out.artifact, summary(out.results))
    except Exception as exc:  # report a safe message; the full traceback stays in the worker log
        logger.exception("training_failed job=%s dataset=%s", job["job_id"], dataset_id)
        training.fail(job["job_id"], f"Training failed ({type(exc).__name__}). The operator can see details in the worker log.")
        return None
    logger.info("training_succeeded job=%s dataset=%s version=%s seconds=%s", job["job_id"], dataset_id, version_id,
                out.results["duration_s"])
    return version_id


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="PharmaIQ training worker")
    parser.add_argument("--once", action="store_true", help="process at most one job and exit")
    parser.add_argument("--poll-seconds", type=float, default=15.0)
    parser.add_argument("--no-prophet", action="store_true", help="skip the optional Prophet models")
    args = parser.parse_args(argv)

    settings = get_settings()
    configure_logging(settings.log_level)
    for noisy in ("cmdstanpy", "prophet"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    url = settings.dataset_store_sqlalchemy_url
    if url is None:
        parser.error("Set DATASET_STORE_URL or DATABASE_URL to the database the API uses for uploads.")
    engine = create_engine(url, pool_pre_ping=True)
    datasets, training = DatasetStore(engine), TrainingStore(engine)
    datasets.ensure_schema()
    worker = f"{socket.gethostname()}:{os.getpid()}"
    logger.info("worker_started worker=%s once=%s", worker, args.once)
    while True:
        recovered = training.recover_stale(STALE_AFTER, MAX_ATTEMPTS)
        if recovered:
            logger.warning("stale_jobs_recovered count=%d", recovered)
        job = training.claim(worker)
        if job is not None:
            logger.info("job_claimed job=%s dataset=%s attempt=%d", job["job_id"], job["dataset_id"], job["attempts"])
            process(job, datasets, training, with_prophet=not args.no_prophet)
        if args.once:
            return 0
        if job is None:
            time.sleep(args.poll_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
