"""Generate data, validate/clean it, and load PostgreSQL. Usage: python scripts/bootstrap.py [--skip-db]"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config  # noqa: E402
from src.data import generate, load, validate  # noqa: E402
from src.data.database import db_available  # noqa: E402

logger = logging.getLogger("bootstrap")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-db", action="store_true", help="generate + validate only")
    args = parser.parse_args()
    config.setup_logging()
    config.ensure_dirs()

    meta = generate.write_outputs()
    report = validate.validate_and_build()
    logger.info("Generated %d clean rows; clean validation passed=%s", meta["n_rows_clean"], report["clean"]["passed"])

    if args.skip_db:
        logger.info("Skipping database load (--skip-db)")
        return 0
    if not db_available():
        logger.error("PostgreSQL is not reachable at DATABASE_URL. Start it with `docker compose up -d`, "
                     "or rerun with --skip-db.")
        return 1
    load.load_clean()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
