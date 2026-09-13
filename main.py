from __future__ import annotations

import argparse
import logging

from src.calibrate import calibrate
from src.generate_dataset import generate
from src.build_ml_datasets import build
from src.validate_dataset import validate


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    parser = argparse.ArgumentParser(description="Run the complete CodeKavach synthetic-data pipeline.")
    parser.add_argument("--cases", type=int, default=None, help="Number of synthetic complaints to generate.")
    parser.add_argument(
        "--skip-calibration",
        action="store_true",
        help="Reuse config/calibration_config.json instead of recalibrating."
    )
    args = parser.parse_args()

    if not args.skip_calibration:
        logging.info("[1/4] Calibrating from reference data...")
        calibrate()

    logging.info("[2/4] Generating synthetic cybercrime data...")
    generate(args.cases)

    logging.info("[3/4] Building ML-ready datasets...")
    build()

    logging.info("[4/4] Validating generated data...")
    validate()

    logging.info("Pipeline complete.")


if __name__ == "__main__":
    main()
