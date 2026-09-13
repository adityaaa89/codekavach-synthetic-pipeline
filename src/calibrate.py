from __future__ import annotations

import argparse
import logging
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import lognorm

from .common import CONFIG_DIR, REFERENCE_DIR, ensure_dirs, load_json, save_json


def normalize(weights: dict[str, float]) -> dict[str, float]:
    total = float(sum(weights.values()))
    if total <= 0:
        raise ValueError("Calibration weights must sum to a positive value.")
    return {str(k): float(v) / total for k, v in weights.items()}


def calibrate(defaults_path: Path | None = None) -> dict:
    ensure_dirs()
    defaults_path = defaults_path or (CONFIG_DIR / "defaults.json")
    defaults = load_json(defaults_path)

    calibration = {
        "seed": int(defaults["seed"]),
        "num_cases": int(defaults["num_cases"]),
        "complaints_per_day": int(defaults["complaints_per_day"]),
        "sources": {},
        "assumptions": {
            "mule_reuse_rate": defaults["mule_reuse_rate"],
            "cross_region_probability": defaults["cross_region_probability"],
            "timing_outlier_rate": defaults["timing_outlier_rate"],
            "chain_depth_probs": defaults["chain_depth_probs"],
            "atm_time_bucket_probs": defaults["atm_time_bucket_probs"],
            "delay_minutes_by_fraud_type": defaults["delay_minutes_by_fraud_type"]
        },
        "entity_attributes": defaults.get("entity_attributes", {
            "banks": ["Bank_A", "Bank_B"],
            "account_types": ["SAVINGS", "CURRENT"],
            "age_brackets": ["18-25", "26-35", "36-50", "51+"],
            "balance_tiers": ["Low", "Medium", "High"]
        })
    }

    ncrb_path = REFERENCE_DIR / "ncrb_stats.csv"
    if ncrb_path.exists():
        ncrb = pd.read_csv(ncrb_path)
        required = {"state", "fraud_type", "cases"}
        if not required.issubset(ncrb.columns):
            raise ValueError(f"{ncrb_path.name} must contain columns: {sorted(required)}")
        ncrb["cases"] = pd.to_numeric(ncrb["cases"], errors="coerce").fillna(0)
        fraud = ncrb.groupby("fraud_type", dropna=True)["cases"].sum().to_dict()
        state = ncrb.groupby("state", dropna=True)["cases"].sum().to_dict()
        calibration["fraud_type_probs"] = normalize(fraud)
        calibration["state_probs"] = normalize(state)
        calibration["sources"]["fraud_type_probs"] = "data/reference/ncrb_stats.csv"
        calibration["sources"]["state_probs"] = "data/reference/ncrb_stats.csv"
    else:
        calibration["fraud_type_probs"] = normalize(defaults["fallback_fraud_type_probs"])
        calibration["state_probs"] = normalize(defaults["fallback_state_probs"])
        calibration["sources"]["fraud_type_probs"] = "fallback assumption"
        calibration["sources"]["state_probs"] = "fallback assumption"

    amount_path = REFERENCE_DIR / "paysim_reference.csv"
    if amount_path.exists():
        amount_df = pd.read_csv(amount_path)
        if "amount" not in amount_df.columns:
            raise ValueError("paysim_reference.csv must contain an 'amount' column.")
        amounts = pd.to_numeric(amount_df["amount"], errors="coerce").dropna()
        amounts = amounts[amounts > 0]
        if len(amounts) < 10:
            raise ValueError("Need at least 10 positive reference amounts for calibration.")
        shape, loc, scale = lognorm.fit(amounts.to_numpy(), floc=0)
        calibration["amount_distribution"] = {
            "type": "lognormal",
            "shape": float(shape),
            "loc": float(loc),
            "scale": float(scale),
            "min": float(amounts.quantile(0.01)),
            "max": float(amounts.quantile(0.99))
        }
        calibration["sources"]["amount_distribution"] = "data/reference/paysim_reference.csv"
    else:
        calibration["amount_distribution"] = {
            "type": "lognormal",
            "shape": 1.0,
            "loc": 0.0,
            "scale": math.exp(10.5),
            "min": 500.0,
            "max": 2000000.0
        }
        calibration["sources"]["amount_distribution"] = "fallback assumption"

    locations_path = REFERENCE_DIR / "cashout_locations.csv"
    if locations_path.exists():
        locations = pd.read_csv(locations_path)
        required = {"location_id", "channel", "bank", "city", "state", "latitude", "longitude"}
        if not required.issubset(locations.columns):
            raise ValueError(f"cashout_locations.csv must contain columns: {sorted(required)}")
        channel_counts = locations["channel"].astype(str).str.upper().value_counts().to_dict()
        calibration["location_inventory"] = {
            "rows": int(len(locations)),
            "channel_counts": {str(k): int(v) for k, v in channel_counts.items()},
            "states": sorted(locations["state"].dropna().astype(str).unique().tolist())
        }
        calibration["sources"]["location_inventory"] = "data/reference/cashout_locations.csv"
    else:
        raise FileNotFoundError(
            "data/reference/cashout_locations.csv is required because withdrawal labels need a candidate location pool."
        )

    out = CONFIG_DIR / "calibration_config.json"
    save_json(calibration, out)
    return calibration


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate CodeKavach simulation parameters from reference files.")
    parser.add_argument("--defaults", type=Path, default=None)
    args = parser.parse_args()
    calibration = calibrate(args.defaults)
    logging.info("Calibration complete.")
    logging.info(f"Saved: {CONFIG_DIR / 'calibration_config.json'}")
    logging.info(f"Fraud types: {calibration['fraud_type_probs']}")
    logging.info(f"States: {calibration['state_probs']}")
    logging.info(f"Amount distribution: {calibration['amount_distribution']}")


if __name__ == "__main__":
    main()
