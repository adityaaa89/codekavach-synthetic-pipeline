from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd
from scipy.stats import chisquare, ks_2samp

from .common import CONFIG_DIR, GENERATED_DIR, REFERENCE_DIR, REPORTS_DIR, ensure_dirs, load_json, save_json


def chi_square_against_probs(series: pd.Series, expected_probs: dict[str, float]) -> dict:
    labels = list(expected_probs.keys())
    observed_counts = series.astype(str).value_counts().reindex(labels, fill_value=0).astype(float)
    total = observed_counts.sum()
    expected_counts = np.array([expected_probs[k] * total for k in labels], dtype=float)
    observed = observed_counts.to_numpy(dtype=float)
    # Numerical normalization protects against tiny floating-point sum differences.
    expected_counts *= observed.sum() / expected_counts.sum()
    stat, p = chisquare(observed, f_exp=expected_counts)
    return {
        "chi_square_statistic": float(stat),
        "p_value": float(p),
        "observed_proportions": {
            label: float(observed_counts[label] / total) if total else 0.0 for label in labels
        },
        "expected_proportions": expected_probs
    }


def validate() -> dict:
    ensure_dirs()
    cfg = load_json(CONFIG_DIR / "calibration_config.json")
    complaints = pd.read_csv(GENERATED_DIR / "complaints.csv")
    withdrawals = pd.read_csv(GENERATED_DIR / "withdrawals.csv")
    transactions = pd.read_csv(GENERATED_DIR / "transactions.csv")

    report = {
        "dataset_sizes": {
            "complaints": int(len(complaints)),
            "transactions": int(len(transactions)),
            "withdrawals": int(len(withdrawals))
        },
        "fraud_type_calibration": chi_square_against_probs(
            complaints["fraud_type"], cfg["fraud_type_probs"]
        ),
        "state_calibration": chi_square_against_probs(
            complaints["state"], cfg["state_probs"]
        )
    }

    ref_amount_path = REFERENCE_DIR / "paysim_reference.csv"
    if ref_amount_path.exists():
        ref = pd.read_csv(ref_amount_path)
        ref_amounts = pd.to_numeric(ref["amount"], errors="coerce").dropna()
        syn_amounts = pd.to_numeric(complaints["amount"], errors="coerce").dropna()
        stat, p = ks_2samp(ref_amounts, syn_amounts)
        report["amount_distribution_ks"] = {
            "ks_statistic": float(stat),
            "p_value": float(p),
            "reference_n": int(len(ref_amounts)),
            "synthetic_n": int(len(syn_amounts)),
            "note": "With a tiny reference sample, use this as a diagnostic rather than a definitive acceptance test."
        }

    report["withdrawal_channel_share"] = {
        str(k): float(v) for k, v in withdrawals["channel"].value_counts(normalize=True).to_dict().items()
    }
    report["withdrawal_time_bucket_share"] = {
        str(k): float(v) for k, v in withdrawals["time_bucket"].value_counts(normalize=True).to_dict().items()
    }
    report["chain_depth_share"] = {
        str(k): float(v) for k, v in (
            pd.read_csv(GENERATED_DIR / "case_account_features.csv")
            .drop_duplicates("complaint_id")["chain_depth"]
            .value_counts(normalize=True)
            .sort_index()
            .to_dict()
        ).items()
    }

    save_json(report, REPORTS_DIR / "validation_report.json")
    logging.info(json.dumps(report, indent=2))
    logging.info(f"Saved: {REPORTS_DIR / 'validation_report.json'}")
    return report


def main() -> None:
    validate()


if __name__ == "__main__":
    main()
