from __future__ import annotations

import logging
import numpy as np
import pandas as pd

from .common import GENERATED_DIR, ML_READY_DIR, ensure_dirs


def build() -> None:
    ensure_dirs()
    complaints = pd.read_csv(GENERATED_DIR / "complaints.csv")
    case_accounts = pd.read_csv(GENERATED_DIR / "case_account_features.csv")
    withdrawals = pd.read_csv(GENERATED_DIR / "withdrawals.csv")
    accounts = pd.read_csv(GENERATED_DIR / "accounts.csv")
    normal = pd.read_csv(GENERATED_DIR / "normal_accounts.csv")

    # 1) Terminal-node classifier: one account per case.
    terminal_cols = [
        "complaint_id", "account_id", "hop_depth", "chain_depth", "incoming_amount",
        "retained_amount_ratio", "holding_time_minutes", "outgoing_activity_in_case",
        "prior_case_count", "mule_reputation", "account_age_days", "ring_size",
        "account_type", "is_terminal_node"
    ]
    case_accounts[terminal_cols].to_csv(ML_READY_DIR / "terminal_node_training.csv", index=False)

    # 2) Channel classifier: one row per complaint, using terminal account features known before location prediction.
    terminal = case_accounts[case_accounts["is_terminal_node"] == 1].copy()
    first_withdrawal = withdrawals.sort_values(["complaint_id", "split_index"]).drop_duplicates("complaint_id")
    channel = terminal.merge(
        complaints[["complaint_id", "fraud_type", "amount", "state", "city"]],
        on="complaint_id",
        how="left"
    ).merge(
        first_withdrawal[["complaint_id", "channel"]],
        on="complaint_id",
        how="left"
    )
    channel_cols = [
        "complaint_id", "fraud_type", "amount", "state", "city", "chain_depth",
        "prior_case_count", "mule_reputation", "account_age_days", "ring_size",
        "account_type", "channel"
    ]
    channel[channel_cols].to_csv(ML_READY_DIR / "withdrawal_channel_training.csv", index=False)

    # 3) Location model: one row per actual cash-out event.
    loc = withdrawals.merge(
        complaints[["complaint_id", "fraud_type", "amount", "state", "city"]].rename(
            columns={"state": "complaint_state", "city": "complaint_city"}
        ),
        on="complaint_id",
        how="left"
    ).merge(
        terminal[["complaint_id", "chain_depth", "prior_case_count", "mule_reputation", "account_age_days", "ring_size", "account_type"]],
        on="complaint_id",
        how="left"
    )
    loc["withdrawal_timestamp"] = pd.to_datetime(loc["withdrawal_timestamp"], errors="coerce")
    loc["hour_of_day"] = loc["withdrawal_timestamp"].dt.hour
    loc["day_of_week"] = loc["withdrawal_timestamp"].dt.dayofweek
    loc_cols = [
        "complaint_id", "fraud_type", "amount", "complaint_state", "complaint_city",
        "chain_depth", "prior_case_count", "mule_reputation", "account_age_days",
        "ring_size", "account_type", "channel", "hour_of_day", "day_of_week",
        "split_index", "split_count", "location_id"
    ]
    loc[loc_cols].to_csv(ML_READY_DIR / "withdrawal_location_training.csv", index=False)

    # 4) Withdrawal-time model.
    complaints_time = complaints[["complaint_id", "complaint_timestamp"]].copy()
    complaints_time["complaint_timestamp"] = pd.to_datetime(complaints_time["complaint_timestamp"], errors="coerce")
    time_df = loc.merge(complaints_time, on="complaint_id", how="left")
    time_df["delay_minutes"] = (
        time_df["withdrawal_timestamp"] - time_df["complaint_timestamp"]
    ).dt.total_seconds() / 60.0
    time_cols = [
        "complaint_id", "fraud_type", "amount", "complaint_state", "complaint_city",
        "chain_depth", "prior_case_count", "mule_reputation", "account_age_days",
        "ring_size", "account_type", "channel", "time_bucket", "delay_minutes"
    ]
    time_df[time_cols].to_csv(ML_READY_DIR / "withdrawal_time_training.csv", index=False)

    # 5) Anomaly-detection dataset. Fraud account behavior is derived from simulated case history.
    fraud_summary = case_accounts.groupby("account_id").agg(
        transaction_velocity_24h=("complaint_id", "count"),
        unique_counterparties_7d=("complaint_id", "nunique"),
        incoming_outgoing_ratio=("retained_amount_ratio", lambda x: float(1 + np.mean(x))),
        avg_holding_time_minutes=("holding_time_minutes", "mean"),
        prior_case_count=("prior_case_count", "max")
    ).reset_index()
    fraud_summary["ring_membership"] = 1
    fraud_summary["is_fraud_account"] = 1

    anomaly_cols = [
        "account_id", "transaction_velocity_24h", "unique_counterparties_7d",
        "incoming_outgoing_ratio", "avg_holding_time_minutes", "ring_membership",
        "prior_case_count", "is_fraud_account"
    ]
    anomaly = pd.concat([fraud_summary[anomaly_cols], normal[anomaly_cols]], ignore_index=True)
    anomaly.to_csv(ML_READY_DIR / "anomaly_detection.csv", index=False)

    logging.info("ML-ready datasets created:")
    for name in [
        "terminal_node_training.csv",
        "withdrawal_channel_training.csv",
        "withdrawal_location_training.csv",
        "withdrawal_time_training.csv",
        "anomaly_detection.csv"
    ]:
        logging.info(f"Saved: {ML_READY_DIR / name}")


def main() -> None:
    build()


if __name__ == "__main__":
    main()
