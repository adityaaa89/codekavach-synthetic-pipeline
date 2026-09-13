from __future__ import annotations

import argparse
import logging
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from faker import Faker
from scipy.stats import lognorm

from .common import CONFIG_DIR, GENERATED_DIR, REFERENCE_DIR, ensure_dirs, load_json



def weighted_choice(rng: np.random.Generator, mapping: dict[str, float]) -> str:
    keys = list(mapping.keys())
    probs = np.array([mapping[k] for k in keys], dtype=float)
    probs = probs / probs.sum()
    return str(rng.choice(keys, p=probs))


def sample_amount(rng: np.random.Generator, cfg: dict) -> float:
    d = cfg["amount_distribution"]
    amount = lognorm.rvs(
        s=float(d["shape"]),
        loc=float(d.get("loc", 0.0)),
        scale=float(d["scale"]),
        random_state=rng
    )
    amount = float(np.clip(amount, d.get("min", 500), d.get("max", 2000000)))
    return round(amount / 100.0) * 100.0


def sample_delay_minutes(rng: np.random.Generator, fraud_type: str, cfg: dict) -> float:
    rules = cfg["assumptions"]["delay_minutes_by_fraud_type"]
    rule = rules.get(fraud_type, rules.get("Other"))
    if rule["distribution"] == "exponential":
        delay = rng.exponential(float(rule["scale"]))
    else:
        delay = rng.gamma(float(rule["shape"]), float(rule["scale"]))
    if rng.random() < float(cfg["assumptions"]["timing_outlier_rate"]):
        delay *= rng.uniform(2.0, 5.0)
    return max(1.0, float(delay))


def choose_city_for_state(rng: np.random.Generator, locations: pd.DataFrame, state: str) -> str:
    cities = locations.loc[locations["state"] == state, "city"].dropna().unique().tolist()
    if not cities:
        cities = locations["city"].dropna().unique().tolist()
    return str(rng.choice(cities))


def choose_channel(rng: np.random.Generator, amount: float, account_type: str, prior_case_count: int) -> str:
    # Hidden simulator rule. This is the label-generation mechanism and must NOT be used as a model feature.
    score = -1.0
    if amount >= 200000:
        score += 1.6
    elif amount >= 100000:
        score += 0.7
    if account_type == "CURRENT":
        score += 0.8
    score += min(prior_case_count, 4) * 0.12
    # Add latent stochastic noise to prevent deterministic prediction
    score += rng.normal(0, 0.5)
    p_branch = 1.0 / (1.0 + np.exp(-score))
    return "BRANCH" if rng.random() < p_branch else "ATM"


def choose_location(
    rng: np.random.Generator,
    locations: pd.DataFrame,
    channel: str,
    home_state: str,
    home_city: str,
    cross_region_probability: float,
) -> pd.Series:
    pool = locations[locations["channel"].str.upper() == channel].copy()
    if pool.empty:
        raise ValueError(f"No locations available for channel {channel}")

    if rng.random() >= cross_region_probability:
        local = pool[(pool["state"] == home_state) & (pool["city"] == home_city)]
        if local.empty:
            local = pool[pool["state"] == home_state]
        if not local.empty:
            pool = local

    # Hidden hotspot mechanism. These weights are intentionally not written to ML-ready data.
    ranks = np.arange(1, len(pool) + 1, dtype=float)
    weights = 1.0 / np.power(ranks, 1.20)
    weights = weights / weights.sum()
    idx = int(rng.choice(np.arange(len(pool)), p=weights))
    return pool.iloc[idx]


def account_reputation(prior_case_count: int) -> float:
    # Leakage-safe: based only on cases that happened earlier in simulation time.
    return round(float(1.0 - np.exp(-0.35 * prior_case_count)), 4)


def generate(num_cases: int | None = None) -> None:
    ensure_dirs()
    cfg_path = CONFIG_DIR / "calibration_config.json"
    if not cfg_path.exists():
        logging.warning("Calibration config not found. Running calibration automatically...")
        from .calibrate import calibrate
        calibrate()
        
    cfg = load_json(cfg_path)
    
    # Extract entity attributes from config
    attrs = cfg.get("entity_attributes", {})
    banks = attrs.get("banks", ["Bank_A", "Bank_B"])
    account_types = attrs.get("account_types", ["SAVINGS", "CURRENT"])
    age_brackets = attrs.get("age_brackets", ["18-25", "26-35", "36-50", "51+"])
    balance_tiers = attrs.get("balance_tiers", ["Low", "Medium", "High"])
    
    seed = int(cfg["seed"])
    rng = np.random.default_rng(seed)
    fake = Faker("en_IN")
    Faker.seed(seed)

    num_cases = int(num_cases or cfg["num_cases"])
    locations = pd.read_csv(REFERENCE_DIR / "cashout_locations.csv")
    locations["channel"] = locations["channel"].astype(str).str.upper()

    fraudster_count = max(50, num_cases // 20)
    ring_count = max(20, num_cases // 50)

    fraudsters = []
    for i in range(fraudster_count):
        fraudsters.append({
            "fraudster_id": f"F{i+1:05d}",
            "home_state": weighted_choice(rng, cfg["state_probs"])
        })

    # Build fraud rings and mule-account pool.
    rings = []
    accounts = []
    ring_accounts: dict[str, list[str]] = {}
    account_state: dict[str, str] = {}
    account_city: dict[str, str] = {}
    account_type: dict[str, str] = {}
    account_bank: dict[str, str] = {}
    account_age_days: dict[str, int] = {}
    prior_case_count: dict[str, int] = defaultdict(int)

    next_account = 1
    for i in range(ring_count):
        ring_id = f"R{i+1:04d}"
        owner = fraudsters[i % len(fraudsters)]
        raw_size = int(rng.zipf(2.0))
        ring_size = int(np.clip(raw_size + 2, 3, 18))
        members = []
        for _ in range(ring_size):
            account_id = f"ACC{next_account:06d}"
            next_account += 1
            state = owner["home_state"] if rng.random() < 0.75 else weighted_choice(rng, cfg["state_probs"])
            city = choose_city_for_state(rng, locations, state)
            a_type = str(rng.choice(account_types, p=[0.82, 0.18][:len(account_types)] + [0] * max(0, len(account_types) - 2)))
            # Re-normalize if length mismatch
            if len(account_types) > 2:
                probs = [0.82, 0.18] + [0.05] * (len(account_types) - 2)
                probs = np.array(probs) / sum(probs)
                a_type = str(rng.choice(account_types, p=probs))
            elif len(account_types) == 1:
                a_type = account_types[0]
            else:
                a_type = str(rng.choice(account_types, p=[0.82, 0.18]))
            bank = str(rng.choice(banks))
            age_days = int(rng.integers(10, 2200))
            members.append(account_id)
            account_state[account_id] = state
            account_city[account_id] = city
            account_type[account_id] = a_type
            account_bank[account_id] = bank
            account_age_days[account_id] = age_days
            accounts.append({
                "account_id": account_id,
                "ring_id": ring_id,
                "bank": bank,
                "state": state,
                "city": city,
                "account_type": a_type,
                "account_age_days": age_days
            })
        ring_accounts[ring_id] = members
        rings.append({
            "ring_id": ring_id,
            "fraudster_id": owner["fraudster_id"],
            "ring_size": ring_size,
            "home_state": owner["home_state"]
        })

    # Ring popularity: a few rings get many cases.
    ring_ids = [r["ring_id"] for r in rings]
    ring_ranks = np.arange(1, len(ring_ids) + 1, dtype=float)
    ring_weights = 1 / np.power(ring_ranks, 1.15)
    ring_weights /= ring_weights.sum()

    victims = []
    complaints = []
    transactions = []
    withdrawals = []
    case_account_features = []

    current_time = datetime(2026, 1, 1, 9, 0, 0)
    rate_per_minute = float(cfg["complaints_per_day"]) / 1440.0

    for case_idx in range(num_cases):
        complaint_id = f"C{case_idx+1:07d}"
        victim_id = f"V{case_idx+1:07d}"
        fraud_type = weighted_choice(rng, cfg["fraud_type_probs"])
        victim_state = weighted_choice(rng, cfg["state_probs"])
        victim_city = choose_city_for_state(rng, locations, victim_state)
        amount = sample_amount(rng, cfg)

        interarrival = rng.exponential(1.0 / max(rate_per_minute, 1e-6))
        current_time += timedelta(minutes=float(interarrival))

        ring_id = str(rng.choice(ring_ids, p=ring_weights))
        members = ring_accounts[ring_id]
        depth_keys = [int(k) for k in cfg["assumptions"]["chain_depth_probs"].keys()]
        depth_probs = np.array([cfg["assumptions"]["chain_depth_probs"][str(k)] for k in depth_keys], dtype=float)
        chain_depth = int(rng.choice(depth_keys, p=depth_probs / depth_probs.sum()))
        chain_depth = min(chain_depth, len(members))

        # Prefer reuse according to configured reuse rate, otherwise mix members randomly.
        reused = [a for a in members if prior_case_count[a] > 0]
        fresh = [a for a in members if prior_case_count[a] == 0]
        
        # Build layers for split-and-converge topology
        # Layer 0: Victim
        # Layer 1: Entry account
        # Layer 2..N-1: Intermediate accounts (can be multiple for smurfing)
        # Layer N: Terminal account
        
        layers = []
        available = set(members)
        
        def pick_accounts(num, exclude):
            pool = [a for a in members if a not in exclude]
            if not pool:
                return []
            use_reused = reused and rng.random() < float(cfg["assumptions"]["mule_reuse_rate"])
            priority = [a for a in pool if prior_case_count[a] > 0] if use_reused else [a for a in pool if prior_case_count[a] == 0]
            if not priority:
                priority = pool
            chosen = list(rng.choice(priority, size=min(num, len(priority)), replace=False))
            return chosen

        used_accounts = set()
        
        # Layer 1 (Entry)
        entry = pick_accounts(1, used_accounts)
        if not entry: entry = [members[0]]
        used_accounts.update(entry)
        layers.append(entry)
        
        # Intermediate layers
        for _ in range(max(0, chain_depth - 2)):
            num_nodes = int(rng.choice([1, 2, 3], p=[0.5, 0.3, 0.2]))
            layer_nodes = pick_accounts(num_nodes, used_accounts)
            if not layer_nodes:
                break
            used_accounts.update(layer_nodes)
            layers.append(layer_nodes)
            
        # Terminal layer
        terminal = pick_accounts(1, used_accounts)
        if not terminal: terminal = [members[-1]]
        used_accounts.update(terminal)
        layers.append(terminal)
        
        terminal_account = layers[-1][0]
        ring_size = len(members)

        victims.append({
            "victim_id": victim_id,
            "state": victim_state,
            "city": victim_city,
            "age_bracket": str(rng.choice(age_brackets)),
            "bank": str(rng.choice(banks)),
            "account_balance_tier": str(rng.choice(balance_tiers)),
            "fraud_type": fraud_type
        })

        complaint_description = (
            f"Victim reported suspected {fraud_type.lower()} fraud of INR {int(amount)} "
            f"from {victim_city}. Transaction details and beneficiary account were provided."
        )
        complaints.append({
            "complaint_id": complaint_id,
            "victim_id": victim_id,
            "fraud_type": fraud_type,
            "amount": amount,
            "state": victim_state,
            "city": victim_city,
            "destination_account": layers[0][0],
            "bank": account_bank[layers[0][0]],
            "complaint_timestamp": current_time.isoformat(),
            "free_text_description": complaint_description,
            "ring_id": ring_id
        })

        tx_time = current_time + timedelta(minutes=float(rng.uniform(1, 5)))
        
        # Generate edges between layers
        node_incoming = {f"VICTIM_{victim_id}": amount}
        layer_nodes = [[f"VICTIM_{victim_id}"]] + layers
        
        for i in range(len(layer_nodes) - 1):
            current_layer = layer_nodes[i]
            next_layer = layer_nodes[i+1]
            hop = i + 1
            
            for src in current_layer:
                inc_amt = node_incoming.get(src, 0)
                if inc_amt <= 0: continue
                
                # Deduct cut if not victim and not terminal
                if src.startswith("VICTIM_"):
                    distribute_amt = inc_amt
                    hold_minutes = 0.0
                else:
                    distribute_amt = inc_amt * float(rng.uniform(0.96, 0.995))
                    hold_minutes = float(rng.uniform(1, 25))
                    
                distribute_amt = round(distribute_amt / 100.0) * 100.0
                
                # Split amounts to next layer
                split_weights = rng.random(len(next_layer))
                split_weights /= split_weights.sum()
                
                for dst, weight in zip(next_layer, split_weights):
                    edge_amt = round(distribute_amt * weight)
                    if edge_amt <= 0: continue
                    
                    transactions.append({
                        "transaction_id": f"T{len(transactions)+1:09d}",
                        "complaint_id": complaint_id,
                        "from_account": src,
                        "to_account": dst,
                        "amount": edge_amt,
                        "hop_number": hop,
                        "timestamp": (tx_time + timedelta(minutes=hold_minutes)).isoformat()
                    })
                    node_incoming[dst] = node_incoming.get(dst, 0) + edge_amt
                
                # Add account features for src (excluding victim)
                if not src.startswith("VICTIM_"):
                    prior = prior_case_count[src]
                    retained_ratio = max(0.0, (inc_amt - distribute_amt) / max(inc_amt, 1.0))
                    case_account_features.append({
                        "complaint_id": complaint_id,
                        "account_id": src,
                        "hop_depth": hop - 1,
                        "chain_depth": len(layers),
                        "incoming_amount": round(inc_amt, 2),
                        "retained_amount_ratio": round(retained_ratio, 4),
                        "holding_time_minutes": round(hold_minutes, 2),
                        "outgoing_activity_in_case": 1,
                        "prior_case_count": prior,
                        "mule_reputation": account_reputation(prior),
                        "account_age_days": account_age_days[src],
                        "ring_size": ring_size,
                        "account_type": account_type[src],
                        "is_terminal_node": 0
                    })
            
            tx_time += timedelta(minutes=hold_minutes + float(rng.uniform(1, 6)))
            
        # Add account features for terminal node
        terminal_inc = node_incoming.get(terminal_account, 0)
        prior = prior_case_count[terminal_account]
        case_account_features.append({
            "complaint_id": complaint_id,
            "account_id": terminal_account,
            "hop_depth": len(layers),
            "chain_depth": len(layers),
            "incoming_amount": round(terminal_inc, 2),
            "retained_amount_ratio": 1.0,
            "holding_time_minutes": 0.0,
            "outgoing_activity_in_case": 0,
            "prior_case_count": prior,
            "mule_reputation": account_reputation(prior),
            "account_age_days": account_age_days[terminal_account],
            "ring_size": ring_size,
            "account_type": account_type[terminal_account],
            "is_terminal_node": 1
        })


        channel = choose_channel(
            rng,
            amount,
            account_type[terminal_account],
            prior_case_count[terminal_account]
        )
        location = choose_location(
            rng,
            locations,
            channel,
            account_state[terminal_account],
            account_city[terminal_account],
            float(cfg["assumptions"]["cross_region_probability"])
        )

        delay_minutes = sample_delay_minutes(rng, fraud_type, cfg)
        withdrawal_time = current_time + timedelta(minutes=delay_minutes)

        if channel == "BRANCH":
            # Force generated branch events into plausible daytime banking hours.
            if withdrawal_time.hour < 10 or withdrawal_time.hour >= 16:
                days_to_add = 1 if withdrawal_time.hour >= 16 else 0
                withdrawal_time = (withdrawal_time + timedelta(days=days_to_add)).replace(
                    hour=int(rng.integers(10, 16)),
                    minute=int(rng.integers(0, 60)),
                    second=0,
                    microsecond=0
                )
            time_bucket = "BankingHours"
        else:
            time_bucket = weighted_choice(rng, cfg["assumptions"]["atm_time_bucket_probs"])
            hour_ranges = {
                "Morning": (6, 12),
                "Afternoon": (12, 17),
                "Evening": (17, 22),
                "Night": (22, 24)
            }
            lo, hi = hour_ranges[time_bucket]
            hour = int(rng.integers(lo, hi))
            withdrawal_time = withdrawal_time.replace(hour=hour, minute=int(rng.integers(0, 60)), second=0, microsecond=0)

        split_count = 1
        if amount >= 300000:
            split_count = int(rng.choice([1, 2, 3], p=[0.45, 0.40, 0.15]))

        for split_idx in range(split_count):
            split_amount = amount / split_count
            withdrawals.append({
                "withdrawal_id": f"W{len(withdrawals)+1:08d}",
                "complaint_id": complaint_id,
                "terminal_account": terminal_account,
                "channel": channel,
                "location_id": location["location_id"],
                "bank": location["bank"],
                "city": location["city"],
                "state": location["state"],
                "latitude": float(location["latitude"]),
                "longitude": float(location["longitude"]),
                "withdrawal_amount": round(split_amount, 2),
                "withdrawal_timestamp": (withdrawal_time + timedelta(minutes=split_idx * int(rng.integers(5, 25)))).isoformat(),
                "time_bucket": time_bucket,
                "split_index": split_idx + 1,
                "split_count": split_count
            })

        # After the case is resolved, update the prior-case history for future cases only.
        for account_id in used_accounts:
            prior_case_count[account_id] += 1

    # Add history fields to account master after generation.
    for row in accounts:
        aid = row["account_id"]
        row["total_case_count"] = prior_case_count[aid]
        row["final_reputation_score"] = account_reputation(prior_case_count[aid])
        row["reused_in_multiple_cases"] = int(prior_case_count[aid] > 1)

    # Generate normal non-fraud accounts for anomaly-detection baseline.
    normal_accounts = []
    normal_count = max(len(accounts), num_cases // 2)
    for i in range(normal_count):
        normal_accounts.append({
            "account_id": f"NACC{i+1:06d}",
            "transaction_velocity_24h": round(float(rng.gamma(2.0, 1.2)), 3),
            "unique_counterparties_7d": int(rng.integers(1, 6)),
            "incoming_outgoing_ratio": round(float(rng.uniform(0.6, 1.5)), 3),
            "avg_holding_time_minutes": round(float(rng.gamma(4.0, 120.0)), 2),
            "ring_membership": 0,
            "prior_case_count": 0,
            "is_fraud_account": 0
        })

    outputs = {
        "victims.csv": victims,
        "fraudsters.csv": fraudsters,
        "accounts.csv": accounts,
        "fraud_rings.csv": rings,
        "complaints.csv": complaints,
        "transactions.csv": transactions,
        "withdrawals.csv": withdrawals,
        "case_account_features.csv": case_account_features,
        "normal_accounts.csv": normal_accounts
    }

    for filename, rows in outputs.items():
        pd.DataFrame(rows).to_csv(GENERATED_DIR / filename, index=False)

    # Copy the reference location inventory into generated outputs for convenience.
    locations.to_csv(GENERATED_DIR / "cashout_locations.csv", index=False)

    logging.info(f"Generated {num_cases} synthetic cases.")
    for filename in outputs:
        logging.info(f"Saved: {GENERATED_DIR / filename}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the CodeKavach synthetic cybercrime world.")
    parser.add_argument("--cases", type=int, default=None, help="Override number of complaints/cases.")
    args = parser.parse_args()
    generate(args.cases)


if __name__ == "__main__":
    main()
