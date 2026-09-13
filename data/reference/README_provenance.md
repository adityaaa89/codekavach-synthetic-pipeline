# Reference data provenance

## ncrb_stats.csv — REAL, sourced
Built from actual NCRB "Crime in India" figures as tabled in a Dec 2025 Lok Sabha
reply (MHA, Unstarred Q.452): https://www.mha.gov.in/MHA1/Par2017/pdfs/par2025-pdfs/LS02122025/452.pdf

Two real NCRB tables were combined:
- State-wise registered "Fraud" cyber-crime cases, 2023 (Annexure-III) — used as
  the real per-state scale.
- National crime-head-wise cyber crime breakdown, 2023 (Annexure-I) — used to
  derive real national proportions across Fraud / Cheating / Forgery / Data
  theft / Fake Profile / Cyber Blackmailing (the six financially-relevant
  IPC heads NCRB actually publishes).

**Important mismatch to know about:** NCRB's published categories are legal
crime-heads (Fraud, Cheating, Forgery...), NOT modus-operandi labels like
"UPI fraud" / "Investment fraud" / "Loan fraud" — that taxonomy comes from
I4C/NCRP (cybercrime.gov.in), and NCRB does not publish a state × modus-operandi
cross-tab. So the per-state split by crime-head here is a documented
**allocation assumption** (state's real Fraud-head total × real national
category proportions), not a directly-published joint distribution. If your
SIH writeup uses UPI/Investment/Loan labels, either relabel these heads
honestly as legal crime-heads, or source the I4C annual report/NCRP dashboard
separately for modus-operandi splits.

## cashout_locations.csv — REAL coordinates, but city-level only, banks are placeholders
Lat/longs are real city centers; `bank` values (Bank_A/Bank_B/Bank_C) are
generic placeholders, same approach as the starter demo file — I have no
verified real bank-branch-to-coordinate mapping. Do not present these as real
bank locations in your SIH writeup. They are NOT real individual ATM/branch
addresses either; I have no network access to query OpenStreetMap's Overpass
API for actual bank locations. To get real point-level ATM/branch data, run
something like this yourself (e.g. via https://overpass-turbo.eu/):

```
[out:json][timeout:60];
area["name"="Maharashtra"]->.a;
(node["amenity"="atm"](area.a); node["amenity"="bank"](area.a););
out body;
```

## paysim_reference.csv — ILLUSTRATIVE, not real PaySim rows
Reshaped to a single `amount` column to match this pipeline's calibrator
(`lognorm.fit` on `df["amount"]`). I cannot download the actual Kaggle PaySim
file (network access is disabled here, and it sits behind Kaggle auth). The
values in this file are representative placeholders in the right shape/scale,
not real dataset rows. Get the real thing yourself:

```bash
pip install kaggle
kaggle datasets download -d ealaxi/paysim1
```
then extract the `amount`, `type`, `step` columns as the original plan describes.

## transaction_behavior.csv — ILLUSTRATIVE domain assumptions
No public dataset publishes "average hops" / "average delay" by fraud category
at this granularity — this is inherently a modeling assumption. The values
here reflect commonly-cited fraud typology (UPI-style fraud cashes out fast;
investment/loan-app scams involve slower layering) but are not citations to a
specific source. For SIH defensibility, cite this as an expert/typology-informed
assumption, not measured data — or look for RBI/I4C annual report language on
modus-operandi timelines to justify specific numbers.
