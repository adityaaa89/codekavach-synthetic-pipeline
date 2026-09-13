# CodeKavach Synthetic Data Pipeline

This project is a highly realistic, relational synthetic cybercrime data generator built for the CodeKavach prototype. It simulates end-to-end money laundering workflows, starting from a victim filing a fraud complaint, mapping the flow of illicit funds through complex "split-and-converge" chains of mule accounts (controlled by fraud rings), and culminating in physical cash-out events (withdrawals at ATMs or bank branches).

## Features

- **Complex Network Topologies:** Generates realistic branching and converging transaction trees (smurfing) rather than simple linear paths.
- **Relational Data Ecosystem:** Outputs a full relational database structure (`victims`, `fraudsters`, `accounts`, `complaints`, `transactions`, `withdrawals`).
- **ML-Ready Datasets:** Automatically transforms the raw relational graphs into specialized datasets engineered for both supervised and unsupervised Machine Learning tasks.
- **Latent Noise Injection:** Features stochastic noise injection to ensure target variables (like Cashout Channel) aren't perfectly deterministic, forcing ML models to learn underlying patterns rather than memorizing formulas.
- **Data-Driven Calibration:** Automatically calibrates baseline probabilities from real-world statistical snapshots (e.g., NCRB stats, Paysim).

## ML-Ready Output Datasets

The pipeline produces the following datasets in `data/ml_ready/` designed for specific predictive tasks:

1. **`anomaly_detection.csv`**: Unsupervised learning (Isolation Forests, Autoencoders) to detect new mule accounts based on aggregated behavioral features (transaction velocity, holding times).
2. **`terminal_node_training.csv`**: Supervised classification (XGBoost, Random Forest) to predict which account in a money-laundering chain will be the final one before cash-out.
3. **`withdrawal_channel_training.csv`**: Binary classification to predict if the fraudster will use an ATM or visit a physical BRANCH.
4. **`withdrawal_location_training.csv`**: Multi-class classification or geospatial ML to predict the physical city/ATM the fraudster will use based on the digital transaction path.
5. **`withdrawal_time_training.csv`**: Regression models to estimate the "delay" (how many minutes law enforcement has to freeze the account before the cash is withdrawn).

## Quick Start

### 1. Set up the Environment
It is highly recommended to use a virtual environment.

**Windows PowerShell:**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**Linux/macOS:**
```bash
python -m venv .venv
source .venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Generate Datasets
Run the main pipeline to generate the relational data and ML-ready datasets. You can specify the number of cases to simulate (e.g., 20,000 cases):

```bash
python main.py --cases 20000
```

*The pipeline will automatically calibrate itself, generate the transaction topologies, compile the ML datasets, and run statistical validation reports against the outputs.*

---
*Note: Do not present the sample reference CSVs included in `data/reference` as official NCRB/RBI/OSM values. Replace them with cleaned, documented public-source data before a final production evaluation.*
