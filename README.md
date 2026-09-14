# TrustV2X WINCOM 2026

This repository contains the reproducibility package for the paper:

**TrustV2X: Explainable Hybrid Learning for Trust-Aware C-ITS/V2X Safety-Message Misbehaviour Detection**

The project investigates trust-aware misbehaviour detection for Cooperative Intelligent Transport Systems (C-ITS) and V2X safety messages. TrustV2X combines machine-learning detection with plausibility-aware and rule-based trust scoring to support explainable message-level decision making at the RSU/edge layer.

---

## Repository Contents

```text
TrustV2X_WINCOM_2026/
│
├── scripts/                 # Dataset generation, attack injection, ML/DL training, validation, trust scoring and explainability
├── sumo/                    # SUMO simulation networks, routes and configuration files
├── paper/                   # Paper draft, final PDF and paper figures
├── results/
│   ├── final_tables/        # Final tables used for reporting
│   ├── final_figures/       # Final selected figures used for reporting
│   ├── final_summary/       # Final experimental summaries
│   ├── tables/              # Controlled SUMO experiment tables
│   ├── veremi/              # VeReMi Extension ML results and preprocessing diagnostics
│   └── veremi_dl/           # VeReMi Extension DL result tables
│
├── requirements.txt         # Python package list
├── environment_info.txt     # Environment and library information
├── .gitignore
└── README.md
