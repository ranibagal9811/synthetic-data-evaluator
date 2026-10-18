"""Shared helpers for all pages: study data, labels, metric definitions."""
import os
import sys

import numpy as np
import pandas as pd
import streamlit as st

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

TEST_MODE = os.environ.get("DASHBOARD_TEST_MODE") == "1"
ICON = {"PASS": "🟢 PASS", "REVIEW": "🟡 REVIEW", "FAIL": "🔴 FAIL"}

STUDY_LABELS = {
    "sdv_overall": "SDV quality score (higher = better)",
    "ks_pass_rate": "KS pass rate (higher = better)",
    "chi2_pass": "Chi-square pass rate on target (higher = better)",
    "tstr_auc": "TSTR AUC (higher = better)",
    "tstr_f1": "TSTR F1 (higher = better)",
    "tstr_auc_gap": "TRTR - TSTR AUC gap (lower = better)",
    "dcr_ratio": "DCR ratio (>= 1 required)",
    "dcr_pct_near": "Near-copy share (lower = better)",
    "ad_gap": "Attribute-disclosure gap (lower = better)",
    "mia_mean": "MIA AUC, mean of 3 attackers (0.5 = private)",
    "mia_auc_lr": "MIA AUC - Logistic Regression",
    "mia_auc_rf": "MIA AUC - Random Forest",
    "mia_auc_mlp": "MIA AUC - MLP",
    "seconds": "Run time (seconds)",
}

# The 7 metrics of the evaluation framework (used on the Methodology and Evaluate pages)
METRICS = [
    dict(n=1, name="KS test (column distributions)", family="Fidelity",
         measures="Whether each numeric column has the same distribution in the real and the synthetic data.",
         how="Two-sample Kolmogorov-Smirnov test on every numeric feature. We report the average KS statistic (0 = identical, 1 = completely different) and the share of columns that pass (p > 0.05).",
         good="Low KS statistic, high pass rate."),
    dict(n=2, name="Chi-square test (class balance)", family="Fidelity",
         measures="Whether the target classes (e.g. diabetic / not diabetic) appear in the same proportions.",
         how="Chi-square test comparing target class counts in real vs synthetic data. PASS if p > 0.05.",
         good="Pass (class ratio preserved)."),
    dict(n=3, name="SDV quality score", family="Fidelity",
         measures="Overall statistical similarity of the synthetic table to the real one.",
         how="Score from 0 to 1 from the SDV library: the average of 'Column Shapes' (single-column distributions) and 'Column Pair Trends' (relationships between pairs of columns).",
         good="Closer to 1."),
    dict(n=4, name="TSTR vs TRTR (utility)", family="Utility",
         measures="Whether a model trained on synthetic data works on real patients.",
         how="Train a Random Forest on synthetic data (TSTR) and, separately, on real data (TRTR); test both on the same real holdout. Report AUC, F1 and accuracy and the TRTR - TSTR gap.",
         good="TSTR close to TRTR (small gap)."),
    dict(n=5, name="Distance to Closest Record (DCR)", family="Privacy",
         measures="Whether synthetic rows are copies or near-copies of real training rows.",
         how="Distance from each synthetic row to its nearest training row, divided by the same distance for unseen real (holdout) rows. Also: share of exact copies and share of 'near-copies' (closer than the 5th percentile of unseen-real distances).",
         good="DCR ratio >= 1, no exact copies, few near-copies."),
    dict(n=6, name="Attribute disclosure", family="Privacy",
         measures="Whether an attacker who knows part of a person's record can learn the rest from the synthetic data.",
         how="The attacker knows about 80% of the columns, finds the nearest synthetic row on those, and guesses the hidden ~20% from it. We compare the guessing success on training records (members) with unseen records (non-members); the gap is the disclosure risk.",
         good="Gap close to 0."),
    dict(n=7, name="Membership inference attack (MIA)", family="Privacy",
         measures="Whether an attacker can tell if a particular person's record was used to train the generator.",
         how="Three attackers of increasing strength (Logistic Regression, Random Forest, MLP) use each record's distances to the nearest synthetic records to separate training rows from unseen rows. Reported as AUC (5-fold cross-validated).",
         good="AUC close to 0.5 (attacker cannot do better than guessing)."),
]


@st.cache_data
def load_study():
    d = pd.read_csv(os.path.join(HERE, "master_results.csv"))
    d = d[d["status"] == "ok"].copy()
    d["mia_mean"] = d[["mia_auc_lr", "mia_auc_rf", "mia_auc_mlp"]].mean(axis=1)
    d["epochs_n"] = pd.to_numeric(d["epochs"], errors="coerce")
    d["pos_label"] = d["pos_rate"].astype(str).map(
        lambda v: "natural (35%)" if v == "natural" else f"{float(v) * 100:.0f}% diabetic")
    return d


def study_subset(d, study):
    if study == "Dataset size":
        s = d[d.studies.str.contains("size") & ((d.model == "copula") | (d.epochs_n == 300))].copy()
        s["level"] = s.n_total.astype(int)
        s["level_label"] = s.level.astype(str) + " rows"
        return s
    if study == "Class imbalance":
        s = d[d.studies.str.contains("imbalance")].copy()
        s["level"] = s.pos_rate.astype(float)
        s["level_label"] = s.pos_label
        return s.sort_values("level", ascending=False)
    s = d[(d.n_total == 768) & (d.pos_rate.astype(str) == "natural")].copy()
    s["level"] = s.epochs_n
    s["level_label"] = s.epochs_n.map(lambda v: "" if pd.isna(v) else f"{int(v)} epochs")
    return s


def sig_stars(p):
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"


def facts(d):
    """Headline numbers from the study, used in the text of several pages."""
    nat = d.pos_rate.astype(str) == "natural"
    cop = d[(d.model == "copula") & (d.n_total == 768)]
    ct = lambda e: d[(d.model == "ctgan") & (d.n_total == 768) & nat & (d.epochs_n == e)]
    f = dict(n_runs=len(d), n_seeds=int(d.seed.nunique()),
             n_cfg=int(d.assign(c=d.model + d.n_total.astype(str) + d.pos_rate.astype(str) + d.epochs.astype(str)).c.nunique()))
    g = lambda frame, col: float(frame[col].mean()) if len(frame) else float("nan")
    for tag, fr in (("cop", cop), ("k300", ct(300)), ("k500", ct(500)), ("k1000", ct(1000)), ("k2000", ct(2000)), ("k3000", ct(3000))):
        for col in ("sdv_overall", "tstr_auc", "tstr_f1", "dcr_ratio", "chi2_pass", "ks_pass_rate", "mia_mean", "seconds",
                    "dcr_pct_near", "pos_rate_synth", "pos_rate_real"):
            f[f"{tag}_{col}"] = g(fr, col)
    f["rho_sdv_mia"] = float(d.sdv_overall.corr(d.mia_mean, method="spearman"))
    f["rho_sdv_dcr"] = float(d.sdv_overall.corr(d.dcr_ratio, method="spearman"))
    f["n_dcr_low"] = int((d.dcr_ratio < 1).sum())
    f["mia_all"] = float(d.mia_mean.mean())
    return f


def gate_by_prefix(gates, prefix):
    return [g for g in gates if g["gate"].startswith(prefix)][0]
