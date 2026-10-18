"""Derives the pass/review/fail thresholds from the controlled-variable study.

Idea ("empirical-null calibration"): none of the study's runs showed detectable membership leakage
(MIA AUC ~ 0.5), so the spread of MIA / attribute-disclosure values across runs describes how a
NON-leaking synthetic dataset behaves.  A new dataset whose value lies beyond the 95th (REVIEW) or
99th (FAIL) percentile of that spread is unusually leaky-looking.  MIA noise shrinks with dataset
size, so percentiles are computed per dataset size and interpolated.

Re-run after updating data/master_results.csv:   python calibrate_thresholds.py
"""
import json
import pandas as pd

d = pd.read_csv("master_results.csv")
d = d[d.status == "ok"].copy()
d["mia_mean"] = d[["mia_auc_lr", "mia_auc_rf", "mia_auc_mlp"]].mean(axis=1)

by_n = {}
for n, g in d.groupby("n_total"):
    by_n[str(int(n))] = {
        "runs": int(len(g)),
        "mia_mean_p95": round(float(g.mia_mean.quantile(.95)), 4),
        "mia_mean_p99": round(float(g.mia_mean.quantile(.99)), 4),
        "ad_gap_p95": round(float(g.ad_gap.quantile(.95)), 4),
        "ad_gap_p99": round(float(g.ad_gap.quantile(.99)), 4),
    }
out = {
    "source": "controlled-variable study, Pima Indians Diabetes, Gaussian Copula + CTGAN",
    "n_runs": int(len(d)),
    "by_total_rows": by_n,
    "dcr_ratio": {"fail_below": 1.0, "note": "synthetic records closer to training data than unseen real records are"},
    "near_copy_share": {"review_above": 0.05, "fail_above": 0.10,
                        "note": "share of synthetic rows closer than the 5th percentile of holdout-to-train distances; ~5% expected by chance"},
    "exact_copies": {"fail_above": 0},
    "reference": {k: {"p05": round(float(d[k].quantile(.05)), 4), "p50": round(float(d[k].quantile(.5)), 4),
                      "p95": round(float(d[k].quantile(.95)), 4)}
                  for k in ["sdv_overall", "tstr_auc", "dcr_ratio", "mia_mean", "ad_gap"]},
}
json.dump(out, open("thresholds.json", "w"), indent=2)
print(json.dumps(out, indent=2))
