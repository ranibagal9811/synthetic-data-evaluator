import json
import os

import dd_common as C
import dd_thresholds as TH
import pandas as pd
import streamlit as st

st.title("🔬 Study Methodology")
st.caption("How the experiment was designed, so the results can be trusted and repeated")

st.header("1. Data")
st.markdown("""
**Pima Indians Diabetes dataset** (Smith et al., 1988): **768** patients, **8 numeric features** (pregnancies, glucose, blood pressure,
skin thickness, insulin, BMI, diabetes pedigree function, age) and a **binary target** (diabetes yes/no; about **35%** positive).
The data is used as distributed: no imputation or cleaning (some columns contain zeros that stand for missing values).
""")

st.header("2. The two generators compared")
st.dataframe(pd.DataFrame({
    "Generator": ["Gaussian Copula", "CTGAN"],
    "Type": ["Statistical model", "Deep-learning GAN"],
    "How it works": ["Learns each column's distribution and the correlations between columns, then samples new rows.",
                     "A generator network and a discriminator network compete; the generator learns to produce realistic rows."],
    "Settings varied": ["None (fits in seconds)", "Training length (epochs): 100, 200, 300, 500, 1000, 2000, 3000"],
}), hide_index=True, width="stretch")
st.caption("Both come from the SDV library (default settings apart from epochs).")

st.header("3. The seven evaluation metrics")
st.dataframe(pd.DataFrame([{"#": m["n"], "Metric": m["name"], "Family": m["family"], "What it measures": m["measures"],
                            "What 'good' looks like": m["good"]} for m in C.METRICS]), hide_index=True, width="stretch")
with st.expander("How each metric is computed"):
    for m in C.METRICS:
        st.markdown(f"**{m['n']}. {m['name']}** - {m['how']}")

st.header("4. Controlled-variable design")
st.markdown("Only **one factor changes at a time**, everything else is held fixed, so any difference can be attributed to that factor.")
st.dataframe(pd.DataFrame({
    "Study": ["Dataset size", "Class imbalance", "CTGAN training length", "Attacker strength"],
    "Factor changed": ["Total rows", "Share of diabetic (minority) patients", "Epochs", "Strength of the privacy attacker"],
    "Levels": ["200 / 400 / 768", "50% / 35% / 10%", "100 / 200 / 300 / 500 / 1000 / 2000 / 3000", "Logistic Regression / Random Forest / MLP"],
    "Held fixed": ["Natural class ratio; CTGAN at 300 epochs", "400 rows; CTGAN at 300 epochs", "768 rows, natural ratio", "Measured inside every run"],
}), hide_index=True, width="stretch")

st.header("5. Protocol for every run")
st.markdown("""
1. Sample the required number of rows (with the required class ratio) and split **80% train / 20% holdout** (stratified).
2. Train the generator on the **training part only**; generate as many synthetic rows as there are training rows.
3. Compute all 7 metrics. Utility and privacy attacks use the **holdout** - real records the generator never saw - as the baseline for "not memorised".
4. Repeat with **30 different random seeds** for every setting (the same seed gives the same data split for both generators, so comparisons are paired).
""")

st.header("6. Statistics")
st.markdown("""
Results are reported as **mean ± standard deviation over 30 seeds**. Differences between generators use **paired t-tests** on matched seeds
(* p<0.05, ** p<0.01, *** p<0.001, ns = not significant). No multiple-comparison correction was applied, so borderline p-values (0.01-0.05) are exploratory.
""")

st.header("7. How privacy thresholds were set")
st.markdown("""
The Act gives no numeric limits, so thresholds were **calibrated from the study itself**. No run showed detectable membership leakage
(attack AUC about 0.5), so the spread of attack scores across runs shows how a *non-leaking* synthetic dataset behaves. A new dataset above the
**95th percentile** of that spread is flagged **REVIEW**, above the **99th** is **FAIL**. Because attack noise shrinks with more data, percentiles are set per dataset size.
""")
rows = [{"Dataset size (rows)": n, "Runs used": v["runs"], "MIA review above": v["mia_mean_p95"], "MIA fail above": v["mia_mean_p99"],
         "Attribute-gap review above": v["ad_gap_p95"], "Attribute-gap fail above": v["ad_gap_p99"]}
        for n, v in TH.T["by_total_rows"].items()]
st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
st.caption("Hard criteria: any exact copy of a training record, or a DCR ratio below 1.0, is a FAIL; a near-copy share above 5% is REVIEW and above 10% is FAIL.")

st.header("8. Limitations")
st.markdown("""
- **One dataset** (Pima, 200-768 rows, 9 columns). Results and thresholds may not transfer to very different data.
- **Membership-inference scores are noisy** with a few hundred records (about ±0.05 around chance per run); finding no leakage does not prove privacy.
- The 10%-diabetic holdouts contain only about 8 positive cases, so utility scores there are noisy.
- No differential-privacy generators were tested - they are the only approach with a formal privacy guarantee.
- Default generator settings only; CTGAN's batch size and other hyperparameters were not tuned.
""")
