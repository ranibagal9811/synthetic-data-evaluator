import dd_common as C
import dd_thresholds as TH
import pandas as pd
import streamlit as st

st.title("⚖️ DPDP Compliance Guide")
st.caption("How the privacy evidence in this app relates to India's Digital Personal Data Protection Act, 2023 - and where it stops")

st.error("**This is not legal advice and not a certificate of compliance.** The DPDP Act 2023 does not mention synthetic data, sets no numeric privacy thresholds, and "
         "(unlike the GDPR) has no separate 'sensitive personal data' category. No regulator has endorsed the thresholds used here. Ask a qualified legal adviser whether a synthetic "
         "dataset falls outside the Act for your purpose.")

st.header("The legal question behind the metrics")
st.markdown("""
The Act applies to **personal data** - data about an individual who is identifiable from it. If a synthetic dataset does **not** reproduce or leak real individuals,
there is an argument that it is not personal data. The metrics in this app give **technical evidence** for that question, not the answer to it.
""")

st.header("Which metric supports which provision?")
st.dataframe(pd.DataFrame([
    ["Is the output 'personal data'? (s.2(t), s.3)", "DCR ratio, near-copies, exact copies, attribute disclosure", "Individual records are not reproduced or re-identifiable"],
    ["Reasonable security safeguards (s.8(5))", "Membership-inference attack AUC", "The generator has not memorised training individuals"],
    ["Erasure when the purpose is served; correction and erasure (s.8(7), s.12)", "Membership-inference attack AUC", "A generator that memorises members could retain data that should be erased"],
    ["Accuracy and completeness (s.8(3))", "SDV score, KS / chi-square, TSTR gap", "Relevant if synthetic data informs decisions about individuals (reported, not gated)"],
    ["Accountability and documentation (s.10 for Significant Data Fiduciaries; DPDP Rules 2025)", "The downloadable PDF report", "Documented evidence of due diligence"],
], columns=["Provision (topic)", "Metric(s)", "Why relevant"]), hide_index=True, width="stretch")

st.header("How PASS / REVIEW / FAIL is decided")
st.markdown("""
Five **privacy gates** are checked; the overall result is the **worst** of them (one FAIL → FAIL; otherwise one REVIEW → REVIEW; otherwise PASS).

| Gate | REVIEW if | FAIL if |
|---|---|---|
| Exact copies of training records | - | any |
| DCR ratio (synthetic vs unseen-real distance to training data) | - | below 1.0 |
| Near-copy share | above 5% | above 10% |
| Membership-inference AUC (mean of 3 attackers) | above the study's 95th percentile for your dataset size | above the 99th percentile |
| Attribute-disclosure gap | above the 95th percentile | above the 99th percentile |
""")
st.subheader("Threshold calculator")
n = st.number_input("Dataset size (total rows)", min_value=50, max_value=100000, value=768, step=50)
c1, c2, c3, c4 = st.columns(4)
c1.metric("MIA: review above", f"{TH._interp(n, 'mia_mean_p95'):.3f}")
c2.metric("MIA: fail above", f"{TH._interp(n, 'mia_mean_p99'):.3f}")
c3.metric("Attribute gap: review above", f"{TH._interp(n, 'ad_gap_p95'):.3f}")
c4.metric("Attribute gap: fail above", f"{TH._interp(n, 'ad_gap_p99'):.3f}")
if TH.outside_range(n):
    st.warning("This size is outside the 200-768-row range used to calibrate the thresholds; they are extrapolated.")
st.caption(f"Calibrated from {TH.T['n_runs']} study runs with no detectable leakage. Smaller datasets get looser thresholds because attack scores are noisier.")

st.header("How to interpret the result")
st.markdown("""
- **PASS** - no sign of leakage that this tool can detect. It is *not* a legal finding that the data is outside the Act.
- **REVIEW** - one gate is above the 95th percentile of non-leaking runs. About **1 in 10 clean datasets** gets REVIEW by chance because attack scores are noisy: re-run with other seeds and look at the flagged metric.
- **FAIL** - a gate is beyond the 99th percentile or a hard criterion (copies, DCR below 1). Do not release the data without investigation.
""")

st.header("What a hospital team should still do")
st.markdown("""
1. **Real data stage:** training the generator is itself processing of personal data. It needs a lawful basis (consent under s.6 or a listed legitimate use under s.7), notice, and security safeguards. This tool does not cover that.
2. **Run the evaluation several times** (different seeds) and keep the PDF reports as evidence.
3. **Consider formal guarantees:** only differential-privacy generators carry a mathematical privacy guarantee; the generators here do not.
4. **Get legal review** before treating synthetic data as outside the Act.
""")

st.header("Limits of the evidence")
st.markdown("""
- Thresholds come from **one public dataset** (Pima, 200-768 rows, 9 columns); other data may behave differently.
- Empirical attacks can **fail to find** leakage but cannot **prove** privacy.
- Single-run attack scores are noisy (about ±0.05 around chance for a few hundred records).
""")
