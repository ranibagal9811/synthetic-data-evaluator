import dd_common as C
import streamlit as st

study = C.load_study()
F = C.facts(study)

st.title("🧭 Model Advisor")
st.caption("Which synthesizer should I use? A recommendation based on the 540-run study")

st.markdown("Answer four questions about your dataset and situation. The advisor applies the selection rule derived from the study and shows the evidence behind it.")
c1, c2 = st.columns(2)
n_rows = c1.number_input("1. How many patient records do you have?", min_value=50, max_value=1_000_000, value=768, step=50)
minority_pct = c2.number_input("2. What % of records belong to the smaller class (e.g. diabetic)?", min_value=1, max_value=50, value=35, step=1)
c3, c4 = st.columns(2)
priority = c3.radio("3. What matters most?", ["Statistical fidelity and preserved class ratio", "Best downstream classifier accuracy (TSTR)"])
hw = c4.radio("4. What computer will you use?", ["CPU only", "GPU - can train CTGAN for 1000+ epochs"])

want_ctgan = priority.startswith("Best") and hw.startswith("GPU") and n_rows >= 500
st.divider()
if want_ctgan:
    st.success("### Recommendation: CTGAN, trained for about 1000 epochs")
    st.markdown(f"""
At 768 rows, CTGAN with 1000 epochs reached a classifier-utility (TSTR AUC) of **{F['k1000_tstr_auc']:.3f}** against **{F['cop_tstr_auc']:.3f}** for Gaussian Copula.
The price: a lower SDV quality score (**{F['k1000_sdv_overall']:.3f}** vs **{F['cop_sdv_overall']:.3f}**), a distorted class ratio (chi-square passed in
**{100 * F['k1000_chi2_pass']:.0f}%** of runs vs **{100 * F['cop_chi2_pass']:.0f}%** for Copula) and about **{F['k1000_seconds'] / max(F['cop_seconds'], 1):.0f}x** the run time.
""")
    st.caption("Beyond ~1000 epochs the study found no further gain in classifier utility (it plateaus, then falls at 3000 epochs) while synthetic records move closer to the real training data "
               f"(DCR ratio {F['k1000_dcr_ratio']:.2f} → {F['k2000_dcr_ratio']:.2f} → {F['k3000_dcr_ratio']:.2f} at 1000 / 2000 / 3000 epochs).")
else:
    st.success("### Recommendation: Gaussian Copula")
    why = []
    if not priority.startswith("Best"):
        why.append("it gave higher fidelity (SDV score, KS tests) than CTGAN at every dataset size and class ratio tested, and it preserves the class ratio")
    else:
        why.append("CTGAN only matches Copula's classifier utility after about 500 epochs and beats it only at 1000+ epochs, which needs a GPU")
    if n_rows < 500:
        why.append("with fewer than ~500 records CTGAN has little data to learn from")
    if hw.startswith("CPU"):
        why.append("it runs in seconds on any computer")
    st.markdown("**Why:** " + "; ".join(why) + ".")
    st.markdown(f"At 768 rows Gaussian Copula scored an SDV quality of **{F['cop_sdv_overall']:.3f}** and a TSTR AUC of **{F['cop_tstr_auc']:.3f}**, "
                f"kept the class ratio (chi-square passed in **{100 * F['cop_chi2_pass']:.0f}%** of runs) and took about **{F['cop_seconds']:.0f} seconds**.")
if minority_pct < 15:
    st.warning("**Small minority class.** At ~10% minority share, neither generator gave reliable classifier utility in the study (TSTR AUC about 0.6, no clear winner). Treat downstream-model results with caution.")
if n_rows > 800 or n_rows < 200:
    st.info("The study covered 200-768 records of one dataset (Pima Indians Diabetes). Extrapolate with care.")
st.info("**Next step:** go to **Generate & Evaluate Data** to generate synthetic data with your choice and check its fidelity and privacy.")

st.subheader("Evidence: CTGAN training length vs Gaussian Copula (768 rows, natural class ratio, average of 30 seeds)")
ep = study[(study.model == "ctgan") & (study.n_total == 768) & (study.pos_rate.astype(str) == "natural")]
cop = study[(study.model == "copula") & (study.n_total == 768)]
if len(ep):
    t = ep.groupby("epochs_n").agg(SDV_quality=("sdv_overall", "mean"), TSTR_AUC=("tstr_auc", "mean"), DCR_ratio=("dcr_ratio", "mean"),
                                   Attack_AUC=("mia_mean", "mean"), Seconds=("seconds", "mean")).round(3)
    t.index = [f"CTGAN, {int(i)} epochs" for i in t.index]
    t.loc["Gaussian Copula"] = cop[["sdv_overall", "tstr_auc", "dcr_ratio", "mia_mean", "seconds"]].mean().round(3).values
    st.dataframe(t, width="stretch")
    st.caption("SDV quality and TSTR AUC: higher is better. DCR ratio: should stay at or above 1. Attack AUC: 0.5 = attacker cannot identify training patients. Seconds: run time on a GPU.")
