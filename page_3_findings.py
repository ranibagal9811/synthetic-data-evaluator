import dd_common as C
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy import stats

study = C.load_study()
F = C.facts(study)

st.title("📊 Study Findings")
st.caption(f"{F['n_runs']} experimental runs · {F['n_cfg']} settings · {F['n_seeds']} random seeds each · Pima Indians Diabetes")

# ---------------------------------------------------------------- headline
c1, c2, c3, c4 = st.columns(4)
c1.metric("Copula fidelity (SDV score, 768 rows)", f"{F['cop_sdv_overall']:.3f}")
c2.metric("CTGAN utility at 1000 epochs (TSTR AUC)", f"{F['k1000_tstr_auc']:.3f}", delta=f"Copula {F['cop_tstr_auc']:.3f}", delta_color="off")
c3.metric("Average attack AUC (0.5 = chance)", f"{F['mia_all']:.3f}")
c4.metric("Runs with DCR ratio below 1", f"{F['n_dcr_low']} of {F['n_runs']}")

# ---------------------------------------------------------------- research questions
st.header("Answers to the research questions")
with st.container(border=True):
    st.markdown("#### RQ1 - Do the generators really differ?  **Yes, mainly in fidelity and utility, not in measured privacy.**")
    st.markdown(f"""
- With the study's default of **300 epochs**, Gaussian Copula had the higher SDV quality score at every dataset size and class ratio (p < 0.001), and higher classifier utility in nearly all of them
  (the exception: at 10% minority share neither was reliable). At 768 rows its SDV score was **{F['cop_sdv_overall']:.3f}** vs **{F['k300_sdv_overall']:.3f}** and its TSTR AUC **{F['cop_tstr_auc']:.3f}** vs **{F['k300_tstr_auc']:.3f}** (p < 0.001).
- CTGAN **catches up with much longer training**: at **1000 epochs** its TSTR AUC (**{F['k1000_tstr_auc']:.3f}**) is above Copula's, and at **3000 epochs** its SDV score ({F['k3000_sdv_overall']:.3f}) is higher too (at 2000 epochs, {F['k2000_sdv_overall']:.3f}, the difference does not survive correction for multiple testing).
- CTGAN **distorts an imbalanced class ratio**: with the natural data, about **{100 * F['k1000_pos_rate_synth']:.0f}%** of its synthetic patients are diabetic vs **{100 * F['k1000_pos_rate_real']:.0f}%** in the real data, at every epoch count
  (it passes only when the data is already 50/50). Copula keeps the ratio (chi-square passed in {100 * F['cop_chi2_pass']:.0f}% of runs vs {100 * F['k1000_chi2_pass']:.0f}% for CTGAN).
- Speed: Copula takes about **{F['cop_seconds']:.0f} s**; CTGAN about **{F['k1000_seconds']:.0f} s** at 1000 epochs and **{F['k3000_seconds']:.0f} s** at 3000 (on a GPU).
""")
with st.container(border=True):
    st.markdown("#### RQ2 - Is there a fidelity-privacy trade-off?  **A mild one, visible in distances to training records but not in the attacks.**")
    st.markdown(f"""
- **Membership-inference attacks found nothing:** the average attack AUC is **{F['mia_all']:.3f}** (0.5 = guessing) for both generators, close to chance at every size, class ratio and training length,
  and it is essentially unrelated to fidelity (Spearman ρ = {F['rho_sdv_mia']:.2f}). The attribute-disclosure gap is also about zero.
- **But synthetic rows get closer to the real training rows as fidelity rises** (ρ = {F['rho_sdv_dcr']:.2f} between SDV score and DCR ratio).
  For CTGAN the DCR ratio falls from **{F['k1000_dcr_ratio']:.2f}** (1000 epochs) to **{F['k2000_dcr_ratio']:.2f}** (2000) and **{F['k3000_dcr_ratio']:.2f}** (3000), and near-copies rise from
  {100 * F['k1000_dcr_pct_near']:.1f}% to {100 * F['k3000_dcr_pct_near']:.1f}% of rows. Only **{F['n_dcr_low']}** of {F['n_runs']} runs fell below DCR = 1.
- **Diminishing returns beyond about 1000-2000 epochs:** CTGAN's fidelity keeps rising, but its classifier utility stops improving (and falls at 3000 epochs) while records move closer to the training data. This is a plateau, not a sharp threshold: no fidelity level was found at which attacks begin to succeed.
- **Proposed compliance criterion:** DCR ratio ≥ 1, no exact copies, and attack scores within the range seen for non-leaking data. No fixed fidelity level was found at which attacks begin to succeed.
""")
with st.container(border=True):
    st.markdown("#### RQ3 - Which generator should I use?  **Default to Gaussian Copula; use CTGAN only for classifier utility, with ~1000 epochs.**")
    st.markdown(f"""
1. **Default: Gaussian Copula** - higher fidelity than CTGAN at 300 epochs or fewer, preserves class balance, runs in seconds, works with small data.
2. **CTGAN only if** your goal is downstream classifier accuracy **and** you can train for about **1000 epochs** (GPU): TSTR AUC **{F['k1000_tstr_auc']:.3f}** vs **{F['cop_tstr_auc']:.3f}** for Copula, with a DCR ratio of **{F['k1000_dcr_ratio']:.2f}** (above the DCR = 1 screening criterion).
3. **Avoid CTGAN at 300 epochs or fewer** - worse than Copula on fidelity and, in nearly all settings, on classifier utility.
4. **At ~10% minority share** neither generator gave reliable classifier utility.
""")
st.caption("The headline numbers above are computed live from the results file. The wording of the conclusions describes the 540-run study.")

# ---------------------------------------------------------------- explorer
st.header("Explore the experiments")
st.info("**How to read the charts:** each point is the **average over 30 random seeds** and the vertical bars show **± one standard deviation** "
        "(how much the result varies between seeds). Blue = Gaussian Copula, red = CTGAN. Where the bars of the two lines overlap, differences are small relative to the noise - "
        "check the paired-test table underneath for the verdict.")
s1, s2 = st.columns(2)
which = s1.selectbox("Which experiment?", ["Dataset size", "Class imbalance", "CTGAN epochs"],
                     help="Dataset size: 200/400/768 rows. Class imbalance: 50/35/10% diabetic. CTGAN epochs: training length.")
metric = s2.selectbox("Which metric?", list(C.STUDY_LABELS), format_func=lambda m: C.STUDY_LABELS[m])
sub = C.study_subset(study, which)
levels = sub.sort_values("level", ascending=(which != "Class imbalance"))["level_label"].drop_duplicates().tolist()
fig = go.Figure()
if which == "CTGAN epochs":
    g = sub[sub.model == "ctgan"].groupby(["level", "level_label"])[metric].agg(["mean", "std"]).reset_index()
    cop = sub[sub.model == "copula"][metric]
    fig.add_scatter(x=g["level"], y=g["mean"], error_y=dict(type="data", array=g["std"]), mode="lines+markers", name="CTGAN", line_color="#d62728")
    if len(cop):
        fig.add_hline(y=cop.mean(), line_color="#1f77b4", annotation_text="Gaussian Copula (average)")
        fig.add_hrect(y0=cop.mean() - cop.std(), y1=cop.mean() + cop.std(), fillcolor="#1f77b4", opacity=.1, line_width=0)
    fig.update_xaxes(type="log", tickvals=sorted(g["level"].unique()), title="CTGAN epochs (768 rows) - log scale")
else:
    g = sub.groupby(["model", "level_label"])[metric].agg(["mean", "std"]).reset_index()
    for m, color in (("copula", "#1f77b4"), ("ctgan", "#d62728")):
        gm = g[g.model == m].set_index("level_label").reindex(levels).reset_index()
        fig.add_scatter(x=gm["level_label"], y=gm["mean"], error_y=dict(type="data", array=gm["std"]), mode="lines+markers",
                        name="Gaussian Copula" if m == "copula" else "CTGAN (300 epochs)", line_color=color)
if metric.startswith("mia"):
    fig.add_hline(y=0.5, line_dash="dash", line_color="grey", annotation_text="0.5 = attacker guessing")
if metric == "dcr_ratio":
    fig.add_hline(y=1, line_dash="dash", line_color="grey", annotation_text="DCR = 1 limit")
fig.update_layout(height=420, yaxis_title=C.STUDY_LABELS[metric], title="Average ± SD over 30 seeds")
st.plotly_chart(fig, width="stretch")

st.markdown("**Is the difference real? Paired t-tests (same seeds)**")
rows = []
if which == "CTGAN epochs":
    cop = sub[sub.model == "copula"].set_index("seed")
    for lv, gg in sub[sub.model == "ctgan"].groupby("level"):
        gg = gg.set_index("seed")
        cm = cop.index.intersection(gg.index)
        if len(cm) > 2:
            p = stats.ttest_rel(cop.loc[cm, metric], gg.loc[cm, metric])[1]
            rows.append({"CTGAN setting": f"{int(lv)} epochs", "Copula": cop.loc[cm, metric].mean(), "CTGAN": gg.loc[cm, metric].mean(), "p-value": p, "Significance": C.sig_stars(p)})
else:
    for lv in levels:
        a = sub[(sub.level_label == lv) & (sub.model == "copula")].set_index("seed")
        b = sub[(sub.level_label == lv) & (sub.model == "ctgan")].set_index("seed")
        cm = a.index.intersection(b.index)
        if len(cm) > 2:
            p = stats.ttest_rel(a.loc[cm, metric], b.loc[cm, metric])[1]
            rows.append({"Setting": lv, "Copula": a.loc[cm, metric].mean(), "CTGAN": b.loc[cm, metric].mean(), "p-value": p, "Significance": C.sig_stars(p)})
if rows:
    st.dataframe(pd.DataFrame(rows).style.format({"Copula": "{:.3f}", "CTGAN": "{:.3f}", "p-value": "{:.4f}"}), hide_index=True, width="stretch")
st.caption("Significance: * p<0.05, ** p<0.01, *** p<0.001, ns = not significant (the two averages cannot be told apart given the seed-to-seed variation).")

# ---------------------------------------------------------------- trade-off
st.header("Fidelity vs privacy: the trade-off chart")
yopt = st.radio("Privacy axis", ["dcr_ratio", "mia_mean", "ad_gap"], horizontal=True, format_func=lambda m: C.STUDY_LABELS[m])
cfg = study.assign(cfg=study.model + " | " + study.n_total.astype(int).astype(str) + " rows | " + study.pos_label + " | " +
                   study.epochs_n.map(lambda v: "" if pd.isna(v) else f"{int(v)} ep"))
P = cfg.groupby("cfg").agg(model=("model", "first"), sdv=("sdv_overall", "mean"), y=(yopt, "mean"), ysd=(yopt, "std")).reset_index()
fig = px.scatter(P, x="sdv", y="y", color="model", error_y="ysd", hover_name="cfg", color_discrete_map={"copula": "#1f77b4", "ctgan": "#d62728"},
                 labels={"sdv": "Fidelity: SDV quality score  (→ better)", "y": C.STUDY_LABELS[yopt]})
if yopt == "dcr_ratio":
    fig.add_hline(y=1, line_dash="dash", line_color="grey", annotation_text="DCR = 1 limit")
    lr = stats.linregress(study.sdv_overall, study.dcr_ratio)
    xs = np.linspace(study.sdv_overall.min(), study.sdv_overall.max(), 20)
    fig.add_scatter(x=xs, y=lr.intercept + lr.slope * xs, mode="lines", line=dict(color="black", width=1), name=f"linear fit (r²={lr.rvalue ** 2:.2f})")
if yopt == "mia_mean":
    fig.add_hline(y=0.5, line_dash="dash", line_color="grey", annotation_text="0.5 = guessing")
fig.update_layout(height=430)
st.plotly_chart(fig, width="stretch")
st.info("""**What this chart shows:** each dot is one setting (generator + size + class ratio + epochs), averaged over 30 seeds. Horizontal position is fidelity (further right = more faithful synthetic data); vertical position is a privacy measure.

**How to read it:** with the **DCR ratio** axis, *lower* means synthetic rows are closer to real training rows; below the dashed line (1.0) they are closer than genuinely new patients would be - the warning zone.
A downward slope means higher fidelity comes with closer copies (a trade-off). With the **attack AUC** axis, dots near the dashed 0.5 line mean attackers cannot tell training patients from others - a flat cloud means fidelity does not raise attack success.""")

# ---------------------------------------------------------------- attackers
st.header("Does a stronger attacker find leakage?")
att = []
for m, name in (("copula", "Gaussian Copula"), ("ctgan", "CTGAN")):
    for col, an in (("mia_auc_lr", "Logistic Regression"), ("mia_auc_rf", "Random Forest"), ("mia_auc_mlp", "MLP (neural net)")):
        v = study[study.model == m][col]
        att.append(dict(Generator=name, Attacker=an, AUC=v.mean(), SD=v.std()))
att = pd.DataFrame(att)
fig = px.bar(att, x="Attacker", y="AUC", color="Generator", barmode="group", error_y="SD", color_discrete_map={"Gaussian Copula": "#1f77b4", "CTGAN": "#d62728"})
fig.add_hline(y=0.5, line_dash="dash", line_color="grey", annotation_text="0.5 = guessing")
fig.update_layout(height=380, yaxis_range=[0.3, 0.7])
st.plotly_chart(fig, width="stretch")
st.info("**What this chart shows:** how well three attackers of increasing strength can tell whether a patient's record was used to train the generator (all runs pooled). "
        "**How to read it:** an AUC of 0.5 means the attacker is just guessing; 1.0 would mean perfect identification. All bars sit at about 0.5 and the error bars (± SD) straddle the line, "
        "so even the strongest attacker found no leakage. Values slightly below 0.5 are statistical noise, not a sign of extra privacy.")

with st.expander("Raw per-run results (all runs)"):
    st.dataframe(study.drop(columns=["run_key"]), width="stretch")
    st.download_button("Download master_results.csv", study.to_csv(index=False).encode(), "master_results.csv")
