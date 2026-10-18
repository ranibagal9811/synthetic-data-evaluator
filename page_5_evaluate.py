import os
import time

import dd_common as C
import dd_metrics as MET
import dd_report as RP
import dd_synth as SY
import dd_thresholds as TH
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.model_selection import train_test_split

HERE = C.HERE
study = C.load_study()
F = C.facts(study)
ICON = C.ICON

st.title("🧪 Generate & Evaluate Synthetic Data")
st.caption("Upload a dataset, generate a synthetic copy, and check its fidelity, utility and privacy with 7 metrics")

st.markdown("""
**How it works**
1. **Upload** a CSV (or use the built-in Pima sample) and pick the target column.
2. **Generate** synthetic data with Gaussian Copula or CTGAN - trained on 80% of your rows.
3. **Evaluate** it against the 20% of real rows the generator never saw, using the 7 metrics.
4. **Read the results** - every metric and every graph is explained - and **download** a PDF report.
""")

# ======================================================================= what can I upload
with st.expander("📁 What kind of data can I upload?", expanded=True):
    st.markdown("""
The app works with **tabular patient-style data in a CSV file** that meets these rules:
""")
    st.dataframe(pd.DataFrame({
        "Requirement": ["File type", "Feature columns", "Target column", "Rows", "Missing values", "Size limit", "Columns"],
        "Rule": ["CSV with a header row", "All **numeric** (integers or decimals) - at least 2",
                 "Exactly **one** column with exactly **2 distinct values** (e.g. 0/1, yes/no, malignant/benign)",
                 "At least **120** rows (files over 5,000 rows are randomly cut to 5,000)",
                 "Rows with missing values are dropped automatically", "Up to 20 MB",
                 "Constant columns (one single value) are removed automatically"],
    }), hide_index=True, width="stretch")
    st.markdown("**Example of a valid file:**")
    st.dataframe(pd.DataFrame({"Glucose": [148, 85, 183], "BloodPressure": [72, 66, 64], "BMI": [33.6, 26.6, 23.3],
                               "Age": [50, 31, 32], "Outcome": [1, 0, 1]}), hide_index=True, width="stretch")
    c1, c2 = st.columns(2)
    c1.markdown("""**Prepare your file first**
- Remove ID, name, date and free-text columns.
- Convert yes/no style features to 0/1.
- Make sure the target has only two classes.""")
    c2.markdown("""**Not supported (the app will tell you)**
- Text or date feature columns
- A target with 3 or more classes
- Fewer than 120 complete rows
- Time-series or multi-table data""")
    st.markdown(f"""**Good test datasets:** the built-in **Pima Indians Diabetes** sample, or other public medical tables such as the **Breast Cancer Wisconsin (Diagnostic)** data
(569 rows, 30 features; drop the `id` column). The privacy thresholds were calibrated on Pima (200-768 rows, 9 columns), so results on very different data are less reliable.""")
    st.warning("**Privacy warning.** If this app is hosted on a public website, **do not upload real patient records**. Use public, de-identified or already-synthetic data for demonstrations. "
               "To evaluate real patient data, run the app inside your own secure environment.")
    st.download_button("⬇ Download the sample CSV (Pima Indians Diabetes)", open(os.path.join(HERE, "diabetes.csv"), "rb").read(),
                       file_name="diabetes_sample.csv", mime="text/csv")

# ======================================================================= inputs
st.header("1. Choose your data and settings")
if not C.TEST_MODE and SY.sdv_version() == "not installed":
    st.error("SDV is not installed in this environment, so synthetic data cannot be generated here. Install the dependencies in requirements.txt.")

src = st.radio("Data source", ["Built-in sample: Pima Indians Diabetes", "Upload my own CSV"], horizontal=True)
df_in, name = None, None
if src.startswith("Built-in"):
    df_in, name = pd.read_csv(os.path.join(HERE, "diabetes.csv")), "Pima Indians Diabetes (sample)"
else:
    up = st.file_uploader("CSV file", type=["csv"])
    if up is not None:
        try:
            df_in, name = pd.read_csv(up), up.name
        except Exception as e:
            st.error(f"Could not read the file as CSV: {e}")

if df_in is not None:
    st.markdown(f"**Preview** - {len(df_in)} rows × {df_in.shape[1]} columns")
    st.dataframe(df_in.head(5), width="stretch")
    cols = list(df_in.columns)
    target = st.selectbox("Target column (the two-class outcome)", cols, index=cols.index("Outcome") if "Outcome" in cols else len(cols) - 1)
    c1, c2, c3 = st.columns(3)
    model_label = c1.selectbox("Generator", ["Gaussian Copula (recommended)", "CTGAN"])
    epochs = c2.select_slider("CTGAN epochs", [100, 200, 300, 500, 1000], value=300, disabled=model_label.startswith("Gaussian"))
    seed = c3.number_input("Random seed", 0, 10_000, 0, help="Changing the seed changes the train/holdout split and the generated data.")
    if model_label == "CTGAN":
        st.info("CTGAN is much slower than Copula (the study measured about 13 s at 300 epochs and 31 s at 1000 epochs on a GPU; on a CPU-only host expect several minutes). "
                "In the study, CTGAN with 300 epochs or fewer was worse than Copula on fidelity and, in nearly all settings, on classifier utility.")
    go_btn = st.button("▶ Generate and evaluate", type="primary")

    if go_btn:
        problems, df = [], df_in.copy()
        if df[target].dropna().nunique() != 2:
            problems.append(f"The target '{target}' must have exactly 2 distinct values (found {df[target].dropna().nunique()}).")
        feats = [c for c in df.columns if c != target]
        bad = [c for c in feats if not pd.api.types.is_numeric_dtype(df[c])]
        if bad:
            problems.append("Non-numeric feature columns are not supported: " + ", ".join(bad) + ". Remove or convert them.")
        if len(feats) < 2:
            problems.append("At least 2 feature columns are needed.")
        n0 = len(df)
        df = df.dropna()
        if len(df) < 120:
            problems.append(f"At least 120 complete rows are needed (found {len(df)}).")
        if problems:
            for p in problems:
                st.error(p)
        else:
            if len(df) > 5000:
                df = df.sample(5000, random_state=int(seed))
                st.info("The dataset was randomly reduced to 5,000 rows to keep the run time reasonable.")
            if len(df) < n0:
                st.info(f"Dropped {n0 - len(df)} rows with missing values.")
            classes = sorted(df[target].unique())
            df[target] = (df[target] == classes[1]).astype(int)
            feats = [c for c in feats if df[c].nunique() > 1]
            df = df[feats + [target]].reset_index(drop=True)
            model = "copula" if model_label.startswith("Gaussian") else "ctgan"
            with st.status("Running the pipeline...", expanded=True) as status:
                t0 = time.time()
                st.write("1/3  Splitting 80% train / 20% holdout (stratified)...")
                train, hold = train_test_split(df, test_size=0.2, stratify=df[target], random_state=int(seed))
                train, hold = train.reset_index(drop=True), hold.reset_index(drop=True)
                st.write(f"2/3  Training {model_label.split(' (')[0]} on {len(train)} rows and generating {len(train)} synthetic rows...")
                try:
                    md = None if C.TEST_MODE else SY.get_metadata(train, target)
                    synth_df = SY.fit_and_sample(model, train, md, epochs, int(seed), len(train),
                                                 use_gpu=os.environ.get("DASHBOARD_USE_GPU") == "1")
                except Exception as e:
                    status.update(label="Generation failed", state="error")
                    st.error(f"{type(e).__name__}: {e}")
                    st.stop()
                st.write("3/3  Computing the 7 metrics (fidelity, utility, distance, attribute disclosure, 3-level membership inference)...")
                res = MET.evaluate_all(train, synth_df, hold, feats, target, md, int(seed))
                gates = TH.privacy_gates(res, len(df))
                verdict = TH.overall(gates)
                status.update(label=f"Done in {time.time() - t0:.0f} seconds", state="complete")
            st.session_state["result"] = dict(
                res=res, gates=gates, verdict=verdict, train=train, hold=hold, synth=synth_df, feats=feats, target=target, classes=classes,
                ctx=dict(dataset_name=name, n_rows=len(df), n_features=len(feats), target=target, model=model_label.split(" (")[0], epochs=epochs,
                         seed=int(seed), n_train=len(train), n_holdout=len(hold),
                         sdv_version=SY.sdv_version() if not C.TEST_MODE else "test-mode", outside_range=TH.outside_range(len(df)), notes=[]))

# ======================================================================= results
R = st.session_state.get("result")
if not R:
    st.info("Results will appear here after you click **Generate and evaluate**.")
    st.stop()

res, gates, verdict = R["res"], R["gates"], R["verdict"]
ctx = R["ctx"]
st.divider()
st.header("2. Your results")
st.caption(f"Dataset: {ctx['dataset_name']} · {ctx['n_rows']} rows · generator: {ctx['model']}" + (f" ({ctx['epochs']} epochs)" if ctx["model"] == "CTGAN" else "") + f" · seed {ctx['seed']}")
banner = {"PASS": st.success, "REVIEW": st.warning, "FAIL": st.error}[verdict]
banner(f"**Overall privacy screening: {verdict}** - " + {
    "PASS": "all five privacy gates are within the range seen for non-leaking synthetic data in the study.",
    "REVIEW": "at least one gate is above the study's 95th percentile. This happens by chance for about 1 in 10 clean datasets - re-run with another seed and check the flagged metric.",
    "FAIL": "at least one gate is beyond the study's 99th percentile or a hard criterion. Investigate before releasing the data."}[verdict])
if ctx["outside_range"]:
    st.caption("⚠ This dataset is outside the 200-768-row range used to calibrate the thresholds; interpret with extra care.")

g_exact = C.gate_by_prefix(gates, "Exact")
g_dcr = C.gate_by_prefix(gates, "DCR")
g_near = C.gate_by_prefix(gates, "Near")
g_mia = C.gate_by_prefix(gates, "Membership")
g_ad = C.gate_by_prefix(gates, "Attribute")
sdv_v = res.get("sdv_overall")
band = TH.fidelity_band(sdv_v)
gap = res["tstr_auc_gap"]
util_label = "close to real" if gap < 0.05 else "moderate gap" if gap < 0.12 else "large gap"

st.subheader("The 7 metrics at a glance")
glance = pd.DataFrame([
    {"#": 1, "Metric": "KS test", "Result": f"mean KS {res['ks_stat_mean']:.3f}; {res['ks_pass_rate']:.0%} of columns pass", "What it tells you": "Column distributions match" if res["ks_pass_rate"] >= 0.5 else "Some columns differ from the real data"},
    {"#": 2, "Metric": "Chi-square (class balance)", "Result": f"p = {res['chi2_p']:.3f}", "What it tells you": "Class ratio preserved ✅" if res["chi2_pass"] else "Class ratio distorted ⚠"},
    {"#": 3, "Metric": "SDV quality score", "Result": "unavailable" if sdv_v is None else f"{sdv_v:.3f}", "What it tells you": "n/a" if sdv_v is None else f"Fidelity {band} compared with the study"},
    {"#": 4, "Metric": "TSTR vs TRTR (utility)", "Result": f"TSTR AUC {res['tstr_auc']:.3f} vs real {res['trtr_auc']:.3f}", "What it tells you": f"Utility gap: {util_label}"},
    {"#": 5, "Metric": "Distance to closest record", "Result": f"DCR ratio {res['dcr_ratio']:.2f}; near-copies {res['dcr_pct_near']:.1%}", "What it tells you": " / ".join(ICON[g["status"]] for g in (g_exact, g_dcr, g_near))},
    {"#": 6, "Metric": "Attribute disclosure", "Result": f"gap {res['ad_gap']:+.3f}", "What it tells you": ICON[g_ad["status"]]},
    {"#": 7, "Metric": "Membership inference", "Result": f"mean AUC {res['mia_mean']:.3f}", "What it tells you": ICON[g_mia["status"]]},
])
st.dataframe(glance, hide_index=True, width="stretch")
st.caption("Metrics 1-4 describe quality and usefulness (not pass/fail gates). Metrics 5-7 are privacy gates and decide the overall result.")


def block(num, title, left_md, fig, what, read):
    with st.container(border=True):
        st.markdown(f"#### {num}. {title}")
        lc, rc = st.columns([2, 3])
        with lc:
            st.markdown(left_md)
        with rc:
            if fig is not None:
                fig.update_layout(height=300, margin=dict(t=30, b=40, l=40, r=10))
                st.plotly_chart(fig, width="stretch")
        st.info(f"**What this graph shows:** {what}\n\n**How to read it:** {read}")


st.subheader("Detailed results and what each graph means")

# ---- 1 KS
ks = pd.Series(res["ks_per_column"]).sort_values()
fig = go.Figure(go.Bar(x=ks.values, y=ks.index, orientation="h", marker_color="#4c78a8"))
fig.update_layout(xaxis_title="KS statistic (0 = identical)")
block(1, "KS test - are the column distributions preserved?",
      f"Average KS statistic: **{res['ks_stat_mean']:.3f}**\n\nColumns passing (p > 0.05): **{res['ks_pass_rate']:.0%}**\n\n"
      f"*In the study, Gaussian Copula passed about {100 * F['cop_ks_pass_rate']:.0f}% of columns and CTGAN (300 epochs) about {100 * F['k300_ks_pass_rate']:.0f}%.*",
      fig, "one bar per numeric column - the KS statistic, i.e. the largest gap between the real and the synthetic cumulative distribution of that column.",
      "0 means the column's distribution is identical; taller bars mean the generator reproduced that column less faithfully. Look for bars far taller than the rest - those are the columns the generator struggled with.")

# ---- 2 Chi-square
fig = go.Figure(go.Bar(x=["Real data", "Synthetic data"], y=[res["pos_rate_real"], res["pos_rate_synth"]], marker_color=["#1f77b4", "#d62728"],
                       text=[f"{res['pos_rate_real']:.1%}", f"{res['pos_rate_synth']:.1%}"], textposition="outside"))
fig.update_layout(yaxis_title=f"Share with {R['target']} = positive class", yaxis_range=[0, min(1, max(res["pos_rate_real"], res["pos_rate_synth"]) * 1.4 + 0.05)])
block(2, "Chi-square test - is the class balance preserved?",
      f"p-value: **{res['chi2_p']:.3f}**\n\nResult: **{'class ratio preserved ✅' if res['chi2_pass'] else 'class ratio distorted ⚠'}**\n\n"
      f"*CTGAN tends to over-represent the smaller class (about {100 * F['k1000_pos_rate_synth']:.0f}% vs {100 * F['k1000_pos_rate_real']:.0f}% in the study); Copula keeps it.*",
      fig, "the share of patients in the positive (smaller) class in the real training data and in the synthetic data.",
      "The two bars should be about the same height. A p-value above 0.05 means the difference is within chance; a clearly taller synthetic bar means the generator inflated the minority class, which can bias models trained on it.")

# ---- 3 SDV
if sdv_v is None:
    fig = None
    left = "SDV quality score is **not available** in this environment" + (f" ({res['sdv_error']})" if res.get("sdv_error") else "") + "."
else:
    vals = {"Overall": sdv_v, "Column shapes": res.get("sdv_column_shapes"), "Column pair trends": res.get("sdv_pair_trends")}
    fig = go.Figure(go.Bar(x=list(vals), y=[v if v is not None else 0 for v in vals.values()], marker_color=["#2ca02c", "#4c78a8", "#f58518"],
                           text=[f"{v:.3f}" if v is not None else "n/a" for v in vals.values()], textposition="outside"))
    ref = TH.T["reference"]["sdv_overall"]
    fig.update_layout(yaxis_range=[0, 1.05], yaxis_title="Score (0-1)")
    fig.add_hrect(y0=ref["p05"], y1=ref["p95"], fillcolor="grey", opacity=0.15, line_width=0, annotation_text="typical range in the study", annotation_position="top left")
    left = (f"Overall quality: **{sdv_v:.3f}** → fidelity **{band}** relative to the study\n\nColumn shapes: **{res.get('sdv_column_shapes', float('nan')):.3f}**\n\n"
            f"Column pair trends: **{res.get('sdv_pair_trends', float('nan')):.3f}**\n\n*Study range (5th-95th percentile): {ref['p05']:.2f}-{ref['p95']:.2f}.*")
block(3, "SDV quality score - overall similarity", left, fig,
      "three scores from 0 to 1: the overall quality, how well single columns match (column shapes), and how well relationships between pairs of columns match (column pair trends). The grey band is the range seen in the study.",
      "higher is better. If 'column shapes' is much higher than 'column pair trends', each column looks right on its own but the relationships between columns (e.g. glucose vs BMI) were weakened.")

# ---- 4 TSTR / TRTR
fig = go.Figure()
fig.add_bar(name="Trained on REAL data (TRTR)", x=["AUC", "F1", "Accuracy"], y=[res["trtr_auc"], res["trtr_f1"], res["trtr_acc"]], marker_color="#1f77b4")
fig.add_bar(name="Trained on SYNTHETIC data (TSTR)", x=["AUC", "F1", "Accuracy"], y=[res["tstr_auc"], res["tstr_f1"], res["tstr_acc"]], marker_color="#ff7f0e")
fig.update_layout(barmode="group", yaxis_range=[0, 1], legend=dict(orientation="h", y=-0.25))
block(4, "TSTR vs TRTR - is the synthetic data useful for training models?",
      f"TRTR AUC (real → real holdout): **{res['trtr_auc']:.3f}**\n\nTSTR AUC (synthetic → real holdout): **{res['tstr_auc']:.3f}**\n\nGap: **{gap:.3f}** ({util_label})\n\n"
      f"TSTR F1: **{res['tstr_f1']:.3f}** · accuracy: **{res['tstr_acc']:.3f}**",
      fig, "how well a Random Forest classifier performs on the same real holdout patients when it was trained on real data (blue) versus on synthetic data (orange).",
      "the closer the orange bars are to the blue ones, the more useful the synthetic data is as a stand-in for the real data. A big gap means models trained on the synthetic data will perform worse on real patients. AUC is the most reliable of the three when classes are imbalanced.")

# ---- 5 DCR
fig = go.Figure()
fig.add_histogram(x=res["dcr_base_values"], name="Unseen real patients → training data", opacity=.55, histnorm="probability", nbinsx=30)
fig.add_histogram(x=res["dcr_synth_values"], name="Synthetic rows → training data", opacity=.55, histnorm="probability", nbinsx=30)
fig.update_layout(barmode="overlay", xaxis_title="distance to the closest real training record (normalised)", legend=dict(orientation="h", y=-0.3))
block(5, "Distance to Closest Record (DCR) - are real records being copied?",
      f"DCR ratio: **{res['dcr_ratio']:.3f}** {ICON[g_dcr['status']]} (limit: ≥ 1.0)\n\nExact copies: **{res['exact_copy_share']:.1%}** {ICON[g_exact['status']]}\n\n"
      f"Near-copy share: **{res['dcr_pct_near']:.1%}** {ICON[g_near['status']]} (review > 5%)",
      fig, "for each record, how far it is from its closest real training record. Orange: synthetic rows. Blue: brand-new real patients the generator never saw (the baseline for 'not copied').",
      "if the orange histogram overlaps or sits to the right of the blue one, synthetic rows are no closer to the training patients than genuinely new patients are - good. Orange shifted clearly to the left, or a spike at zero, means the generator is reproducing training records. The DCR ratio is the average orange distance divided by the average blue distance.")

# ---- 6 attribute disclosure
fig = go.Figure(go.Bar(x=["Training patients (members)", "Unseen patients (non-members)"], y=[res["ad_member"], res["ad_nonmember"]], marker_color=["#d62728", "#1f77b4"],
                       text=[f"{res['ad_member']:.1%}", f"{res['ad_nonmember']:.1%}"], textposition="outside"))
fig.update_layout(yaxis_title="Share of hidden values guessed correctly", yaxis_range=[0, min(1, max(res["ad_member"], res["ad_nonmember"]) * 1.5 + 0.05)])
block(6, "Attribute disclosure - can hidden details be inferred?",
      f"Success on training patients: **{res['ad_member']:.1%}**\n\nSuccess on unseen patients: **{res['ad_nonmember']:.1%}**\n\nGap: **{res['ad_gap']:+.3f}** {ICON[g_ad['status']]}\n\n(review if above {g_ad['review'].replace('> ', '')}, fail if above {g_ad['fail'].replace('> ', '')})",
      fig, "an attacker who knows about 80% of a person's data tries to guess the remaining 20% from the closest synthetic record. The bars show how often the guess is right for people who were in the training data (red) and for people who were not (blue).",
      "equal bars mean being in the training data gives no extra exposure. The gap (red minus blue) is the real risk: a clearly taller red bar means the synthetic data reveals more about the people it was trained on. Both bars can be well above zero simply because many values are easy to guess - only the gap matters.")

# ---- 7 MIA
fig = go.Figure(go.Bar(x=["Logistic Regression", "Random Forest", "MLP"], y=[res["mia_auc_lr"], res["mia_auc_rf"], res["mia_auc_mlp"]], marker_color="#6a5acd",
                       text=[f"{res['mia_auc_lr']:.3f}", f"{res['mia_auc_rf']:.3f}", f"{res['mia_auc_mlp']:.3f}"], textposition="outside"))
fig.add_hline(y=0.5, line_dash="dash", line_color="grey", annotation_text="0.5 = guessing")
fig.add_hline(y=float(g_mia["review"].replace("> ", "")), line_dash="dot", line_color="orange", annotation_text="review level")
fig.update_layout(yaxis_range=[0.3, 0.8], yaxis_title="Attack AUC")
block(7, "Membership inference attack - was a person's record used for training?",
      f"Logistic Regression: **{res['mia_auc_lr']:.3f}**\n\nRandom Forest: **{res['mia_auc_rf']:.3f}**\n\nMLP: **{res['mia_auc_mlp']:.3f}**\n\nMean: **{res['mia_mean']:.3f}** {ICON[g_mia['status']]}\n\n"
      f"(review above {g_mia['review'].replace('> ', '')}, fail above {g_mia['fail'].replace('> ', '')})",
      fig, "how well three attackers of increasing strength can tell, from a record's distance to the synthetic data, whether that record was in the generator's training set.",
      "0.5 means the attacker is just guessing (nothing leaks); 1.0 would mean perfect identification. Bars near 0.5 are good. The dotted orange line is where results become unusual compared with non-leaking datasets of your size. Scores from a single run are noisy (about ±0.05), so small excursions are common.")

# ---- additional diagnostic plots
st.subheader("Extra diagnostic plots")
v1, v2 = st.columns(2)
with v1:
    pick = st.multiselect("Compare column distributions", R["feats"], default=R["feats"][:3])
    if pick:
        long = pd.concat([R["train"][pick].assign(Source="Real (training)"), R["synth"][pick].assign(Source="Synthetic")]).melt(id_vars="Source", var_name="Feature", value_name="Value")
        fig = px.histogram(long, x="Value", color="Source", facet_row="Feature", barmode="overlay", histnorm="probability density", opacity=0.55, nbins=30,
                           color_discrete_map={"Real (training)": "#1f77b4", "Synthetic": "#d62728"})
        fig.update_yaxes(matches=None, showticklabels=False)
        fig.update_layout(height=230 * len(pick) + 60, margin=dict(t=30))
        fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
        st.plotly_chart(fig, width="stretch")
        st.info("**What this shows:** the value distribution of each chosen column in the real training data (blue) and in the synthetic data (red). "
                "**How to read it:** the two shapes should overlap closely. Missing humps, shifted peaks or different spread show where the generator misrepresents a column.")
with v2:
    cr, cs = R["train"].corr().values, R["synth"].corr().values
    names = list(R["train"].columns)
    fig = px.imshow(np.abs(cr - cs), x=names, y=names, color_continuous_scale="Reds", zmin=0, zmax=0.6)
    fig.update_layout(height=420, margin=dict(t=30))
    st.plotly_chart(fig, width="stretch")
    st.info("**What this shows:** for every pair of columns, how much their correlation differs between real and synthetic data (|real − synthetic|). "
            "**How to read it:** white/pale squares mean the relationship between those two columns was preserved; dark red squares mean it was weakened or distorted. It explains a low 'column pair trends' score.")

# ---- gates + downloads
st.subheader("Privacy gates and DPDP topics")
st.dataframe(pd.DataFrame([{"Gate": g["gate"], "Value": g["fmt"].format(g["value"]), "Review if": g["review"] or "-", "Fail if": g["fail"], "Result": ICON[g["status"]],
                            "Related DPDP topic": g["provision"]} for g in gates]), hide_index=True, width="stretch")
st.caption("The overall result is the worst of these five gates. See the **DPDP Compliance Guide** page for what this does and does not mean.")

st.subheader("Downloads")
d1, d2 = st.columns(2)
pdf_bytes = RP.build_pdf(dict(ctx, res=res, gates=gates, verdict=verdict))
d1.download_button("⬇ PDF evaluation report", pdf_bytes, file_name="synthetic_data_evaluation_report.pdf", mime="application/pdf", type="primary")
out = R["synth"].copy()
out[R["target"]] = out[R["target"]].map({0: R["classes"][0], 1: R["classes"][1]})
d2.download_button("⬇ Synthetic data (CSV)", out.to_csv(index=False).encode(), file_name="synthetic_data.csv", mime="text/csv")
