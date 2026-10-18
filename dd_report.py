"""Generates the PDF evaluation / DPDP-alignment report (fpdf2, core fonts -> ASCII only)."""
import datetime as dt

from fpdf import FPDF
from fpdf.enums import XPos, YPos

import dd_thresholds as TH

COLORS = {"PASS": (46, 125, 50), "REVIEW": (230, 140, 0), "FAIL": (198, 40, 40)}
NL = dict(new_x=XPos.LMARGIN, new_y=YPos.NEXT)


def _a(s):
    """ASCII-safe text for core PDF fonts."""
    rep = {"\u2265": ">=", "\u2264": "<=", "\u00b1": "+/-", "\u2013": "-", "\u2014": "-", "\u2019": "'", "\u2018": "'",
           "\u201c": '"', "\u201d": '"', "\u00d7": "x", "\u2192": "->", "\u2248": "~", "\u20b9": "Rs "}
    s = str(s)
    for k, v in rep.items():
        s = s.replace(k, v)
    return s.encode("latin-1", "replace").decode("latin-1")


class PDF(FPDF):
    def header(self):
        self.set_font("helvetica", "B", 9)
        self.set_text_color(110, 110, 110)
        self.cell(0, 6, "Synthetic Data Evaluation Report - DPDP Act 2023 alignment (indicative, not legal advice)",
                  **NL)
        self.set_text_color(0, 0, 0)
        self.ln(1)

    def footer(self):
        self.set_y(-12)
        self.set_font("helvetica", "I", 8)
        self.set_text_color(110, 110, 110)
        self.cell(0, 6, f"Page {self.page_no()}", align="C")


def _h(pdf, text):
    pdf.ln(2)
    pdf.set_font("helvetica", "B", 12)
    pdf.set_fill_color(235, 240, 248)
    pdf.cell(0, 7, _a(text), fill=True, **NL)
    pdf.set_font("helvetica", "", 9.5)
    pdf.ln(1)


def _p(pdf, text, size=9.5, style=""):
    pdf.set_font("helvetica", style, size)
    pdf.multi_cell(0, 4.8, _a(text), **NL)


def _table(pdf, header, rows, widths, status_col=None):
    from fpdf.enums import MethodReturnValue
    LH = 5.0
    pdf.set_font("helvetica", "B", 8.5)
    pdf.set_fill_color(220, 226, 238)
    for h, w in zip(header, widths):
        pdf.cell(w, 6.5, _a(h), border=1, fill=True)
    pdf.ln()
    pdf.set_font("helvetica", "", 8.5)
    for r in rows:
        n_lines = max(len(pdf.multi_cell(w - 2, LH, _a(c), dry_run=True, output=MethodReturnValue.LINES))
                      for c, w in zip(r, widths))
        h = n_lines * LH + 2
        if pdf.get_y() + h > pdf.h - 18:
            pdf.add_page()
        x0, y0 = pdf.l_margin, pdf.get_y()
        for i, (c, w) in enumerate(zip(r, widths)):
            x = x0 + sum(widths[:i])
            pdf.rect(x, y0, w, h)
            pdf.set_xy(x + 1, y0 + 1)
            if status_col is not None and i == status_col and str(c) in COLORS:
                pdf.set_text_color(*COLORS[str(c)])
                pdf.set_font("helvetica", "B", 8.5)
            pdf.multi_cell(w - 2, LH, _a(c), new_x=XPos.RIGHT, new_y=YPos.TOP)
            pdf.set_text_color(0, 0, 0)
            pdf.set_font("helvetica", "", 8.5)
        pdf.set_xy(x0, y0 + h)
    pdf.ln(1)


def build_pdf(ctx):
    """ctx keys: dataset_name, n_rows, n_features, target, model, epochs, seed, n_train, n_holdout, res (metrics),
    gates, verdict, sdv_version, notes (list[str])"""
    res, gates, verdict = ctx["res"], ctx["gates"], ctx["verdict"]
    pdf = PDF(format="A4")
    pdf.set_auto_page_break(auto=True, margin=16)
    pdf.add_page()
    pdf.set_font("helvetica", "B", 17)
    pdf.cell(0, 10, "Synthetic Data Evaluation Report", **NL)
    pdf.set_font("helvetica", "", 10)
    pdf.cell(0, 6, _a(f"Generated {dt.date.today().isoformat()}  |  Dataset: {ctx['dataset_name']}"), **NL)
    pdf.ln(2)

    _h(pdf, "1. Overall privacy screening result")
    pdf.set_font("helvetica", "B", 20)
    pdf.set_text_color(*COLORS[verdict])
    pdf.cell(0, 11, verdict, **NL)
    pdf.set_text_color(0, 0, 0)
    msg = {"PASS": "All five privacy screening gates are within the range observed for non-leaking synthetic data in the calibration study.",
           "REVIEW": "At least one gate is above the 95th percentile of the calibration study's non-leaking runs. This happens by chance in roughly 1 in 10 clean datasets: repeat the evaluation with other seeds and inspect the flagged metric before relying on the data.",
           "FAIL": "At least one gate is beyond the 99th percentile (or a hard criterion) of the calibration study. Do not release this synthetic dataset without investigation."}[verdict]
    _p(pdf, msg)

    _h(pdf, "2. Dataset and generation settings")
    rows = [["Dataset", ctx["dataset_name"]], ["Rows / features", f"{ctx['n_rows']} rows, {ctx['n_features']} features + target '{ctx['target']}'"],
            ["Train / holdout split", f"{ctx['n_train']} / {ctx['n_holdout']} (stratified 80/20; generator sees TRAIN only)"],
            ["Generator", ctx["model"] + (f", {ctx['epochs']} epochs" if ctx["model"].lower().startswith("ctgan") else "")],
            ["Random seed", str(ctx["seed"])], ["SDV version", ctx["sdv_version"]],
            ["Synthetic rows", str(ctx["n_train"])]]
    _table(pdf, ["Item", "Value"], rows, [45, 145])

    _h(pdf, "3. Privacy screening gates")
    grows = []
    for g in gates:
        thr = " / ".join(x for x in [f"review {g['review']}" if g["review"] else "", f"fail {g['fail']}"] if x)
        grows.append([g["gate"], g["fmt"].format(g["value"]), thr, g["provision"], g["status"]])
    _table(pdf, ["Metric", "Value", "Thresholds", "Related DPDP provision", "Result"], grows, [62, 18, 38, 52, 20], status_col=4)
    _p(pdf, "Thresholds are calibrated to dataset size using the 95th (review) and 99th (fail) percentiles of a controlled study of "
            f"{TH.T['n_runs']} synthetic-data runs in which no membership leakage was detectable. DCR ratio < 1 means synthetic records "
            "sit closer to the training data than unseen real records do.", 8.5, "I")

    _h(pdf, "4. Fidelity and utility (for fitness-for-purpose, not gated)")
    f = lambda v, k=3: "n/a" if v is None or v != v else f"{v:.{k}f}"
    frows = [["SDV quality score (0-1)", f(res.get("sdv_overall")), "Overall statistical similarity (column shapes + pair trends)"],
             ["KS test pass rate", f(res["ks_pass_rate"]), "Share of numeric columns whose distribution is not significantly different (p > 0.05)"],
             ["Mean KS statistic", f(res["ks_stat_mean"]), "Lower is better"],
             ["Chi-square on target (p)", f(res["chi2_p"]), "p > 0.05: class ratio preserved"],
             ["Target share real / synthetic", f"{res['pos_rate_real']:.1%} / {res['pos_rate_synth']:.1%}", "Positive-class share"],
             ["TRTR AUC (real -> real holdout)", f(res["trtr_auc"]), "Utility ceiling"],
             ["TSTR AUC (synthetic -> real holdout)", f(res["tstr_auc"]), "Utility of synthetic data; gap to TRTR: " + f(res["tstr_auc_gap"])],
             ["TSTR F1 / accuracy", f"{f(res['tstr_f1'])} / {f(res['tstr_acc'])}", "Random Forest trained on synthetic data"]]
    _table(pdf, ["Metric", "Value", "Meaning"], frows, [58, 32, 100])
    if res.get("sdv_error"):
        _p(pdf, "Note: the SDV quality score could not be computed in this environment (" + res["sdv_error"] + ").", 8.5, "I")

    _h(pdf, "5. Membership-inference detail")
    mrows = [[a, f(res[f"mia_auc_{k}"])] for k, a in (("lr", "Logistic Regression attacker"), ("rf", "Random Forest attacker"), ("mlp", "MLP attacker"))]
    mrows.append(["Mean of three attackers (gated)", f(res["mia_mean"])])
    _table(pdf, ["Attacker", "AUC (0.5 = cannot tell members from non-members)"], mrows, [80, 110])

    _h(pdf, "6. How the metrics relate to the DPDP Act 2023 (indicative mapping)")
    maprows = [
        ["Is the output 'personal data'? (s.2(t), s.3)", "DCR ratio, near-copy share, exact copies, attribute disclosure", "Evidence that individual records are not reproduced or re-identifiable"],
        ["Reasonable security safeguards (s.8(5))", "Membership-inference AUC", "Evidence the generator has not memorised training individuals"],
        ["Erasure when purpose is served / correction and erasure (s.8(7), s.12)", "Membership-inference AUC", "A generator that memorises members could retain data that should be erased"],
        ["Accuracy and completeness (s.8(3))", "SDV score, KS / chi-square, TSTR gap", "Relevant when synthetic data informs decisions about individuals"],
        ["Accountability / documentation (s.10 for Significant Data Fiduciaries; DPDP Rules 2025)", "This report", "Documented evidence of due diligence"],
    ]
    _table(pdf, ["Provision (topic)", "Metric(s)", "Why it is relevant"], maprows, [62, 56, 72])

    _h(pdf, "7. Limitations and disclaimer")
    for t in [
        "This report is a technical screening aid, not legal advice and not a certification of compliance. The DPDP Act 2023 does not mention synthetic data, prescribes no numeric privacy thresholds, and (unlike the GDPR) has no separate 'sensitive personal data' category; no regulator has endorsed the thresholds used here. The DPDP Rules 2025 were notified on 13 November 2025.",
        f"Thresholds were calibrated on one public dataset (Pima Indians Diabetes, {TH.T['n_runs']} runs, 160-614 training rows, Gaussian Copula and CTGAN). Datasets that differ in size, dimensionality or class balance may behave differently" + (" - this dataset is outside the calibrated size range." if ctx.get("outside_range") else "."),
        "Membership-inference scores from a single run are noisy (about +/-0.05 around chance for a few hundred records). About 1 in 10 non-leaking datasets will trigger REVIEW by chance; repeat with several seeds.",
        "Empirical attacks can only fail to find leakage; they do not prove privacy. Differential privacy is the only formal guarantee and is not used by the generators here.",
        "Use a qualified legal adviser to decide whether a synthetic dataset falls outside the Act's definition of personal data for your purpose.",
    ]:
        pdf.set_x(pdf.l_margin)
        pdf.set_font("helvetica", "", 9)
        pdf.multi_cell(0, 4.6, _a("- " + t), **NL)
        pdf.ln(0.8)
    for n in ctx.get("notes", []):
        _p(pdf, "Note: " + n, 8.5, "I")
    return bytes(pdf.output())
