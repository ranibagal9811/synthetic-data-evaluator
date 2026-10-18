"""Synthetic Healthcare Data Evaluator - multi-page Streamlit app (entry point).
Run:  streamlit run app.py
"""
import os
import sys

import streamlit as st

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

st.set_page_config(page_title="Synthetic Healthcare Data Evaluator", page_icon="🩺", layout="wide")

try:
    import dd_common, dd_metrics, dd_report, dd_synth, dd_thresholds  # noqa: F401
except ModuleNotFoundError as _e:
    st.error(f"Could not import one of the app's modules ({_e}). Some dd_*.py files are missing from the repository.")
    st.write("Files found next to app.py:")
    st.code("\n".join(sorted(os.listdir(HERE))))
    st.info("Fix: the repository must contain app.py, dd_common.py, dd_metrics.py, dd_report.py, dd_synth.py, "
            "dd_thresholds.py, thresholds.json, master_results.csv, diabetes.csv and all page_*.py files in the same folder. "
            "Then reboot the app.")
    st.stop()

pages = [
    st.Page("page_1_overview.py", title="Project Overview", icon="🏥", default=True),
    st.Page("page_2_methodology.py", title="Study Methodology", icon="🔬"),
    st.Page("page_3_findings.py", title="Study Findings", icon="📊"),
    st.Page("page_4_advisor.py", title="Model Advisor", icon="🧭"),
    st.Page("page_5_evaluate.py", title="Generate & Evaluate Data", icon="🧪"),
    st.Page("page_6_dpdp.py", title="DPDP Compliance Guide", icon="⚖️"),
]
nav = st.navigation(pages)

with st.sidebar:
    st.divider()
    st.caption("🩺 **Synthetic Healthcare Data Evaluator**\n\nResearch prototype. A technical screening aid - "
               "not legal advice and not a DPDP certification.")

nav.run()
