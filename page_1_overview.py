import dd_common as C
import streamlit as st

F = C.facts(C.load_study())

st.title("🏥 Project Overview")
st.caption("Privacy-compliant synthetic healthcare data: a unified fidelity-privacy evaluation framework with DPDP Act benchmarking")

st.header("Why this project matters")
st.markdown("""
**Healthcare AI needs data that hospitals cannot freely share.** Patient records power diagnosis models, research and
quality improvement, but they are personal data. In India the **Digital Personal Data Protection (DPDP) Act, 2023**
(with the **DPDP Rules, 2025** notified in November 2025) places duties on anyone who processes such data: a lawful basis such as
consent, reasonable security safeguards, and erasure when the purpose is served. For many teams - small hospitals, start-ups, student
researchers - that makes it slow or impossible to share or experiment with real records.

**Synthetic data is a promising workaround.** A generative model learns the statistical patterns of a real dataset and produces
*artificial* patient records that keep those patterns without describing any real person. If done well, such data can be shared more freely.
""")

st.header("The problem this project tackles")
c1, c2, c3 = st.columns(3)
c1.markdown("##### 1. Quality is not guaranteed\nSynthetic data can look plausible yet distort distributions, break relationships between variables, "
            "or change the class balance, so models trained on it fail on real patients.")
c2.markdown("##### 2. Privacy is not guaranteed\nA generator can memorise and leak real records. Without testing, 'synthetic' does not mean 'safe'.")
c3.markdown("##### 3. No simple way to choose and check\nThere are many generators and many metrics, and no widely agreed standard. "
            "Teams cannot easily tell which generator suits their data, or how to document that it is safe enough.")

st.header("What we built and asked")
st.markdown("""
We built a **single evaluation framework** that scores any synthetic dataset on **7 metrics** - three for fidelity and utility, four for privacy - and used it
in a **controlled experiment** comparing two popular generators, **Gaussian Copula** (statistical) and **CTGAN** (deep learning), on the
Pima Indians Diabetes dataset. Three research questions guided the work:
""")
st.markdown("""
- **RQ1 - Do the generators really differ?** Can one framework reveal differences between Gaussian Copula and CTGAN in fidelity, utility and privacy?
- **RQ2 - Is there a fidelity-privacy trade-off, and where does it become risky?** As synthetic data gets more faithful, does it start to leak, and can we define a practical compliance threshold?
- **RQ3 - Which generator should I use?** Can the results be turned into a simple, evidence-based selection rule?
""")

st.header("What you can do in this app")
st.dataframe({
    "Page": ["🔬 Study Methodology", "📊 Study Findings", "🧭 Model Advisor", "🧪 Generate & Evaluate Data", "⚖️ DPDP Compliance Guide"],
    "What it gives you": [
        "How the experiment was designed: data, generators, the 7 metrics, how privacy thresholds were set.",
        "Interactive results of the study and the answers to the three research questions.",
        "A quick recommendation of Gaussian Copula vs CTGAN for your situation, backed by the study.",
        "Upload a CSV, generate synthetic data, see all 7 metrics with explanations, and download a PDF report.",
        "How the privacy metrics relate to DPDP provisions, what PASS / REVIEW / FAIL means, and the limits."],
}, hide_index=True, width="stretch")

st.header("Study at a glance")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Experimental runs", f"{F['n_runs']}")
m2.metric("Settings compared", f"{F['n_cfg']}")
m3.metric("Random seeds per setting", f"{F['n_seeds']}")
m4.metric("Evaluation metrics", "7")

st.header("Who it is for")
st.markdown("""
- **Hospital data teams and clinical researchers** who want to share or analyse data with less privacy risk.
- **AI developers and students** who need realistic healthcare-like data for experiments.
- **Compliance and governance staff** who need documented evidence that a synthetic dataset was tested.
""")

st.warning("**Important.** This is a research prototype. It is a *technical screening aid*, not legal advice and not a DPDP certification. "
           "The DPDP Act does not mention synthetic data or set numeric privacy thresholds. See the **DPDP Compliance Guide** page for the limits.")
