# Synthetic Healthcare Data Evaluator (multi-page Streamlit app, flat layout)

All files sit in ONE folder (no sub-folders) so they can be uploaded through the GitHub web page.

Pages (sidebar menu):
1. Project Overview        - why the project matters
2. Study Methodology       - data, generators, the 7 metrics, experiment design, thresholds, limitations
3. Study Findings          - answers to the research questions + interactive charts
4. Model Advisor           - Gaussian Copula vs CTGAN recommendation
5. Generate & Evaluate Data - upload a CSV, generate synthetic data, 7 metrics, graph explanations, PDF report
6. DPDP Compliance Guide   - provision mapping, PASS/REVIEW/FAIL rules, limits

Run locally (Python 3.11 or 3.12):
    pip install -r requirements.txt
    streamlit run app.py

Streamlit Community Cloud: main file `app.py`, Python 3.12 (Advanced settings).
If the build fails, copy the contents of requirements-lite.txt into requirements.txt
(everything works except live generation on the Generate & Evaluate page).

Update the study data: replace master_results.csv, run `python calibrate_thresholds.py`, push.
The numbers on the pages are computed live from master_results.csv; the wording of the conclusions describes the 540-run study.

Research prototype - a technical screening aid, not legal advice or DPDP certification.
