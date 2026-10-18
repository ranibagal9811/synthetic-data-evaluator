"""Loads study-calibrated thresholds and turns metric values into PASS / REVIEW / FAIL."""
import json
import math
import os

_PATH = os.path.join(os.path.dirname(__file__), "thresholds.json")
with open(_PATH) as f:
    T = json.load(f)

ORDER = {"PASS": 0, "REVIEW": 1, "FAIL": 2}


def _interp(n_total, key):
    """Interpolate a per-size percentile on log(rows); clamp outside the studied 200-768 range."""
    pts = sorted((int(n), v[key]) for n, v in T["by_total_rows"].items())
    if n_total <= pts[0][0]:
        return pts[0][1]
    if n_total >= pts[-1][0]:
        return pts[-1][1]
    for (n0, v0), (n1, v1) in zip(pts[:-1], pts[1:]):
        if n0 <= n_total <= n1:
            w = (math.log(n_total) - math.log(n0)) / (math.log(n1) - math.log(n0))
            return v0 + w * (v1 - v0)


def outside_range(n_total):
    ns = [int(n) for n in T["by_total_rows"]]
    return n_total < min(ns) * 0.75 or n_total > max(ns) * 1.5


def privacy_gates(res, n_total):
    """Return list of dict(gate, value, review, fail, status, provision) for the 5 gating metrics."""
    mia_r, mia_f = _interp(n_total, "mia_mean_p95"), _interp(n_total, "mia_mean_p99")
    ad_r, ad_f = _interp(n_total, "ad_gap_p95"), _interp(n_total, "ad_gap_p99")
    nc = T["near_copy_share"]

    def grade(v, review, fail, higher_is_worse=True):
        if v is None or v != v:
            return "REVIEW"
        if higher_is_worse:
            return "FAIL" if v > fail else "REVIEW" if v > review else "PASS"
        return "FAIL" if v < fail else "PASS"

    gates = [
        dict(gate="Exact copies of training records", value=res["exact_copy_share"], review=None, fail="> 0",
             status="FAIL" if res["exact_copy_share"] > 0 else "PASS",
             provision="Identifiability (s.2(t), s.3)", fmt="{:.1%}"),
        dict(gate="DCR ratio (synthetic vs unseen-real distance to training data)", value=res["dcr_ratio"],
             review=None, fail="< 1.0", status=grade(res["dcr_ratio"], None, T["dcr_ratio"]["fail_below"], False),
             provision="Identifiability (s.2(t), s.3)", fmt="{:.3f}"),
        dict(gate="Near-copy share (closer than 5th pct. of unseen-real)", value=res["dcr_pct_near"],
             review=f"> {nc['review_above']:.0%}", fail=f"> {nc['fail_above']:.0%}",
             status=grade(res["dcr_pct_near"], nc["review_above"], nc["fail_above"]),
             provision="Identifiability (s.2(t), s.3)", fmt="{:.1%}"),
        dict(gate="Membership-inference AUC (mean of LR/RF/MLP attackers)", value=res["mia_mean"],
             review=f"> {mia_r:.3f}", fail=f"> {mia_f:.3f}", status=grade(res["mia_mean"], mia_r, mia_f),
             provision="Safeguards and erasure (s.8(5), s.8(7), s.12)", fmt="{:.3f}"),
        dict(gate="Attribute-disclosure gap (members minus non-members)", value=res["ad_gap"],
             review=f"> {ad_r:.3f}", fail=f"> {ad_f:.3f}", status=grade(res["ad_gap"], ad_r, ad_f),
             provision="Identifiability / safeguards (s.2(t), s.8(5))", fmt="{:.3f}"),
    ]
    return gates


def overall(gates):
    worst = max(gates, key=lambda g: ORDER[g["status"]])["status"]
    return worst


def fidelity_band(sdv):
    ref = T["reference"]["sdv_overall"]
    if sdv is None or sdv != sdv:
        return "n/a"
    return "high" if sdv >= ref["p95"] - 0.02 else "moderate" if sdv >= ref["p05"] + 0.03 else "low"
