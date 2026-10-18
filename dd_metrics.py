"""The 7-metric evaluation framework, generalised to any numeric tabular dataset with a binary target.
Definitions are identical to the controlled-variable study (controlled_variable_study.py)."""
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.neighbors import NearestNeighbors
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def _auc(y, p):
    try:
        return float(roc_auc_score(y, p))
    except ValueError:
        return float("nan")


def sdv_quality(real, synth, metadata):
    """SDV quality score. Returns (overall, column_shapes, pair_trends, error_message)."""
    try:
        try:
            from sdv.evaluation.single_table import evaluate_quality
        except ImportError:
            from sdv.evaluation import evaluate_quality  # type: ignore
        try:
            rep = evaluate_quality(real, synth, metadata, verbose=False)
        except Exception as e1:
            if "dictionary" not in str(e1):
                raise
            rep = evaluate_quality({"table": real}, {"table": synth}, metadata, verbose=False)
        props = rep.get_properties().set_index("Property")["Score"]
        return (float(rep.get_score()), float(props.get("Column Shapes", np.nan)),
                float(props.get("Column Pair Trends", np.nan)), None)
    except Exception as e:  # pragma: no cover
        return None, None, None, f"{type(e).__name__}: {str(e)[:160]}"


def fidelity_and_utility(real, synth, holdout, features, target, metadata, seed):
    out = {}
    ks = [stats.ks_2samp(real[c], synth[c]) for c in features]
    out["ks_stat_mean"] = float(np.mean([k.statistic for k in ks]))
    out["ks_pass_rate"] = float(np.mean([k.pvalue > 0.05 for k in ks]))
    out["ks_per_column"] = {c: float(k.statistic) for c, k in zip(features, ks)}
    table = np.array([[(real[target] == v).sum() for v in (0, 1)], [(synth[target] == v).sum() for v in (0, 1)]])
    try:
        p = float(stats.chi2_contingency(table)[1])
    except ValueError:
        p = 0.0
    out["chi2_p"], out["chi2_pass"] = p, float(p > 0.05)
    out["pos_rate_real"], out["pos_rate_synth"] = float((real[target] == 1).mean()), float((synth[target] == 1).mean())
    sdv, cs, pt, err = sdv_quality(real, synth, metadata)
    out.update(sdv_overall=sdv, sdv_column_shapes=cs, sdv_pair_trends=pt, sdv_error=err)
    Xh, yh = holdout[features], holdout[target]
    for tag, tr in (("trtr", real), ("tstr", synth)):
        if tr[target].nunique() < 2:
            acc = f1 = auc = float("nan")
        else:
            rf = RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=-1).fit(tr[features], tr[target])
            acc, f1, auc = (accuracy_score(yh, rf.predict(Xh)), f1_score(yh, rf.predict(Xh)),
                            _auc(yh, rf.predict_proba(Xh)[:, 1]))
        out[f"{tag}_acc"], out[f"{tag}_f1"], out[f"{tag}_auc"] = float(acc), float(f1), auc
    out["tstr_auc_gap"] = out["trtr_auc"] - out["tstr_auc"]
    return out


def privacy(real, synth, holdout, features, target, seed):
    out = {}
    cols = features + [target]
    sc = MinMaxScaler().fit(real[cols])
    R, S, H = (sc.transform(d[cols]) for d in (real, synth, holdout))

    # (5) DCR
    nn_real = NearestNeighbors(n_neighbors=1).fit(R)
    d_syn = nn_real.kneighbors(S)[0][:, 0]
    d_base = nn_real.kneighbors(H)[0][:, 0]
    out["dcr_synth_mean"], out["dcr_base_mean"] = float(d_syn.mean()), float(d_base.mean())
    out["dcr_ratio"] = float(d_syn.mean() / max(d_base.mean(), 1e-12))
    out["dcr_pct_near"] = float((d_syn < np.percentile(d_base, 5)).mean())
    out["exact_copy_share"] = float((d_syn < 1e-9).mean())
    out["dcr_synth_values"], out["dcr_base_values"] = d_syn, d_base

    rng = np.random.RandomState(seed)
    m = len(H)
    R_bal = R[rng.choice(len(R), size=min(m, len(R)), replace=False)]
    H_bal = H[:len(R_bal)]

    # (6) Attribute disclosure: ~80% of columns known, ~20% guessed
    n_hidden = max(1, int(round(0.2 * len(cols))))
    sig = real[cols].std().values
    raw_scale = np.where(sc.data_max_ - sc.data_min_ == 0, 1, sc.data_max_ - sc.data_min_)
    categorical = [real[c].nunique() <= 2 for c in cols]
    succ = {"member": [], "non": []}
    for _ in range(5):
        hidden = rng.choice(len(cols), size=n_hidden, replace=False)
        known = np.setdiff1d(np.arange(len(cols)), hidden)
        nn_s = NearestNeighbors(n_neighbors=1).fit(S[:, known])
        for name, Q in (("member", R_bal), ("non", H_bal)):
            idx = nn_s.kneighbors(Q[:, known])[1][:, 0]
            guess, true = S[idx][:, hidden], Q[:, hidden]
            ok = []
            for j, h in enumerate(hidden):
                if categorical[h]:
                    ok.append(np.abs(guess[:, j] - true[:, j]) < 0.5)
                else:
                    ok.append(np.abs(guess[:, j] - true[:, j]) <= 0.25 * sig[h] / raw_scale[h])
            succ[name].append(np.mean(ok))
    out["ad_member"], out["ad_nonmember"] = float(np.mean(succ["member"])), float(np.mean(succ["non"]))
    out["ad_gap"] = out["ad_member"] - out["ad_nonmember"]

    # (7) MIA at three attacker strengths
    k = min(5, len(S))
    nn_syn = NearestNeighbors(n_neighbors=k).fit(S)
    idx_cols = [0, min(2, k - 1), k - 1]
    feat = lambda Q: nn_syn.kneighbors(Q)[0][:, idx_cols]
    X = np.vstack([feat(R_bal), feat(H_bal)])
    y = np.r_[np.ones(len(R_bal)), np.zeros(len(H_bal))]
    attackers = {
        "lr": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
        "rf": RandomForestClassifier(n_estimators=100, max_depth=5, random_state=seed, n_jobs=-1),
        "mlp": make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=500, random_state=seed)),
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    for name, clf in attackers.items():
        p = cross_val_predict(clf, X, y, cv=cv, method="predict_proba")[:, 1]
        out[f"mia_auc_{name}"] = _auc(y, p)
    out["mia_mean"] = float(np.nanmean([out[f"mia_auc_{a}"] for a in attackers]))
    out["mia_auc_max"] = float(np.nanmax([out[f"mia_auc_{a}"] for a in attackers]))
    return out


def evaluate_all(real, synth, holdout, features, target, metadata, seed):
    res = fidelity_and_utility(real, synth, holdout, features, target, metadata, seed)
    res.update(privacy(real, synth, holdout, features, target, seed))
    return res
