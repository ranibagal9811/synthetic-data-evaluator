"""Synthesizer wrappers (SDV). Version-tolerant: handles the SDV API changes met during the study."""
import os

import numpy as np
import pandas as pd

TABLE = "table"


def sdv_version():
    try:
        from importlib.metadata import version
        return version("sdv")
    except Exception:
        return "not installed"


def get_metadata(df, target):
    md, tname = None, None
    from sdv.metadata import Metadata  # SDV >= 1.2x
    for name, call in (("detect_from_dataframe", lambda: Metadata.detect_from_dataframe(df, table_name=TABLE)),
                       ("detect_from_dataframes", lambda: Metadata.detect_from_dataframes({TABLE: df}))):
        if hasattr(Metadata, name):
            try:
                md, tname = call(), TABLE
                break
            except Exception:
                continue
    if md is None:
        from sdv.metadata import SingleTableMetadata
        md = SingleTableMetadata()
        md.detect_from_dataframe(df)
    for kwargs in ({"column_name": target, "sdtype": "categorical", "table_name": tname},
                   {"column_name": target, "sdtype": "categorical"}):
        if tname is None and "table_name" in kwargs:
            continue
        try:
            md.update_column(**kwargs)
            break
        except TypeError:
            continue
    return md


def _dict_err(e):
    return "dictionary" in str(e) and "table" in str(e)


def fit_and_sample(model, train, metadata, epochs, seed, n_out, use_gpu=False):
    """model: 'copula' | 'ctgan'.  DASHBOARD_TEST_MODE=1 swaps in a trivial sampler (unit tests only)."""
    if os.environ.get("DASHBOARD_TEST_MODE") == "1":
        rng = np.random.RandomState(seed)
        x = rng.multivariate_normal(train.mean().values, np.cov(train.values.T), n_out)
        s = pd.DataFrame(x, columns=train.columns)
        for c in train.columns:
            if train[c].nunique() <= 2:
                s[c] = (s[c] > train[c].mean()).astype(int)
        return s
    import inspect

    import torch
    from sdv.single_table import CTGANSynthesizer, GaussianCopulaSynthesizer
    np.random.seed(seed)
    torch.manual_seed(seed)
    if model == "copula":
        synth = GaussianCopulaSynthesizer(metadata)
    else:
        params = inspect.signature(CTGANSynthesizer.__init__).parameters
        kw = dict(epochs=int(epochs), verbose=False)
        if "cuda" in params:
            kw["cuda"] = use_gpu
        elif "enable_gpu" in params:
            kw["enable_gpu"] = use_gpu
        synth = CTGANSynthesizer(metadata, **kw)
    try:
        synth.fit(train)
    except Exception as e:
        if not _dict_err(e):
            raise
        synth.fit({TABLE: train})
    np.random.seed(seed)
    torch.manual_seed(seed)
    out = synth.sample(num_rows=n_out)
    return next(iter(out.values())) if isinstance(out, dict) else out
