"""
Shared utilities for the stroke + SLP timing study.

Keep manuscript-output scripts thin: database paths, output paths, the
J18/J69 aspiration-related pneumonia derivation, and the time-varying Cox
implementation live here.
"""

import os
from pathlib import Path

try:
    from dotenv import load_dotenv as _load_dotenv
except ModuleNotFoundError:
    def _load_dotenv(path):
        import re

        path = Path(path)
        if not path.exists():
            return False
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            value = re.sub(
                r"\$\{([^}]+)\}",
                lambda match: os.environ.get(match.group(1), ""),
                value,
            )
            os.environ.setdefault(key, value)
        return True


def find_project_root(start):
    """Return the nearest ancestor containing .env."""
    for parent in [Path(start).resolve(), *Path(start).resolve().parents]:
        if (parent / ".env").exists():
            return parent
    return Path(__file__).resolve().parents[3]


PROJECT_DIR = Path(__file__).resolve().parents[1]
ROOT_DIR = find_project_root(PROJECT_DIR)
_load_dotenv(ROOT_DIR / ".env")

DB_PATH = Path(os.getenv("duckdb_database", "cms_data.duckdb"))
OUT_DIR = Path(os.getenv("project_paths", PROJECT_DIR.parent)) / "stroke_SLP" / "output_files"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MAX_FOLLOW = 365
TV_COVARIATES = ["age_at_adm", "van_walraven_score", "index_los"]


def connect(read_only=True):
    import duckdb

    con = duckdb.connect(str(DB_PATH), read_only=read_only)
    con.execute("SET memory_limit='24GB'; SET threads=12;")
    return con


def add_aspiration_related_pna(df):
    """Add the primary J18/J69 aspiration-related pneumonia event-day column."""
    import numpy as np

    df = df.copy()
    df["days_to_asp_related"] = np.where(
        df["first_pneumonia_code"].isin(["J18", "J69"]),
        df["days_to_pneumonia"],
        np.nan,
    )
    return df


def standardize_covariates(df, covariates=TV_COVARIATES):
    df = df.copy()
    for col in covariates:
        if col not in df.columns:
            continue
        sd = df[col].std()
        if sd and sd > 0:
            df[col] = (df[col] - df[col].mean()) / sd
    return df


def build_time_varying_df(
    df,
    event_col,
    competing_col,
    treat_grp,
    group_col="slp_timing_group",
    covariates=TV_COVARIATES,
    max_follow=MAX_FOLLOW,
):
    """Split person-time at days_to_slp_outpt for TV Cox models."""
    import numpy as np
    import pandas as pd

    records = []
    for _, row in df.iterrows():
        slp_day = float(row["days_to_slp_outpt"])
        ev_day = row[event_col]
        comp_day = row[competing_col] if competing_col and pd.notna(row[competing_col]) else np.nan

        candidates = [float(max_follow)]
        if pd.notna(ev_day):
            candidates.append(float(ev_day))
        if pd.notna(comp_day):
            candidates.append(float(comp_day))
        end_time = min(candidates)

        final_event = int(
            pd.notna(ev_day)
            and float(ev_day) <= max_follow
            and float(ev_day) == end_time
        )
        group_flag = 1 if row[group_col] == treat_grp else 0
        base = {
            col: float(row[col]) if pd.notna(row[col]) else 0.0
            for col in covariates
        }

        if end_time <= slp_day:
            records.append({
                "id": row["DSYSRTKY"],
                "start": 0.0,
                "stop": max(end_time, 0.5),
                "trt": 0,
                "event": final_event,
                **base,
            })
        else:
            if slp_day > 0:
                records.append({
                    "id": row["DSYSRTKY"],
                    "start": 0.0,
                    "stop": slp_day,
                    "trt": 0,
                    "event": 0,
                    **base,
                })
            records.append({
                "id": row["DSYSRTKY"],
                "start": slp_day,
                "stop": max(end_time, slp_day + 0.5),
                "trt": group_flag,
                "event": final_event,
                **base,
            })

    return pd.DataFrame(records)


def run_tv_cox(
    df,
    event_col,
    competing_col,
    treat_grp,
    group_col="slp_timing_group",
    covariates=TV_COVARIATES,
    max_follow=MAX_FOLLOW,
    min_events=10,
    penalizer=None,
):
    """
    Fit the shared time-varying Cox model.

    Returns: ((hr, lo95, hi95, p), n_patients, n_events). The result tuple is
    None when there are too few events or model fitting fails.
    """
    tv = build_time_varying_df(
        df,
        event_col,
        competing_col,
        treat_grp,
        group_col=group_col,
        covariates=covariates,
        max_follow=max_follow,
    )
    if tv.empty:
        return None, 0, 0

    tv = tv.dropna(subset=covariates)
    tv = standardize_covariates(tv, covariates)
    n_pts = tv["id"].nunique()
    n_evts = int(tv["event"].sum())
    if n_evts < min_events:
        return None, n_pts, n_evts

    try:
        import numpy as np
        from lifelines import CoxTimeVaryingFitter

        kwargs = {} if penalizer is None else {"penalizer": penalizer}
        ctv = CoxTimeVaryingFitter(**kwargs)
        ctv.fit(
            tv,
            id_col="id",
            start_col="start",
            stop_col="stop",
            event_col="event",
            show_progress=False,
        )
        row = ctv.summary.loc["trt"]
        result = (
            np.exp(row["coef"]),
            np.exp(row["coef lower 95%"]),
            np.exp(row["coef upper 95%"]),
            row["p"],
        )
        return result, n_pts, n_evts
    except Exception as exc:
        print(f"    Cox failed: {exc}")
        return None, n_pts, n_evts
