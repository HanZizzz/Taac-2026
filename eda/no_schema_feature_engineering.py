#!/usr/bin/env python3
"""Build leakage-aware sidecar features without editing schema.json.

The script reads TAAC2026 flat-layout parquet files and writes a sidecar
parquet table keyed by row order.  It intentionally does not mutate the
competition parquet files and does not require changing schema.json.

Typical use:
    python eda/no_schema_feature_engineering.py \
        --input-dir /path/to/train_parquet_dir \
        --schema-path /path/to/schema.json \
        --output-path /tmp/taac_sidecar_features.parquet

The generated features are intended for EDA, GBDT baselines, feature ablation,
or later runtime fusion.  If you use frequency/count features for validation or
test, fit the statistics on the training side only and transform holdout/test
with the frozen table to avoid leakage.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

np = None
pd = None


def _ensure_deps() -> None:
    """Import optional runtime dependencies after CLI parsing.

    Keeping these imports lazy allows `--help` and `py_compile` to work in
    minimal environments while still failing clearly when feature generation is
    actually requested without pandas/numpy installed.
    """
    global np, pd
    import numpy as _np
    import pandas as _pd

    np = _np
    pd = _pd


WINDOW_SECONDS: Tuple[int, ...] = (
    5 * 60,
    60 * 60,
    6 * 60 * 60,
    24 * 60 * 60,
    3 * 24 * 60 * 60,
    7 * 24 * 60 * 60,
    30 * 24 * 60 * 60,
)


def _as_list(value: Any) -> List[Any]:
    """Return a Python list for scalar/list/ndarray/null parquet cells."""
    if value is None:
        return []
    if isinstance(value, float) and np.isnan(value):
        return []
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _valid_values(value: Any) -> List[int]:
    """Return positive integer values; TAAC uses <=0 as padding/missing."""
    out: List[int] = []
    for x in _as_list(value):
        if x is None:
            continue
        try:
            xi = int(x)
        except (TypeError, ValueError):
            continue
        if xi > 0:
            out.append(xi)
    return out


def _is_missing_cell(value: Any) -> bool:
    """Treat null/NaN/empty list as missing for sidecar feature purposes."""
    if value is None:
        return True
    if isinstance(value, float) and np.isnan(value):
        return True
    if isinstance(value, np.ndarray):
        return len(value) == 0
    if isinstance(value, (list, tuple)):
        return len(value) == 0
    return False


def _read_parquet(input_path: Path) -> pd.DataFrame:
    if input_path.is_dir():
        files = sorted(input_path.glob("*.parquet"))
        if not files:
            raise FileNotFoundError(f"No parquet files found under {input_path}")
        return pd.concat((pd.read_parquet(p) for p in files), ignore_index=True)
    return pd.read_parquet(input_path)


def _discover_columns(df: pd.DataFrame) -> Dict[str, List[str]]:
    cols = list(df.columns)
    return {
        "user_int": sorted([c for c in cols if c.startswith("user_int_feats_")], key=_fid_sort_key),
        "user_dense": sorted([c for c in cols if c.startswith("user_dense_feats_")], key=_fid_sort_key),
        "item_int": sorted([c for c in cols if c.startswith("item_int_feats_")], key=_fid_sort_key),
        "seq_a": sorted([c for c in cols if c.startswith("domain_a_seq_")], key=_fid_sort_key),
        "seq_b": sorted([c for c in cols if c.startswith("domain_b_seq_")], key=_fid_sort_key),
        "seq_c": sorted([c for c in cols if c.startswith("domain_c_seq_")], key=_fid_sort_key),
        "seq_d": sorted([c for c in cols if c.startswith("domain_d_seq_")], key=_fid_sort_key),
    }


def _fid_sort_key(name: str) -> int:
    return int(name.rsplit("_", 1)[1])


def _load_seq_timestamp_columns(schema_path: Optional[Path]) -> Dict[str, str]:
    """Map seq_a/seq_b/... to timestamp column names using schema.json.

    This reads schema metadata but does not modify it.  If schema_path is not
    provided or a domain has no ts_fid, time-window features are skipped for
    that domain.
    """
    if schema_path is None:
        return {}
    raw = json.loads(schema_path.read_text(encoding="utf-8"))
    mapping: Dict[str, str] = {}
    for domain, cfg in raw.get("seq", {}).items():
        ts_fid = cfg.get("ts_fid")
        prefix = cfg.get("prefix")
        if ts_fid is None or prefix is None:
            continue
        mapping[domain] = f"{prefix}_{ts_fid}"
    return mapping


def _missing_features(df: pd.DataFrame, columns: Mapping[str, Sequence[str]]) -> pd.DataFrame:
    result: Dict[str, Any] = {}
    for group in ("user_int", "user_dense", "item_int"):
        miss_cols = []
        for col in columns[group]:
            name = f"fe_{col}__missing"
            values = df[col].map(_is_missing_cell).astype("int8")
            result[name] = values
            miss_cols.append(name)
        if miss_cols:
            stacked = pd.concat([result[c] for c in miss_cols], axis=1)
            result[f"fe_{group}__missing_cnt"] = stacked.sum(axis=1).astype("int16")
            result[f"fe_{group}__missing_ratio"] = stacked.mean(axis=1).astype("float32")
    return pd.DataFrame(result, index=df.index)


def _multi_value_features(df: pd.DataFrame, columns: Sequence[str], prefix: str) -> pd.DataFrame:
    result: Dict[str, Any] = {}
    for col in columns:
        vals = df[col].map(_valid_values)
        lengths = vals.map(len)
        uniques = vals.map(lambda xs: len(set(xs)))
        result[f"fe_{col}__len"] = lengths.astype("int16")
        result[f"fe_{col}__non_empty"] = (lengths > 0).astype("int8")
        result[f"fe_{col}__unique_cnt"] = uniques.astype("int16")
        result[f"fe_{col}__dup_ratio"] = [
            np.float32(0.0 if n == 0 else 1.0 - (u / n))
            for n, u in zip(lengths, uniques)
        ]
        result[f"fe_{col}__first"] = vals.map(lambda xs: xs[0] if xs else 0).astype("int64")
        result[f"fe_{col}__top1"] = vals.map(_top1_value).astype("int64")
    if columns:
        len_cols = [f"fe_{c}__len" for c in columns]
        len_df = pd.DataFrame({c: result[c] for c in len_cols}, index=df.index)
        result[f"fe_{prefix}__total_list_len"] = len_df.sum(axis=1).astype("int32")
        result[f"fe_{prefix}__non_empty_list_cnt"] = (len_df > 0).sum(axis=1).astype("int16")
    return pd.DataFrame(result, index=df.index)


def _top1_value(values: Sequence[int]) -> int:
    if not values:
        return 0
    return Counter(values).most_common(1)[0][0]


def _sequence_features(
    df: pd.DataFrame,
    columns: Mapping[str, Sequence[str]],
    seq_timestamp_cols: Mapping[str, str],
) -> pd.DataFrame:
    result: Dict[str, Any] = {}
    domain_len_cols: List[str] = []
    for domain in ("seq_a", "seq_b", "seq_c", "seq_d"):
        seq_cols = list(columns[domain])
        if not seq_cols:
            continue
        per_col_lengths = []
        for col in seq_cols:
            lengths = df[col].map(lambda x: len(_valid_values(x))).astype("int32")
            result[f"fe_{col}__len"] = lengths
            result[f"fe_{col}__non_empty"] = (lengths > 0).astype("int8")
            result[f"fe_{col}__unique_cnt"] = df[col].map(lambda x: len(set(_valid_values(x)))).astype("int32")
            per_col_lengths.append(lengths)

        # Domain length is the max side-info length in the domain, matching the
        # training dataset's length construction.
        domain_lengths = pd.concat(per_col_lengths, axis=1).max(axis=1).astype("int32")
        domain_len_name = f"fe_{domain}__len"
        domain_len_cols.append(domain_len_name)
        result[domain_len_name] = domain_lengths
        result[f"fe_{domain}__log1p_len"] = np.log1p(domain_lengths).astype("float32")
        result[f"fe_{domain}__non_empty"] = (domain_lengths > 0).astype("int8")

        ts_col = seq_timestamp_cols.get(domain)
        if ts_col and ts_col in df.columns and "timestamp" in df.columns:
            _add_time_window_features(result, df, domain, ts_col)

    if domain_len_cols:
        len_df = pd.DataFrame({c: result[c] for c in domain_len_cols}, index=df.index)
        total_len = len_df.sum(axis=1).replace(0, np.nan)
        result["fe_all_seq__total_len"] = len_df.sum(axis=1).astype("int32")
        result["fe_all_seq__max_len"] = len_df.max(axis=1).astype("int32")
        result["fe_all_seq__non_empty_domain_cnt"] = (len_df > 0).sum(axis=1).astype("int8")
        for c in domain_len_cols:
            ratio_name = c.replace("__len", "__len_ratio")
            result[ratio_name] = (len_df[c] / total_len).fillna(0).astype("float32")
    return pd.DataFrame(result, index=df.index)


def _add_time_window_features(result: Dict[str, Any], df: pd.DataFrame, domain: str, ts_col: str) -> None:
    label_ts = df["timestamp"].astype("int64").to_numpy()
    seq_ts = df[ts_col].map(_valid_values)
    gaps: List[np.ndarray] = []
    for now, values in zip(label_ts, seq_ts):
        if not values:
            gaps.append(np.asarray([], dtype=np.int64))
            continue
        arr = np.asarray(values, dtype=np.int64)
        gaps.append(np.maximum(now - arr, 0))

    result[f"fe_{domain}__last_gap"] = np.asarray(
        [int(g.min()) if len(g) else -1 for g in gaps], dtype=np.int64
    )
    result[f"fe_{domain}__mean_gap"] = np.asarray(
        [np.float32(g.mean()) if len(g) else np.float32(-1.0) for g in gaps], dtype=np.float32
    )
    result[f"fe_{domain}__std_gap"] = np.asarray(
        [np.float32(g.std()) if len(g) else np.float32(0.0) for g in gaps], dtype=np.float32
    )
    for window in WINDOW_SECONDS:
        result[f"fe_{domain}__cnt_{window}s"] = np.asarray(
            [int((g <= window).sum()) for g in gaps], dtype=np.int32
        )
    cnt_1h = result.get(f"fe_{domain}__cnt_{60 * 60}s")
    cnt_7d = result.get(f"fe_{domain}__cnt_{7 * 24 * 60 * 60}s")
    if cnt_1h is not None and cnt_7d is not None:
        result[f"fe_{domain}__burst_1h_over_7d"] = (
            np.asarray(cnt_1h, dtype=np.float32) / np.maximum(np.asarray(cnt_7d, dtype=np.float32), 1.0)
        ).astype(np.float32)


def _frequency_features(df: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    """In-file frequency features for EDA only.

    For production validation/test, compute counts on the training period and
    apply the frozen mapping to holdout/test to avoid target leakage through
    future availability.
    """
    result: Dict[str, Any] = {}
    for col in columns:
        if col not in df.columns:
            continue
        counts = df[col].value_counts(dropna=False)
        freq = df[col].map(counts).fillna(0).astype("int32")
        result[f"fe_{col}__freq_in_input"] = freq
        result[f"fe_{col}__log1p_freq_in_input"] = np.log1p(freq).astype("float32")
    return pd.DataFrame(result, index=df.index)


def build_sidecar_features(df: pd.DataFrame, schema_path: Optional[Path] = None) -> pd.DataFrame:
    if np is None or pd is None:
        _ensure_deps()
    columns = _discover_columns(df)
    seq_ts_cols = _load_seq_timestamp_columns(schema_path)

    frames = [
        pd.DataFrame({"row_id": np.arange(len(df), dtype=np.int64)}, index=df.index),
        _missing_features(df, columns),
        _multi_value_features(
            df,
            [
                c for c in columns["user_int"] + columns["item_int"]
                if df[c].map(lambda x: isinstance(x, (list, tuple, np.ndarray))).any()
            ],
            "multi_int",
        ),
        _sequence_features(df, columns, seq_ts_cols),
        _frequency_features(df, ["user_id", "item_id"]),
    ]
    if "label_type" in df.columns:
        frames.append(pd.DataFrame({"label": (df["label_type"].fillna(0).astype("int64") == 2).astype("int8")}, index=df.index))
    return pd.concat(frames, axis=1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build TAAC2026 no-schema-change sidecar features")
    parser.add_argument("--input-dir", type=Path, required=True, help="Input parquet file or directory")
    parser.add_argument("--schema-path", type=Path, default=None, help="Optional schema.json path, read-only")
    parser.add_argument("--output-path", type=Path, required=True, help="Output sidecar parquet path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _ensure_deps()
    df = _read_parquet(args.input_dir)
    features = build_sidecar_features(df, args.schema_path)
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(args.output_path, index=False)
    print(f"Wrote {features.shape[1]} sidecar features for {features.shape[0]} rows to {args.output_path}")


if __name__ == "__main__":
    main()


"""
python eda/no_schema_feature_engineering.py \
  --input-dir ../industrual_data/ \
  --schema-path /path/to/schema.json \
  --output-path /tmp/taac_sidecar_features.parquet
"""
