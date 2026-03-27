import re
import math
from pathlib import Path
from typing import Dict, List, Any, Optional

import pandas as pd
import numpy as np


# =========================
# Config
# =========================
LOG_PATH = "D:\\Desktop\\Link\\baseline1\\socket_run_log.txt"   # 改成你的日志路径
OUT_DIR = "D:\\Desktop\\Link\\baseline1"      # 输出目录


# =========================
# Utility
# =========================
def to_number(s: Optional[str]) -> Optional[float]:
    if s is None:
        return None
    s = s.strip()
    if s == "":
        return None
    try:
        return float(s)
    except Exception:
        return None


def safe_int_from_floatlike(s: Optional[str]) -> Optional[int]:
    v = to_number(s)
    if v is None or math.isnan(v):
        return None
    return int(round(v))


def parse_status_token(s: Optional[str]) -> Optional[str]:
    if s is None:
        return None
    return s.strip()


def block_search(pattern: str, text: str, flags=re.MULTILINE | re.DOTALL):
    m = re.search(pattern, text, flags)
    return m.groups() if m else None


def extract_first(pattern: str, text: str, cast=None, flags=re.MULTILINE):
    m = re.search(pattern, text, flags)
    if not m:
        return None
    val = m.group(1)
    if cast is None:
        return val
    try:
        return cast(val)
    except Exception:
        return None


def describe_series(s: pd.Series) -> Dict[str, Any]:
    s = pd.to_numeric(s, errors="coerce").dropna()
    if len(s) == 0:
        return {
            "count": 0,
            "mean": np.nan,
            "std": np.nan,
            "min": np.nan,
            "q25": np.nan,
            "median": np.nan,
            "q75": np.nan,
            "max": np.nan,
        }
    return {
        "count": int(s.count()),
        "mean": s.mean(),
        "std": s.std(ddof=1) if len(s) > 1 else 0.0,
        "min": s.min(),
        "q25": s.quantile(0.25),
        "median": s.median(),
        "q75": s.quantile(0.75),
        "max": s.max(),
    }


# =========================
# Parsing core
# =========================
SCRIPT_BLOCK_RE = re.compile(
    r"=+\n"
    r"Script:\s*(?P<script>[^\n]+)\n"
    r"Seed:\s*(?P<seed>\d+)\n"
    r"Return code:\s*(?P<return_code>-?\d+)\n"
    r"Finished at:\s*(?P<finished_at>[^\n]+)\n"
    r"Duration seconds:\s*(?P<duration>[0-9.eE+-]+)\n"
    r"-+\n"
    r"(?P<body>.*?)"
    r"(?=(?:\n=+\nScript:)|(?:\n#+\nSeed batch)|\Z)",
    re.DOTALL
)

PARAM_LINE_RE = re.compile(
    r"^\s*([A-Za-z_]\w*)\s*=\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*$",
    re.MULTILINE
)

TOP_CAND_RE = re.compile(
    r"^\s*1\)\s*cost=([^\s,]+),\s*nfev=([^\s,]+),\s*status=([^\n]+)$",
    re.MULTILINE
)

GP_PEAK_RE = re.compile(
    r"GP structure peaks \((Re|Im|Phase)\):\s*([^\n]+)"
)

HEADER_SEED_RE = re.compile(
    r"Seed batch\s+\d+/\d+\s*\nSeed:\s*(\d+)",
    re.MULTILINE
)


def parse_optimized_parameters(body: str) -> Dict[str, float]:
    params = {}
    m = re.search(
        r"===== Optimized parameters =====\s*(.*?)\s*(?:(?:===== Top-5 candidates \(by cost\) =====)|(?:\[STDERR\])|(?:\Z))",
        body,
        re.DOTALL
    )
    if not m:
        return params

    param_block = m.group(1)
    for name, value in PARAM_LINE_RE.findall(param_block):
        params[name] = to_number(value)
    return params


def parse_top1_candidate(body: str) -> Dict[str, Any]:
    out = {
        "top1_cost": None,
        "top1_nfev": None,
        "top1_status": None,
    }
    m = TOP_CAND_RE.search(body)
    if m:
        out["top1_cost"] = to_number(m.group(1))
        out["top1_nfev"] = safe_int_from_floatlike(m.group(2))
        out["top1_status"] = parse_status_token(m.group(3))
    return out


def parse_complexity_metrics(body: str) -> Dict[str, Any]:
    keys = {
        "p": r"^\s*p\s*=\s*([0-9.eE+-]+)\s*$",
        "N_freq_fit": r"N_freq_fit\s*=\s*([0-9.eE+-]+)",
        "N_residual_dim": r"N_residual_dim\s*=\s*([0-9.eE+-]+)",
        "model_eval": r"model_eval\s*=\s*([0-9.eE+-]+)",
        "residual_calls": r"residual_calls\s*=\s*([0-9.eE+-]+)",
        "objective_calls": r"objective_calls\s*=\s*([0-9.eE+-]+)",
        "T_fit_total": r"T_fit_total\s*=\s*([0-9.eE+-]+)\s*s",
        "T_global": r"T_global\s*=\s*([0-9.eE+-]+)\s*s",
        "T_local": r"T_local\s*=\s*([0-9.eE+-]+)\s*s",
        "de_popsize": r"de_popsize\s*=\s*([0-9.eE+-]+)",
        "de_maxiter": r"de_maxiter\s*=\s*([0-9.eE+-]+)",
        "n_starts": r"n_starts\s*=\s*([0-9.eE+-]+)",
        "top_k": r"top_k\s*=\s*([0-9.eE+-]+)",
    }

    out = {}
    for k, pat in keys.items():
        val = extract_first(pat, body, cast=to_number)
        if k in {"p", "N_freq_fit", "N_residual_dim", "model_eval", "residual_calls",
                 "objective_calls", "de_popsize", "de_maxiter", "n_starts", "top_k"}:
            out[k] = safe_int_from_floatlike(str(val)) if val is not None else None
        else:
            out[k] = val
    return out


def parse_raw_metrics(body: str) -> Dict[str, Any]:
    out = {
        "SSE_raw": extract_first(r"SSE_raw\s*=\s*([0-9.eE+-]+)", body, cast=to_number),
        "RMSE_raw": extract_first(r"RMSE_raw\s*=\s*([0-9.eE+-]+)", body, cast=to_number),
        "AIC_raw": extract_first(r"AIC_raw\s*=\s*([0-9.eE+-]+)", body, cast=to_number),
        "BIC_raw": extract_first(r"BIC_raw\s*=\s*([0-9.eE+-]+)", body, cast=to_number),
        "n": extract_first(r"\bn\s*=\s*([0-9.eE+-]+)", body, cast=lambda x: safe_int_from_floatlike(x)),
    }
    return out


def parse_gp_peaks(body: str) -> Dict[str, Any]:
    out = {}
    for domain, values_str in GP_PEAK_RE.findall(body):
        vals = [v.strip().replace(" Hz", "") for v in values_str.split(",")]
        vals = [to_number(v) for v in vals[:3]]
        while len(vals) < 3:
            vals.append(None)
        prefix = domain.lower()  # re / im / phase
        out[f"gp_{prefix}_peak1_hz"] = vals[0]
        out[f"gp_{prefix}_peak2_hz"] = vals[1]
        out[f"gp_{prefix}_peak3_hz"] = vals[2]
    return out


def parse_ablation_name(body: str) -> Optional[str]:
    return extract_first(r"^\s*Ablation\s*=\s*(.+?)\s*$", body, cast=str, flags=re.MULTILINE)


def parse_one_block(m: re.Match) -> Dict[str, Any]:
    d = m.groupdict()
    body = d["body"]

    row = {
        "script": d["script"].strip(),
        "seed": int(d["seed"]),
        "return_code": int(d["return_code"]),
        "finished_at": d["finished_at"].strip(),
        "duration_seconds": to_number(d["duration"]),
        "ablation_name": parse_ablation_name(body),
    }

    row.update(parse_top1_candidate(body))
    row.update(parse_complexity_metrics(body))
    row.update(parse_raw_metrics(body))
    row.update(parse_gp_peaks(body))

    params = parse_optimized_parameters(body)
    for k, v in params.items():
        row[f"param_{k}"] = v

    return row


def parse_log(text: str) -> pd.DataFrame:
    rows = []
    for m in SCRIPT_BLOCK_RE.finditer(text):
        rows.append(parse_one_block(m))

    if not rows:
        raise ValueError("没有匹配到任何 Script 块，请检查日志格式或正则。")

    df = pd.DataFrame(rows)

    # 统一列顺序：基础列在前，其余列在后
    base_cols = [
        "script", "ablation_name", "seed", "return_code", "finished_at", "duration_seconds",
        "top1_cost", "top1_nfev", "top1_status",
        "p", "N_freq_fit", "N_residual_dim", "model_eval", "residual_calls", "objective_calls",
        "T_fit_total", "T_global", "T_local",
        "de_popsize", "de_maxiter", "n_starts", "top_k",
        "SSE_raw", "RMSE_raw", "AIC_raw", "BIC_raw", "n",
        "gp_re_peak1_hz", "gp_re_peak2_hz", "gp_re_peak3_hz",
        "gp_im_peak1_hz", "gp_im_peak2_hz", "gp_im_peak3_hz",
        "gp_phase_peak1_hz", "gp_phase_peak2_hz", "gp_phase_peak3_hz",
    ]
    param_cols = sorted([c for c in df.columns if c.startswith("param_")])
    other_cols = [c for c in df.columns if c not in base_cols and c not in param_cols]
    ordered_cols = [c for c in base_cols if c in df.columns] + other_cols + param_cols
    df = df[ordered_cols]

    return df


# =========================
# Summaries
# =========================
def summarize_by_script(df: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = [
        "duration_seconds", "top1_cost", "top1_nfev",
        "model_eval", "residual_calls", "objective_calls",
        "T_fit_total", "T_global", "T_local",
        "SSE_raw", "RMSE_raw", "AIC_raw", "BIC_raw"
    ]

    rows = []
    for script, g in df.groupby("script", dropna=False):
        row = {
            "script": script,
            "ablation_name": g["ablation_name"].dropna().iloc[0] if g["ablation_name"].notna().any() else None,
            "n_runs": len(g),
            "n_success_returncode0": int((g["return_code"] == 0).sum()),
            "success_rate_returncode0": float((g["return_code"] == 0).mean()),
            "n_unique_seeds": g["seed"].nunique(),
        }

        if "top1_status" in g.columns:
            status_counts = g["top1_status"].astype(str).value_counts(dropna=False).to_dict()
            for k, v in status_counts.items():
                row[f"top1_status_count__{k}"] = v

        for col in numeric_cols:
            if col in g.columns:
                stats = describe_series(g[col])
                for stat_name, value in stats.items():
                    row[f"{col}__{stat_name}"] = value

        rows.append(row)

    out = pd.DataFrame(rows).sort_values("script").reset_index(drop=True)
    return out


def summarize_parameters_by_script(df: pd.DataFrame) -> pd.DataFrame:
    param_cols = [c for c in df.columns if c.startswith("param_")]
    rows = []

    for script, g in df.groupby("script", dropna=False):
        for col in param_cols:
            stats = describe_series(g[col])
            row = {
                "script": script,
                "parameter": col.replace("param_", ""),
            }
            row.update(stats)
            rows.append(row)

    out = pd.DataFrame(rows).sort_values(["script", "parameter"]).reset_index(drop=True)
    return out


def summarize_gp_peaks_by_script(df: pd.DataFrame) -> pd.DataFrame:
    peak_cols = [c for c in df.columns if c.startswith("gp_") and c.endswith("_hz")]
    rows = []

    for script, g in df.groupby("script", dropna=False):
        for col in peak_cols:
            stats = describe_series(g[col])
            row = {
                "script": script,
                "gp_peak": col,
            }
            row.update(stats)
            rows.append(row)

    out = pd.DataFrame(rows).sort_values(["script", "gp_peak"]).reset_index(drop=True)
    return out


# =========================
# Main
# =========================
def main():
    log_path = Path(LOG_PATH)
    out_dir = Path(OUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    text = log_path.read_text(encoding="utf-8", errors="replace")
    df = parse_log(text)

    summary_df = summarize_by_script(df)
    param_summary_df = summarize_parameters_by_script(df)
    gp_summary_df = summarize_gp_peaks_by_script(df)

    # 额外生成一个紧凑版主表，便于看主要结论
    compact_cols = [
        "script", "ablation_name", "seed", "return_code",
        "duration_seconds", "top1_cost", "top1_nfev", "top1_status",
        "model_eval", "residual_calls", "objective_calls",
        "T_fit_total", "T_global", "T_local",
        "SSE_raw", "RMSE_raw", "AIC_raw", "BIC_raw",
    ]
    compact_cols = [c for c in compact_cols if c in df.columns]
    compact_df = df[compact_cols].copy()

    # 保存
    df.to_csv(out_dir / "parsed_runs.csv", index=False, encoding="utf-8-sig")
    compact_df.to_csv(out_dir / "parsed_runs_compact.csv", index=False, encoding="utf-8-sig")
    summary_df.to_csv(out_dir / "summary_by_script.csv", index=False, encoding="utf-8-sig")
    param_summary_df.to_csv(out_dir / "parameter_summary_by_script.csv", index=False, encoding="utf-8-sig")
    gp_summary_df.to_csv(out_dir / "gp_peak_summary_by_script.csv", index=False, encoding="utf-8-sig")

    # 控制台打印一点结果
    print("=" * 80)
    print("Parsed runs:")
    print(df[["script", "seed", "return_code"]].head(10).to_string(index=False))
    print("=" * 80)
    print("Summary by script:")
    with pd.option_context("display.max_columns", 200, "display.width", 200):
        print(summary_df.to_string(index=False))
    print("=" * 80)
    print(f"输出目录: {out_dir.resolve()}")
    print("已生成文件：")
    print(" - parsed_runs.csv")
    print(" - parsed_runs_compact.csv")
    print(" - summary_by_script.csv")
    print(" - parameter_summary_by_script.csv")
    print(" - gp_peak_summary_by_script.csv")


if __name__ == "__main__":
    main()