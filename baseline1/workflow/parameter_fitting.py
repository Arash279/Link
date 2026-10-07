from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from Parameter_Fitting import Csf as pf_csf
from Parameter_Fitting import Llr as pf_llr
from Parameter_Fitting import Lm as pf_lm
from Parameter_Fitting import fit_rr as pf_rr
from Parameter_Fitting import fr_fa as pf_fr_fa

from .config import ExtractionConfig, InitialGuessConfig, Paths
from .initial_params import DEFAULT_INITIAL_INPUTS, InitialParamInputs


def _median(values: list[float]) -> float:
    finite = [float(v) for v in values if np.isfinite(v)]
    return float(np.median(finite)) if finite else float("nan")


def _mean(values: list[float]) -> float:
    finite = [float(v) for v in values if np.isfinite(v)]
    return float(np.mean(finite)) if finite else float("nan")


def _prefer(value: float, fallback: float) -> float:
    return float(value) if np.isfinite(value) else float(fallback)


def build_resonance_report(paths: Paths, cfg: ExtractionConfig) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    groups = (
        ("y_short", cfg.y_short_tables),
        ("y_long", cfg.y_long_tables),
        ("delta", cfg.delta_tables),
    )
    with sqlite3.connect(str(paths.db_path)) as conn:
        for group, tables in groups:
            for table in tables:
                result = pf_fr_fa.process_table(conn, table)
                rows.append(
                    {
                        "group": group,
                        "table": table,
                        "fr_hz": result["fr"],
                        "zmax_ohm": result["Zmax"],
                        "fa_hz": result["fa"],
                        "zanti_ohm": result["Zanti"],
                        "fr_idx": result["fr_idx"],
                        "fa_idx": result["fa_idx"],
                        "search_lo_hz": result["search_lo"],
                        "search_hi_hz": result["search_hi"],
                    }
                )
    return pd.DataFrame(rows)


def build_leakage_report(paths: Paths, cfg: ExtractionConfig) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    groups = (
        ("y_short", cfg.y_short_tables, "Y"),
        ("y_long", cfg.y_long_tables, "Y"),
        ("delta", cfg.delta_tables, "Delta"),
    )
    with sqlite3.connect(str(paths.db_path)) as conn:
        for group, tables, topology in groups:
            for table in tables:
                rec, err = pf_llr.process_table(conn, table, topology, do_plot=False)
                if rec is None:
                    rows.append(
                        {
                            "group": group,
                            "table": table,
                            "topology": topology,
                            "status": err["status"] if err else "FAILED",
                        }
                    )
                    continue

                quality_ok = bool(
                    np.isfinite(rec["err"])
                    and rec["err"] <= pf_llr.QLTY_ERR_MAX
                    and np.isfinite(rec["span_dec"])
                    and rec["span_dec"] >= pf_llr.QLTY_SPAN_MIN
                )
                rows.append(
                    {
                        "group": group,
                        "table": table,
                        "topology": topology,
                        "status": rec["status"],
                        "f_lo_hz": rec["f_lo"],
                        "f_hi_hz": rec["f_hi"],
                        "span_dec": rec["span_dec"],
                        "n_points": rec["n_pts"],
                        "leq_h": rec["Leq_win"],
                        "l_sigma_h": rec["L_sigma"],
                        "err": rec["err"],
                        "quality_ok": quality_ok,
                    }
                )
    return pd.DataFrame(rows)


def build_csf_report(paths: Paths, cfg: ExtractionConfig) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    with sqlite3.connect(str(paths.db_path)) as conn:
        for table in cfg.single_phase_tables:
            freq, mag, phase = pf_csf.fetch_table(conn, table)
            cap_curve = pf_csf.to_Cp(freq, mag, phase)
            lf_win = pf_csf.find_platform_segmented(freq, cap_curve, phase, kind="LF")
            hf_win = pf_csf.find_platform_segmented(freq, cap_curve, phase, kind="HF")
            csf_lf_f, lf_err = pf_csf.summarize_Cp(cap_curve, lf_win)
            csf_hf_f, hf_err = pf_csf.summarize_Cp(cap_curve, hf_win)
            rows.append(
                {
                    "table": table,
                    "csf_lf_f": csf_lf_f,
                    "csf_hf_f": csf_hf_f,
                    "csf0_calc_f": csf_lf_f - 3.0 * csf_hf_f,
                    "lf_err": lf_err,
                    "hf_err": hf_err,
                    "lf_found": lf_win is not None,
                    "hf_found": hf_win is not None,
                }
            )
    return pd.DataFrame(rows)


def build_rr_report(paths: Paths, cfg: ExtractionConfig, lls_h: float, llr_h: float) -> pd.DataFrame:
    old_lls, old_llr = pf_rr.Lls, pf_rr.Llr
    pf_rr.Lls, pf_rr.Llr = lls_h, llr_h
    rows: list[dict[str, object]] = []
    try:
        for table in cfg.rr_tables:
            try:
                freq, z_meas = pf_rr.read_data(str(paths.db_path), table, cfg.rr_fmin_hz, cfg.rr_fmax_hz)
                rrs_ohm = float(pf_rr.fit_Rr(freq, z_meas))
                z_fit = pf_rr.Z_model(freq, rrs_ohm)
                mse = float(np.mean(np.abs(z_meas - z_fit) ** 2))
                rows.append(
                    {
                        "table": table,
                        "rrs_ohm": rrs_ohm,
                        "mse": mse,
                        "n_points": int(freq.size),
                        "status": "OK",
                    }
                )
            except Exception as exc:  # pragma: no cover - report row is enough
                rows.append({"table": table, "status": f"FAILED: {exc}"})
    finally:
        pf_rr.Lls, pf_rr.Llr = old_lls, old_llr
    return pd.DataFrame(rows)


def build_lm_report(paths: Paths, cfg: ExtractionConfig, lls_h: float) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for group, tables in (("primary", cfg.lm_primary_tables), ("check", cfg.lm_check_tables)):
        for table in tables:
            result = pf_lm.compute_group_lm(str(paths.db_path), table, lls_h)
            if result is None:
                rows.append({"group": group, "table": table, "status": "NO_BAND"})
                continue
            rows.append(
                {
                    "group": group,
                    "table": table,
                    "status": "OK",
                    "lm_median_h": result.lm_median_h,
                    "lm_mean_h": result.lm_mean_h,
                    "lm_std_h": result.lm_std_h,
                    "lm_iqr_h": result.lm_iqr_h,
                    "n_points": result.n_points,
                    "band_lo_hz": result.band_lo_hz,
                    "band_hi_hz": result.band_hi_hz,
                }
            )
    return pd.DataFrame(rows)


def assemble_initial_inputs(
    resonance: pd.DataFrame,
    leakage: pd.DataFrame,
    csf: pd.DataFrame,
    rr: pd.DataFrame,
    lm: pd.DataFrame,
    priors: InitialGuessConfig,
) -> InitialParamInputs:
    leak_ok = leakage[leakage.get("quality_ok", False) == True]
    ys = _median(leak_ok.loc[leak_ok["group"] == "y_short", "l_sigma_h"].tolist())
    yl = _median(leak_ok.loc[leak_ok["group"] == "y_long", "l_sigma_h"].tolist())
    dd = _median(leak_ok.loc[leak_ok["group"] == "delta", "l_sigma_h"].tolist())
    y_combined = _median([ys, yl])
    l_sigma_final = _median([y_combined, dd])
    lls_h = _prefer(0.5 * l_sigma_final, DEFAULT_INITIAL_INPUTS.lls_h)
    llr_h = _prefer(0.5 * l_sigma_final, DEFAULT_INITIAL_INPUTS.llr_h)

    res_y = resonance[resonance["group"] == "y_short"]
    fr_hz = _prefer(_median(res_y["fr_hz"].tolist()), DEFAULT_INITIAL_INPUTS.fr_hz)
    zmax_ohm = _prefer(_median(res_y["zmax_ohm"].tolist()), DEFAULT_INITIAL_INPUTS.zmax_ohm)
    fa_hz = _prefer(_median(res_y["fa_hz"].tolist()), DEFAULT_INITIAL_INPUTS.fa_hz)
    zanti_ohm = _prefer(_median(res_y["zanti_ohm"].tolist()), DEFAULT_INITIAL_INPUTS.zanti_ohm)

    csf_hf_f = _prefer(_median(csf["csf_hf_f"].tolist()), DEFAULT_INITIAL_INPUTS.csf_hf_f)
    csf_lf_f = _prefer(_median(csf["csf_lf_f"].tolist()), DEFAULT_INITIAL_INPUTS.csf_lf_f)

    rr_ok = rr[rr["status"] == "OK"]
    rrs_ohm = _prefer(
        _mean(rr_ok["rrs_ohm"].tolist()) if "rrs_ohm" in rr_ok.columns else float("nan"),
        DEFAULT_INITIAL_INPUTS.rrs_ohm,
    )

    lm_ok = lm[(lm["group"] == "primary") & (lm["status"] == "OK")]
    lm_h = _median(lm_ok["lm_median_h"].tolist()) if "lm_median_h" in lm_ok.columns else float("nan")
    if not np.isfinite(lm_h):
        lm_fallback = lm[lm["status"] == "OK"]
        lm_h = _median(lm_fallback["lm_median_h"].tolist()) if "lm_median_h" in lm_fallback.columns else float("nan")
    lm_h = _prefer(lm_h, DEFAULT_INITIAL_INPUTS.lm_h)

    return InitialParamInputs(
        hp=priors.hp,
        connection=priors.connection,
        lls_h=lls_h,
        llr_h=llr_h,
        lm_h=lm_h,
        rrs_ohm=rrs_ohm,
        csf_hf_f=csf_hf_f,
        csf_lf_f=csf_lf_f,
        fr_hz=fr_hz,
        zmax_ohm=zmax_ohm,
        fa_hz=fa_hz,
        zanti_ohm=zanti_ohm,
        nlls_init_h=priors.nlls_init_h,
        lad_h=priors.lad_h,
    )


def run_parameter_fitting_stage(
    paths: Paths,
    extraction_cfg: ExtractionConfig,
    priors: InitialGuessConfig,
) -> dict[str, object]:
    resonance = build_resonance_report(paths, extraction_cfg)
    leakage = build_leakage_report(paths, extraction_cfg)
    leak_ok = leakage[leakage.get("quality_ok", False) == True]
    ys = _median(leak_ok.loc[leak_ok["group"] == "y_short", "l_sigma_h"].tolist())
    yl = _median(leak_ok.loc[leak_ok["group"] == "y_long", "l_sigma_h"].tolist())
    dd = _median(leak_ok.loc[leak_ok["group"] == "delta", "l_sigma_h"].tolist())
    l_sigma_final = _median([_median([ys, yl]), dd])
    lls_h = 0.5 * l_sigma_final
    llr_h = 0.5 * l_sigma_final

    csf = build_csf_report(paths, extraction_cfg)
    rr = build_rr_report(paths, extraction_cfg, lls_h=lls_h, llr_h=llr_h)
    lm = build_lm_report(paths, extraction_cfg, lls_h=lls_h)
    initial_inputs = assemble_initial_inputs(resonance, leakage, csf, rr, lm, priors)
    return {
        "resonance": resonance,
        "leakage": leakage,
        "csf": csf,
        "rr": rr,
        "lm": lm,
        "initial_inputs": initial_inputs,
    }


def write_parameter_fitting_outputs(paths: Paths, results: dict[str, object]) -> dict[str, Path]:
    paths.output_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, Path] = {}
    for key in ("resonance", "leakage", "csf", "rr", "lm"):
        frame = results[key]
        if not isinstance(frame, pd.DataFrame):
            continue
        out_path = paths.output_dir / f"stage1_{key}_report.csv"
        frame.to_csv(out_path, index=False, encoding="utf-8-sig")
        outputs[key] = out_path

    inputs = results["initial_inputs"]
    if isinstance(inputs, InitialParamInputs):
        csv_path = paths.output_dir / "stage1_initial_inputs.csv"
        json_path = paths.output_dir / "stage1_initial_inputs.json"
        pd.DataFrame(
            [{"field": key, "value": value} for key, value in asdict(inputs).items()]
        ).to_csv(csv_path, index=False, encoding="utf-8-sig")
        json_path.write_text(json.dumps(asdict(inputs), indent=2), encoding="utf-8")
        outputs["initial_inputs_csv"] = csv_path
        outputs["initial_inputs_json"] = json_path
    return outputs
