import sqlite3
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


DB_PATH = r"D:\Desktop\EE5003\data\AP_1p5.db"
PRIMARY_GROUPS = ["exp_10", "exp_11", "exp_12"]
CHECK_GROUPS = ["exp_18", "exp_19", "exp_20"]

LLS = 2.55e-2

F_LO_INIT = 5_000.0
F_HI_INIT = 80_000.0
F_LO_MIN = 2_000.0
F_HI_MAX = 100_000.0

SLOPE_MIN = -1.2
SLOPE_MAX = -0.8
IM_OVER_RE_RATIO = 2.0
LM_CV_MAX = 0.15
MIN_POINTS = 20


@dataclass(frozen=True)
class BandResult:
    table: str
    lm_median_h: float
    lm_mean_h: float
    lm_std_h: float
    lm_iqr_h: float
    n_points: int
    band_lo_hz: float
    band_hi_hz: float


def load_table(db_path: str, table: str) -> pd.DataFrame:
    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql_query(f"SELECT Freq, Zabs, Phase FROM {table}", conn)
    df = df.dropna()
    df = df[(df["Freq"] > 0) & (df["Zabs"] > 0)]
    return df.sort_values("Freq").reset_index(drop=True)


def complex_impedance_from_abs_phase(zabs: np.ndarray, phase_deg: np.ndarray) -> np.ndarray:
    phi = np.deg2rad(phase_deg)
    return zabs * (np.cos(phi) + 1j * np.sin(phi))


def derivative_loglog(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    mask = (np.isfinite(y)) & (np.isfinite(x)) & (np.abs(y) > 0) & (x > 0)
    out = np.full_like(x, np.nan, dtype=float)
    if mask.sum() < 3:
        return out

    yy = np.log(np.abs(y[mask]))
    xx = np.log(x[mask])
    dydx = np.empty_like(xx)
    dydx[1:-1] = (yy[2:] - yy[:-2]) / (xx[2:] - xx[:-2])
    dydx[0] = (yy[1] - yy[0]) / (xx[1] - xx[0])
    dydx[-1] = (yy[-1] - yy[-2]) / (xx[-1] - xx[-2])
    out[np.where(mask)[0]] = dydx
    return out


def pick_band_by_criteria(
    f: np.ndarray,
    ymag: np.ndarray,
    slope: np.ndarray,
    f_lo_init: float,
    f_hi_init: float,
    f_lo_min: float,
    f_hi_max: float,
    slope_min: float,
    slope_max: float,
    im_over_re_ratio: float,
    lm_cv_max: float,
    min_points: int,
) -> Optional[tuple[np.ndarray, np.ndarray, tuple[float, float]]]:
    def select_in_window(f_lo: float, f_hi: float):
        mask0 = (f >= f_lo) & (f <= f_hi)
        if mask0.sum() < min_points:
            return None

        ywin = ymag[mask0]
        sw = slope[mask0]
        fwin = f[mask0]
        good = (
            np.isfinite(sw)
            & (sw >= slope_min)
            & (sw <= slope_max)
            & (np.abs(np.imag(ywin)) >= im_over_re_ratio * np.abs(np.real(ywin)))
        )
        if good.sum() < min_points:
            return None

        omega = 2.0 * np.pi * fwin[good]
        bm = np.imag(ywin[good])
        valid = np.abs(bm) > 0
        if valid.sum() < min_points:
            return None

        lm_f = 1.0 / (omega[valid] * np.abs(bm[valid]))
        if lm_f.size < min_points:
            return None
        med = np.median(lm_f)
        if med <= 0:
            return None
        cv = float(np.std(lm_f) / med)
        if cv > lm_cv_max:
            return None
        return fwin[good][valid], ywin[good][valid], (f_lo, f_hi)

    for f_lo, f_hi in (
        (f_lo_init, f_hi_init),
        (f_lo_min, f_hi_init),
        (f_lo_init, f_hi_max),
        (f_lo_min, f_hi_max),
    ):
        picked = select_in_window(f_lo, f_hi)
        if picked is not None:
            return picked
    return None


def compute_group_lm(
    db_path: str,
    table: str,
    lls_h: float,
    f_lo_init: float = F_LO_INIT,
    f_hi_init: float = F_HI_INIT,
) -> Optional[BandResult]:
    df = load_table(db_path, table)
    if df.empty:
        return None

    freq = df["Freq"].to_numpy(dtype=float)
    zabs = df["Zabs"].to_numpy(dtype=float)
    phase = df["Phase"].to_numpy(dtype=float)
    z_meas = complex_impedance_from_abs_phase(zabs, phase)

    omega = 2.0 * np.pi * freq
    zmag = z_meas - 1j * omega * lls_h
    with np.errstate(divide="ignore", invalid="ignore"):
        ymag = 1.0 / zmag

    slope = derivative_loglog(np.abs(np.imag(ymag)), freq)
    picked = pick_band_by_criteria(
        f=freq,
        ymag=ymag,
        slope=slope,
        f_lo_init=f_lo_init,
        f_hi_init=f_hi_init,
        f_lo_min=F_LO_MIN,
        f_hi_max=F_HI_MAX,
        slope_min=SLOPE_MIN,
        slope_max=SLOPE_MAX,
        im_over_re_ratio=IM_OVER_RE_RATIO,
        lm_cv_max=LM_CV_MAX,
        min_points=MIN_POINTS,
    )
    if picked is None:
        return None

    f_sel, y_sel, (flo, fhi) = picked
    omega_sel = 2.0 * np.pi * f_sel
    bm = np.imag(y_sel)
    mask = np.isfinite(omega_sel) & np.isfinite(bm) & (np.abs(bm) > 0)
    if mask.sum() < MIN_POINTS:
        return None

    lm_f = 1.0 / (omega_sel[mask] * np.abs(bm[mask]))
    q1, q3 = np.percentile(lm_f, [25, 75])
    return BandResult(
        table=table,
        lm_median_h=float(np.median(lm_f)),
        lm_mean_h=float(np.mean(lm_f)),
        lm_std_h=float(np.std(lm_f)),
        lm_iqr_h=float(q3 - q1),
        n_points=int(lm_f.size),
        band_lo_hz=float(flo),
        band_hi_hz=float(fhi),
    )


def summarize_groups(results: list[BandResult], tag: str) -> dict[str, float]:
    if not results:
        return {}
    meds = np.array([r.lm_median_h for r in results], dtype=float)
    means = np.array([r.lm_mean_h for r in results], dtype=float)
    return {
        f"{tag}_lm_median_h": float(np.median(meds)),
        f"{tag}_lm_mean_h": float(np.mean(means)),
        f"{tag}_lm_iqr_of_medians_h": float(np.percentile(meds, 75) - np.percentile(meds, 25)),
        f"{tag}_lm_std_of_medians_h": float(np.std(meds)),
        f"{tag}_n_groups": int(len(results)),
    }


def main() -> None:
    primary = [compute_group_lm(DB_PATH, table, LLS) for table in PRIMARY_GROUPS]
    check = [compute_group_lm(DB_PATH, table, LLS) for table in CHECK_GROUPS]
    rows = [
        {
            "group": "primary" if result.table in PRIMARY_GROUPS else "check",
            "table": result.table,
            "lm_median_h": result.lm_median_h,
            "lm_mean_h": result.lm_mean_h,
            "lm_std_h": result.lm_std_h,
            "lm_iqr_h": result.lm_iqr_h,
            "n_points": result.n_points,
            "band_lo_hz": result.band_lo_hz,
            "band_hi_hz": result.band_hi_hz,
        }
        for result in [r for r in primary + check if r is not None]
    ]
    frame = pd.DataFrame(rows)
    if not frame.empty:
        print(frame.to_string(index=False))
    print(summarize_groups([r for r in primary if r is not None], "primary"))
    print(summarize_groups([r for r in check if r is not None], "check"))


if __name__ == "__main__":
    main()
