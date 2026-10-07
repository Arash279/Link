from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal


ConnectionType = Literal["Y", "Delta"]


@dataclass(frozen=True)
class InitialParamInputs:
    # Direct inputs or independently estimated values.
    hp: float
    connection: ConnectionType
    lls_h: float
    llr_h: float
    lm_h: float
    rrs_ohm: float
    csf_hf_f: float
    csf_lf_f: float
    fr_hz: float
    zmax_ohm: float
    fa_hz: float
    zanti_ohm: float
    nlls_init_h: float
    lad_h: float = 1.3e-7


DEFAULT_INITIAL_INPUTS = InitialParamInputs(
    hp=1.5,
    connection="Y",
    lls_h=2.55e-2,
    llr_h=2.55e-2,
    lm_h=5.5e-2,
    rrs_ohm=28.0,
    csf_hf_f=2.461e-10,
    csf_lf_f=1.516e-9,
    fr_hz=36311.2,
    zmax_ohm=2.50e4,
    fa_hz=66734.6,
    zanti_ohm=4.11e3,
    nlls_init_h=1.8e-10,
    lad_h=1.3e-7,
)


def rcore_from_hp(hp: float) -> float:
    return 6300.0 * (hp ** (-0.6958))


def csw_from_fr_y(csf_hf_f: float, lls_h: float, fr_hz: float) -> float:
    omega = 2.0 * math.pi * fr_hz
    numerator = 2.0 * (omega**2) * lls_h * csf_hf_f - 1.0
    denominator = (omega**2) * lls_h * ((omega**2) * lls_h * csf_hf_f - 1.0)
    return numerator / denominator


def csw_from_fr_delta(csf_hf_f: float, lls_h: float, fr_hz: float) -> float:
    omega = 2.0 * (2.0 * math.pi * fr_hz)
    numerator = 2.0 * (omega**2) * lls_h * csf_hf_f - 1.0
    denominator = (omega**2) * lls_h * ((omega**2) * lls_h * csf_hf_f - 1.0)
    return numerator / denominator


def eta_lls_from_fa(csf_hf_f: float, fa_hz: float) -> float:
    return 1.0 / (csf_hf_f * (2.0 * math.pi * fa_hz) ** 2)


def rsf_from_zanti(zanti_ohm: float) -> float:
    return (2.0 / 3.0) * zanti_ohm


def parallel_mag_r_and_jx(r_ohm: float, x_ohm: float) -> float:
    return (r_ohm * abs(x_ohm)) / math.sqrt(r_ohm**2 + x_ohm**2)


def rsw_from_zmax_rcore_llr(fr_hz: float, zmax_ohm: float, rcore_ohm: float, llr_h: float) -> float:
    x_l = 2.0 * math.pi * fr_hz * llr_h
    z_par = parallel_mag_r_and_jx(rcore_ohm, x_l)
    return (2.0 / 3.0) * zmax_ohm - z_par


def csf0_from_lf_hf(csf_lf_f: float, csf_hf_f: float) -> float:
    return csf_lf_f - 3.0 * csf_hf_f


def build_initial_param_values(inputs: InitialParamInputs) -> dict[str, float]:
    rcore_ohm = rcore_from_hp(inputs.hp)
    if inputs.connection == "Y":
        csw_f = csw_from_fr_y(inputs.csf_hf_f, inputs.lls_h, inputs.fr_hz)
    elif inputs.connection == "Delta":
        csw_f = csw_from_fr_delta(inputs.csf_hf_f, inputs.lls_h, inputs.fr_hz)
    else:
        raise ValueError("connection must be 'Y' or 'Delta'")

    # Keep the paper formula available for reference, but do not feed it into nLls.
    # In the current CurVer model, nLls is still a manually fixed small initial value.
    _eta_lls_formula_h = eta_lls_from_fa(inputs.csf_hf_f, inputs.fa_hz)

    return {
        "Lls": inputs.lls_h,
        "Csw": csw_f,
        "Rsw": rsw_from_zmax_rcore_llr(inputs.fr_hz, inputs.zmax_ohm, rcore_ohm, inputs.llr_h),
        "Llr": inputs.llr_h,
        "Rrs": inputs.rrs_ohm,
        "Rcore": rcore_ohm,
        "Lm": inputs.lm_h,
        "nLls": inputs.nlls_init_h,
        "Csf": inputs.csf_hf_f,
        "Rsf": rsf_from_zanti(inputs.zanti_ohm),
        "Csf0": csf0_from_lf_hf(inputs.csf_lf_f, inputs.csf_hf_f),
        "Lad": inputs.lad_h,
    }
