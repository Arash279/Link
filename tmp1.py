import math

def lc_values(fL, fH, g, ZL=50.0, kind="series"):
    """Return (L, C) for one resonator.
    kind="series"  ->  L/ZL = g/dw,  C*ZL = dw/(w0^2 g)
    kind="shunt"   ->  C*ZL = g/dw,  L/ZL = dw/(w0^2 g)
    """
    wL, wH = 2*math.pi*fL, 2*math.pi*fH
    dw = wH - wL
    w0 = math.sqrt(wH*wL)

    if kind == "series":
        L = (g/dw) * ZL
        C = (dw/(w0*w0*g)) / ZL
    elif kind == "shunt":
        C = (g/dw) / ZL
        L = (dw/(w0*w0*g)) * ZL
    else:
        raise ValueError("kind must be 'series' or 'shunt'")
    return L, C  # in Henry, Farad

# ----- inputs -----
f1 = 4.845e9    # GHz -> Hz （下边界）
f2 = 5.355e9    # 上边界
g1 = g4 = 0.7654
g2 = g3 = 1.8478
ZL = 50.0

# Map到四阶 ladder：Series(g1) – Shunt(g2) – Series(g3) – Shunt(g4)
L1, C1 = lc_values(f1, f2, g1, ZL, "series")
L2, C2 = lc_values(f1, f2, g2, ZL, "shunt")
L3, C3 = lc_values(f1, f2, g3, ZL, "series")
L4, C4 = lc_values(f1, f2, g4, ZL, "shunt")

print(L1, C1, L2, C2, L3, C3, L4, C4)
