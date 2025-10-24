# -*- coding: utf-8 -*-
# pip install lcapy schemdraw
import os
import lcapy, schemdraw
print("lcapy:", lcapy.__version__)
print("schemdraw:", schemdraw.__version__)

from lcapy import Circuit

NET = r"""
.params R1=100 L1=10e-3 C1=1e-6

V1   VA   0    AC 1
R1   VA   VB   R1
L1   VB   VC   L1

C1   VC   G    C1
Wg   G    0
"""

# 关键：VC/G/0 同 x，不同 y，让电容垂直向下
hints = {
    "VA": (0.0, 0.0),
    "VB": (2.8, 0.0),
    "VC": (5.6, 0.0),
    "G":  (5.6, -2.0),
    "0":  (5.6, -3.6),
}

cct = Circuit(NET)

# ——诊断：节点匹配——
nodes_in_circuit = {str(n) for n in cct.nodes}
print("NODES:", sorted(nodes_in_circuit))

# ——核心区别：直接取 sch 属性，再 draw，手动指定 hints——
sch = cct.sch
os.makedirs("figs", exist_ok=True)

sch.draw(
    engine="schemdraw",
    hints=hints,           # 在这里传 hints
    label_nodes=True,
    label_values=False,
    tex=False,
    dpi=220,
    filename="figs/series_RLC_vertical_cap.png",
    show=False,
)
print("[OK] 已保存 figs/series_RLC_vertical_cap.png")
