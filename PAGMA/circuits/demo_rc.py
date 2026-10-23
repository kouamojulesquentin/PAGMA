"""Demo circuit: RC low-pass filter, simulated in pure Python (no Spectre required).

Serves as an end-to-end test of the pipeline (generate -> label -> dataset -> train).
The model parses the netlist (R1, C1, R_SHORT_*, R_OPEN_*), so defects have a real effect.
"""
import math
import re
from pathlib import Path

import numpy as np

from ..faults import (Dut, find_targets, generate_parametric, read_lines,
                      two_terminal_defects, write_lines)
from ..stimuli import StimConfig
from .base import Circuit, relative_verdict

_SUF = {"t": 1e12, "g": 1e9, "meg": 1e6, "k": 1e3, "m": 1e-3, "u": 1e-6, "n": 1e-9, "p": 1e-12, "f": 1e-15}


def _num(s: str) -> float:
    m = re.fullmatch(r"([-+]?[\d.]+(?:e[-+]?\d+)?)([a-z]*)", s.lower())
    if not m:
        raise ValueError(f"Unreadable SPICE value: {s}")
    return float(m.group(1)) * _SUF.get(m.group(2), 1.0)


def _value(tok, key):
    for t in tok[3:]:
        if t.lower().startswith(key + "="):
            return _num(t.split("=", 1)[1])
    return _num(tok[3])


def _network(path):
    rs, rp, c = [], [], 0.0
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s[0] in "*.;$+":
            continue
        tok = s.split()
        nodes = {tok[1].lower(), tok[2].lower()}
        kind = tok[0][0].lower()
        if kind == "r":
            val = _value(tok, "r")
            (rs if nodes == {"in", "out"} else rp).append(val)
        elif kind == "c" and nodes == {"out", "0"}:
            c += _value(tok, "c")
    par = lambda xs: 1.0 / sum(1.0 / x for x in xs) if xs else math.inf
    return par(rs), par(rp), c


def _thevenin(path):
    Rs, Rp, C = _network(path)
    gain = 1.0 if math.isinf(Rp) else Rp / (Rs + Rp)
    Rth = Rs if math.isinf(Rp) else Rs * Rp / (Rs + Rp)
    return gain, Rth, C


class DemoRC(Circuit):
    name = "demo_rc"
    description = "RC low-pass filter (demo, pure Python, no Spectre)"
    corners = ("",)
    relative_specs = True
    measure_names = ("gain_dc", "fc")
    default_n_param = 40
    required_files = ("DEMO_RC.sub",)
    TOL = 0.05

    def stim_config(self):
        return StimConfig(v_low=0.0, v_high=1.0, bit_time=50e-6, window=1e-3,
                          n_points=200, rise=1e-6)

    def generate(self, out_dir, n_param, rng):
        lines = read_lines(self.root / "DEMO_RC.sub")
        out = Path(out_dir)
        nominal = out / "DEMO_RC_NOMINAL.sub"
        write_lines(nominal, lines)
        duts = [Dut(nominal.name, str(nominal), "nominal", "")]
        rule = lambda l: (l.split()[0], "R" if l.split()[0][0].lower() == "r" else "C") \
            if l.split()[0][0].lower() in "rc" else None
        targets = find_targets(lines, rule)
        duts += generate_parametric(lines, targets, n_param, 2, rng, out,
                                    lambda i, safe: f"DEMO_RC_MC_{i:04d}.sub")
        duts += two_terminal_defects(lines, lambda l: l.split()[0][0].lower() in "rc", out,
                                     "DEMO_RC_{kind}_{name}.sub", "100", "1000meg")
        return duts

    def measure(self, dut_path, corner, work, timeout):
        gain, Rth, C = _thevenin(dut_path)
        fc = 1.0 / (2 * math.pi * Rth * C) if C > 0 else math.inf
        return {"status": "OK", "values": {"gain_dc": gain, "fc": fc}, "checker": None}

    def verdict(self, m, nominal):
        return relative_verdict(m["values"], nominal, self.measure_names, self.TOL)

    def waveform(self, dut_path, corner, stim, work, timeout):
        gain, Rth, C = _thevenin(dut_path)
        pts = np.asarray(stim.points)
        tt = np.linspace(0.0, stim.t_stop, stim.n_points * 20)
        vth = np.interp(tt, pts[:, 0], pts[:, 1]) * gain
        if C <= 0:
            return tt, vth
        a = math.exp(-(tt[1] - tt[0]) / (Rth * C))
        v = np.empty_like(vth)
        v[0] = vth[0]
        for k in range(len(tt) - 1):
            v[k + 1] = a * v[k] + (1 - a) * vth[k + 1]
        return tt, v
