"""OPAMP1: parametric defects + MOS defects; labeled according to absolute specifications (Spectre)."""
import re
from pathlib import Path

from ..faults import (Dut, find_targets, generate_parametric, mos_defects, read_lines, write_lines)
from ..spectre_io import (absolutize_includes, find_measures, find_print, run_spectre, sub_once)
from ..stimuli import StimConfig
from .base import Circuit, absolute_verdict

SPECS = {
    "voffset": (-0.001, 0.001),     # ±1 mV
    "vpp@1mhz": (0.700, 1.000),     # 700 mV à 1 V
    "delay": (None, 1.2e-7),        # max 120 ns
    "iddq": (None, 3.0e-7),         # max 300 nA
}


def _rule(line):
    """A parameter that should be adjusted depending on the type of component."""
    parts = line.split()
    if not parts:
        return None
    n = parts[0].lower()
    if n.startswith(("m", "r", "xr")):
        return n, "W"
    if n.startswith(("c", "xc", "xi", "xd")):
        return n, "AREA"
    if n.startswith(("q", "d", "xq")):
        return (n, "AREA") if "area=" in line.lower() else (n, "M")
    return None


class Opamp1(Circuit):
    name = "opamp1"
    description = "OPAMP1 (Spectre) : specs voffset / vpp / delay / iddq"
    measure_names = tuple(SPECS)
    default_n_param = 3000
    R_SHORT, R_OPEN = "0.1", "1000meg"
    DEFECTS = True  
    required_files = ("subcircuit_files/OPAMP1.sub", "OPAMP1.circuit", "OPAMP1_ML.circuit")

    def stim_config(self):
        
        return StimConfig(v_low=1.45, v_high=1.85, bit_time=50e-9, window=7e-6, n_points=701,
                          t_sig_start=0.5e-6, rise=2e-9)

    def generate(self, out_dir, n_param, rng):
        lines = read_lines(self.root / "subcircuit_files/OPAMP1.sub")
        out = Path(out_dir)
        nom = out / "OPAMP1_NOMINAL.sub"
        write_lines(nom, lines)
        duts = [Dut(nom.name, str(nom), "nominal", "")]
        targets = find_targets(lines, _rule)
        duts += generate_parametric(lines, targets, n_param, 5, rng, out,
                                    lambda i, safe: f"OPAMP1_MC_{i:04d}.sub")
        if self.DEFECTS:
            duts += mos_defects(lines, out, "OPAMP1_{kind}_{name}.sub", self.R_SHORT, self.R_OPEN)
        return duts

    def _bench(self, template, dut_path, work, name):
        text = (self.root / template).read_text(encoding="utf-8", errors="ignore")
        text = sub_once(r"\.INCLUDE.*subcircuit_files/OPAMP1\.sub", f'.INCLUDE "{Path(dut_path).resolve()}"',
                        text, "include du DUT")
        text = absolutize_includes(text, self.root)
        (Path(work) / name).write_text(text, encoding="utf-8")
        return text

    def measure(self, dut_path, corner, work, timeout):
        self._bench("OPAMP1.circuit", dut_path, work, "OPAMP1.circuit")
        st = run_spectre("OPAMP1.circuit", work, timeout)
        if st != "OK":
            return {"status": st, "values": {}, "checker": None}
        raw = find_measures(work)
        return {"status": "OK", "values": raw, "checker": None, "raw_keys": sorted(raw)}

    def verdict(self, m, nominal):
        return absolute_verdict(m["values"], SPECS)

    def waveform(self, dut_path, corner, stim, work, timeout):
        text = (self.root / "OPAMP1_ML.circuit").read_text(encoding="utf-8", errors="ignore")
        if "simulator lang=spice" not in text.lower():
            text = "simulator lang=spice\n" + text
        text = sub_once(r"\.INCLUDE.*subcircuit_files/OPAMP1\.sub", f'.INCLUDE "{Path(dut_path).resolve()}"',
                        text, "include du DUT")
        text = sub_once(r"^Vin\s+.*", f"Vin inp 0 PWL ({stim.pwl})", text, "source Vin")
        dt = (stim.t_stop - stim.t_obs_start) / (stim.n_points - 1)
        text = sub_once(r"^\.TRAN.*", f".TRAN {dt:.6g} {stim.t_stop:.6g} 0 strobeperiod={dt:.6g}", text, ".TRAN")
        text = absolutize_includes(text, self.root)
        (Path(work) / "OPAMP1_ML.circuit").write_text(text, encoding="utf-8")
        st = run_spectre("OPAMP1_ML.circuit", work, timeout)
        if st != "OK":
            return st
        t, v = find_print(work)
        return (t, v) if len(t) > 10 else "NO_DATA"
