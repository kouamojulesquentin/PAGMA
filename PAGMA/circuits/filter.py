"""FILTER: parametric defects + shorts/opens, 5 process corners; spec = nominal value of the corner ± tolerance."""
import re
from pathlib import Path

from ..faults import (Dut, find_targets, generate_parametric, read_lines,
                      two_terminal_defects, write_lines)
from ..spectre_io import absolutize_includes, find_measures, find_print, run_spectre
from ..stimuli import StimConfig
from .base import Circuit, relative_verdict

CORNERS = ("typical", "fastN_fastP", "fastN_slowP", "slowN_fastP", "slowN_slowP")

BENCH = """* ML dataset (FILTER)
.TEMP 27
.PARAM vddnom=3.3
.PARAM vref_val=1.65
VDD vdda 0 DC vddnom
VSS vssa 0 DC 0
VREF vref 0 DC vref_val
V_pd  pd  0 DC 0
V_xpd xpd 0 DC vddnom
Vin V_in 0 PWL ({pwl})
.INCLUDE "{dut}"
Xdut V_in V_out vdda vssa vref pd xpd FILTER
.INCLUDE "{prim}"
.INCLUDE "{opamp}"
.INCLUDE "{proc}"
.TRAN {tstep:.6g} {tstop:.6g}
.PRINT TRAN V(V_out)
.END
"""


def _passive_rule(line):
    l = line.lower()
    n = l.split()[0]
    if n.startswith("xr") and "primitive_rpoly2" in l:
        return n, "L"
    if n.startswith("xc") and "primitive_cpoly" in l:
        return n, "AREA"
    return None


class Filter(Circuit):
    name = "filter"
    description = "Filtre actif (Spectre), 5 coins de process : gain_20k, gain_100k, freq_cutoff"
    corners = CORNERS
    relative_specs = True
    measure_names = ("gain_20k", "gain_100k", "freq_cutoff")
    MEASURE_ALIASES = {"freq_cutoff": ("freq_coupure",)}   
    default_n_param = 1000
    TOL = 0.05                       
    R_SHORT, R_OPEN = "100", "1000meg"
    required_files = ("subcircuit_files/FILTER.sub", "subcircuit_files/PRIMITIVE.sub",
                      "subcircuit_files/OPAMP1.sub", "filter.circuit") + \
        tuple(f"process_models/{c}.process" for c in CORNERS)

    def stim_config(self):
        
        return StimConfig(v_low=0.0, v_high=0.5, bit_time=5e-6, window=100e-6, n_points=200, rise=0.1e-6)

    def hard_label(self, fault_class, spec_label, checker):
       
        if fault_class in ("short", "open", "param_extreme"):
            return 1
        return 0   # param_safe et nominal

    def generate(self, out_dir, n_param, rng):
        lines = read_lines(self.root / "subcircuit_files/FILTER.sub")
        out = Path(out_dir)
        nom = out / "FILTER_NOMINAL.sub"
        write_lines(nom, lines)
        duts = [Dut(nom.name, str(nom), "nominal", "")]
        duts += generate_parametric(lines, find_targets(lines, _passive_rule), n_param, 5, rng, out,
                                    lambda i, safe: f"FILTER_MC_{i:04d}.sub")
        duts += two_terminal_defects(lines, lambda l: _passive_rule(l) is not None, out,
                                     "FILTER_{kind}_{name}.sub", self.R_SHORT, self.R_OPEN)
        return duts

    def _proc(self, corner):
        p = self.root / "process_models" / f"{corner}.process"
        return p if p.exists() else self.root / "process_models" / "typical.process"

    def measure(self, dut_path, corner, work, timeout):
        out = []
        for line in (self.root / "filter.circuit").read_text(encoding="utf-8", errors="ignore").splitlines():
            low = line.lower()
            if ".include" in low and "process" in low:
                out.append(f'.INCLUDE "{self._proc(corner).resolve()}"')
            elif ".include" in low and "filter.sub" in low:
                out.append(f'.INCLUDE "{Path(dut_path).resolve()}"')
            else:
                out.append(line)
        text = absolutize_includes("\n".join(out) + "\n", self.root)
        (Path(work) / "filter.circuit").write_text(text, encoding="utf-8")
        st = run_spectre("filter.circuit", work, timeout, extra=("-64",))
        if st != "OK":
            return {"status": st, "values": {}, "checker": None}
        raw = find_measures(work)
        vals = {}
        for n in self.measure_names:
            for cand in (n,) + tuple(self.MEASURE_ALIASES.get(n, ())):
                if cand in raw:            
                    vals[n] = raw[cand]
                    break
        
        return {"status": "OK", "values": vals, "checker": None, "raw_keys": sorted(raw)}

    def verdict(self, m, nominal):
        return relative_verdict(m["values"], nominal, self.measure_names, self.TOL)

    def waveform(self, dut_path, corner, stim, work, timeout):
        r = self.root / "subcircuit_files"
        tstep = (stim.t_stop / (stim.n_points)) / 10
        text = BENCH.format(pwl=stim.pwl, dut=Path(dut_path).resolve(), prim=(r / "PRIMITIVE.sub").resolve(),
                            opamp=(r / "OPAMP1.sub").resolve(), proc=self._proc(corner).resolve(),
                            tstep=tstep, tstop=stim.t_stop)
        (Path(work) / "sim_ml.circuit").write_text(text, encoding="utf-8")
        st = run_spectre("sim_ml.circuit", work, timeout, extra=("-64", "-format", "psfascii"))
        if st != "OK":
            return st
        t, v = find_print(work)
        return (t, v) if len(t) > 5 else "NO_DATA"
