"""BANDGAP: HSPICE/Spectre testbench + Verilog-A checker; stimuli applied to the power supplies."""
import re
from pathlib import Path

import numpy as np

from ..faults import Dut, find_targets, generate_parametric, read_lines, write_lines
from ..spectre_io import absolutize_includes, run_spectre
from ..stimuli import StimConfig
from .base import Circuit, absolute_verdict

MARKER = "** End of top cell TB_bandgap."

# Specs DC (Table 2)
T_IDLE, T_DC, T_STAB = 10e-3, 50e-3, 100e-3
SPECS = {"idle_10ms": (None, 0.010), "dc_50ms": (1.395, 1.405), "stab_diff": (None, 0.007)}

SPY = """`include "disciplines.vams"
`include "constants.vams"
module csv_writer(vin, vfail);
    input vin, vfail;
    electrical vin, vfail;
    integer fid;
    analog begin
        @(initial_step) fid = $fopen("__CSV__");
        @(timer(0, __PERIOD__)) $fstrobe(fid, "%e %e %e", $abstime, V(vin), V(vfail));
        @(final_step) $fclose(fid);
    end
endmodule
"""

SUPPLIES = """
* --- INJECTION STIMULUS ---
Vforce_enable net033 0 DC 0
Vprbs_ana VDDa! 0 PWL ({pwl})
Vprbs_bulk VDDba! 0 PWL ({pwl})
Vprbs_dig VDDC! 0 PWL ({pwl})
Vprbs_bulk_dig VDDbc! 0 PWL ({pwl})
Vgnd_ana VSS! 0 DC 0
Vgnd_bulk VSSb! 0 DC 0
"""


def _rule(line):
    n = line.split()[0].lower()
    if n.startswith("m"):
        return n, "W"
    if n.startswith("r"):
        return n, "R"
    if n.startswith("c"):
        return n, "C"
    if n.startswith("q"):
        return n, "M"
    return None


def _inject_open(lines, parts):
    """Open a terminal: The terminal is connected to a floating node via 1 GΩ. Returns “None” if the component cannot be found."""
    full, model, term = parts[2], parts[3].lower(), parts[4].lower().strip()
    comp = (full.split(".")[1] if "." in full else full).split("-")[0]
    if any(x in model for x in ("nana", "pana", "mos")):
        pin = {"d": 1, "g": 2, "s": 3, "b": 4}.get(term, -1)
    elif any(x in model for x in ("latpnp", "pnp", "npn")):
        pin = {"c": 1, "b": 2, "e": 3}.get(term, -1)
    elif any(x in model for x in ("resistor", "capacitor")):
        pin = {"1": 1, "2": 2}.get(term, -1)
    else:
        pin = -1
    if pin == -1:
        return None
    out, done = [], False
    for line in lines:
        if not done and re.match(rf"^\s*{re.escape(comp)}\s+", line, re.I):
            tk = line.split()
            if len(tk) > pin:
                old, new = tk[pin], f"OPEN_{comp}_{term}"
                tk[pin] = new
                out += [f"* DEFECT: {line.strip()}\n", " ".join(tk) + "\n",
                        f"R_OPEN_{comp}_{term} {new} {old} 1000MEG\n"]
                done = True
                continue
        out.append(line)
    return out if done else None


class Bandgap(Circuit):
    name = "bandgap"
    description = "Bandgap (Spectre + Verilog-A checker): idle/DC/stability specifications, VDD stimuli"
    dut_ext = ".hsp"
    measure_names = tuple(SPECS)
    default_n_param = 2000
    required_files = ("src/tb_bandgap.hsp", "src/top.hsp", "ref/defect_short.txt", "ref/defect_open.txt")

    def stim_config(self):
        """Returns the stimulus configuration for the bandgap circuit."""
        return StimConfig(v_low=2.3, v_high=2.7, bit_time=10e-6, window=127 * 10e-6, n_points=500,
                          t_sig_start=11e-6, t_obs_start=11e-6, v_init=2.5, startup_from_zero=True,
                          dc_window=110e-3, dc_points=1100)

    # ---- population ------------------------------------------------------
    def generate(self, out_dir, n_param, rng):
        src = self.root / "src/tb_bandgap.hsp"
        lines = read_lines(src)
        out = Path(out_dir)
        nom = out / "bandgap_NOMINAL.hsp"
        write_lines(nom, lines)
        duts = [Dut(nom.name, str(nom), "nominal", "")]
        targets = find_targets(lines, _rule, subckt="bandgap")
        duts += generate_parametric(
            lines, targets, n_param, 20, rng, out,
            lambda i, safe: f"bandgap_{'MC_SAFE' if safe else 'MC_EXTREME'}_{i:04d}.hsp")
        text = "".join(lines)
        if MARKER not in text:
            raise ValueError(f"'{MARKER}' tag missing from {src}")
        for l in read_lines(self.root / "ref/defect_short.txt"):
            if not l.startswith("Defect_"):
                continue
            p = [x.strip() for x in l.split(",")]
            fn = f"bandgap_{p[0]}.hsp"
            body = text.replace(MARKER, f"** INJECTED SHORT: {p[0]}\nR_FAULT {p[1]} {p[2]} 0.1\n{MARKER}")
            write_lines(out / fn, [body])
            duts.append(Dut(fn, str(out / fn), "short", f"{p[1]}-{p[2]}"))
        skipped = 0
        for l in read_lines(self.root / "ref/defect_open.txt"):
            if not l.startswith("Defect_"):
                continue
            p = [x.strip() for x in l.split(",")]
            new = _inject_open(lines, p)
            if new is None:           
                skipped += 1
                continue
            fn = f"bandgap_{p[0]}.hsp"
            write_lines(out / fn, new)
            duts.append(Dut(fn, str(out / fn), "open", f"{p[2]}:{p[4]}"))
        if skipped:
            print(f"  [bandgap] {skipped} “Open” faults ignored (component or terminal not found).")
        return duts

    # ---- simulation ----------------------------------------------------------
    def _run(self, dut_path, work, stim, period, timeout):
        work = Path(work)
        csv_path = work / "data.csv"
        (work / "csv_writer.va").write_text(
            SPY.replace("__CSV__", str(csv_path)).replace("__PERIOD__", f"{period:.6g}"), encoding="utf-8")
        tb = Path(dut_path).read_text(encoding="utf-8", errors="ignore")
        if MARKER not in tb:
            raise ValueError(f"Marqueur '{MARKER}' absent de {dut_path}")
        tb = tb.replace(MARKER, f"\nXcspy bg fail csv_writer\n\n{MARKER}")
        if stim.pwl:
            for pat in (r"^(XI91\s+.*supplies)", r"^(V3\s+net033.*)"):
                tb, n = re.subn(pat, r"*\1", tb, flags=re.M)
                if n == 0:
                    raise ValueError(f"Source à neutraliser introuvable dans le testbench: {pat}")
            tb += SUPPLIES.format(pwl=stim.pwl)
        (work / "tb.hsp").write_text(tb, encoding="utf-8")

        top = (self.root / "src/top.hsp").read_text(encoding="utf-8", errors="ignore")
        top = top.replace('.include "tb_bandgap.hsp"', f'.hdl "{work / "csv_writer.va"}"\n.include "{work / "tb.hsp"}"')
        if stim.pwl:  
            first = [True]

            def repl(m):
                if first[0]:
                    first[0] = False
                    return f".tran 1u {stim.t_stop:.6g}"
                return ""

            top, n = re.subn(r"^\s*\.(tran|dc)\b.*$", repl, top, flags=re.I | re.M)
            if n == 0:
                raise ValueError("No analysis .tran/.dc in top.hsp")
        top = absolutize_includes(top, self.root / "src")
        (work / "run.sp").write_text(top, encoding="utf-8")

        st = run_spectre("run.sp", work, timeout)
        if st != "OK":
            return st
        rows = []
        if csv_path.exists():
            for l in csv_path.read_text(encoding="utf-8", errors="ignore").splitlines():
                p = l.split()
                if len(p) >= 3:
                    try:
                        rows.append([float(x) for x in p[:3]])
                    except ValueError:
                        pass
        return np.array(rows) if len(rows) > 10 else "NO_DATA"

    def measure(self, dut_path, corner, work, timeout):
        from ..stimuli import make_stimulus
        stim = make_stimulus("DC", self.stim_config())
        res = self._run(dut_path, work, stim, 500e-6, timeout)
        if isinstance(res, str):
            return {"status": res, "values": {}, "checker": None}
        t, bg, fail = res[:, 0], res[:, 1], res[:, 2]
        at = lambda x: bg[np.abs(t - x).argmin()]
        vals = {"idle_10ms": at(T_IDLE), "dc_50ms": at(T_DC), "stab_diff": abs(at(T_STAB) - at(T_DC))}
        return {"status": "OK", "values": vals, "checker": int(fail[-1] > 0.5)}

    def verdict(self, m, nominal):
        return absolute_verdict(m["values"], SPECS)

    def hard_label(self, fault_class, spec_label, checker):
        if fault_class in ("short", "open"):
            return 1
        if fault_class == "param_safe":
            return 0
        return checker if checker is not None else spec_label   # EXTREME/nominal 

    def waveform(self, dut_path, corner, stim, work, timeout):
        span = stim.t_stop - stim.t_obs_start
        res = self._run(dut_path, work, stim, span / (2 * stim.n_points), timeout)
        if isinstance(res, str):
            return res
        return res[:, 0], res[:, 1]
