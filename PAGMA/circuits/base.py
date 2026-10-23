"""Common interface for all circuits + specification checks."""
import math
from pathlib import Path

from ..stimuli import StimConfig


class Circuit:
    name = ""
    description = ""
    dut_ext = ".sub"
    corners: tuple = ("",)            # process corners; (“”,) = none
    measure_names: tuple = ()         # Measured values for the “spec” label
    default_n_param = 100
    required_files: tuple = ()        # related to circuits/<name>/
    relative_specs = False            # True: specs = nominal ± tol

    def __init__(self, circuits_dir):
        self.root = Path(circuits_dir) / self.name

    # ---- to implement -------------------------------------------------
    def stim_config(self) -> StimConfig:
        raise NotImplementedError

    def generate(self, out_dir: Path, n_param: int, rng) -> list:
        
        """ Write DUTs (nominal + parametric + defects) and return a list of Dut."""
        raise NotImplementedError

    def measure(self, dut_path: str, corner: str, work: Path, timeout: int) -> dict:
        """Returns {“status”: “OK”|..., "values": {name: val}, "checker": 0|1|None}."""
        
        raise NotImplementedError

    def verdict(self, m: dict, nominal):
        """(spec_label 0=PASS/1=FAIL, raisons)."""
        raise NotImplementedError

    def waveform(self, dut_path: str, corner: str, stim, work: Path, timeout: int):
        """Returns (t, v) or an error status string."""
        raise NotImplementedError

    # ---- policy --------------------------------------------
    def hard_label(self, fault_class: str, spec_label: int, checker) -> int:
        """By default: fault->1, SAFE->0, otherwise spec label."""
        if fault_class in ("short", "open"):
            return 1
        if fault_class == "param_safe":
            return 0
        if fault_class == "nominal" and self.relative_specs:
            return 0          # nominal is always PASS for relative specs
        return spec_label

    # ---- verification ---------------------------------------------------
    def missing_files(self):
        return [f for f in self.required_files if not (self.root / f).exists()]


def _bad(x) -> bool:
    return x is None or not math.isfinite(x)


def absolute_verdict(values: dict, specs: dict):
    """specs = {nom: (min|None, max|None)}."""
    if not values:
        return 1, "NO_MEASURE"
    fails = []
    for k, (lo, hi) in specs.items():
        v = values.get(k)
        if _bad(v):
            fails.append(f"{k}(NaN)")
            continue
        if lo is not None and v < lo:
            fails.append(f"{k}(Low:{v:.2e})")
        if hi is not None and v > hi:
            fails.append(f"{k}(High:{v:.2e})")
    return (1, ";".join(fails)) if fails else (0, "NONE")


def relative_verdict(values: dict, nominal, names, tol: float):
    """Specs = nominal value of the corner ± tolerance (relative). A measurement with a nominal reference of NaN cannot
    be verified: it is ignored and flagged (“unchecked=...”) instead of causing all circuits to fail."""
    if not nominal:
        return 1, "NO_NOMINAL"
    if not values:
        return 1, "NO_MEASURE"
    fails, unchecked = [], []
    for k in names:
        v, ref = values.get(k), nominal.get(k)
        if _bad(ref):
            unchecked.append(k)
            continue
        if _bad(v):
            fails.append("%s(NaN)" % k)
            continue
        lo, hi = sorted((ref * (1 - tol), ref * (1 + tol)))
        if v < lo or v > hi:
            fails.append("%s(%.3e vs %.3e)" % (k, v, ref))
    if len(unchecked) == len(names):
        return 1, "NO_REFERENCE"
    note = (";unchecked=" + ",".join(unchecked)) if unchecked else ""
    return (1, ";".join(fails) + note) if fails else (0, "NONE" + note)
