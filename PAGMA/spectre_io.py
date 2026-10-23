"""Spectre Inputs/Outputs: launch, parsing .print / .measure / .mt0, includes."""
import os
import re
import shlex
import signal
import subprocess
from pathlib import Path

import numpy as np

UNITS = {"f": 1e-15, "p": 1e-12, "n": 1e-9, "u": 1e-6, "m": 1e-3,
         "k": 1e3, "M": 1e6, "G": 1e9, "s": 1.0}


def spectre_cmd() -> str:
    return os.environ.get("SPECTRE_CMD", "spectre")


def _num(parts, i):
    v = float(parts[i])
    if len(parts) > i + 1 and parts[i + 1] in UNITS:
        return v * UNITS[parts[i + 1]], i + 2
    return v, i + 1


def parse_print(path):

    p = Path(path)
    t, v = [], []
    if not p.exists():
        return np.array([]), np.array([])
    for line in p.read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = line.split()
        if not parts or not (parts[0][0].isdigit() or parts[0][0] in ".-"):
            continue
        try:
            tv, i = _num(parts, 0)
            if len(parts) <= i:
                continue
            vv, _ = _num(parts, i)
        except ValueError:
            continue
        t.append(tv)
        v.append(vv)
    t, v = np.array(t), np.array(v)
    order = np.argsort(t, kind="stable")
    return t[order], v[order]


def resample(t, v, grid):
    
    if len(t) < 2:
        return None
    span = grid[-1] - grid[0]
    if t[-1] < grid[-1] - 0.02 * span:
        return None
    return np.interp(grid, t, v)


_SPICE = {"t": 1e12, "g": 1e9, "x": 1e6, "k": 1e3, "m": 1e-3, "u": 1e-6, "n": 1e-9, "p": 1e-12, "f": 1e-15, "a": 1e-18}
_BAD = ("nan", "-nan", "failed", "fail", "inf", "-inf", "+inf", "undefined")
_NUMRE = re.compile(r"([-+]?(?:\d+\.?\d*|\.\d+)(?:e[-+]?\d+)?)([a-z]*)")


def spice_float(tok: str) -> float:

    t = tok.strip().strip(",;").lower()
    if t in _BAD:
        return float("nan")
    m = _NUMRE.fullmatch(t)
    if not m:
        raise ValueError(tok)
    suf = m.group(2)
    mult = 1e6 if suf.startswith("meg") else (_SPICE.get(suf[:1], 1.0) if suf else 1.0)
    return float(m.group(1)) * mult


_KV = re.compile(r"([A-Za-z0-9_@\.]+)\s*=\s*([^\s,;]+)")


def parse_measure(path) -> dict:

    p = Path(path)
    if not p.exists():
        return {}
    text = p.read_text(encoding="utf-8", errors="ignore")
    out = {}
    lines = [l for l in text.splitlines() if l.strip() and not l.lstrip().startswith(("*", "#", "//", "$", "."))]
    if len(lines) >= 2:
        names, vals = lines[-2].split(), lines[-1].split()
        if len(names) == len(vals) and "=" not in lines[-2]:
            for n, v in zip(names, vals):
                try:
                    out[n.lower()] = spice_float(v)
                except ValueError:
                    pass
    for k, v in _KV.findall(text):
        try:
            out[k.lower()] = spice_float(v)
        except ValueError:
            pass
    return out


def find_measures(work: Path) -> dict:
    out = {}
    for pat in ("*.mt0", "*.measure"):
        for f in sorted(Path(work).glob(pat)):
            out.update(parse_measure(f))
    return out


def find_print(work: Path):
    files = sorted(Path(work).glob("*.print"))
    return parse_print(files[0]) if files else (np.array([]), np.array([]))


_INC = re.compile(r"^(\s*\.(?:include|inc|hdl)\s+)([\"']?)([^\"'\s]+)\2", re.I | re.M)


def absolutize_includes(text: str, base_dir) -> str:

    base = Path(base_dir).resolve()

    def fix(m):
        p = Path(m.group(3))
        if not p.is_absolute():
            p = (base / p)
        return f'{m.group(1)}"{p}"'

    return _INC.sub(fix, text)


def sub_once(pattern, repl, text, what, flags=re.I | re.M):
    new, n = re.subn(pattern, lambda _m: repl, text, flags=flags)
    if n == 0:
        raise ValueError(f"Motif introuvable dans le testbench ({what}): {pattern}")
    return new


_CHILD = [None]


def _kill_child():
    p = _CHILD[0]
    if p is not None and p.poll() is None:
        try:
            os.killpg(p.pid, signal.SIGKILL)   # tue Spectre et ses sous-processus
        except OSError:
            pass


def install_worker_signals():

    signal.signal(signal.SIGINT, signal.SIG_IGN)

    def on_term(*_a):
        _kill_child()
        os._exit(1)

    signal.signal(signal.SIGTERM, on_term)


def run_spectre(netlist: str, cwd, timeout: int, extra=()) -> str:
    cmd = shlex.split(spectre_cmd()) + list(extra) + [netlist]
    with open(str(Path(cwd) / "spectre.log"), "w") as lf:
        try:

            p = subprocess.Popen(cmd, cwd=str(cwd), stdout=lf, stderr=subprocess.STDOUT, start_new_session=True)
        except FileNotFoundError:
            raise RuntimeError("Spectre executable not found (%s). Set SPECTRE_CMD." % cmd[0])
        _CHILD[0] = p
        try:
            p.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            _kill_child()
            p.wait()
            return "TIMEOUT"
        except BaseException:
            _kill_child()
            raise
        finally:
            _CHILD[0] = None
    return "OK" if p.returncode == 0 else "CRASH"
