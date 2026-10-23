"""Fault injection: parametric (Monte Carlo SAFE/EXTREME) and faults (short/open)."""
import re
from dataclasses import dataclass
from pathlib import Path

NUM = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?"

# Profils de variation multiplicative
SAFE = (0.95, 1.05)                       # ±5 % : mismatch 
EXTREME_LOW = (0.50, 0.70)                # -50 % à -30 %
EXTREME_HIGH = (1.30, 1.60)               # +30 % à +60 %


@dataclass
class Dut:
    filename: str
    path: str
    fault_class: str   # nominal | param_safe | param_extreme | short | open
    detail: str = ""


@dataclass
class Target:
    index: int
    name: str
    param: str


def read_lines(path):
    lines = Path(path).read_text(encoding="utf-8", errors="ignore").splitlines(keepends=True)
    if lines and not lines[-1].endswith("\n"):
        lines[-1] += "\n"
    return lines


def write_lines(path, lines) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("".join(lines), encoding="utf-8", errors="ignore")


def is_netlist_line(line: str) -> bool:
    s = line.strip()
    return bool(s) and s[0] not in "*.+;$"


def param_pattern(param: str):
    return re.compile(rf"(\b{param}\s*=\s*)(['\"]?)({NUM})([a-zA-Z]*)(['\"]?)", re.I)


def has_param(line: str, param: str) -> bool:
    return bool(param_pattern(param).search(line))


def scale_param(line: str, param: str, mult: float):
    
    m = param_pattern(param).search(line)
    if not m:
        return line, None
    new = float(m.group(3)) * mult
    rep = f"{m.group(1)}{m.group(2)}{new:.6g}{m.group(4)}{m.group(5)}"
    return line[:m.start()] + rep + line[m.end():], m.group(0)


def draw_multiplier(rng, safe: bool) -> float:
    if safe:
        return rng.uniform(*SAFE)
    return rng.uniform(*EXTREME_LOW) if rng.random() < 0.5 else rng.uniform(*EXTREME_HIGH)


def find_targets(lines, rule, subckt=None):
    """rule(line) -> (name, parameter) | None."""
    targets, inside = [], subckt is None
    for i, line in enumerate(lines):
        s = line.strip().lower()
        if subckt is not None:
            if s.startswith(f".subckt {subckt.lower()}"):
                inside = True
                continue
            if s.startswith(".ends"):
                inside = False
                continue
        if not inside or not is_netlist_line(line):
            continue
        r = rule(line)
        if r and has_param(line, r[1]):
            targets.append(Target(i, r[0], r[1]))
    return targets


def generate_parametric(lines, targets, n, max_mods, rng, out_dir, namer):
   
    if not targets:
        raise ValueError("No editable components were found in the netlist.")
    duts = []
    for i in range(1, n + 1):
        safe = i <= n // 2
        picked = rng.sample(targets, rng.randint(1, min(max_mods, len(targets))))
        cur, det = list(lines), []
        for t in picked:
            mult = draw_multiplier(rng, safe)
            cur[t.index], _ = scale_param(cur[t.index], t.param, mult)
            det.append(f"{t.name}.{t.param}:{'H' if mult > 1 else 'L'}({(mult - 1) * 100:+.0f}%)")
        fn = namer(i, safe)
        path = Path(out_dir) / fn
        write_lines(path, cur)
        duts.append(Dut(fn, str(path), "param_safe" if safe else "param_extreme", ";".join(det)))
    return duts


def two_terminal_defects(lines, is_target, out_dir, name_fmt, r_short, r_open):
    
    duts = []
    for idx, line in enumerate(lines):
        if not is_netlist_line(line) or not is_target(line):
            continue
        tok = line.split()
        if len(tok) < 3:
            continue
        name, n1, n2 = tok[0], tok[1], tok[2]
        ls = list(lines)
        ls.insert(idx + 1, f"R_SHORT_{name} {n1} {n2} {r_short}\n")
        fn = name_fmt.format(kind="SHORT", name=name)
        write_lines(Path(out_dir) / fn, ls)
        duts.append(Dut(fn, str(Path(out_dir) / fn), "short", f"{name}:{n1}-{n2}"))
        lo = list(lines)
        lo[idx] = "* " + lo[idx]
        lo.insert(idx + 1, f"R_OPEN_{name} {n1} {n2} {r_open}\n")
        fn = name_fmt.format(kind="OPEN", name=name)
        write_lines(Path(out_dir) / fn, lo)
        duts.append(Dut(fn, str(Path(out_dir) / fn), "open", f"{name}:{n1}-{n2}"))
    return duts


def mos_defects(lines, out_dir, name_fmt, r_short, r_open):
   
    duts = []
    for idx, line in enumerate(lines):
        if not is_netlist_line(line):
            continue
        tok = line.split()
        if len(tok) < 6 or not tok[0].lower().startswith("m"):
            continue
        name = tok[0]
        node = {"d": tok[1], "g": tok[2], "s": tok[3]}
        for a, b in (("d", "s"), ("g", "s"), ("d", "g")):
            if node[a] == node[b]:
                continue
            ls = list(lines)
            ls.insert(idx + 1, f"R_SHORT_{name}_{a}{b} {node[a]} {node[b]} {r_short}\n")
            fn = name_fmt.format(kind="SHORT", name=f"{name}_{a}{b}")
            write_lines(Path(out_dir) / fn, ls)
            duts.append(Dut(fn, str(Path(out_dir) / fn), "short", f"{name}:{a}-{b}"))
        for term, pos in (("d", 1), ("g", 2), ("s", 3)):
            tk = list(tok)
            new = f"OPEN_{name}_{term}"
            tk[pos] = new
            lo = list(lines)
            lo[idx] = (f"* DEFECT: {line.strip()}\n{' '.join(tk)}\n"
                       f"R_OPEN_{name}_{term} {new} {node[term]} {r_open}\n")
            fn = name_fmt.format(kind="OPEN", name=f"{name}_{term}")
            write_lines(Path(out_dir) / fn, lo)
            duts.append(Dut(fn, str(Path(out_dir) / fn), "open", f"{name}:{term}"))
    return duts
