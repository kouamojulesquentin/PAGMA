"""Pipeline stages: generate -> label -> dataset"""
import csv
import datetime
import json
import random
import shutil
import sys
import time
import os
from collections import Counter, defaultdict
from dataclasses import dataclass
from multiprocessing import Pool
from pathlib import Path
from typing import Optional

from .circuits import get_circuit
from .spectre_io import install_worker_signals, resample
from .stimuli import make_stimulus

LABEL_COLS = ["filename", "corner", "fault_class", "detail", "status",
              "hard_label", "spec_label", "checker_label", "failed_params"]
DATA_KEYS = ["filename", "corner", "fault_class", "hard_label", "spec_label", "checker_label"]
MAX_KEPT_FAILS = 10


class PipelineError(Exception):
    """User error """


@dataclass
class Opts:
    circuits_dir: str = "circuits"
    runs_dir: str = "runs"
    work_root: str = "/tmp/amsml_work"
    jobs: int = 1
    limit: Optional[int] = None
    seed: int = 0
    timeout: int = 300
    resume: bool = False


def run_dir(o, name):
    return Path(o.runs_dir) / name


def read_csv(path):
    with open(str(path), newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def fmt_t(sec):
    return str(datetime.timedelta(seconds=int(sec)))


def stratified(rows, limit, key="fault_class"):
    """Balanced selection across classes (quick test using --limit)."""
    if not limit or limit >= len(rows):
        return rows
    groups = defaultdict(list)
    for r in rows:
        groups[r[key]].append(r)
    out, i = [], 0
    while len(out) < limit and any(groups.values()):
        for g in list(groups.values()):
            if i < len(g) and len(out) < limit:
                out.append(g[i])
        i += 1
    return out


def select(rows, o):
    """--limit: stratified sample; otherwise, shuffled order."""
    nom = [r for r in rows if r["fault_class"] == "nominal"]
    rest = [r for r in rows if r["fault_class"] != "nominal"]
    if o.limit:
        return nom + stratified(rest, o.limit)
    random.Random(o.seed).shuffle(rest)
    return nom + rest


# ------------------------------------------------------------------ progression
class Progress(object):

    def __init__(self, total, desc):
        self.total, self.desc, self.n = total, desc, 0
        self.t0, self.last = time.time(), 0.0
        self.status = Counter()
        self.tty = sys.stdout.isatty()

    def _line(self):
        el = time.time() - self.t0
        rate = self.n / el if el > 0 else 0.0
        eta = (self.total - self.n) / rate if rate > 0 else 0
        frac = self.n / float(self.total)
        k = int(frac * 12)
        stats = " ".join("%s:%d" % kv for kv in sorted(self.status.items()))
        line = "[%s] [%s%s] %d/%d %.1f%% %.1f/s ETA %s %s" % (
            self.desc, "#" * k, "-" * (12 - k), self.n, self.total, frac * 100, rate, fmt_t(eta), stats)
        if self.tty:   # une ligne qui deborde du terminal ne peut plus etre reecrite sur place
            return line[:max(20, shutil.get_terminal_size((80, 24)).columns - 1)]
        return "%s | ecoule %s" % (line, fmt_t(el))

    def show(self):
        if self.tty:
            sys.stdout.write("\r" + self._line() + "\033[K")
        else:
            print(self._line())
        sys.stdout.flush()

    def update(self, status):
        self.n += 1
        self.status[status.split(":")[0]] += 1
        now = time.time()
        if now - self.last >= (0.5 if self.tty else 30) or self.n == self.total:
            self.last = now
            self.show()

    def note(self, msg):
        if self.tty:
            sys.stdout.write("\r\033[K")
        print(msg)
        if self.tty:
            self.show()

    def close(self):
        if self.tty:
            sys.stdout.write("\n")
            sys.stdout.flush()


def pmap(fn, tasks, jobs, desc, status_of):

    total = len(tasks)
    if total == 0:
        return
    prog = Progress(total, desc)
    pool = Pool(jobs, initializer=install_worker_signals) if jobs > 1 else None
    gen = pool.imap_unordered(fn, tasks, chunksize=1) if pool else map(fn, tasks)
    seen_err = [0]
    try:
        for r in gen:
            st = status_of(r)
            prog.update(st)
            if st.startswith("ERR") and seen_err[0] < 3:
                seen_err[0] += 1
                prog.note("  ! %s" % st[:300])
            yield r
        if pool:
            pool.close()
        prog.close()
    except BaseException:
        if pool:
            pool.terminate()
        prog.close()
        raise
    finally:
        if pool:
            pool.join()


# ------------------------------------------------------------------ workers
def _workdir(root, tag):
    p = Path(root) / ("%s_%d_%d" % (tag, os.getpid(), int(time.time() * 1e6)))
    p.mkdir(parents=True, exist_ok=True)
    return p


def _keep_fail(work, fail_dir, tag):

    try:
        d = Path(fail_dir)
        d.mkdir(parents=True, exist_ok=True)
        if len([x for x in d.iterdir() if x.is_dir()]) >= MAX_KEPT_FAILS:
            return
        dst = d / tag
        dst.mkdir(exist_ok=True)
        for pat in ("*.log", "*.circuit", "*.sp", "*.hsp", "*.va", "*.mt0", "*.measure"):
            for f in Path(work).glob(pat):
                shutil.copy(str(f), str(dst / f.name))
    except OSError:
        pass


def _label_worker(t):
    circ = get_circuit(t["circuit"], t["circuits_dir"])
    work = _workdir(t["work_root"], "lab")
    try:
        try:
            m = circ.measure(t["dut_path"], t["corner"], work, t["timeout"])
        except Exception as e:  # noqa: BLE001
            m = {"status": "ERR:%s:%s" % (type(e).__name__, e), "values": {}, "checker": None}
        if m["status"] != "OK":
            _keep_fail(work, t["fail_dir"], "%s_%s_label" % (Path(t["filename"]).stem, t["corner"]))
        elif t["fault_class"] == "nominal":
            _keep_fail(work, t["fail_dir"].replace("failed_logs", "nominal_debug"), t["corner"] or "nominal")
        return t, m
    finally:
        shutil.rmtree(str(work), ignore_errors=True)


def _wave_worker(t):
    circ = get_circuit(t["circuit"], t["circuits_dir"])
    work = _workdir(t["work_root"], "wav")
    try:
        try:
            res = circ.waveform(t["dut_path"], t["corner"], t["stim"], work, t["timeout"])
            if isinstance(res, str):
                st, vec = res, None
            else:
                vec = resample(res[0], res[1], t["stim"].grid())
                st = "OK" if vec is not None else "INCOMPLETE"
        except Exception as e:  # noqa: BLE001
            st, vec = "ERR:%s:%s" % (type(e).__name__, e), None
        if st != "OK":
            _keep_fail(work, t["fail_dir"], "%s_%s_%s" % (Path(t["filename"]).stem, t["corner"], t["stim"].name))
        return t, st, vec
    finally:
        shutil.rmtree(str(work), ignore_errors=True)


def _base(circ, name, o):
    return dict(circuit=name, circuits_dir=o.circuits_dir, work_root=o.work_root, timeout=o.timeout,
                fail_dir=str(run_dir(o, name) / "failed_logs"))


# ------------------------------------------------------------------ generate
def generate(name, o, n_param=None):
    circ = get_circuit(name, o.circuits_dir)
    rd = run_dir(o, name)
    if o.resume and (rd / "manifest.csv").exists():
        print("[%s]  existing manifest (--resume)." % name)
        return None
    missing = circ.missing_files()
    if missing:
        raise PipelineError("[%s] missing files in %s: %s" % (name, circ.root, missing))
    duts_dir = rd / "duts"
    if duts_dir.exists():
        shutil.rmtree(str(duts_dir))
    duts_dir.mkdir(parents=True)
    for stale in ("labels.csv", "nominal.json"):   
        if (rd / stale).exists():
            (rd / stale).unlink()
            print("[%s] %s Remove (obsolete after regeneration)." % (name, stale))
    duts = circ.generate(duts_dir, n_param or circ.default_n_param, random.Random(o.seed))
    with open(str(rd / "manifest.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["filename", "fault_class", "detail"])
        for d in duts:
            w.writerow([d.filename, d.fault_class, d.detail])
    print("[%s] %d DUT generated -> %s  %s" % (name, len(duts), rd / "manifest.csv",
                                           dict(Counter(d.fault_class for d in duts))))
    return duts


# ------------------------------------------------------------------ label
def _tasks(circ, name, o, rows):
    base = _base(circ, name, o)
    for r in rows:
        for c in circ.corners:
            yield dict(base, dut_path=str((run_dir(o, name) / "duts" / r["filename"]).resolve()),
                       filename=r["filename"], corner=c, fault_class=r["fault_class"], detail=r.get("detail", ""))


def label(name, o):
    circ = get_circuit(name, o.circuits_dir)
    rd = run_dir(o, name)
    if not (rd / "manifest.csv").exists():
        raise PipelineError("[%s] manifest.csv not found: run the step first 'generate'." % name)
    manifest = read_csv(rd / "manifest.csv")
    noms = [r for r in manifest if r["fault_class"] == "nominal"]
    others = select([r for r in manifest if r["fault_class"] != "nominal"], o)
    lpath, npath = rd / "labels.csv", rd / "nominal.json"
    cols = LABEL_COLS + ["meas_%s" % k for k in circ.measure_names]

    done, nominal = set(), {}
    resuming = o.resume and lpath.exists()
    if resuming:
        done = set((r["filename"], r["corner"]) for r in read_csv(lpath))
        if npath.exists():
            nominal = json.loads(npath.read_text(encoding="utf-8"))
        print("[%s] Resume: %d steps already taken." % (name, len(done)))
    f = open(str(lpath), "a" if resuming else "w", newline="", encoding="utf-8")
    w = csv.DictWriter(f, fieldnames=cols)
    if not resuming:
        w.writeheader()

    def emit(t, m):
        if m["status"] == "OK":
            spec, why = circ.verdict(m, nominal.get(t["corner"]))
        else:
            spec, why = 1, m["status"]
        chk = m.get("checker")
        row = {"filename": t["filename"], "corner": t["corner"], "fault_class": t["fault_class"],
               "detail": t["detail"], "status": m["status"],
               "hard_label": circ.hard_label(t["fault_class"], spec, chk), "spec_label": spec,
               "checker_label": "" if chk is None else chk, "failed_params": why}
        for k in circ.measure_names:
            v = m["values"].get(k)
            row["meas_%s" % k] = "" if v is None else "%.6e" % v
        w.writerow(row)
        f.flush()

    t0 = time.time()
    try:
        nom_tasks = [t for t in _tasks(circ, name, o, noms) if (t["filename"], t["corner"]) not in done]
        if nom_tasks:
            print("[%s] nominal circuit measurement (%d simulations)..." % (name, len(nom_tasks)))
            results = list(pmap(_label_worker, nom_tasks, o.jobs, "nominal", lambda r: r[1]["status"]))
            for t, m in results:
                if m["status"] == "OK":
                    nominal[t["corner"]] = m["values"]
            bad = [(t, m) for t, m in results if m["status"] != "OK"]
            if bad:
                raise PipelineError(
                    "[%s] The NOMINAL circuit does not simulate (corner '%s': %s). There is no point in continuing. : "
                    "voir %s/failed_logs/ (spectre.log + netlist)." % (
                        name, bad[0][0]["corner"], bad[0][1]["status"][:200], rd))
            absent = [n for n in circ.measure_names if any(n not in m["values"] for _, m in results)]
            if absent:
                keys = next((m.get("raw_keys") for _, m in results if m.get("raw_keys") is not None), [])
                raise PipelineError(
                    "[%s] measurement(s) %s MISSING from the results Nominal circuit spectrum.\n"
                    "  Available measurements: %s\n  Correct the name in the bench file (.MEAS) or in amsml/circuits/%s.py "
                    "(measure_names / MEASURE_ALIASES). Raw files: %s/nominal_debug/" % (
                        name, ", ".join(absent), ", ".join(keys) or "(none)", name, rd))
            npath.write_text(json.dumps(nominal, indent=2), encoding="utf-8")
            for t, m in results:
                emit(t, m)
        for c, vals in sorted(nominal.items()):
            print("  nominal[%s] : %s" % (c or "-", ", ".join("%s=%.4g" % kv for kv in sorted(vals.items()))))
        nan_c = [c for c, v in sorted(nominal.items()) if not v or any(x != x for x in v.values())]
        if nan_c:
            dbg = rd / "nominal_debug"
            print("  ! WARNING: Nominal measurements are NaN or missing (corners: %s). These measurements will NOT be verified." % ", ".join(nan_c))
            print("    Raw files saved in %s/<coin>/ . Extrait (%s) :" % (dbg, nan_c[0] or "nominal"))
            for f_ in sorted((dbg / (nan_c[0] or "nominal")).glob("*.m*"))[:2]:
                print("    --- %s" % f_.name)
                for l_ in f_.read_text(encoding="utf-8", errors="ignore").splitlines()[:12]:
                    print("    | " + l_[:150])

        tasks = [t for t in _tasks(circ, name, o, others) if (t["filename"], t["corner"]) not in done]
        print("[%s] %d Labeling simulations (%d in parallel). Ctrl-C = clean shutdown, "
              "Restart with --resume to continue." % (name, len(tasks), o.jobs))
        for t, m in pmap(_label_worker, tasks, o.jobs, "label", lambda r: r[1]["status"]):
            emit(t, m)
    finally:
        f.close()

    rows = read_csv(lpath)
    print("[%s] labels -> %s (%d lines, %s)" % (name, lpath, len(rows), fmt_t(time.time() - t0)))
    print("  statuts   :", dict(Counter(r["status"].split(":")[0] for r in rows)))
    print("  hard_label:", dict((c, dict(Counter(r["hard_label"] for r in rows if r["fault_class"] == c)))
                               for c in sorted(set(r["fault_class"] for r in rows))))
    nb_bad = sum(1 for r in rows if r["status"] != "OK")
    if rows and nb_bad > 0.2 * len(rows):
        print("  ! %d/%d simulations no-OK : show %s/failed_logs/" % (nb_bad, len(rows), rd))
    return rows


# ------------------------------------------------------------------ dataset
def dataset(name, stimuli, o):
    circ = get_circuit(name, o.circuits_dir)
    rd = run_dir(o, name)
    if not (rd / "labels.csv").exists():
        raise PipelineError("[%s] labels.csv not found: First run the “generate” step, then the “label” step." % name)
    rows = select(read_csv(rd / "labels.csv"), o)
    out_dir = rd / "datasets"
    out_dir.mkdir(exist_ok=True)
    outputs = []
    for kind in stimuli:
        try:
            stim = make_stimulus(kind, circ.stim_config())
        except ValueError as e:
            if kind.upper() == "DC":
                print("[%s] DC stimulus ignored (%s)" % (name, e))
                continue
            raise PipelineError("[%s] stimulus %s: %s" % (name, kind, e))
        path = out_dir / ("DATASET_%s.csv" % stim.name)
        header = DATA_KEYS + ["t_%d" % i for i in range(stim.n_points)]
        done, resuming = set(), o.resume and path.exists()
        if resuming:
            with open(str(path), newline="", encoding="utf-8") as fr:
                rd_ = csv.reader(fr)
                if next(rd_) != header:
                    raise PipelineError("%s : If the header is different, delete the file or restart without --resume." % path)
                done = set((r[0], r[1]) for r in rd_)
        tasks = [dict(_base(circ, name, o), stim=stim, dut_path=str((rd / "duts" / r["filename"]).resolve()),
                      **dict((k, r[k]) for k in DATA_KEYS))
                 for r in rows if (r["filename"], r["corner"]) not in done]
        print("[%s] stimulus %s: %d simulations (%d already completed), %d points/vector" % (
            name, stim.name, len(tasks), len(done), stim.n_points))
        ok, status, t0 = len(done), Counter(), time.time()
        f = open(str(path), "a" if resuming else "w", newline="", encoding="utf-8")
        try:
            w = csv.writer(f)
            if not resuming:
                w.writerow(header)
            for t, st, vec in pmap(_wave_worker, tasks, o.jobs, stim.name, lambda r: r[1]):
                status[st.split(":")[0]] += 1
                if vec is not None:
                    w.writerow([t[k] for k in DATA_KEYS] + ["%.6g" % x for x in vec])
                    f.flush()
                    ok += 1
        finally:
            f.close()
        inp = stim.sample_input()
        meta = {"circuit": name, "stimulus": stim.name, "t_start": stim.t_obs_start, "t_stop": stim.t_stop,
                "n_points": stim.n_points, "n_rows": ok, "status_last_run": dict(status),
                "time": stim.grid().tolist(), "input": None if inp is None else inp.tolist()}
        path.with_suffix(".meta.json").write_text(json.dumps(meta), encoding="utf-8")
        print("  -> %s  (%d lignes, %s ; %s)" % (path, ok, dict(status), fmt_t(time.time() - t0)))
        outputs.append(path)
    return outputs
