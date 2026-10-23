"""CLI : list | generate | label | dataset | all | train | interactive."""
import argparse
import shutil
import sys
import time
import unicodedata

from . import pipeline as P
from .circuits import REGISTRY, get_circuit
from .stimuli import STIMULI

ALL_STEPS = ("generate", "label", "dataset")


class _AsciiFold(object):
    """Terminal non-UTF8 (locale C) : on retire les accents au lieu de planter."""

    def __init__(self, stream):
        self._s = stream

    def write(self, text):
        enc = getattr(self._s, "encoding", None) or "ascii"
        try:
            text.encode(enc)
        except (UnicodeEncodeError, LookupError):
            text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
        return self._s.write(text)

    def __getattr__(self, name):
        return getattr(self._s, name)


def _opts(a):
    return P.Opts(circuits_dir=a.circuits_dir, runs_dir=a.runs_dir, work_root=a.work_root, jobs=a.jobs,
                  limit=a.limit, seed=a.seed, timeout=a.timeout, resume=a.resume)


def _stimuli(arg):
    return list(STIMULI) if (not arg or arg == ["all"]) else [s.upper() for s in arg]


def cmd_list(a):
    o = _opts(a)
    sp = shutil.which("spectre")
    print("Spectre : %s" % ("OK (%s)" % sp if sp else "Not found in the PATH (SPECTRE_CMD variable)"))
    print("Stimuli : %s\n" % ", ".join(STIMULI))
    for name, cls in REGISTRY.items():
        c = cls(a.circuits_dir)
        miss = c.missing_files()
        rd = P.run_dir(o, name)
        state = ["manifest" if (rd / "manifest.csv").exists() else None,
                 "labels" if (rd / "labels.csv").exists() else None]
        state += [p.stem.replace("DATASET_", "ds:") for p in sorted((rd / "datasets").glob("DATASET_*.csv"))] \
            if (rd / "datasets").exists() else []
        print("  %-10s %s" % (name, c.description))
        print("  %-10s fichiers: %s | already produced: %s\n" % (
            "", "OK" if not miss else "MISSING -> " + ", ".join(miss),
            ", ".join(x for x in state if x) or "rien"))


def with_prereqs(circuit, steps, o):
    """Add the missing steps (e.g., dataset without labels.csv => label, or even generate)."""
    rd = P.run_dir(o, circuit)
    need = set(steps)
    if "dataset" in need and not (rd / "labels.csv").exists():
        need.add("label")
    if "label" in need and not (rd / "manifest.csv").exists():
        need.add("generate")
    full = [s for s in ALL_STEPS if s in need]
    added = [s for s in full if s not in steps]
    if added:
        print("[%s] Steps added automatically (prerequisites missing) : %s" % (circuit, ", ".join(added)))
    return full


def run_steps(circuit, steps, stimuli, o, n_param=None):
    for step in steps:
        t0 = time.time()
        if step == "generate":
            P.generate(circuit, o, n_param)
        elif step == "label":
            P.label(circuit, o)
        elif step == "dataset":
            P.dataset(circuit, stimuli, o)
        print("[%s] etape '%s' ended in %s\n" % (circuit, step, P.fmt_t(time.time() - t0)))


def cmd_train(a, circuit):
    try:
        from . import train as T
    except ImportError as e:
        raise P.PipelineError("scikit-learn/pandas missing (%s). Install : pip install -r requirements-train.txt" % e)
    o = _opts(a)
    ddir = P.run_dir(o, circuit) / "datasets"
    wanted = [s.upper() for s in a.stimuli] if a.stimuli and a.stimuli != ["all"] else None
    paths = sorted(ddir.glob("DATASET_*.csv")) if ddir.exists() else []
    if wanted:
        have = dict((p.stem.replace("DATASET_", ""), p) for p in paths)
        for w in wanted:
            if w not in have:
                print("[%s] ! dataset %s not found (start with 'dataset')" % (circuit, w))
        paths = [have[w] for w in wanted if w in have]
    if not paths:
        raise P.PipelineError("[%s] no dataset found in %s : Start by running the step 'dataset'." % (circuit, ddir))
    rd = P.run_dir(o, circuit)
    try:
        T.run(circuit, paths, label=a.label, models=a.models, features=a.features, residual=a.residual, cv=a.cv,
              test_size=a.test_size, seed=o.seed, results_dir=rd / "results", model_dir=rd / "models")
    except ValueError as e:
        raise P.PipelineError("[%s] %s" % (circuit, e))


def _pick(prompt, items):
    sel = input(prompt).strip().lower()
    return list(items) if sel == "all" else [items[int(x) - 1] for x in sel.split(",")]


def cmd_interactive(a):
    names = list(REGISTRY)
    print("Available circuits  :")
    for i, n in enumerate(names, 1):
        print("  %d. %-10s %s%s" % (i, n, REGISTRY[n].description,
                                    "   [missing files]" if get_circuit(n, a.circuits_dir).missing_files() else ""))
    chosen = _pick("Circuits (ex: 1,3 ou 'all') : ", names)
    print("Stimuli : " + ", ".join("%d.%s" % (i, s) for i, s in enumerate(STIMULI, 1)))
    stim = _pick("Stimuli (ex: 2,3 ou 'all') : ", list(STIMULI))
    allsteps = list(ALL_STEPS) + ["train"]
    print("Steps : " + ", ".join("%d.%s" % (i, s) for i, s in enumerate(allsteps, 1)))
    steps = _pick("Steps (ex: 1,2,3 ou 'all') : ", allsteps)
    lim = input("Limit to N circuits for a quick test (empty = all) : ").strip()
    o = _opts(a)
    o.limit = int(lim) if lim else None
    if not o.resume and input("Resume existing results if available ? [y/N] ").strip().lower() in ("o", "y"):
        o.resume = True
    if input("Run %s / %s / %s ? [y/N] " % (chosen, stim, steps)).strip().lower() not in ("o", "y", "oui", "yes"):
        return
    for c in chosen:
        run_steps(c, with_prereqs(c, [x for x in steps if x != "train"], o) if set(steps) - {"train"} else [], stim, o)
        if "train" in steps:
            ns = argparse.Namespace(**vars(a))
            ns.stimuli, ns.label, ns.models, ns.features = stim, "hard_label", ["logreg", "random_forest", "mlp"], "raw"
            ns.residual, ns.cv, ns.test_size = False, 0, 0.2
            cmd_train(ns, c)


def build_parser():
    p = argparse.ArgumentParser(prog="run_pipeline.py", description=__doc__)
    p.add_argument("--circuits-dir", default="circuits")
    p.add_argument("--runs-dir", default="runs")
    p.add_argument("--work-root", default="/tmp/amsml_work", help="temporary directory for Spectre simulations (default: /tmp/amsml_work)")
    p.add_argument("--jobs", type=int, default=1, help="simultaneous Spectre simulations (default: 1)")
    p.add_argument("--limit", type=int, default=None, help="limit to N circuits for a quick test (default: all)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--timeout", type=int, default=300, help="seconds max for a single Spectre simulation ")
    p.add_argument("--resume", action="store_true", help="resume : do not re-run what is already in the CSV files")
    sub = p.add_subparsers(dest="cmd")
    sub.required = True
    sub.add_parser("list")
    sub.add_parser("interactive")
    st = sub.add_parser("train", help="phase 2 : train ML models on the datasets (phase 1 must be done first)")
    st.add_argument("--circuit", nargs="+", required=True)
    st.add_argument("--stimuli", nargs="+", default=["all"], help="datasets to use (default: all)")
    st.add_argument("--label", default="hard_label", choices=["hard_label", "spec_label", "checker_label", "fault_class"])
    st.add_argument("--models", nargs="+", default=["logreg", "random_forest", "mlp"],
                    choices=["logreg", "svm", "random_forest", "gboost", "mlp"])
    st.add_argument("--features", default="raw", choices=["raw", "stats", "raw+stats"])
    st.add_argument("--residual", action="store_true", help="subtracts the response of the nominal circuit (same corner)")
    st.add_argument("--cv", type=int, default=0, help="K-fold cross-validation (default: a single holdout set)")
    st.add_argument("--test-size", type=float, default=0.2)
    for name in ("generate", "label", "dataset", "all"):
        s = sub.add_parser(name)
        s.add_argument("--circuit", nargs="+", required=True, help="name(s) or 'all'")
        if name in ("generate", "all"):
            s.add_argument("--n-param", type=int, default=None, help="number of parametric DUTs to generate ")
        if name in ("dataset", "all"):
            s.add_argument("--stimuli", nargs="+", default=["all"], help="between %s or all" % ", ".join(STIMULI))
        if name == "all":
            s.add_argument("--notify", action="store_true", help="e-mail notification (variables SMTP_*)")
    return p


def main(argv=None):
    sys.stdout, sys.stderr = _AsciiFold(sys.stdout), _AsciiFold(sys.stderr)
    a = build_parser().parse_args(argv)
    try:
        if a.cmd == "list":
            return cmd_list(a)
        if a.cmd == "interactive":
            return cmd_interactive(a)
        circuits = list(REGISTRY) if a.circuit == ["all"] else a.circuit
        if a.cmd == "train":
            for c in circuits:
                cmd_train(a, c)
            return
        o = _opts(a)
        steps = ALL_STEPS if a.cmd == "all" else (a.cmd,)
        t0 = time.time()
        for c in circuits:
            run_steps(c, steps, _stimuli(getattr(a, "stimuli", None)), o, n_param=getattr(a, "n_param", None))
        if getattr(a, "notify", False):
            from .notify import send
            send("[amsml] termine: %s" % ", ".join(circuits), "Duree: %s" % P.fmt_t(time.time() - t0))
    except KeyboardInterrupt:
        print("\nInterrupted (Ctrl-C). Results that have already been calculated are preserved: rerun the same command"
              "with --resume (global option, before the subcommand).")
        sys.exit(130)
    except (P.PipelineError, KeyError) as e:
        print("\nERREUR : %s" % (e.args[0] if e.args else e))
        sys.exit(2)


if __name__ == "__main__":
    main()
