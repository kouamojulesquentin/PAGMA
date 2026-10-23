import random
import re
from pathlib import Path

import numpy as np
import pytest

from amsml import pipeline as P
from amsml.circuits import get_circuit
from amsml.circuits import bandgap as bg
from amsml.faults import find_targets, scale_param
from amsml.spectre_io import absolutize_includes, parse_measure, parse_print, resample
from amsml.stimuli import STIMULI, StimConfig, make_stimulus, prbs7

ROOT = Path(__file__).resolve().parents[1]


def cfg(**kw):
    d = dict(v_low=0.0, v_high=1.0, bit_time=50e-6, window=1e-3, n_points=200, rise=1e-6)
    d.update(kw)
    return StimConfig(**d)


def test_prbs7_period_and_balance():
    s = prbs7()
    assert len(s) == 127 and sum(s) == 64
    states = {tuple(s[i:i + 7]) for i in range(127)}
    assert len(states) == 127  # séquence maximale


@pytest.mark.parametrize("kind", [k for k in STIMULI if k != "DC"])
def test_stimuli_valid(kind):
    st = make_stimulus(kind, cfg())
    t = np.array([p[0] for p in st.points])
    v = np.array([p[1] for p in st.points])
    assert np.all(np.diff(t) > 0) and t[-1] >= st.t_stop - 1e-12
    assert v.min() >= -1e-9 and v.max() <= 1 + 1e-9
    assert st.sample_input().shape == (200,)


def test_APRMLS_has_8_levels():
    st = make_stimulus("APRMLS", cfg(window=127 * 50e-6, n_points=2000))
    assert len({round(v, 6) for _, v in st.points}) == 8


def test_nyquist_guard():
    with pytest.raises(ValueError):
        make_stimulus("SINUS", cfg(sine_freq=1e6))


def test_scale_param_keeps_suffix_and_quotes():
    l, old = scale_param("M1 a b c d nmos W=1.5u L=1u\n", "W", 2.0)
    assert "W=3u" in l and "L=1u" in l
    l, _ = scale_param("R1 a b R='400n'\n", "R", 0.5)
    assert "R='200n'" in l


def test_parsers(tmp_path):
    p = tmp_path / "x.print"
    p.write_text("time v\n0 1.65\n5 u 1.66\n10 u 500 m\n")
    t, v = parse_print(p)
    assert np.allclose(t, [0, 5e-6, 10e-6]) and np.allclose(v, [1.65, 1.66, 0.5])
    assert resample(t, v, np.linspace(0, 10e-6, 5)) is not None
    assert resample(t, v, np.linspace(0, 100e-6, 5)) is None  
    m = tmp_path / "a.measure"
    m.write_text("voffset = 1.2e-4\ndelay = failed\n")
    assert parse_measure(m)["voffset"] == 1.2e-4 and np.isnan(parse_measure(m)["delay"])
    m2 = tmp_path / "b.mt0"
    m2.write_text("* c\ngain_20k gain_100k\n1.04 1.56\n")
    assert parse_measure(m2) == {"gain_20k": 1.04, "gain_100k": 1.56}


def test_absolutize_includes(tmp_path):
    out = absolutize_includes('.include "model/a.sp"\n.INC b.sp\n.include "/abs/c.sp"\n', tmp_path)
    assert str(tmp_path.resolve() / "model/a.sp") in out and "/abs/c.sp" in out and "b.sp" in out


def test_demo_end_to_end(tmp_path):
    o = P.Opts(circuits_dir=str(ROOT / "circuits"), runs_dir=str(tmp_path), work_root=str(tmp_path / "w"), seed=1)
    P.generate("demo_rc", o, 20)
    rows = P.label("demo_rc", o)
    by = {r["fault_class"]: r for r in rows}
    assert by["nominal"]["spec_label"] == "0" and by["short"]["hard_label"] == "1"
    assert all(r["hard_label"] == "0" for r in rows if r["fault_class"] == "param_safe")
    out = P.dataset("demo_rc", ["PRBS", "APRMLS"], o)
    assert all(p.exists() for p in out)
    head = out[0].read_text().splitlines()
    assert head[0].endswith("t_199") and len(head) == 1 + len(rows)


def test_filter_generation(tmp_path):
    c = tmp_path / "filter" / "subcircuit_files"
    c.mkdir(parents=True)
    (c / "FILTER.sub").write_text(
        ".subckt FILTER in out\nxr1 in n1 primitive_rpoly2 L=10u W=1u\nxr2 n1 out primitive_rpoly2 L=20u W=1u\n"
        "xc1 out 0 primitive_cpoly AREA=100p\n.ends\n")
    circ = get_circuit("filter", tmp_path)
    duts = circ.generate(tmp_path / "o", 10, random.Random(0))
    kinds = [d.fault_class for d in duts]
    assert kinds.count("short") == 3 and kinds.count("open") == 3 and kinds.count("nominal") == 1
    assert circ.hard_label("param_extreme", 0, None) == 1


def test_opamp_generation_and_bench(tmp_path):
    c = tmp_path / "opamp1"
    (c / "subcircuit_files").mkdir(parents=True)
    (c / "subcircuit_files/OPAMP1.sub").write_text(
        ".subckt OPAMP1 inp inn out\nM1 d1 inp s b nmos W=2u L=1u\nM2 out d1 vss vss nmos W=4u L=1u\n"
        "C1 out 0 AREA=1p\n.ends\n")
    (c / "OPAMP1_ML.circuit").write_text(
        '.INCLUDE "subcircuit_files/OPAMP1.sub"\n.INCLUDE "process.sp"\nVin inp 0 DC 1.65\n.TRAN 1n 1u\n.PRINT TRAN V(out)\n')
    circ = get_circuit("opamp1", tmp_path)
    duts = circ.generate(tmp_path / "o", 6, random.Random(0))
    assert sum(d.fault_class == "short" for d in duts) == 6 and sum(d.fault_class == "open" for d in duts) == 6
    import amsml.circuits.opamp1 as mod
    mod.run_spectre = lambda *a, **k: "OK"
    w = tmp_path / "w"
    w.mkdir()
    (w / "x.print").write_text("\n".join(f"{i} n {1.5}" for i in range(20)))
    stim = make_stimulus("PRBS", circ.stim_config())
    t, v = circ.waveform(duts[0].path, "", stim, w, 10)
    txt = (w / "OPAMP1_ML.circuit").read_text()
    assert "PWL (" in txt and duts[0].filename in txt and str(c.resolve() / "process.sp") in txt
    assert re.search(r"^\.TRAN 1e-08 7e-06 0 strobeperiod=1e-08", txt, re.M)


def test_bandgap_generation_and_patching(tmp_path, monkeypatch):
    c = tmp_path / "bandgap"
    (c / "src").mkdir(parents=True)
    (c / "ref").mkdir()
    (c / "src/tb_bandgap.hsp").write_text(
        ".subckt bandgap vdd out\nM1 a b c d nana W=1u L=1u\nR1 a out R='400n'\nQ1 a b c latpnp M=8\n.ends\n"
        "XI91 x y supplies\nV3 net033 0 DC 2.5\n" + bg.MARKER + "\n")
    (c / "src/top.hsp").write_text('.inc "model/m.sp"\n.include "tb_bandgap.hsp"\n.dc temp 0 1 1\n')
    (c / "ref/defect_short.txt").write_text("Defect_S1, a, b\n")
    (c / "ref/defect_open.txt").write_text("Defect_O1, x, I0.M1-x, nana, d\nDefect_O2, x, I0.MZ-x, nana, d\n")
    circ = get_circuit("bandgap", tmp_path)
    duts = circ.generate(tmp_path / "o", 4, random.Random(0))
    kinds = [d.fault_class for d in duts]
    assert kinds.count("short") == 1 and kinds.count("open") == 1   
    assert "R='" in Path(duts[1].path).read_text()                  

    def fake_spectre(net, cwd, timeout, extra=()):
        rows = [f"{i*1e-4:e} {0.0 if i*1e-4 < 0.02 else 1.4} 0.0" for i in range(1100)]
        (Path(cwd) / "data.csv").write_text("\n".join(rows))
        return "OK"

    monkeypatch.setattr(bg, "run_spectre", fake_spectre)
    w = tmp_path / "w"
    w.mkdir()
    m = circ.measure(duts[0].path, "", w, 5)
    assert m["status"] == "OK" and m["checker"] == 0 and circ.verdict(m, None)[0] == 0
    w2 = tmp_path / "w2"
    w2.mkdir()
    stim = make_stimulus("PRBS", circ.stim_config())
    circ.waveform(duts[0].path, "", stim, w2, 5)
    tb, top = (w2 / "tb.hsp").read_text(), (w2 / "run.sp").read_text()
    assert "*XI91" in tb and "*V3 net033" in tb and "Vprbs_ana VDDa! 0 PWL" in tb and "Xcspy bg fail" in tb
    assert top.count(".tran") == 1 and ".dc" not in top and str(c.resolve() / "src/model/m.sp") in top


def _opts(tmp_path, **kw):
    return P.Opts(circuits_dir=str(ROOT / "circuits"), runs_dir=str(tmp_path / "runs"),
                  work_root=str(tmp_path / "w"), seed=1, **kw)


def test_friendly_errors_when_prereqs_missing(tmp_path):
    with pytest.raises(P.PipelineError):
        P.dataset("demo_rc", ["PRBS"], _opts(tmp_path))
    with pytest.raises(P.PipelineError):
        P.label("demo_rc", _opts(tmp_path))


def test_resume_and_parallel(tmp_path):
    P.generate("demo_rc", _opts(tmp_path), 20)
    P.label("demo_rc", _opts(tmp_path, jobs=2))
    o = _opts(tmp_path, jobs=2, limit=10)
    P.dataset("demo_rc", ["PRBS"], o)
    ds = tmp_path / "runs/demo_rc/datasets/DATASET_PRBS.csv"
    n1 = len(ds.read_text().splitlines())
    assert n1 == 12   # en-tete + nominal (toujours conserve) + 10
    o2 = _opts(tmp_path, jobs=2, resume=True)         # reprise sans limite : complete sans doublons
    P.dataset("demo_rc", ["PRBS"], o2)
    keys = [tuple(l.split(",")[:2]) for l in ds.read_text().splitlines()[1:]]
    assert len(keys) == len(set(keys)) == 25
    P.label("demo_rc", o2)                            # reprise du label : rien a refaire
    assert len(P.read_csv(tmp_path / "runs/demo_rc/labels.csv")) == 25


def test_run_spectre_timeout_and_crash(tmp_path, monkeypatch):
    from amsml import spectre_io as sio
    fake = tmp_path / "fake_spectre.sh"
    fake.write_text("#!/bin/sh\nif [ \"$1\" = slow ]; then sleep 30; fi\nif [ \"$1\" = bad ]; then exit 3; fi\nexit 0\n")
    fake.chmod(0o755)
    monkeypatch.setenv("SPECTRE_CMD", str(fake))
    assert sio.run_spectre("ok", tmp_path, 5) == "OK"
    assert sio.run_spectre("bad", tmp_path, 5) == "CRASH"
    import time
    t0 = time.time()
    assert sio.run_spectre("slow", tmp_path, 1) == "TIMEOUT" and time.time() - t0 < 5
    monkeypatch.setenv("SPECTRE_CMD", str(tmp_path / "nope"))
    with pytest.raises(RuntimeError):
        sio.run_spectre("x", tmp_path, 1)


def test_nominal_failure_aborts_fast(tmp_path, monkeypatch):
    from amsml.circuits import demo_rc
    P.generate("demo_rc", _opts(tmp_path), 6)
    monkeypatch.setattr(demo_rc.DemoRC, "measure", lambda self, *a, **k: {"status": "CRASH", "values": {}, "checker": None})
    with pytest.raises(P.PipelineError):
        P.label("demo_rc", _opts(tmp_path))


def test_relative_verdict_ignores_nan_reference():
    from amsml.circuits.base import relative_verdict
    nom = {"a": 1.0, "b": float("nan")}
    assert relative_verdict({"a": 1.01, "b": float("nan")}, nom, ("a", "b"), 0.05) == (0, "NONE;unchecked=b")
    assert relative_verdict({"a": 1.2, "b": 3.0}, nom, ("a", "b"), 0.05)[0] == 1
    assert relative_verdict({"a": 1.0}, {"a": float("nan")}, ("a",), 0.05)[1] == "NO_REFERENCE"


def test_nominal_label_is_pass_for_relative_circuits():
    for n in ("filter", "demo_rc"):
        assert get_circuit(n, ROOT / "circuits").hard_label("nominal", 1, None) == 0


def test_measure_parsing_suffix_and_failed(tmp_path):
    (tmp_path / "a.mt0").write_text("$DATA1 SOURCE='S' temp=27\n.TITLE 'x'\ngain_20k freq_coupure\n1.04 145.35k\n")
    m = parse_measure(tmp_path / "a.mt0")
    assert m["gain_20k"] == 1.04 and abs(m["freq_coupure"] - 145350) < 1e-6
    (tmp_path / "b.measure").write_text("voffset = 1.2m\nfreq_coupure = failed\n")
    m = parse_measure(tmp_path / "b.measure")
    assert m["voffset"] == 0.0012 and np.isnan(m["freq_coupure"]) and "1.2m" not in m


def test_progress_line_fits_terminal(monkeypatch):
    import shutil
    monkeypatch.setattr(shutil, "get_terminal_size", lambda d=None: __import__("os").terminal_size((80, 24)))
    pr = P.Progress(5180, "label")
    pr.tty = True
    pr.n = 1234
    pr.status.update({"OK": 1200, "CRASH": 30, "TIMEOUT": 4})
    assert len(pr._line()) <= 79


def _demo_datasets(tmp_path, n=80):
    o = _opts(tmp_path, jobs=1)
    P.generate("demo_rc", o, n)
    P.label("demo_rc", o)
    P.dataset("demo_rc", ["PRBS", "APRMLS"], o)
    return o


def test_train_holdout_cv_multiclass(tmp_path):
    pytest.importorskip("sklearn")
    pytest.importorskip("pandas")
    from amsml import train as T
    o = _demo_datasets(tmp_path)
    rd = tmp_path / "runs/demo_rc"
    paths = sorted((rd / "datasets").glob("DATASET_*.csv"))
    kw = dict(results_dir=rd / "results", model_dir=rd / "models", seed=0)
    res = T.run("demo_rc", paths, models=("logreg", "random_forest"), residual=True, **kw)
    assert len(res) == 4 and all(r["f1"] > 0.85 for r in res)
    assert (rd / "models/PRBS_random_forest_hard_label.joblib").exists()
    res = T.run("demo_rc", paths[:1], models=("random_forest",), features="raw+stats", cv=3, **kw)
    assert res[0]["folds"] == 3 and "f1_std" in res[0]
    res = T.run("demo_rc", paths[:1], label="fault_class", models=("random_forest",), features="stats", **kw)
    assert res[0]["accuracy"] > 0.5
    assert len((rd / "results/summary.csv").read_text().splitlines()) == 1 + 4 + 1 + 1


def test_split_is_by_circuit_and_same_across_stimuli(tmp_path):
    pytest.importorskip("pandas")
    from amsml import train as T
    _demo_datasets(tmp_path)
    d = tmp_path / "runs/demo_rc/datasets"
    a, _ = T.load_dataset(d / "DATASET_PRBS.csv")
    b, _ = T.load_dataset(d / "DATASET_APRMLS.csv")
    a, b = a[a.fault_class != "nominal"].reset_index(drop=True), b[b.fault_class != "nominal"].reset_index(drop=True)
    fa, fb = T.assign_folds(a, 5, 0), T.assign_folds(b, 5, 0)
    assert (fa == fb).all()
    for _, g in a.assign(f=fa).groupby("filename"):
        assert g["f"].nunique() == 1


def test_train_errors_are_friendly(tmp_path):
    from amsml.cli import main
    with pytest.raises(SystemExit) as e:
        main(["--runs-dir", str(tmp_path), "train", "--circuit", "demo_rc"])
    assert e.value.code == 2


def test_missing_measure_name_aborts_with_available_keys(tmp_path, monkeypatch):
    from amsml.circuits import demo_rc
    P.generate("demo_rc", _opts(tmp_path), 6)
    monkeypatch.setattr(demo_rc.DemoRC, "measure", lambda self, *a, **k: {
        "status": "OK", "values": {"gain_dc": 1.0}, "checker": None, "raw_keys": ["gain_dc", "freq_cutoff"]})
    with pytest.raises(P.PipelineError) as e:
        P.label("demo_rc", _opts(tmp_path))
    msg = str(e.value)
    assert "fc" in msg and "freq_cutoff" in msg


def test_filter_measure_reads_freq_cutoff_from_real_spectre_files(tmp_path, monkeypatch):
    import amsml.circuits.filter as mod
    root = tmp_path / "filter"
    (root / "subcircuit_files").mkdir(parents=True)
    (root / "process_models").mkdir()
    (root / "filter.circuit").write_text(
        '.INCLUDE "subcircuit_files/PRIMITIVE.sub"\n.INCLUDE "subcircuit_files/FILTER.sub"\n'
        '.INCLUDE "process_models/typical.process"\n.AC DEC 20 1k 10Meg\n')
    (root / "process_models/typical.process").write_text("* p\n")

    def fake(net, cwd, timeout, extra=()):
        (cwd / "filter.mt0").write_text("$ Simulator: spectre\n.title 'x'\ngain_20k gain_100k freq_cutoff temper alter#\n"
                                        "1.01678 1.42619 218331 27 1\n")
        (cwd / "filter.measure").write_text("Measurement Name   :  acrun1\nfreq_cutoff     = 218331 \n"
                                            "gain_100k       = 1.42619 \ngain_20k        = 1.01678 \n")
        return "OK"

    monkeypatch.setattr(mod, "run_spectre", fake)
    c = get_circuit("filter", tmp_path)
    w = tmp_path / "w"
    w.mkdir()
    m = c.measure(str(tmp_path / "x.sub"), "typical", w, 5)
    assert m["values"] == {"gain_20k": 1.01678, "gain_100k": 1.42619, "freq_cutoff": 218331.0}
