# PAGMA
## Population Automation & Generation for Mixed-sytem Analysis & Classification.

Generation of populations of analog circuits **with faults** (parametric + short/open faults),
simulation using **Spectre** with multiple **stimuli**, construction of **datasets** labeled PASS/FAIL.
Then **training and comparison of classifiers** (phase 2). Same steps for each circuit.

```
generate ──► label ──► dataset ──► train
 DUT + manifest.csv    specs + hard_label     one CSV file per stimulus     classifieurs, comparison
                       (labels.csv)           (DATASET_<STIM>.csv)    stimulus x models (summary.csv)
```

| Circuit   | Simulator | Corners | Label `spec`                                  | Label `hard`                               |
|-----------|-----------|-------|-----------------------------------------------|--------------------------------------------|
| `demo_rc` | Python | –    | nominal ±5 %                                  | construction (default→1, SAFE→0, otherwise spec) |
| `opamp1`  | Spectre   | –     | specs absolues (voffset, vpp, delay, iddq)    | construction (EXTREME→1)                                     |
| `filter`  | Spectre   | 5     | nominal value of the corner ±5 % | construction (EXTREME→1)                   |
| `bandgap` | Spectre + Verilog-A | – | specs DC (idle 10 ms, DC 50 ms, stability 100 ms) | EXTREME/nominal → checker |



## Quickstart

```bash
git clone <your-repo> pagma && cd pagma
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt            # >= Python 3.6

python3 run_pipeline.py list               # circuits, existing files, detected spectrum, already produced
python3 run_pipeline.py --jobs 4 all --circuit demo_rc --stimuli PRBS APRMLS CHIRP
pip install -r requirements-train.txt      
python3 run_pipeline.py train --circuit demo_rc --residual
# tests : pip install -r requirements-dev.txt && python3 -m pytest -q     (27 tests)
```
Résults in `runs/demo_rc/` : `manifest.csv`, `labels.csv`, `datasets/DATASET_*.csv` (+ `.meta.json`).

## Simulations with spectre

1. Place your files in `circuits/<circuit>/` (`list` shows which ones are missing):
   - `opamp1/` : `subcircuit_files/OPAMP1.sub`, `OPAMP1.circuit` (bench specs), `OPAMP1_ML.circuit` (transient bench with `Vin ...` and `.TRAN`)
   - `filter/` : `subcircuit_files/{FILTER,PRIMITIVE,OPAMP1}.sub`, `filter.circuit`, `process_models/{typical,fastN_fastP,fastN_slowP,slowN_fastP,slowN_slowP}.process`
   - `bandgap/` : `src/{tb_bandgap.hsp,top.hsp,model/,checker/}`, `ref/{defect_short.txt,defect_open.txt}`
2. `export SPECTRE_CMD=spectre` if necessary, then **always start with a short test**:
```bash
python3 run_pipeline.py --limit 12 all --circuit filter --stimuli PRBS        # stratified sample
python3 run_pipeline.py --jobs 8 all --circuit filter --stimuli PRBS APRMLS    # full campaign
python3 run_pipeline.py interactive                                           # step-by-step menu
```
Global Options (`--jobs --limit --seed --resume --timeout --runs-dir --circuits-dir --work-root`)
are placed **before** the subcommand. `--jobs` = simultaneous simulations (Spectre licenses).
Order of magnitude: filter = 1,037 circuits × 5 corners ≈ 5,200 simulations per `label` step and per stimulus.

## Monitoring Large Simulations, Stopping, and Resuming

- **Live progress bar**: `[label] [#####---] 1234/5180 23.8% | 1.9 sim/s | elapsed … | ETA … | OK:1230 CRASH:4`
  (on terminal: line rewritten; redirected to a file/`nohup`: one line every 30 s).
- **Ctrl-C = clean stop** (running Spectre processes killed, no orphaned processes). Each result is written **in real time**:
  nothing is lost. Resume: rerun the same command with `--resume` (only repeats what’s missing).
- `--timeout S`: maximum duration per simulation (Spectre and child processes terminated).
- First failed runs are saved in `runs/<circuit>/failed_logs/` (`spectre.log` + netlist) for diagnostics.
- The **nominal** circuit is simulated first and its measurements are displayed; if it fails to simulate, the program stops
  immediately with the log path (no need to run thousands of simulations with a broken benchmark).
- **Missing measurement** (name different from the bench’s `.MEAS`): program stops immediately with a list of available measurements and the file/variable to correct (`measure_names` / `MEASURE_ALIASES` in `PAGMA/circuits/<c>.py`).
- **NaN nominal measurements** (e.g., `freq_coupure=nan`): flagged in red; raw `.mt0/.measure` files are saved in
  `runs/<circuit>/nominal_debug/<corner>/`, and their start points are displayed. A measurement without a valid reference is **ignored** (marked
  `unchecked=...` in `failed_params`) instead of causing all circuits to fail.
- Long runs: `nohup python3 run_pipeline.py --jobs 8 all --circuit filter --stimuli all > filter.log 2>&1 &` followed by `tail -f filter.log`.

## Commands

| Command | Role |
|---|---|
| `list` | circuits, missing files, Spectre, results already generated |
| `generate --circuit C [--n-param N]` | DUT + `manifest.csv` (50% SAFE ±5%, 50% EXTREME −50…−30% / +30…+60%, 1 to 5 components (20 for bandgap), + shorts/opens). Removes obsolete `labels.csv`. |
| `label --circuit C` | measurements + `spec_label`, `hard_label`, `checker_label` → `labels.csv` |
| `dataset --circuit C --stimuli ...` | one CSV per stimulus + `.meta.json` (time grid + input signal) |
| `all` | `generate → label → dataset` (`--notify`: email via `SMTP_*`) |
| `train --circuit C [options]` | phase 2, see below |
| `interactive` | guided selection; automatically adds any missing prerequisite steps |

## Stimuli

`DC` (bandgap only), `PRBS` (PRBS-7), **`APRMLS`** (8 levels evenly distributed between `v_low` and `v_high`, period 127 symbols),
`STEP`, `SINUS`, `CHIRP`, `MULTISINE`. All share the same window and number of circuit points (`stim_config()` in
`PAGMA/circuits/<c>.py`); frequencies derived from the output sampling rate, anti-aliasing guard.

## Dataset Format

`filename, corner, fault_class, hard_label, spec_label, checker_label, t_0 … t_{N-1}` — `1 = FAIL`,
`fault_class ∈ {nominal, param_safe, param_extreme, short, open}`. Failed/truncated simulations are **excluded** (never zeros).

## Training (Phase 2)

```bash
pip install -r requirements-train.txt
python3 run_pipeline.py train --circuit filter --stimuli PRBS APRMLS CHIRP --residual            # stimulus comparison
python3 run_pipeline.py train --circuit filter --models logreg random_forest mlp svm --cv 5      # cross-validation
python3 run_pipeline.py train --circuit filter --label fault_class --features raw+stats          # multi-class
```
| Option | Role |
|---|---|
| `--stimuli` | datasets to use (default: all those in `runs/<c>/datasets/`) |
| `--label` | `hard_label` (default), `spec_label`, `checker_label`, or `fault_class` (multi-class) |
| `--models` | `logreg`, `random_forest`, `mlp` (default), `svm`, `gboost` |
| `--features` | `raw` (waveform, default), `stats` (12 descriptors + energy per band), `raw+stats` |
| `--residual` | subtracts the response of the **nominal** circuit from the same corner: generally provides the clearest improvement in fault detection |
| `--cv K` / `--test-size` | K-fold cross-validation, or a holdout set (default 20%) |

Methodological guarantees: the split is performed **per circuit** (a DUT and its test points are never both in-progress and under test at the same time),
stratified by error class, and **identical across stimuli** (fair comparison); the `nominal` rows serve
only as a reference. Outputs: `runs/<c>/results/summary.csv` (cumulative: accuracy, balanced_acc, precision, recall, F1, AUC,
and for each fault class, the predicted FAIL proportion `pred_fail_<class>` = detection rate, or false alarms for `param_safe`),
models `runs/<c>/models/*.joblib` (holdout mode), F1 ranking displayed at the end.

## Add a circuit

Subclass `PAGMA.circuits.base.Circuit` (`stim_config`, `generate`, `measure`, `verdict`, `waveform`, optionally
`hard_label`) and add it to `REGISTRY`. `demo_rc.py` is the shortest example.
