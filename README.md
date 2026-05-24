# neural_seismic

A Python package for automated seismic first-break picking using neural networks. Implements and compares three neural network architectures: backpropagation (BPNN), 1D convolutional (CVNN), and long short-term memory (LSTM); against the classical Coppens method, with a focus on making model decisions interpretable to non-ML practitioners.

---

## Background

First-break picking is the identification of the arrival time of the first seismic wave in a refraction survey trace. Manual picking is time-consuming and inconsistent across operators. This package frames the problem as a binary sequence classification task: train a model to output a step function that transitions from 0 to 1 at the first-break time. The argmax of the normalised differential of that output gives the pick.

Feature engineering uses a discrete wavelet transform (Daubechies db2, 5 levels) with radial basis function interpolation to produce 33 input features per time step: the raw signal, the low-pass filtered signal, and 32 wavelet packet coefficients. Models are tested on 1-feature (raw only), 2-feature (raw + filter), and 33-feature inputs.

---

## Repository Structure

```
neural_seismic/
├── main.py                          # CLI entry point
├── config.json                      # configurable defaults (CLI flags override)
├── pyproject.toml                   # package metadata and install config
├── neural_seismic_environment.yml   # conda environment spec
├── neural_seismic_requirements.txt  # pip requirements
├── notebooks/
│   ├── Comparison of Methods for Comprehendable first break Detection.ipynb
│   └── Technical_Revisions_OGR.ipynb
├── Scripts/                         # original scripts (legacy, superseded by src/)
└── src/
    └── neural_seismic/
        ├── __init__.py              # public API
        ├── trace.py                 # Trace data class
        ├── models.py                # model builders and training orchestration
        ├── picking.py               # first-break picking algorithms
        ├── io.py                    # data loading and result serialisation
        ├── db.py                    # SQLite experiment database
        ├── viz.py                   # visualisation functions
        └── utils.py                 # utilities (progress bar, normalisation, etc.)
```

---

## Installation

**Recommended: conda environment**

```bash
conda env create -f neural_seismic_environment.yml
conda activate neural_seismic
pip install -e .
```

**pip only**

```bash
pip install -r neural_seismic_requirements.txt
pip install -e .
```

The `-e` (editable) install makes `neural_seismic` importable from anywhere in the environment without path manipulation.

---

## Usage

### Workflow overview

All ingested data lives in `experiment.db` (SQLite). Each training run produces an `exp_id_<id>/` output folder containing `results.db` and the serialised model files. The workflow is two steps:

1. **Ingest** — load the CSV into the database and compute feature spaces. This only needs to run once; re-running is a no-op for rows already present.
2. **Run** — create a new experiment (random train/vali/test split recorded by primary key), train all models, and write results to `exp_id_<id>/`.

### Command line

```bash
# Step 1 — ingest CSV and preprocess (run once)
python main.py ingest data/traces.csv

# Step 2 — run a new experiment
python main.py run

# Quick test on a small subset (splits are sized automatically)
python main.py run --num-samples 100

# Target a different database file
python main.py --db /path/to/project.db ingest data/traces.csv
python main.py --db /path/to/project.db run

# Reuse the exact train/vali/test split of a previous experiment
python main.py run --experiment abc12345

# List all experiments
python main.py list

# Show row counts
python main.py stats
```

**`ingest` options**

| Flag | Default | Description |
|---|---|---|
| `--num-traces N` | all | Maximum traces to load from the CSV |
| `--force-preprocess` | off | Recompute and overwrite existing feature spaces |

**`run` options**

| Flag | Default | Description |
|---|---|---|
| `--experiment ID` | — | Reuse an existing experiment's split instead of creating a new one |
| `--num-samples N` | all | Cap total traces used (useful for quick tests); train/vali sizes scale automatically |
| `--train-size N` | 6000 | Training set size; auto-scaled to 80% of the train+vali pool if it exceeds available data |
| `--coppens-window N` | 50 | Window size for the Coppens baseline |
| `--notes TEXT` | — | Free-text label stored with the experiment |

### Configuration

Defaults for the CLI and model training are read from `config.json` in the repository root. CLI flags always override the file. If the file is absent the built-in defaults apply.

```json
{
  "database": {
    "path": "experiment.db"       // default --db path
  },
  "experiment": {
    "train_vali_split": 0.6,      // fraction of traces used for train+vali
    "num_samples": null,          // cap total traces (null = all); set to e.g. 100 for quick tests
    "train_size": 6000,           // default --train-size
    "coppens_window": 50          // default --coppens-window
  },
  "training": {
    "optimizer": "Adamax",        // Keras optimizer name
    "hidden_neurons": 10,         // neurons per hidden layer
    "hidden_layers": 1,           // number of hidden layers
    "steps_per_epoch": 100,       // generator steps per epoch
    "epochs": 5                   // training epochs per model
  }
}
```

### Database schema

**`experiment.db`** — ingested data and run index

| Table | Description |
|---|---|
| `traces` | One row per trace — metadata, coordinates, signal BLOBs |
| `features` | One row per trace — serialised feature space DataFrame + uncertainty metrics |
| `experiments` | One row per training run — ID, timestamp, split sizes, Coppens window, notes |
| `splits` | One row per (experiment, trace) — records which split each trace belongs to |
| `predictions` | One row per (experiment, trace, model) — predicted first-break value in ms |

**`exp_id_<id>/results.db`** — per-run outputs

| Table | Description |
|---|---|
| `predictions` | One row per (trace, model) — `trace_id` matches `traces.id` in `experiment.db` |
| `training_history` | One row per (model, epoch, metric) — loss and accuracy for every training epoch |

**`exp_id_<id>/`** also contains one `<MODEL>_model.p` file (dill-serialised Keras model) per architecture.

### Querying results

```python
from neural_seismic.db import ExperimentDB

with ExperimentDB('experiment.db') as db:
    db.list_experiments()
    df = db.get_predictions('abc12345')   # DataFrame: trace_id, model, predicted_value
    meta = db.get_experiment_meta('abc12345')
```

### In a notebook (without the database)

```python
from neural_seismic import import_traces, save_exp_instance, mean_residual_calcs
from neural_seismic.models import run_all_models_consis
from neural_seismic.picking import test_coppens
import sklearn.model_selection

# Load data from a known path (no GUI dialog)
traces = import_traces(5000, 'Rio', path='data/traces.csv')

for t in traces:
    t.calc_metrics()
    t.gen_feat_space()

traces = [t for t in traces if t.FB_Picks[1] != 0 and t.FB_Picks[0] != 0]
train_vali, test = sklearn.model_selection.train_test_split(traces, train_size=0.6)
train, vali = sklearn.model_selection.train_test_split(train_vali, train_size=6000)

models, predictions, histories = run_all_models_consis(train, vali, test)
coppens = test_coppens(test, window=50)
# Results written to exp_id_<run_id>/results.db and exp_id_<run_id>/<MODEL>_model.p
test_out = save_exp_instance(run_id='my_run', test=test, models=models,
                             histories=histories, preds=predictions, coppens=coppens)
```

---

## Package Modules

### `trace.py` — `Trace` class

The core data structure. Each instance holds metadata, spatial coordinates, raw and filtered signals, wavelet feature space, operator first-break picks, and prediction slots for all models.

Key methods:

| Method | Description |
|---|---|
| `calc_metrics()` | Computes three uncertainty metrics: operator pick disparity, noise-to-signal ratio, first-break clarity |
| `gen_feat_space()` | Builds the 33-feature DataFrame using DWT + RBF interpolation |
| `plot_comp(title)` | Plots raw vs filtered signal with first-break pick |
| `plot_scalo(state)` | Plots the wavelet scalogram |

### `models.py` — model builders and training

Model builder functions return compiled Keras `Sequential` models. All use mean absolute error loss and the Adamax optimiser.

| Function | Architecture | Input features |
|---|---|---|
| `BPNN_1/2/33` | Dense → tanh hidden layers → Dense(1) | 1, 2, or 33 |
| `CONV_1/2/33` | Conv1D → MaxPool → Conv1D → Dense(1) | 1, 2, or 33 |
| `LSTM_1/2/33` | LSTM(20) → hidden LSTM layers → Dense(1) | 1, 2, or 33 |

Training orchestration:

- `run_all_models(train, vali, test)` — trains the 6 primary variants
- `run_all_models_consis(train, vali, test, ...)` — trains all 9 variants; returns `(models, predictions, histories)`. Accepts keyword arguments `optimizer`, `hidden_neurons`, `hidden_layers`, `steps_per_epoch`, `epochs` (all configurable via `config.json`).

### `picking.py` — first-break picking

| Function | Description |
|---|---|
| `process_fb_calc(predicted_set, predictions)` | Converts raw model output to first-break picks via running max → differential → normalised PDF → argmax |
| `convert_series(prediction_array)` | Converts a single prediction array to a pseudo-PDF |
| `test_coppens(traces, window)` | Coppens energy-ratio method baseline |
| `mean_residual_calcs(test_set)` | Mean prediction residual (ms) for each of the 9 models |
| `confidence_calcs(pred_array, model_array)` | Area-under-peak confidence score for each prediction |

### `io.py` — data loading and serialisation

| Function | Description |
|---|---|
| `load_csv()` | Loads a trace CSV via a GUI file dialog |
| `load_csv_path(path)` | Loads a trace CSV from a file path (for scripted use) |
| `import_traces(n, fb_type, path=None)` | Constructs `Trace` objects from a CSV. Missing optional columns are substituted with defaults. |
| `save_exp_instance(run_id, test, models, histories, preds, coppens)` | Post-processes raw model outputs into first-break picks, writes `exp_id_<run_id>/results.db` (predictions + training history) and one `<MODEL>_model.p` per architecture |

### `db.py` — SQLite experiment database

`ExperimentDB` wraps a single SQLite file and provides the full ingest → preprocess → experiment → results lifecycle. Supports use as a context manager (`with ExperimentDB(...) as db`).

| Method | Description |
|---|---|
| `ingest(csv_path, num_traces=None)` | Load a CSV into the `traces` table. Idempotent — skips rows already present. |
| `preprocess(force=False)` | Compute feature spaces and uncertainty metrics for all unprocessed traces. Resumable — skips traces already in `features`. |
| `new_experiment(train_ids, vali_ids, test_ids, ...)` | Record a train/vali/test split by primary key and return an 8-char experiment ID. |
| `get_split_ids(experiment_id)` | Return `{'train': [...], 'vali': [...], 'test': [...]}` for an experiment. |
| `load_traces(trace_ids)` | Bulk-load `Trace` objects with pre-computed feature spaces from the database. |
| `save_predictions(experiment_id, test_traces)` | Write per-model predicted first-break values to the `predictions` table. |
| `get_predictions(experiment_id)` | Return predictions as a DataFrame with columns `[trace_id, model, predicted_value]`. |
| `get_experiment_meta(experiment_id)` | Return experiment metadata as a dict. |
| `list_experiments()` | Print a summary table of all experiments. |
| `stats()` | Print row counts for all tables. |

### `viz.py` — visualisation

| Function | Description |
|---|---|
| `trace_comparison(comp_trace, ids)` | Side-by-side plot of up to 3 traces |
| `plot_scalogram(trace)` | Two-panel wavelet scalogram with signal overlay |
| `plot_scalogram_alt(trace)` | Single-panel scalogram with twin-axis signal |
| `plot_predictions(ids, traces, name)` | Multi-trace plot with all model predictions as vertical lines |
| `visualise_classifier(trace, ...)` | Plots the raw classifier series, CDF, PDF, and model prediction |
| `plot_uncertainty(traces, method)` | Contour map of an uncertainty metric (`UM1`, `UM2`, or `UM3`) over the survey area |
| `plot_violin(dataframe)` | Violin plot of prediction residuals by model and feature set |
| `plot_viola(test)` | Builds the long-format DataFrame for `plot_violin` |
| `model_plot(gather, suptitle)` | Gather-style plot of raw/filtered signals with optional ML prediction overlay |
| `area_plot(traces)` | Scatter plot of receiver locations with interactive area selection |

### `utils.py` — utilities

| Function | Description |
|---|---|
| `printProgressBar(...)` | Terminal progress bar |
| `notify()` | Plays a completion sound (Windows only) |
| `keras_input(learn_set, feat_num)` | Stacks and normalises feature spaces into arrays for Keras |
| `reject_outliers(data, m)` | Removes values more than `m` standard deviations from the mean |
| `calc_iter(seismic_df)` | Counts unique station locations from a raw DataFrame |

---

## CSV Format

The CSV must have a leading index column (unnamed) and the following columns. Optional columns are used when present and substituted with defaults when absent.

**Required**

| Column | Description |
|---|---|
| `Station` | Station identifier |
| `xr`, `yr`, `zr` | Receiver coordinates |
| `afb` | Contractor first-break pick (seconds) |
| `rfb` | Company first-break pick (seconds) |
| `R1_000`–`R1_499` | Raw signal (500 samples) |

**Optional**

| Column | Default | Description |
|---|---|---|
| `F1_000`–`F1_499` | raw signal | Low-pass filtered signal — if absent, raw signal is used in its place |
| `R2_000`–`R2_499` | — | Raw repeat-shot signal — if absent, repeat mode is disabled |
| `F2_000`–`F2_499` | — | Filtered repeat-shot signal |
| `date`, `time` | `None` | Acquisition date and time |
| `repeat` | `False` | Whether a repeat shot exists |
| `K_Type` | `''` | Trace type classification |
| `Grav_sel` | `0` | Gravity selection flag |
| `xs`, `ys`, `zs` | `0` | Source coordinates |

Any additional columns (e.g. `L4_PROSPECT`, `Basment_Elevation`) are ignored.

---

## Dependencies

Key dependencies (see `neural_seismic_environment.yml` for pinned versions):

- **TensorFlow / Keras** — model training and inference
- **NumPy / Pandas** — data handling
- **PyWavelets** — discrete wavelet transform
- **SciPy** — RBF interpolation and signal processing
- **scikit-learn** — train/test splitting and normalisation
- **Matplotlib / Seaborn / Colorcet** — visualisation
- **dill** — serialisation of trained Keras models to `.p` files
- **sqlite3** — experiment database (Python standard library, no install required)
