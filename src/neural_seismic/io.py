import os
import pickle
import sqlite3

import dill
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import filedialog

from .trace import trace
from .utils import printProgressBar


def _parse_seismic_df(df):
    if 'repeat' in df.columns:
        df['repeat'] = df['repeat'].apply(bool)
    if 'rfb' in df.columns:
        df['rfb'] = pd.to_numeric(df['rfb'], errors='coerce').fillna(0)
    if 'Grav_sel' in df.columns:
        df['Grav_sel'] = pd.to_numeric(df['Grav_sel'], errors='coerce').fillna(0)
    return df


def load_csv():
    """Loads the CSV of seismic trace data via a GUI file dialog."""
    root = tk.Tk()
    root.lift()
    file_path1 = filedialog.askopenfilename(filetypes=[('.csvfiles', '.csv')], title='Select Trace File')
    root.destroy()
    df = pd.read_csv(file_path1, index_col=0)
    return _parse_seismic_df(df)


def load_csv_path(path):
    """Loads the CSV of seismic trace data from a file path."""
    df = pd.read_csv(path, index_col=0)
    return _parse_seismic_df(df)


def _get(row, col, default=None):
    return row[col] if col in row.index else default


def _signal(row, start, end):
    """Slice a signal block by column label range, returning None if absent."""
    if start not in row.index:
        return None
    return row.loc[start:end]


def import_traces(num2import, fb_type, path=None):
    comp_trace = list()
    seismic_df = load_csv_path(path) if path else load_csv()
    cols = set(seismic_df.columns)

    has_filt1 = 'F1_000' in cols
    has_raw2  = 'R2_000' in cols
    has_filt2 = 'F2_000' in cols

    l = num2import
    if isinstance(num2import, str):
        l = len(seismic_df)

    printProgressBar(0, l, prefix='Progress:', suffix='Complete', length=50)
    for i in range(l):
        row = seismic_df.iloc[i]

        Raw_1 = row.loc['R1_000':'R1_499']
        Flt_1 = row.loc['F1_000':'F1_499'] if has_filt1 else Raw_1
        Raw_2 = row.loc['R2_000':'R2_499'] if has_raw2  else None
        Flt_2 = row.loc['F2_000':'F2_499'] if has_filt2 else None
        Repeat = bool(_get(row, 'repeat', False)) and has_raw2

        trace_obj = trace(
            iD      = i,
            Date    = _get(row, 'date'),
            Time    = _get(row, 'time'),
            Station = row['Station'],
            Repeat  = Repeat,
            Atl_fb  = row['afb'],
            Rio_fb  = row['rfb'],
            Gravty  = _get(row, 'Grav_sel', 0),
            K_type  = _get(row, 'K_Type', ''),
            xr=row['xr'], yr=row['yr'], zr=row['zr'],
            xs=_get(row, 'xs', 0), ys=_get(row, 'ys', 0), zs=_get(row, 'zs', 0),
            Raw_1=Raw_1, Flt_1=Flt_1, Raw_2=Raw_2, Flt_2=Flt_2,
        )
        comp_trace.append(trace_obj)
        printProgressBar(i + 1, l, prefix='Progress:', suffix='Complete', length=50)
    return comp_trace


_RESULTS_SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
    trace_id        INTEGER NOT NULL,
    model           TEXT NOT NULL,
    predicted_value INTEGER,
    PRIMARY KEY (trace_id, model)
);

CREATE TABLE IF NOT EXISTS training_history (
    model   TEXT NOT NULL,
    epoch   INTEGER NOT NULL,
    metric  TEXT NOT NULL,
    value   REAL,
    PRIMARY KEY (model, epoch, metric)
);
"""


def save_exp_instance(run_id, test, models, histories, preds, coppens):
    """Post-process predictions and write results.db + model files into {run_id}/."""
    model_keys = ['BPNN01', 'BPNN02', 'BPNN33', 'CVNN01', 'CVNN02', 'CVNN33', 'LSTM01', 'LSTM02', 'LSTM33']

    if len(test) != len(preds[0]):
        raise Exception('Missmatched number of predictions and traces.')
    if len(model_keys) != len(preds):
        raise Exception('Missmatched models and predictions.')

    # --- post-process raw model output onto trace objects ---
    for method_i, key_i in enumerate(model_keys):
        print('Processing - ' + key_i)
        pred = preds[method_i]
        for index in range(len(pred)):
            pred_series = pred[index].flatten()
            edited = []
            for ts in range(len(pred_series)):
                edited.append(max(pred_series[:ts + 1]))
            dif_ = np.diff(edited, 1)
            area = np.trapezoid(dif_)
            nrml = dif_ / area
            fbpk = int(np.argmax(nrml))
            trace_i = test[index]
            trace_i.prediction_series[key_i] = pred_series
            trace_i.prediction__value[key_i] = fbpk

    for j, trace_i in enumerate(test):
        trace_i.prediction_series['Coppens'] = coppens[1][j]
        trace_i.prediction__value['Coppens'] = coppens[0][j]

    # --- create output folder ---
    out_dir = f'exp_id_{run_id}'
    os.makedirs(out_dir, exist_ok=True)

    # --- results.db ---
    db_path = os.path.join(out_dir, 'results.db')
    conn = sqlite3.connect(db_path)
    conn.executescript(_RESULTS_SCHEMA)

    pred_rows = []
    for trace_i in test:
        for key in model_keys + ['Coppens']:
            val = trace_i.prediction__value[key]
            pred_rows.append((trace_i.iD, key, int(val) if val is not None else None))
    conn.executemany(
        "INSERT OR REPLACE INTO predictions (trace_id, model, predicted_value) VALUES (?,?,?)",
        pred_rows,
    )

    hist_rows = []
    for key, hist in zip(model_keys, histories):
        for metric, values in hist.history.items():
            for epoch, value in enumerate(values):
                hist_rows.append((key, epoch, metric, float(value)))
    conn.executemany(
        "INSERT OR REPLACE INTO training_history (model, epoch, metric, value) VALUES (?,?,?,?)",
        hist_rows,
    )

    conn.commit()
    conn.close()
    print(f"Results written to {db_path}")

    # --- save models ---
    for i, key in enumerate(model_keys):
        model_path = os.path.join(out_dir, key + '_model.p')
        with open(model_path, 'wb') as f:
            dill.dump(models[i], f, pickle.HIGHEST_PROTOCOL)
    print(f"Models saved to {out_dir}/")

    return test
