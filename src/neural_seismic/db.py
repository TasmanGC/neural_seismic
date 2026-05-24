"""SQLite-backed experiment database for trace storage, preprocessing, and run tracking."""

import io
import json
import pickle
import sqlite3
import uuid
import datetime

import numpy as np

from .trace import trace as Trace
from .utils import printProgressBar


_SCHEMA = """
CREATE TABLE IF NOT EXISTS traces (
    id          INTEGER PRIMARY KEY,
    station     TEXT,
    xr REAL, yr REAL, zr REAL,
    xs REAL, ys REAL, zs REAL,
    afb         REAL,
    rfb         REAL,
    date        TEXT,
    time        TEXT,
    k_type      TEXT,
    grav_sel    REAL,
    repeat      INTEGER,
    raw_1       BLOB NOT NULL,
    flt_1       BLOB NOT NULL,
    raw_2       BLOB,
    flt_2       BLOB
);

CREATE TABLE IF NOT EXISTS features (
    trace_id    INTEGER PRIMARY KEY REFERENCES traces(id),
    feat_space  BLOB NOT NULL,
    unc_1       REAL,
    unc_2       REAL,
    unc_3       TEXT
);

CREATE TABLE IF NOT EXISTS experiments (
    id              TEXT PRIMARY KEY,
    created_at      TEXT NOT NULL,
    train_size      INTEGER,
    vali_size       INTEGER,
    test_size       INTEGER,
    coppens_window  INTEGER,
    notes           TEXT
);

CREATE TABLE IF NOT EXISTS splits (
    experiment_id   TEXT REFERENCES experiments(id),
    trace_id        INTEGER REFERENCES traces(id),
    split           TEXT NOT NULL CHECK (split IN ('train','vali','test')),
    PRIMARY KEY (experiment_id, trace_id)
);

CREATE TABLE IF NOT EXISTS predictions (
    experiment_id   TEXT REFERENCES experiments(id),
    trace_id        INTEGER REFERENCES traces(id),
    model           TEXT NOT NULL,
    predicted_value REAL,
    PRIMARY KEY (experiment_id, trace_id, model)
);
"""

_MODEL_KEYS = [
    'Coppens',
    'BPNN01', 'BPNN02', 'BPNN33',
    'CVNN01',  'CVNN02',  'CVNN33',
    'LSTM01',  'LSTM02',  'LSTM33',
]


# ---------------------------------------------------------------------------
# Serialisation helpers

def _arr_to_blob(arr):
    if arr is None:
        return None
    buf = io.BytesIO()
    np.save(buf, np.asarray(arr, dtype=float))
    return buf.getvalue()


def _blob_to_arr(blob):
    if blob is None:
        return None
    return np.load(io.BytesIO(blob))


def _df_to_blob(df):
    return pickle.dumps(df)


def _blob_to_df(blob):
    return pickle.loads(blob)


# ---------------------------------------------------------------------------

class ExperimentDB:

    def __init__(self, path):
        self.path = path
        self.conn = sqlite3.connect(path)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA foreign_keys=ON")
        self.conn.executescript(_SCHEMA)
        self.conn.commit()

    # ------------------------------------------------------------------
    # Ingestion

    def ingest(self, csv_path, num_traces=None):
        """Load a CSV into the traces table. Safe to call repeatedly — skips existing rows."""
        from .io import load_csv_path

        df = load_csv_path(csv_path)
        cols = set(df.columns)
        has_filt1 = 'F1_000' in cols
        has_raw2  = 'R2_000' in cols
        has_filt2 = 'F2_000' in cols

        n = num_traces if num_traces is not None else len(df)
        existing = {r[0] for r in self.conn.execute("SELECT id FROM traces")}
        todo = [i for i in range(n) if i not in existing]

        if not todo:
            print(f"All {n} traces already in database, skipping ingest.")
            return

        print(f"Ingesting {len(todo)} traces into {self.path} ...")
        printProgressBar(0, len(todo), prefix='Ingest:', suffix='Complete', length=50)

        batch = []
        for count, i in enumerate(todo):
            row = df.iloc[i]
            raw_1_vals = row.loc['R1_000':'R1_499'].values
            flt_1_vals = row.loc['F1_000':'F1_499'].values if has_filt1 else raw_1_vals

            batch.append((
                i,
                str(row['Station']),
                float(row['xr']), float(row['yr']), float(row['zr']),
                float(row.get('xs', 0) or 0),
                float(row.get('ys', 0) or 0),
                float(row.get('zs', 0) or 0),
                float(row['afb']),
                float(row['rfb']),
                str(row.get('date', '') or ''),
                str(row.get('time', '') or ''),
                str(row.get('K_Type', '') or ''),
                float(row.get('Grav_sel', 0) or 0),
                int(bool(row.get('repeat', False)) and has_raw2),
                _arr_to_blob(raw_1_vals),
                _arr_to_blob(flt_1_vals),
                _arr_to_blob(row.loc['R2_000':'R2_499'].values) if has_raw2 else None,
                _arr_to_blob(row.loc['F2_000':'F2_499'].values) if has_filt2 else None,
            ))

            if len(batch) >= 500:
                self._insert_trace_batch(batch)
                batch = []
            printProgressBar(count + 1, len(todo), prefix='Ingest:', suffix='Complete', length=50)

        if batch:
            self._insert_trace_batch(batch)

        print(f"Ingested {len(todo)} traces.")

    def _insert_trace_batch(self, batch):
        self.conn.executemany("""
            INSERT OR IGNORE INTO traces
              (id, station, xr, yr, zr, xs, ys, zs, afb, rfb,
               date, time, k_type, grav_sel, repeat,
               raw_1, flt_1, raw_2, flt_2)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, batch)
        self.conn.commit()

    # ------------------------------------------------------------------
    # Preprocessing

    def preprocess(self, force=False):
        """Compute and store feature spaces + uncertainty metrics for all unprocessed traces.

        Only processes traces with rfb != 0 and afb != 0. Safe to interrupt and resume.
        """
        if force:
            self.conn.execute("DELETE FROM features")
            self.conn.commit()

        done = {r[0] for r in self.conn.execute("SELECT trace_id FROM features")}
        all_ids = [
            r[0] for r in self.conn.execute(
                "SELECT id FROM traces WHERE rfb != 0 AND afb != 0 ORDER BY id"
            )
        ]
        todo = [i for i in all_ids if i not in done]

        if not todo:
            print("All eligible traces already preprocessed.")
            return

        print(f"Preprocessing {len(todo)} traces ...")
        printProgressBar(0, len(todo), prefix='Preprocess:', suffix='Complete', length=50)

        batch = []
        for count, tid in enumerate(todo):
            t = self._load_trace_row(tid)
            t.calc_metrics()
            t.gen_feat_space()

            unc_1 = float(t.Unc_Metrics[0]) if t.Unc_Metrics and t.Unc_Metrics[0] is not None else None
            unc_2 = float(t.Unc_Metrics[1]) if t.Unc_Metrics and t.Unc_Metrics[1] is not None else None
            unc_3 = json.dumps(t.Unc_Metrics[2]) if t.Unc_Metrics else None
            batch.append((tid, _df_to_blob(t.feat_space), unc_1, unc_2, unc_3))

            if len(batch) >= 200:
                self._insert_feature_batch(batch)
                batch = []
            printProgressBar(count + 1, len(todo), prefix='Preprocess:', suffix='Complete', length=50)

        if batch:
            self._insert_feature_batch(batch)

        print("Preprocessing complete.")

    def _insert_feature_batch(self, batch):
        self.conn.executemany(
            "INSERT OR IGNORE INTO features (trace_id, feat_space, unc_1, unc_2, unc_3) VALUES (?,?,?,?,?)",
            batch,
        )
        self.conn.commit()

    # ------------------------------------------------------------------
    # Trace loading

    def _load_trace_row(self, trace_id):
        """Load a Trace from the traces table only (no feature data)."""
        row = self.conn.execute("SELECT * FROM traces WHERE id=?", (trace_id,)).fetchone()
        if row is None:
            raise KeyError(f"Trace {trace_id} not found in database")

        (id_, station, xr, yr, zr, xs, ys, zs, afb, rfb,
         date, time_, k_type, grav_sel, repeat,
         raw_1_blob, flt_1_blob, raw_2_blob, flt_2_blob) = row

        return Trace(
            iD=id_, Date=date, Time=time_, Station=station,
            Repeat=bool(repeat), Atl_fb=afb, Rio_fb=rfb,
            Gravty=grav_sel, K_type=k_type,
            xr=xr, yr=yr, zr=zr, xs=xs, ys=ys, zs=zs,
            Raw_1=_blob_to_arr(raw_1_blob),
            Flt_1=_blob_to_arr(flt_1_blob),
            Raw_2=_blob_to_arr(raw_2_blob),
            Flt_2=_blob_to_arr(flt_2_blob),
        )

    def load_traces(self, trace_ids):
        """Load Trace objects with pre-computed feature spaces from the database."""
        if not trace_ids:
            return []

        placeholders = ','.join('?' * len(trace_ids))

        # load raw trace rows
        trace_rows = {
            r[0]: r for r in self.conn.execute(
                f"SELECT * FROM traces WHERE id IN ({placeholders})", trace_ids
            )
        }

        # load feature rows
        feat_rows = {
            r[0]: r for r in self.conn.execute(
                f"SELECT trace_id, feat_space, unc_1, unc_2, unc_3 FROM features WHERE trace_id IN ({placeholders})",
                trace_ids,
            )
        }

        traces = []
        for tid in trace_ids:
            row = trace_rows[tid]
            (id_, station, xr, yr, zr, xs, ys, zs, afb, rfb,
             date, time_, k_type, grav_sel, repeat,
             raw_1_blob, flt_1_blob, raw_2_blob, flt_2_blob) = row

            t = Trace(
                iD=id_, Date=date, Time=time_, Station=station,
                Repeat=bool(repeat), Atl_fb=afb, Rio_fb=rfb,
                Gravty=grav_sel, K_type=k_type,
                xr=xr, yr=yr, zr=zr, xs=xs, ys=ys, zs=zs,
                Raw_1=_blob_to_arr(raw_1_blob),
                Flt_1=_blob_to_arr(flt_1_blob),
                Raw_2=_blob_to_arr(raw_2_blob),
                Flt_2=_blob_to_arr(flt_2_blob),
            )

            if tid in feat_rows:
                _, feat_blob, unc_1, unc_2, unc_3_json = feat_rows[tid]
                t.feat_space = _blob_to_df(feat_blob)
                t.Unc_Metrics = [unc_1, unc_2, json.loads(unc_3_json) if unc_3_json else None]

            traces.append(t)

        return traces

    # ------------------------------------------------------------------
    # Experiments

    def new_experiment(self, train_ids, vali_ids, test_ids, coppens_window=50, notes=None):
        """Record a new train/vali/test split and return the experiment ID."""
        exp_id = uuid.uuid4().hex[:8]
        now = datetime.datetime.now().isoformat(timespec='seconds')

        self.conn.execute("""
            INSERT INTO experiments (id, created_at, train_size, vali_size, test_size, coppens_window, notes)
            VALUES (?,?,?,?,?,?,?)
        """, (exp_id, now, len(train_ids), len(vali_ids), len(test_ids), coppens_window, notes))

        rows = (
            [(exp_id, tid, 'train') for tid in train_ids] +
            [(exp_id, tid, 'vali')  for tid in vali_ids]  +
            [(exp_id, tid, 'test')  for tid in test_ids]
        )
        self.conn.executemany(
            "INSERT INTO splits (experiment_id, trace_id, split) VALUES (?,?,?)", rows
        )
        self.conn.commit()
        return exp_id

    def get_split_ids(self, experiment_id):
        """Return {'train': [...], 'vali': [...], 'test': [...]} for an experiment."""
        rows = self.conn.execute(
            "SELECT trace_id, split FROM splits WHERE experiment_id=? ORDER BY trace_id",
            (experiment_id,),
        ).fetchall()
        if not rows:
            raise KeyError(f"Experiment '{experiment_id}' not found or has no splits")
        splits = {'train': [], 'vali': [], 'test': []}
        for tid, split in rows:
            splits[split].append(tid)
        return splits

    def get_experiment_meta(self, experiment_id):
        row = self.conn.execute(
            "SELECT id, created_at, train_size, vali_size, test_size, coppens_window, notes FROM experiments WHERE id=?",
            (experiment_id,),
        ).fetchone()
        if row is None:
            raise KeyError(f"Experiment '{experiment_id}' not found")
        keys = ['id', 'created_at', 'train_size', 'vali_size', 'test_size', 'coppens_window', 'notes']
        return dict(zip(keys, row))

    def list_experiments(self):
        rows = self.conn.execute(
            "SELECT id, created_at, train_size, vali_size, test_size, notes FROM experiments ORDER BY created_at"
        ).fetchall()
        if not rows:
            print("No experiments in database.")
            return
        print(f"{'ID':<10}  {'Created':<20}  {'Train':>6}  {'Vali':>6}  {'Test':>6}  Notes")
        print("-" * 72)
        for r in rows:
            notes = (r[5] or '')[:30]
            print(f"{r[0]:<10}  {r[1]:<20}  {r[2]:>6}  {r[3]:>6}  {r[4]:>6}  {notes}")

    # ------------------------------------------------------------------
    # Predictions

    def save_predictions(self, experiment_id, test_traces):
        """Write prediction values from trace objects into the predictions table."""
        rows = []
        for t in test_traces:
            for model in _MODEL_KEYS:
                val = t.prediction__value.get(model)
                if val is not None:
                    rows.append((experiment_id, t.iD, model, float(val)))
        self.conn.executemany("""
            INSERT OR REPLACE INTO predictions (experiment_id, trace_id, model, predicted_value)
            VALUES (?,?,?,?)
        """, rows)
        self.conn.commit()

    def get_predictions(self, experiment_id):
        """Return predictions as a DataFrame with columns [trace_id, model, predicted_value]."""
        import pandas as pd
        rows = self.conn.execute(
            "SELECT trace_id, model, predicted_value FROM predictions WHERE experiment_id=? ORDER BY trace_id, model",
            (experiment_id,),
        ).fetchall()
        return pd.DataFrame(rows, columns=['trace_id', 'model', 'predicted_value'])

    # ------------------------------------------------------------------

    def stats(self):
        """Print a summary of database contents."""
        n_traces  = self.conn.execute("SELECT COUNT(*) FROM traces").fetchone()[0]
        n_feat    = self.conn.execute("SELECT COUNT(*) FROM features").fetchone()[0]
        n_exp     = self.conn.execute("SELECT COUNT(*) FROM experiments").fetchone()[0]
        n_preds   = self.conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
        print(f"Traces: {n_traces}  |  Preprocessed: {n_feat}  |  Experiments: {n_exp}  |  Predictions: {n_preds}")

    def close(self):
        self.conn.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
