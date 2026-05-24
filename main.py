"""
neural_seismic CLI

Commands
--------
  ingest  Load a CSV into the experiment database and preprocess traces.
  run     Train all models against a database, creating or targeting an experiment.
  list    Show all experiments in the database.
  stats   Show database content counts.

Examples
--------
  python main.py ingest data/traces.csv
  python main.py run
  python main.py run --experiment abc12345
  python main.py --db /path/to/other.db run --notes "first run"
  python main.py list
"""

import argparse
import json
import os
import sys

import sklearn.model_selection

from neural_seismic.db import ExperimentDB
from neural_seismic.io import save_exp_instance
from neural_seismic.models import run_all_models_consis
from neural_seismic.picking import test_coppens, mean_residual_calcs
from neural_seismic.utils import notify


_MODEL_KEYS = ['BPNN01', 'BPNN02', 'BPNN33', 'CVNN01', 'CVNN02', 'CVNN33',
               'LSTM01', 'LSTM02', 'LSTM33']

_CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'config.json')


def _load_config(path=_CONFIG_PATH):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def build_parser(cfg):
    db_cfg  = cfg.get('database', {})
    exp_cfg = cfg.get('experiment', {})

    parser = argparse.ArgumentParser(
        prog='main.py',
        description='neural_seismic experiment runner',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        '--db', default=db_cfg.get('path', 'experiment.db'), metavar='PATH',
        help='SQLite experiment database (default: %(default)s)',
    )

    sub = parser.add_subparsers(dest='command', metavar='command')

    # ---- ingest ----------------------------------------------------------
    p_ingest = sub.add_parser('ingest', help='Load a CSV and preprocess traces into the database')
    p_ingest.add_argument('csv', help='Path to seismic trace CSV')
    p_ingest.add_argument(
        '--num-traces', type=int, default=None,
        help='Maximum number of traces to load (default: all)',
    )
    p_ingest.add_argument(
        '--force-preprocess', action='store_true',
        help='Recompute and overwrite existing feature spaces',
    )

    # ---- run -------------------------------------------------------------
    p_run = sub.add_parser('run', help='Run a training experiment')
    p_run.add_argument(
        '--experiment', default=None, metavar='ID',
        help='Target an existing experiment ID (reuse its train/vali/test split)',
    )
    p_run.add_argument(
        '--num-samples', type=int, default=exp_cfg.get('num_samples', None),
        help='Cap the total number of traces used (drawn from the start of the DB; default: all)',
    )
    p_run.add_argument(
        '--train-size', type=int, default=exp_cfg.get('train_size', 6000),
        help='Training set size when creating a new experiment (default: %(default)s)',
    )
    p_run.add_argument(
        '--coppens-window', type=int, default=exp_cfg.get('coppens_window', 50),
        help='Window size for the Coppens baseline (default: %(default)s)',
    )
    p_run.add_argument('--notes', default=None, help='Free-text notes stored with the experiment')

    # ---- list ------------------------------------------------------------
    sub.add_parser('list', help='List all experiments in the database')

    # ---- stats -----------------------------------------------------------
    sub.add_parser('stats', help='Show database content counts')

    return parser


def cmd_ingest(db, args):
    db.ingest(args.csv, args.num_traces)
    db.preprocess(force=args.force_preprocess)


def cmd_list(db, _args):
    db.list_experiments()


def cmd_stats(db, _args):
    db.stats()


def cmd_run(db, args, cfg):
    train_vali_split = cfg.get('experiment', {}).get('train_vali_split', 0.6)
    train_cfg = cfg.get('training', {})

    if args.experiment:
        meta = db.get_experiment_meta(args.experiment)
        split_ids = db.get_split_ids(args.experiment)
        exp_id = args.experiment
        coppens_window = meta['coppens_window']
        print(
            f"Resuming experiment {exp_id} — "
            f"train: {len(split_ids['train'])}, "
            f"vali: {len(split_ids['vali'])}, "
            f"test: {len(split_ids['test'])}"
        )
    else:
        all_ids = [
            r[0] for r in db.conn.execute(
                "SELECT trace_id FROM features ORDER BY trace_id"
            )
        ]
        if not all_ids:
            sys.exit("No preprocessed traces in database. Run 'ingest' first.")

        if args.num_samples is not None:
            all_ids = all_ids[:args.num_samples]
            print(f"Limiting to {len(all_ids)} traces (--num-samples).")

        if len(all_ids) < 3:
            sys.exit(f"Not enough preprocessed traces ({len(all_ids)}) — need at least 3.")

        train_vali_ids, test_ids = sklearn.model_selection.train_test_split(
            all_ids, train_size=train_vali_split, random_state=None
        )

        train_size = args.train_size
        if train_size >= len(train_vali_ids):
            train_size = max(1, int(len(train_vali_ids) * 0.8))
            print(f"train_size auto-scaled to {train_size} (80% of {len(train_vali_ids)} train+vali traces).")

        train_ids, vali_ids = sklearn.model_selection.train_test_split(
            train_vali_ids, train_size=train_size, random_state=None
        )

        coppens_window = args.coppens_window
        exp_id = db.new_experiment(
            train_ids, vali_ids, test_ids,
            coppens_window=coppens_window,
            notes=args.notes,
        )
        split_ids = {'train': train_ids, 'vali': vali_ids, 'test': test_ids}
        print(
            f"New experiment {exp_id} — "
            f"train: {len(train_ids)}, "
            f"vali: {len(vali_ids)}, "
            f"test: {len(test_ids)}"
        )

    print("Loading traces from database ...")
    train = db.load_traces(split_ids['train'])
    vali  = db.load_traces(split_ids['vali'])
    test  = db.load_traces(split_ids['test'])

    print("Training models ...")
    models, predictions, histories = run_all_models_consis(
        train, vali, test,
        optimizer=train_cfg.get('optimizer', 'Adamax'),
        hidden_neurons=train_cfg.get('hidden_neurons', 10),
        hidden_layers=train_cfg.get('hidden_layers', 1),
        steps_per_epoch=train_cfg.get('steps_per_epoch', 100),
        epochs=train_cfg.get('epochs', 5),
    )
    notify()

    print("Running Coppens method ...")
    coppens = test_coppens(test, coppens_window)

    print("Saving results ...")
    test_out = save_exp_instance(run_id=exp_id, test=test, models=models, histories=histories, preds=predictions, coppens=coppens)
    db.save_predictions(exp_id, test_out)

    residuals = mean_residual_calcs(test_out)
    print(f"\nExperiment {exp_id} — mean residuals (ms):")
    for key, res in zip(_MODEL_KEYS, residuals):
        print(f"  {key:8s}  {res:+.1f}")


def main():
    cfg = _load_config()
    parser = build_parser(cfg)
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    with ExperimentDB(args.db) as db:
        if args.command == 'ingest':
            cmd_ingest(db, args)
        elif args.command == 'list':
            cmd_list(db, args)
        elif args.command == 'stats':
            cmd_stats(db, args)
        elif args.command == 'run':
            cmd_run(db, args, cfg)


if __name__ == '__main__':
    main()
