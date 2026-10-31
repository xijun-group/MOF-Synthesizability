"""Adapted from /home/cheyue/MOFSimplify/batch/predict_batch.py: thermal/solvent predictions keep full precision (no rounding).

Original docstring: Batch prediction with MOFSimplify models (thermal ANN, solvent ANN,
water RF, acid RF) over featurized descriptor CSVs produced by
featurize_shards.py.

Usage:
    python predict_batch.py --desc-dir DESCRIPTORS --out predictions.csv \
        [--start --end] (chunk of descriptor CSVs, for parallel shards)
"""
import argparse
import glob
import os

import joblib
import numpy as np
import pandas as pd

MOFSIMPLIFY = os.path.expanduser('~/MOFSimplify/')

RACs = ['D_func-I-0-all', 'D_func-I-1-all', 'D_func-I-2-all', 'D_func-I-3-all',
        'D_func-S-0-all', 'D_func-S-1-all', 'D_func-S-2-all', 'D_func-S-3-all',
        'D_func-T-0-all', 'D_func-T-1-all', 'D_func-T-2-all', 'D_func-T-3-all',
        'D_func-Z-0-all', 'D_func-Z-1-all', 'D_func-Z-2-all', 'D_func-Z-3-all',
        'D_func-chi-0-all', 'D_func-chi-1-all', 'D_func-chi-2-all',
        'D_func-chi-3-all', 'D_lc-I-0-all', 'D_lc-I-1-all', 'D_lc-I-2-all',
        'D_lc-I-3-all', 'D_lc-S-0-all', 'D_lc-S-1-all', 'D_lc-S-2-all',
        'D_lc-S-3-all', 'D_lc-T-0-all', 'D_lc-T-1-all', 'D_lc-T-2-all',
        'D_lc-T-3-all', 'D_lc-Z-0-all', 'D_lc-Z-1-all', 'D_lc-Z-2-all',
        'D_lc-Z-3-all', 'D_lc-chi-0-all', 'D_lc-chi-1-all', 'D_lc-chi-2-all',
        'D_lc-chi-3-all', 'D_mc-I-0-all', 'D_mc-I-1-all', 'D_mc-I-2-all',
        'D_mc-I-3-all', 'D_mc-S-0-all', 'D_mc-S-1-all', 'D_mc-S-2-all',
        'D_mc-S-3-all', 'D_mc-T-0-all', 'D_mc-T-1-all', 'D_mc-T-2-all',
        'D_mc-T-3-all', 'D_mc-Z-0-all', 'D_mc-Z-1-all', 'D_mc-Z-2-all',
        'D_mc-Z-3-all', 'D_mc-chi-0-all', 'D_mc-chi-1-all', 'D_mc-chi-2-all',
        'D_mc-chi-3-all', 'f-I-0-all', 'f-I-1-all', 'f-I-2-all', 'f-I-3-all',
        'f-S-0-all', 'f-S-1-all', 'f-S-2-all', 'f-S-3-all', 'f-T-0-all', 'f-T-1-all',
        'f-T-2-all', 'f-T-3-all', 'f-Z-0-all', 'f-Z-1-all', 'f-Z-2-all', 'f-Z-3-all',
        'f-chi-0-all', 'f-chi-1-all', 'f-chi-2-all', 'f-chi-3-all', 'f-lig-I-0',
        'f-lig-I-1', 'f-lig-I-2', 'f-lig-I-3', 'f-lig-S-0', 'f-lig-S-1', 'f-lig-S-2',
        'f-lig-S-3', 'f-lig-T-0', 'f-lig-T-1', 'f-lig-T-2', 'f-lig-T-3', 'f-lig-Z-0',
        'f-lig-Z-1', 'f-lig-Z-2', 'f-lig-Z-3', 'f-lig-chi-0', 'f-lig-chi-1',
        'f-lig-chi-2', 'f-lig-chi-3', 'func-I-0-all', 'func-I-1-all',
        'func-I-2-all', 'func-I-3-all', 'func-S-0-all', 'func-S-1-all',
        'func-S-2-all', 'func-S-3-all', 'func-T-0-all', 'func-T-1-all',
        'func-T-2-all', 'func-T-3-all', 'func-Z-0-all', 'func-Z-1-all',
        'func-Z-2-all', 'func-Z-3-all', 'func-chi-0-all', 'func-chi-1-all',
        'func-chi-2-all', 'func-chi-3-all', 'lc-I-0-all', 'lc-I-1-all', 'lc-I-2-all',
        'lc-I-3-all', 'lc-S-0-all', 'lc-S-1-all', 'lc-S-2-all', 'lc-S-3-all',
        'lc-T-0-all', 'lc-T-1-all', 'lc-T-2-all', 'lc-T-3-all', 'lc-Z-0-all',
        'lc-Z-1-all', 'lc-Z-2-all', 'lc-Z-3-all', 'lc-chi-0-all', 'lc-chi-1-all',
        'lc-chi-2-all', 'lc-chi-3-all', 'mc-I-0-all', 'mc-I-1-all', 'mc-I-2-all',
        'mc-I-3-all', 'mc-S-0-all', 'mc-S-1-all', 'mc-S-2-all', 'mc-S-3-all',
        'mc-T-0-all', 'mc-T-1-all', 'mc-T-2-all', 'mc-T-3-all', 'mc-Z-0-all',
        'mc-Z-1-all', 'mc-Z-2-all', 'mc-Z-3-all', 'mc-chi-0-all', 'mc-chi-1-all',
        'mc-chi-2-all', 'mc-chi-3-all']
GEO = ['Df', 'Di', 'Dif', 'GPOAV', 'GPONAV', 'GPOV', 'GSA', 'POAV', 'POAV_vol_frac',
       'PONAV', 'PONAV_vol_frac', 'VPOV', 'VSA', 'cell_v']

WATER_FEATS = ['mc-Z-3-all', 'D_mc-Z-3-all', 'D_mc-Z-2-all', 'D_mc-Z-1-all',
               'mc-chi-3-all', 'mc-Z-1-all', 'mc-Z-0-all', 'D_mc-chi-2-all',
               'f-lig-Z-2', 'GSA', 'f-lig-I-0', 'func-S-1-all']
ACID_FEATS = ['mc-chi-3-all', 'Dif', 'mc-Z-2-all', 'Di', 'f-T-2-all',
              'D_mc-chi-2-all', 'lc-chi-3-all', 'D_mc-S-1-all']


def precision_(y_true, y_pred):
    import tensorflow.keras.backend as K
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    return true_positives / (predicted_positives + K.epsilon())


def recall_(y_true, y_pred):
    import tensorflow.keras.backend as K
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    return true_positives / (possible_positives + K.epsilon())


def f1_(y_true, y_pred):
    p = precision_(y_true, y_pred)
    r = recall_(y_true, y_pred)
    return 2 * ((p * r) / (p + r + K_epsilon()))


def K_epsilon():
    import tensorflow.keras.backend as K
    return K.epsilon()


def train_match(train_df, new_row, skip_cols=('name', 'cif_file', 'Dif')):
    """Vectorized version of app.py's in-train check: a train row matches if
    |row - new| <= 5% of |new| for all checked columns."""
    cols = [c for c in new_row.index if c in train_df.columns and c not in skip_cols]
    t = train_df[cols].to_numpy(dtype=float)
    n = new_row[cols].to_numpy(dtype=float)
    with np.errstate(invalid='ignore', divide='ignore'):
        diff = np.abs(t - n)
        tol = 0.05 * np.abs(n)
        match = (diff <= tol) | (np.isnan(n) & np.isnan(t))
        ok = np.all(match, axis=1)
    # require new row values to be finite for a match (NaN new value can't confirm match)
    if np.any(np.isnan(n)):
        ok &= np.zeros_like(ok)
    idx = np.where(ok)[0]
    return idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--desc-dir', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--nshards', type=int, default=1)
    args = ap.parse_args()

    import tensorflow as tf
    from sklearn.preprocessing import StandardScaler

    deps = {'precision': precision_, 'recall': recall_, 'f1': f1_,
            'precision_': precision_, 'recall_': recall_, 'f1_': f1_}
    solvent_model = tf.keras.models.load_model(
        MOFSIMPLIFY + 'model/solvent/ANN/final_model_flag_few_epochs.h5', custom_objects=deps)
    thermal_model = tf.keras.models.load_model(
        MOFSIMPLIFY + 'model/thermal/ANN/final_model_T_few_epochs.h5', custom_objects=deps)

    water_model = joblib.load(MOFSIMPLIFY + 'model/water_and_acid/models/water_model.joblib')
    acid_model = joblib.load(MOFSIMPLIFY + 'model/water_and_acid/models/acid_model.joblib')
    water_scaler = joblib.load(MOFSIMPLIFY + 'model/water_and_acid/models/water_scaler.joblib')
    acid_scaler = joblib.load(MOFSIMPLIFY + 'model/water_and_acid/models/acid_scaler.joblib')

    # thermal training setup
    t_train = pd.read_csv(MOFSIMPLIFY + 'model/thermal/ANN/train.csv')
    t_train = t_train.loc[:, (t_train != t_train.iloc[0]).any()]
    t_feats = [c for c in t_train.columns.values if c in RACs + GEO]
    t_train_d = t_train.dropna(subset=t_feats + ['T'])
    x_scaler_t = StandardScaler().fit(t_train_d[t_feats].values)
    y_scaler_t = StandardScaler().fit(t_train_d[['T']].values)

    # solvent training setup
    s_train = pd.read_csv(MOFSIMPLIFY + 'model/solvent/ANN/dropped_connectivity_dupes/train.csv')
    s_train = s_train.loc[:, (s_train != s_train.iloc[0]).any()]
    s_feats = [c for c in s_train.columns.values if c in RACs + GEO]
    s_train_d = s_train.dropna(subset=s_feats + ['flag']).copy()
    s_train_d['flag'] = [1 if x == 1 else 0 for x in s_train_d['flag']]
    x_scaler_s = StandardScaler().fit(s_train_d[s_feats].values)

    files = sorted(glob.glob(os.path.join(args.desc_dir, '*_descriptors.csv')))[args.shard::args.nshards]
    print(f'predict shard {args.shard}/{args.nshards}: {len(files)} descriptor files', flush=True)

    rows = []
    for i, f in enumerate(files):
        name = os.path.basename(f).replace('_descriptors.csv', '')
        df = pd.read_csv(f)
        new = df.iloc[0]
        rec = {'name': name}

        # ---- thermal ----
        if not df[t_feats].isna().any().any():
            m = train_match(t_train_d, new)
            if len(m) > 0:
                rec['thermal_in_train'] = True
                rec['thermal_T_C'] = float(t_train_d.iloc[m[0]]['T'])
            else:
                Xn = x_scaler_t.transform(df[t_feats].values)
                pred = y_scaler_t.inverse_transform(thermal_model.predict(Xn, verbose=0))
                rec['thermal_in_train'] = False
                rec['thermal_T_C'] = float(pred[0][0])
        else:
            rec['thermal_in_train'] = False
            rec['thermal_T_C'] = np.nan

        # ---- solvent ----
        if not df[s_feats].isna().any().any():
            m = train_match(s_train_d, new)
            if len(m) > 0:
                rec['solvent_in_train'] = True
                rec['solvent_flag'] = int(s_train_d.iloc[m[0]]['flag'])
            else:
                Xn = x_scaler_s.transform(df[s_feats].values)
                pred = solvent_model.predict(Xn, verbose=0)
                rec['solvent_in_train'] = False
                rec['solvent_flag'] = float(pred[0][0])
        else:
            rec['solvent_in_train'] = False
            rec['solvent_flag'] = np.nan

        # ---- water ----
        if not df[WATER_FEATS].isna().any().any():
            Xw = water_scaler.transform(df[WATER_FEATS].to_numpy())
            rec['water_proba'] = round(float(water_model.predict_proba(Xw)[:, 1][0]), 2)
        else:
            rec['water_proba'] = np.nan

        # ---- acid ----
        if not df[ACID_FEATS].isna().any().any():
            Xa = acid_scaler.transform(df[ACID_FEATS].to_numpy())
            rec['acid_proba'] = round(float(acid_model.predict_proba(Xa)[:, 1][0]), 2)
        else:
            rec['acid_proba'] = np.nan

        rows.append(rec)
        if (i + 1) % 200 == 0:
            print(f'shard {args.shard}: {i+1}/{len(files)}', flush=True)

    out_df = pd.DataFrame(rows)
    out_df.to_csv(args.out, index=False)
    print(f'shard {args.shard} DONE wrote {len(out_df)} rows to {args.out}', flush=True)


if __name__ == '__main__':
    main()
