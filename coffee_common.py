"""Shared machinery for scripts 3 to 9.

WHY THIS FILE EXISTS
Scripts 3 to 6 each need the same feature list, the same class codes, the same
region names, the same pixel sampler and the same splits. When those lived in
four separate copies they drifted, twice, and both times it was silent:

  * one copy still held the old three-design experiment while the later scripts
    read output from the four-design one,
  * and the region names disagreed, so one script labelled a cluster Magdalena
    while the next called the same cluster Santander.

Neither raised an error. Both would have reached the paper. So the constants
and the machinery live here once, and every script imports them.

Settings that depend on WHERE YOUR FILES ARE live in gee_local_config.py, the
same file scripts 1 and 2 already use. Settings that are METHOD, such as the
seeds, the tree count and the pixel caps, live in the script that uses them,
because those belong with the experiment rather than with your machine.

Nothing here fits a model or writes a file. Importing it is cheap.
"""

from pathlib import Path
import gc
import importlib.util
import json
import os

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, f1_score, cohen_kappa_score,
                             confusion_matrix, classification_report)


# =============================================================================
# LOCAL CONFIG. Same contract as scripts 1 and 2.
# =============================================================================
def load_local_config(path, required):
    """Load required settings; dataset-specific hooks stay optional."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing {path}. Copy gee_config.example.py to "
            "gee_local_config.py in the same folder as the scripts and fill "
            "in your settings."
        )
    spec = importlib.util.spec_from_file_location('gee_local_config', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    missing = [name for name in required if not hasattr(module, name)]
    if missing:
        raise ValueError(
            f"Missing settings in {path.name}: {', '.join(missing)}")
    return module


SCRIPT_DIR = (Path(__file__).resolve().parent if '__file__' in globals()
              else Path.cwd())
_cfg = load_local_config(SCRIPT_DIR / 'gee_local_config.py',
                         ('OUTPUT_DIR', 'ANNUAL_OUTPUT_NAME'))

OUTPUT_DIR = Path(_cfg.OUTPUT_DIR)
ANNUAL_FILE = OUTPUT_DIR / _cfg.ANNUAL_OUTPUT_NAME
# Where scripts 3 to 9 write. Defaults beside the data if not set.
ANALYSIS_DIR = Path(getattr(_cfg, 'ANALYSIS_DIR', OUTPUT_DIR / 'analysis'))


def out_dir(*parts):
    """A subfolder of ANALYSIS_DIR, created if needed."""
    p = ANALYSIS_DIR.joinpath(*parts)
    p.mkdir(parents=True, exist_ok=True)
    return p


# =============================================================================
# THE EXPERIMENT'S FIXED VOCABULARY. One copy, imported everywhere.
# =============================================================================
FEATURE_COLUMNS = [
    'canopy_height_tolan_variance_k5',
    'elevation',
    'B3_contrast_k1',
    'GCVI_variance_k1',
    'GCVI_variance_k5',
    'VARI_variance_k5',
    'B11_variance_k5',
    'B6',
    'B6_contrast_k4',
    'elevation_contrast_k3',
    'VARI_variance_k1',
    'B11',
    'B2_variance_k4',
    'slope',
    'B8_contrast_k1',
]

# The parquet carries the original nine-label scheme collapsed to five codes
# with a gap at 3. Remapped to 0..4 so array indexing is safe.
RAW_TO_CLEAN = {0: 0, 1: 1, 2: 2, 4: 3, 5: 4}
CLASS_NAMES = {0: 'Non-Shade Coffee', 1: 'Shade Coffee', 2: 'Forest',
               3: 'Open', 4: 'Urban'}
CLASS_SHORT = {0: 'Sun coffee', 1: 'Shade coffee', 2: 'Forest',
               3: 'Open', 4: 'Urban'}
ALL_CLASSES = sorted(CLASS_NAMES)
K = len(ALL_CLASSES)
NSC, SHADE_COFFEE, FOREST, OPEN, URBAN = 0, 1, 2, 3, 4
COFFEE = [NSC, SHADE_COFFEE]
NON_COFFEE = [FOREST, OPEN, URBAN]

ID_COL = 'unique_id'
SOURCE_COL = 'poly_source'
CLUSTER_COL = 'cluster_id'
COORD_PREF = [('easting', 'northing'), ('lon', 'lat')]

N_REGIONS = 4
# Checked against the cluster centroids. The scripts verify this at load and
# warn if the nearest department disagrees, because an earlier version had
# 1, 2 and 3 rotated and every region label in the output was wrong.
REGION_NAME_HINT = {0: 'Cauca', 1: 'Cundinamarca', 2: 'Magdalena',
                    3: 'Santander'}
DEPT_LONLAT = {'Cauca': (-76.5, 2.5), 'Cundinamarca': (-74.0, 5.0),
               'Santander': (-73.0, 7.0), 'Magdalena': (-74.0, 10.5)}

# Magdalena carries about 8 coffee test polygons per split, as few as 5. It is
# reported but excluded from per-region claims, and the scripts say so.
SMALL_REGIONS = [2]

# Shared defaults. A script may override any of these locally.
MIN_PIXELS_PER_POLYGON = 10
SEEDS = [7, 12, 35, 53, 56, 64, 67, 75, 84, 100]
FINAL_PIXEL_CAP = 50
TEST_PIXEL_CAP = None
FINAL_TREES = 500
RF_MAX_DEPTH = 30
FINAL_WEIGHT = 'balanced'
FINAL_TEST_FRAC = 0.25
PREDICT_BATCH_ROWS = 2000
RF_N_JOBS = -1

# Every script writes and reads under these tags. Change one and the chain
# breaks loudly rather than quietly, because 4, 5 and 6 verify the splits.
TAG_DESIGNS = 'annual15_nested'
TAG_DISTANCE = 'annual15_distance'
TAG_NSC = 'annual15_nsc'


def banner(title, char='='):
    print('\n' + char * 78)
    print(title)
    print(char * 78)


def mode_of(s):
    return s.mode().iloc[0] if not s.mode().empty else np.nan


# =============================================================================
# DATA
# =============================================================================
class Data:
    """The pixel table, the polygon table and the clusters.

    Built once per script. Every downstream helper takes one of these rather
    than reaching for a global, so two scripts cannot disagree about what was
    loaded.
    """

    def __init__(self, df, poly_df, coord_cols, lonlat, cluster_source):
        self.df = df
        self.poly_df = poly_df
        self.coord_cols = coord_cols
        self.lonlat = lonlat
        self.cluster_source = cluster_source
        self.all_ids = set(poly_df[ID_COL])
        self.clusters = sorted(poly_df[CLUSTER_COL].unique())
        self.cluster_ids = {
            c: set(poly_df.loc[poly_df[CLUSTER_COL] == c, ID_COL])
            for c in self.clusters}
        self.region_of_cluster = dict(
            poly_df.groupby(CLUSTER_COL)['region'].first())
        self.class_of = dict(zip(poly_df[ID_COL], poly_df['class']))
        self.cluster_of = dict(zip(poly_df[ID_COL], poly_df[CLUSTER_COL]))
        self.region_of = dict(zip(poly_df[ID_COL], poly_df['region']))
        self.npix_of = dict(zip(poly_df[ID_COL], poly_df['n_pixels']))
        self._id_arr = df[ID_COL].to_numpy()
        self._cap_cache = {}

    # ---- pixel sampling ---------------------------------------------------
    def capped_positions(self, seed, cap):
        """Positions of the pixels this seed selects, per polygon.

        The selection depends on the seed and the polygon ONLY. It does not
        depend on which polygons were asked for, which is what lets every
        design see identical pixels for any polygon they share. An earlier
        version shuffled a list whose length depended on the id set, so a
        polygon in both the pooled and the local training set got a different
        sample in each, and only 12 of 75 shared polygons matched.
        """
        key = (seed, cap)
        if key in self._cap_cache:
            return self._cap_cache[key]
        if cap is None:
            pos = np.arange(len(self.df))
        else:
            perm = np.random.RandomState(seed).permutation(len(self.df))
            pid = self._id_arr[perm]
            rank = pd.Series(pid).groupby(pid, sort=False).cumcount().to_numpy()
            pos = np.sort(perm[rank < cap])
        self._cap_cache[key] = pos
        return pos

    def capped(self, ids, seed, cap):
        pos = self.capped_positions(seed, cap)
        keep = pd.Series(self._id_arr[pos]).isin(ids).to_numpy()
        return self.df.iloc[pos[keep]]

    # ---- splits -----------------------------------------------------------
    def strat_pick(self, ids, frac, seed, by=('class',)):
        frame = self.poly_df[self.poly_df[ID_COL].isin(ids)]
        rng = np.random.RandomState(seed)
        picked = []
        for _, cell in frame.groupby(list(by), observed=True):
            k = min(max(1, int(np.ceil(frac * len(cell)))), len(cell))
            picked.extend(rng.choice(cell[ID_COL].to_numpy(), size=k,
                                     replace=False).tolist())
        return set(picked)

    def build_splits(self, seeds, frac=FINAL_TEST_FRAC):
        """{seed: (train_ids, test_ids)}. Deterministic given the polygon set."""
        return {s: (self.all_ids - self.strat_pick(self.all_ids, frac, s),
                    self.strat_pick(self.all_ids, frac, s)) for s in seeds}

    def distance_matrix_km(self):
        """Polygon centroid distances in km. Requires projected coordinates."""
        if self.coord_cols != ('easting', 'northing'):
            raise ValueError(
                'Distances need projected coordinates in metres. Found '
                f'{self.coord_cols}. Add easting and northing, or project '
                'first. Degrees would make every distance meaningless.')
        xy = self.poly_df[['x', 'y']].to_numpy(float)
        D = np.hypot(xy[:, 0][:, None] - xy[:, 0][None, :],
                     xy[:, 1][:, None] - xy[:, 1][None, :]) / 1000.0
        return D, {u: i for i, u in enumerate(self.poly_df[ID_COL])}


def load_data(features=None, cluster_source='auto', verbose=True):
    """Read the annual parquet, filter it, and build polygons and clusters.

    Identical filtering in every script, so the splits reproduce. Any change
    here changes every downstream result, which is the point of it being in
    one place.
    """
    features = list(features or FEATURE_COLUMNS)
    available = pq.ParquetFile(ANNUAL_FILE).schema_arrow.names

    required = [ID_COL, SOURCE_COL, 'class', 'elevation']
    missing = [c for c in required + features if c not in available]
    if missing:
        raise ValueError(f'missing columns in {ANNUAL_FILE.name}: '
                         f'{sorted(set(missing))}')
    coords = next((p for p in COORD_PREF
                   if all(c in available for c in p)), None)
    if coords is None:
        raise ValueError(f'no coordinate pair found: {COORD_PREF}')
    lonlat = [c for c in ('lon', 'lat') if c in available]
    has_file_cluster = CLUSTER_COL in available
    if cluster_source == 'file' and not has_file_cluster:
        raise ValueError(f"cluster_source is 'file' but {CLUSTER_COL} is not "
                         f'a column in {ANNUAL_FILE.name}')

    load = list(dict.fromkeys(
        required + list(coords) + lonlat
        + ([CLUSTER_COL] if has_file_cluster else []) + features))
    if verbose:
        banner('LOADING')
        print(f'  {ANNUAL_FILE}')
        print(f'  {len(load)} of {len(available)} columns, '
              f'{CLUSTER_COL} present: {has_file_cluster}')
    df = pd.read_parquet(ANNUAL_FILE, columns=load, engine='pyarrow',
                         use_threads=False, pre_buffer=False)

    bad = df.loc[~df['class'].isin(RAW_TO_CLEAN), 'class'].unique()
    if len(bad):
        raise ValueError(f'class values not in RAW_TO_CLEAN: {bad}')
    df['class'] = df['class'].map(RAW_TO_CLEAN).astype(int)
    n0, p0 = len(df), df[ID_COL].nunique()

    complete = df[features].notna().all(axis=1)
    df = df.loc[complete].copy()
    del complete
    gc.collect()
    if df.empty:
        raise RuntimeError('the completeness filter removed every row')
    px = df.groupby(ID_COL).size()
    thin = px[px < MIN_PIXELS_PER_POLYGON].index
    if len(thin):
        df = df[~df[ID_COL].isin(thin)].copy()
        gc.collect()
    if verbose:
        print(f'  rows     {n0:>9,} -> {len(df):>9,}')
        print(f'  polygons {p0:>9,} -> {df[ID_COL].nunique():>9,}  '
              f'(complete across {len(features)} features, '
              f'>= {MIN_PIXELS_PER_POLYGON} pixels)')

    coord_cols = tuple(coords)
    agg = {'class': ('class', mode_of), 'source': (SOURCE_COL, mode_of),
           'elevation': ('elevation', 'mean'),
           'x': (coord_cols[0], 'mean'), 'y': (coord_cols[1], 'mean'),
           'n_pixels': (ID_COL, 'size')}
    for c in lonlat:
        agg[c] = (c, 'mean')
    if has_file_cluster:
        agg['file_cluster'] = (CLUSTER_COL, mode_of)
    poly_df = df.groupby(ID_COL).agg(**agg).reset_index()

    use_file = (has_file_cluster and cluster_source in ('auto', 'file')
                and poly_df['file_cluster'].notna().all())
    if use_file:
        poly_df[CLUSTER_COL] = poly_df['file_cluster'].astype(int)
        src = f'{CLUSTER_COL} column in the file'
    else:
        XY = StandardScaler().fit_transform(
            poly_df[['x', 'y']].to_numpy(float))
        poly_df[CLUSTER_COL] = KMeans(n_clusters=N_REGIONS, n_init=25,
                                      random_state=0).fit_predict(XY)
        order = (poly_df.groupby(CLUSTER_COL)['x'].mean()
                 .sort_values().index.tolist())
        poly_df[CLUSTER_COL] = poly_df[CLUSTER_COL].map(
            {o: n for n, o in enumerate(order)})
        del XY
        gc.collect()
        src = f'KMeans k={N_REGIONS} on standardised {coord_cols}'
    poly_df['region'] = poly_df[CLUSTER_COL].map(
        lambda i: f"R{i} ({REGION_NAME_HINT.get(i, '?')})")
    if verbose:
        print(f'  clusters from {src}')

    data = Data(df, poly_df, coord_cols, lonlat, src)
    if verbose:
        _check_region_names(poly_df, lonlat)
        print(f'\n  {len(poly_df)} polygons by class and region')
        print(pd.crosstab(poly_df['class'].map(CLASS_NAMES),
                          poly_df['region'], margins=True).to_string())
    return data


def _check_region_names(poly_df, lonlat):
    """Cluster labels are arbitrary. Naming them wrong mislabels every table,
    which happened once, so it is checked rather than trusted."""
    if len(lonlat) != 2:
        return
    gc_ = poly_df.groupby(CLUSTER_COL)[['lon', 'lat']].mean()
    rows = []
    for i, row in gc_.iterrows():
        d = {n: np.hypot(row['lon'] - ll[0], row['lat'] - ll[1])
             for n, ll in DEPT_LONLAT.items()}
        best = min(d, key=d.get)
        rows.append({CLUSTER_COL: i, 'lon': round(row['lon'], 3),
                     'lat': round(row['lat'], 3),
                     'named': REGION_NAME_HINT.get(i, '?'),
                     'nearest': best, 'deg': round(d[best], 2),
                     'agrees': REGION_NAME_HINT.get(i) == best})
    t = pd.DataFrame(rows).set_index(CLUSTER_COL)
    print('\n  Cluster centres against department locations')
    print(t.to_string())
    if not t['agrees'].all():
        print('\n  WARNING. REGION_NAME_HINT disagrees with the nearest')
        print('  department. The modelling is unaffected but every region')
        print('  NAME in every table would be wrong. Fix it in')
        print('  coffee_common.py before reading anything below.')


# =============================================================================
# MODEL
# =============================================================================
def prep_X(rows, feats):
    return np.nan_to_num(rows[feats].to_numpy(dtype=np.float32, copy=True),
                         copy=False, nan=0.0)


def fit_rf(rows, feats, y, seed, n_trees=FINAL_TREES, weight=FINAL_WEIGHT,
           max_depth=RF_MAX_DEPTH, n_jobs=RF_N_JOBS, quiet=True):
    gc.collect()
    if not quiet:
        print(f'    fit: seed {seed}, {len(rows):,} pixels, {len(feats)} '
              f'features, {n_trees} trees', flush=True)
    Xs = prep_X(rows, feats)
    sc = StandardScaler(copy=False)
    Xs = sc.fit_transform(Xs).astype(np.float32, copy=False)
    rf = RandomForestClassifier(n_estimators=n_trees, max_depth=max_depth,
                                random_state=seed, n_jobs=n_jobs,
                                class_weight=weight)
    rf.fit(Xs, y)
    del Xs
    gc.collect()
    return rf, sc


def predict_batched(rf, sc, rows, feats, want_proba=False,
                    batch=PREDICT_BATCH_ROWS):
    out = np.empty(len(rows), dtype=rf.classes_.dtype)
    P = np.zeros((len(rows), K), dtype=np.float32) if want_proba else None
    cols = np.searchsorted(ALL_CLASSES, rf.classes_.astype(int))
    for s in range(0, len(rows), batch):
        e = min(s + batch, len(rows))
        M = sc.transform(prep_X(rows.iloc[s:e], feats),
                         copy=False).astype(np.float32, copy=False)
        if want_proba:
            pr = rf.predict_proba(M).astype(np.float32)
            P[s:e][:, cols] = pr
            out[s:e] = rf.classes_[pr.argmax(axis=1)]
        else:
            out[s:e] = rf.predict(M)
        del M
    return out, P


def polygon_labels(rows, y_pred, P=None, vote='hard'):
    ids = rows[ID_COL].to_numpy()
    y_true = rows['class'].to_numpy()
    if vote == 'soft':
        if P is None:
            raise RuntimeError('soft vote asked for without probabilities')
        order = np.argsort(ids, kind='stable')
        ids_s, P_s, yt_s = ids[order], P[order], y_true[order]
        starts = np.flatnonzero(np.r_[True, ids_s[1:] != ids_s[:-1]])
        S = np.add.reduceat(P_s, starts, axis=0)
        return (ids_s[starts], yt_s[starts],
                np.asarray(ALL_CLASSES)[S.argmax(axis=1)])
    tmp = pd.DataFrame({ID_COL: ids, 'y_true': y_true, 'y_pred': y_pred})
    g = tmp.groupby(ID_COL).agg(y_true=('y_true', 'first'),
                                y_pred=('y_pred', mode_of))
    return g.index.to_numpy(), g['y_true'].to_numpy(), g['y_pred'].to_numpy()


def metrics(y_true, y_pred):
    """Per-class F1 is one-vs-rest against ALL five classes.

    cf_f1 and sc_nsc_f1 are NOT. They are scored only on polygons whose truth
    is one of the two classes named, so they exclude false positives arriving
    from the other three. Report them beside the one-vs-rest scores, never
    instead of them.
    """
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    out = {'oa': accuracy_score(y_true, y_pred)}
    if len(np.unique(y_true)) > 1:
        out['kappa'] = cohen_kappa_score(y_true, y_pred)
    f1s = []
    for c in ALL_CLASSES:
        if (y_true == c).sum() == 0:
            out[f'f1_{CLASS_NAMES[c]}'] = np.nan
            continue
        v = f1_score(y_true == c, y_pred == c, zero_division=0)
        out[f'f1_{CLASS_NAMES[c]}'] = v
        f1s.append(v)
    out['macro_f1'] = float(np.mean(f1s)) if f1s else np.nan
    cof = [out[f'f1_{CLASS_NAMES[c]}'] for c in COFFEE
           if not np.isnan(out.get(f'f1_{CLASS_NAMES[c]}', np.nan))]
    out['coffee_mean_f1'] = float(np.mean(cof)) if cof else np.nan
    out['cf_f1'] = _restricted_f1(y_true, y_pred, SHADE_COFFEE, FOREST)
    out['sc_nsc_f1'] = _restricted_f1(y_true, y_pred, SHADE_COFFEE, NSC)
    out['nsc_vs_open_f1'] = _restricted_f1(y_true, y_pred, NSC, OPEN)
    m = np.isin(y_true, COFFEE)
    out['sc_nsc_n'] = int(m.sum())
    return out


def _restricted_f1(y_true, y_pred, positive, other):
    """F1 for `positive` among polygons that truly are one of the two."""
    m = np.isin(y_true, [positive, other])
    if m.sum() == 0 or len(np.unique(y_true[m])) < 2:
        return np.nan
    return f1_score(y_true[m] == positive, y_pred[m] == positive,
                    zero_division=0)


def polygon_metrics(rows, y_pred, P=None, vote='hard'):
    _, yt, yp = polygon_labels(rows, y_pred, P, vote)
    return metrics(yt, yp)


def print_confusion(y_true, y_pred, title):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    if not len(y_true):
        print(f'\n  {title}: nothing to show')
        return
    labs = [c for c in ALL_CLASSES
            if (y_true == c).sum() or (y_pred == c).sum()]
    cm = confusion_matrix(y_true, y_pred, labels=labs)
    names = [CLASS_NAMES[c] for c in labs]
    print(f'\n  {title}, rows are truth')
    print(pd.DataFrame(cm, index=names, columns=names).to_string())
    print()
    print(classification_report(y_true, y_pred, labels=labs,
                                target_names=names, zero_division=0,
                                digits=3))


# =============================================================================
# SPLITS ON DISK. How scripts 6, 7 and 8 know they match script 5.
# =============================================================================
def splits_path(tag=TAG_DESIGNS):
    return out_dir(tag, 'splits') / f'S_{tag}_splits.json'


def save_splits(splits, tag=TAG_DESIGNS):
    payload = {str(s): {'f0': {'train': sorted(int(i) for i in tr),
                               'test': sorted(int(i) for i in te)}}
               for s, (tr, te) in splits.items()}
    p = splits_path(tag)
    if p.exists():
        try:
            if json.loads(p.read_text()) == payload:
                print('  splits match the ones already on disk')
            else:
                print('  WARNING. The splits on disk DIFFER from the ones')
                print('  just built. The polygon set or the seeds changed.')
                print('  Earlier results in this folder are not comparable.')
        except Exception:
            pass
    p.write_text(json.dumps(payload))
    return p


def verify_splits(splits, tag=TAG_DESIGNS, quiet=False):
    """Refuse to continue unless these splits are the ones script 5 used.

    Without this a script silently scores different polygons and the whole
    chain stops being paired, which is the failure this project spent a week
    chasing the first time.
    """
    p = splits_path(tag)
    if not p.exists():
        print(f'  WARNING. No splits file at {p}')
        print('  Rebuilt with the same recipe. If the polygon set changed')
        print('  since script 5 ran, nothing here pairs with it.')
        return False
    disk = json.loads(p.read_text())
    bad = []
    for seed, (_, te) in splits.items():
        d = disk.get(str(seed), {}).get('f0')
        if d is None:
            bad.append(f'seed {seed} absent from the splits file')
        elif set(d['test']) != {int(i) for i in te}:
            bad.append(f'seed {seed} test set differs')
    if bad:
        raise RuntimeError(
            'Script 5 splits could not be reproduced, so the test polygons '
            'would not match and nothing here would be comparable.\n  '
            + '\n  '.join(bad))
    if not quiet:
        print(f'  reproduced script 5 splits exactly for all '
              f'{len(splits)} seeds')
    return True


def save_table(frame, tag, name, index=False):
    p = out_dir(tag) / name
    frame.to_csv(p, index=index)
    return p
