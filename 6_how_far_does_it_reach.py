"""QUESTION 2. How accurate is the map really, and how far does a training
polygon reach?

Two things are tangled together and this separates them.

    A buffer sweep. The test polygons are FIXED and identical to script 5's
    held-out share. Training polygons within d km of any test polygon are
    removed. A control arm removes the SAME NUMBER with the SAME CLASS MIX
    from any distance. Both arms lose the same data, so the gap between them
    is the part caused by distance rather than by having less data.

    A grouped split. Polygons within DEDUP_M metres that share a class are
    treated as ONE unit when the split is drawn, so a plot can never be
    tested against its own near-duplicate. A completely different method,
    aimed at the same question.

WHAT EACH OUTCOME MEANS
    distance effect near zero everywhere   the benefit was data volume
    large at small buffers, then flat      near-field, a validation artifact
    still growing at the widest buffer     reference data stays informative
                                           at that range, which is the
                                           sampling result

Also reports what the closest polygon pairs look like, which costs nothing
and distinguishes duplicate labels from ordinary neighbouring land.

The ladder is dense below 500 m because the first run showed the median
nearest neighbour at 208 m, so everything that mattered was happening under
the old first step.

WHAT THIS WRITES
    T8_*_sweep.csv                         the curve, figure F2 and F3
    P_*_predictions_by_buffer.csv          script 7 cuts this by distance
    T9_*_near_pairs.csv, T9b_*_marginal.csv, T10_*_grouped.csv

Run time about 45 minutes.
"""

import gc
import time

import numpy as np
import pandas as pd

import coffee_common as cc
from coffee_common import (ID_COL, CLUSTER_COL, CLASS_NAMES, SEEDS,
                           FINAL_PIXEL_CAP, TEST_PIXEL_CAP, NSC,
                           SHADE_COFFEE, TAG_DISTANCE, TAG_DESIGNS, banner)

# ---- method settings -------------------------------------------------------
TAG = TAG_DISTANCE
BUFFERS_KM = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0]
N_SWEEP_SEEDS = 5     # region SD is 0.043, seed SD 0.005, so seeds are cheap
DEDUP_M = 200.0       # None to skip the grouped pass
MIN_TRAIN_POLY = 30
MIN_TEST_POLY = 5
RUN_RANDOM_ARM = True
SAVE_PREDICTIONS = True

SWEEP_SEEDS = SEEDS[:N_SWEEP_SEEDS]
t0 = time.time()

banner('SCRIPT 6. HOW FAR FROM A TRAINING POLYGON DOES IT STILL WORK')
_arms = 2 if RUN_RANDOM_ARM else 1
_nf = len(SWEEP_SEEDS) * cc.N_REGIONS * ((len(BUFFERS_KM) - 1) * _arms + 1)
_ng = len(SWEEP_SEEDS) * cc.N_REGIONS if DEDUP_M is not None else 0
print(f'  buffers {BUFFERS_KM} km')
print(f'  {len(SWEEP_SEEDS)} seeds, {_nf + _ng} forest fits')

D = cc.load_data()
SPLITS = D.build_splits(SWEEP_SEEDS)
banner('SPLITS')
cc.verify_splits(SPLITS, TAG_DESIGNS)

DKM, POS = D.distance_matrix_km()
CLS = D.poly_df['class'].to_numpy()
SRC = D.poly_df['source'].astype(str).to_numpy()


# =============================================================================
# HOW FAR APART ARE THE POLYGONS
# =============================================================================
banner('NEAREST NEIGHBOUR DISTANCES BETWEEN POLYGONS, KM')
_D = DKM.copy()
np.fill_diagonal(_D, np.inf)
print(pd.Series(_D.min(axis=1)).describe(
    percentiles=[.05, .25, .5, .75, .95]).round(3).to_string())
rows = []
for c in D.clusters:
    idx = [POS[u] for u in D.cluster_ids[c]]
    sub = _D[np.ix_(idx, idx)]
    v = sub.min(axis=1)
    rows.append({'region': D.region_of_cluster[c], 'n_poly': len(idx),
                 'nn_p05': np.percentile(v, 5), 'nn_median': np.median(v),
                 'nn_p95': np.percentile(v, 95),
                 'extent_km': sub[np.isfinite(sub)].max()})
print('\n  Same, within each cluster')
print(pd.DataFrame(rows).round(2).to_string(index=False))
print('\n  If most neighbours sit inside your smallest buffer, the ladder is')
print('  too coarse at the bottom and the first step will look like a cliff.')

banner('WHAT THE CLOSEST POLYGON PAIRS LOOK LIKE')
print('  Two polygons a few metres apart that share a class and a source are')
print('  very likely the same field digitised twice. If that is what sits')
print('  under the first buffer steps, the drop there is a split artifact')
print('  rather than short-range signal.')
iu = np.triu_indices(len(D.poly_df), k=1)
pd_km, same_cls = DKM[iu], CLS[iu[0]] == CLS[iu[1]]
same_src = SRC[iu[0]] == SRC[iu[1]]
rows, prev = [], 0.0
for hi in [0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 1.0]:
    m = (pd_km >= prev) & (pd_km < hi)
    n = int(m.sum())
    rows.append({'band_km': f'{prev:g} to {hi:g}', 'pairs': n,
                 'polygons_involved': int(len(np.unique(np.concatenate(
                     [iu[0][m], iu[1][m]])))) if n else 0,
                 'pct_same_class': round(100 * same_cls[m].mean(), 1)
                 if n else np.nan,
                 'pct_same_source': round(100 * same_src[m].mean(), 1)
                 if n else np.nan})
    prev = hi
PAIRS = pd.DataFrame(rows)
del rows
print('\n' + PAIRS.to_string(index=False))
print(f'\n  For reference, {100 * same_cls.mean():.1f} percent of ALL pairs')
print('  share a class by chance. A band far above that is the duplicate')
print('  signature. A band near it is ordinary neighbouring land.')
cc.save_table(PAIRS, TAG, f'T9_{TAG}_near_pairs.csv')

PRED_ROWS = []


def score(tr_ids, te_ids, seed, tag=None):
    tr = D.capped(tr_ids, seed, FINAL_PIXEL_CAP)
    te = D.capped(te_ids, seed, TEST_PIXEL_CAP)
    rf, sc = cc.fit_rf(tr, cc.FEATURE_COLUMNS, tr['class'].values, seed)
    yp, _ = cc.predict_batched(rf, sc, te, cc.FEATURE_COLUMNS)
    pid, pyt, pyp = cc.polygon_labels(te, yp)
    rec = {f'poly_{a}': b for a, b in cc.metrics(pyt, pyp).items()}
    rec.update({'n_train_poly': len(tr_ids), 'n_test_poly': len(te_ids),
                'n_train_px': len(tr)})
    if SAVE_PREDICTIONS:
        p = pd.DataFrame({ID_COL: pid, 'y_true': pyt, 'y_pred': pyp})
        p['seed'] = seed
        for k, v in (tag or {}).items():
            p[k] = v
        PRED_ROWS.append(p)
    del rf, sc, tr, te
    gc.collect()
    return rec


# =============================================================================
# THE SWEEP
# =============================================================================
banner('BUFFER SWEEP, TEST POLYGONS HELD FIXED')
print('  buffer d   drop training polygons within d km of ANY test polygon')
print('  random d   drop the same NUMBER and CLASS MIX at random, any distance')
print('  Both arms lose the same data, so the gap between them is distance.')

rows, skipped = [], []
for seed in SWEEP_SEEDS:
    tr_ids, te_ids = SPLITS[seed]
    for c in D.clusters:
        lte = D.cluster_ids[c] & te_ids
        if len(lte) < MIN_TEST_POLY:
            skipped.append({'seed': seed, CLUSTER_COL: c,
                            'reason': f'{len(lte)} test polygons'})
            continue
        pool = np.array(sorted(tr_ids))
        dmin = DKM[np.ix_([POS[u] for u in pool],
                          [POS[u] for u in lte])].min(axis=1)
        pool_cls = np.array([D.class_of[u] for u in pool])
        for d in BUFFERS_KM:
            keep = dmin >= d if d > 0 else np.ones(len(pool), bool)
            removed = int((~keep).sum())
            base = {'seed': seed, CLUSTER_COL: c,
                    'region': D.region_of_cluster[c], 'buffer_km': d,
                    'n_removed': removed, 'n_test_poly': len(lte)}
            arms = [('buffer', set(pool[keep]))]
            if RUN_RANDOM_ARM and d > 0 and removed > 0:
                rng = np.random.RandomState(
                    abs(hash((int(seed), int(c), float(d)))) % (2 ** 31))
                drop = []
                for cl in np.unique(pool_cls):
                    n_cl = int(((~keep) & (pool_cls == cl)).sum())
                    if n_cl:
                        drop.extend(rng.choice(pool[pool_cls == cl],
                                               size=n_cl,
                                               replace=False).tolist())
                arms.append(('random', set(pool) - set(drop)))
            for arm, trset in arms:
                if len(trset) < MIN_TRAIN_POLY or \
                        len({D.class_of[u] for u in trset}) < 2:
                    skipped.append({'seed': seed, CLUSTER_COL: c,
                                    'buffer_km': d, 'arm': arm,
                                    'reason': 'too few training polygons'})
                    continue
                rec = score(trset, lte, seed,
                            {'buffer_km': d, 'arm': arm, CLUSTER_COL: c,
                             'region': D.region_of_cluster[c]})
                rows.append({**base, 'arm': arm, **rec})
                if d == 0.0:
                    # nothing removed, so both arms are the same model.
                    # Recorded under each name so every curve starts equal.
                    rows.append({**base, 'arm': 'random', **rec})
    print(f'    seed {seed:>4} done', flush=True)

SWEEP = pd.DataFrame(rows)
del rows
if skipped:
    print(f'\n  {len(skipped)} cells skipped')
    print(pd.DataFrame(skipped)['reason'].value_counts().head().to_string())


# =============================================================================
# REPORT
# =============================================================================
SHOW = ['poly_macro_f1', f'poly_f1_{CLASS_NAMES[SHADE_COFFEE]}',
        f'poly_f1_{CLASS_NAMES[NSC]}', 'poly_sc_nsc_f1']
NICE = {'poly_macro_f1': 'macro F1',
        f'poly_f1_{CLASS_NAMES[SHADE_COFFEE]}': 'shade coffee F1',
        f'poly_f1_{CLASS_NAMES[NSC]}': 'sun coffee F1',
        'poly_sc_nsc_f1': 'shade vs sun'}

banner('HOW MUCH TRAINING DATA EACH BUFFER REMOVES')
rm = (SWEEP[SWEEP['arm'] == 'buffer'].groupby('buffer_km')
      .agg(n_removed=('n_removed', 'mean'), n_train=('n_train_poly', 'mean'),
           n_test=('n_test_poly', 'mean'), cells=('seed', 'size')).round(1))
rm['pct_removed'] = (100 * rm['n_removed']
                     / (rm['n_removed'] + rm['n_train'])).round(1)
print(rm.to_string())

for col in SHOW:
    if col not in SWEEP.columns:
        continue
    banner(f'{NICE[col].upper()} AGAINST BUFFER DISTANCE')
    piv = SWEEP.pivot_table(index='buffer_km', columns='arm', values=col)
    if {'buffer', 'random'} <= set(piv.columns):
        piv['distance effect'] = piv['random'] - piv['buffer']
        piv['data-loss effect'] = piv['random'].iloc[0] - piv['random']
    print(piv.round(4).to_string())
print('\n  distance effect   how much worse the buffered model is than')
print('    losing the same data at random. This is the distance term.')
print('  data-loss effect  how much the random arm alone fell. This is the')
print('    cost of having less data, whatever its location.')

banner('COST PER TRAINING POLYGON REMOVED, BY DISTANCE BAND')
print('  This settles whether the near ones are special. Each row takes one')
print('  step up the ladder and divides the score lost by the polygons that')
print('  step removed. FLAT means a polygon 8 km away is worth about as much')
print('  as one 300 m away. A SPIKE in the first band or two means the')
print('  nearest carry outsized weight, the duplicate signature.')
bc = (SWEEP[SWEEP['arm'] == 'buffer'].groupby('buffer_km')
      .agg(**{'n_removed': ('n_removed', 'mean'),
              **{c: (c, 'mean') for c in SHOW if c in SWEEP.columns}})
      .sort_index())
marg, bk = [], bc.index.to_list()
for i in range(1, len(bk)):
    dn = bc['n_removed'].iloc[i] - bc['n_removed'].iloc[i - 1]
    r = {'band_km': f'{bk[i - 1]:g} to {bk[i]:g}', 'polys_removed': round(dn, 1)}
    for c in SHOW:
        if c in bc.columns:
            r[NICE[c]] = (round((bc[c].iloc[i - 1] - bc[c].iloc[i]) / dn, 5)
                          if dn >= 1 else np.nan)
    marg.append(r)
MARGINAL = pd.DataFrame(marg)
del marg
print('\n' + MARGINAL.to_string(index=False))
print('\n  Bands that removed fewer than 1 polygon on average are blank.')

banner('HOW MANY COFFEE POLYGONS EACH REGION ACTUALLY TESTS')
print('  poly_sc_nsc_f1 is computed on these polygons alone. A region with a')
print('  handful cannot support a per-region claim, however clean it looks.')
if 'poly_sc_nsc_n' in SWEEP.columns:
    print('\n' + SWEEP[SWEEP['arm'] == 'buffer'].groupby('region')
          .agg(coffee_test_polys=('poly_sc_nsc_n', 'mean'),
               min_seen=('poly_sc_nsc_n', 'min'),
               all_test_polys=('n_test_poly', 'mean')).round(1).to_string())
    _small = ', '.join(f'R{c} ({cc.REGION_NAME_HINT.get(c, "?")})'
                       for c in cc.SMALL_REGIONS)
    print(f'\n  {_small} is too small for a per-region claim.')

cc.save_table(SWEEP, TAG, f'T8_{TAG}_sweep.csv')
cc.save_table(rm, TAG, f'T8b_{TAG}_removal.csv', index=True)
cc.save_table(MARGINAL, TAG, f'T9b_{TAG}_marginal.csv')


# =============================================================================
# GROUPED SPLIT. Near-duplicates cannot straddle training and test.
# =============================================================================
GROUPED = pd.DataFrame()
if DEDUP_M is not None:
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    banner('GROUPED SPLIT, NEAR-DUPLICATES KEPT TOGETHER')
    print(f'  Polygons within {DEDUP_M:g} m of each other that also share a')
    print('  class are treated as one unit when the split is drawn, so a')
    print('  polygon can never be tested against its own near-duplicate.')
    print('  These splits are NOT script 5 s. Read the DIFFERENCE column')
    print('  against its own baseline below, never the level against the')
    print('  sweep above.')
    near = (DKM <= DEDUP_M / 1000.0) & (CLS[:, None] == CLS[None, :])
    ncomp, lab = connected_components(coo_matrix(near), directed=False)
    D.poly_df['group'] = lab
    gs = pd.Series(lab).value_counts()
    in_group = int(gs[gs > 1].sum())
    print(f'\n  {len(D.poly_df)} polygons collapse into {ncomp} groups')
    print(f'  largest {int(gs.max())}, {int((gs > 1).sum())} hold more than one')
    print(f'  {in_group} polygons ({100 * in_group / len(D.poly_df):.1f} '
          f'percent) share a group with at least one other')
    if ncomp == len(D.poly_df):
        print('  Nothing grouped at this radius, so this pass reproduces the')
        print('  ungrouped baseline. Raise DEDUP_M.')

    grp = (D.poly_df.groupby('group')
           .agg(**{'class': ('class', 'first')}).reset_index())
    ids_of = D.poly_df.groupby('group')[ID_COL].apply(set).to_dict()
    rows = []
    for seed in SWEEP_SEEDS:
        rng = np.random.RandomState(seed)
        picked = []
        for _, cell in grp.groupby('class', observed=True):
            k = min(max(1, int(np.ceil(cc.FINAL_TEST_FRAC * len(cell)))),
                    len(cell))
            picked.extend(rng.choice(cell['group'].to_numpy(), size=k,
                                     replace=False).tolist())
        te_ids = set().union(*[ids_of[g] for g in picked])
        tr_ids = D.all_ids - te_ids
        for c in D.clusters:
            lte = D.cluster_ids[c] & te_ids
            if len(lte) < MIN_TEST_POLY or len(tr_ids) < MIN_TRAIN_POLY:
                continue
            rows.append({'seed': seed, CLUSTER_COL: c,
                         'region': D.region_of_cluster[c], 'arm': 'grouped',
                         **score(tr_ids, lte, seed,
                                 {'buffer_km': -1.0, 'arm': 'grouped',
                                  CLUSTER_COL: c,
                                  'region': D.region_of_cluster[c]})})
        print(f'    seed {seed:>4} done', flush=True)
    GROUPED = pd.DataFrame(rows)
    del rows

    if len(GROUPED):
        ung = (SWEEP[(SWEEP['arm'] == 'buffer') & (SWEEP['buffer_km'] == 0)]
               .groupby('region')[SHOW].mean())
        gr = GROUPED.groupby('region')[SHOW].mean()
        narrow = [c for c in ('poly_macro_f1', 'poly_sc_nsc_f1')
                  if c in ung.columns]
        print('\n  Polygon scores by region. Two metrics for width, all in')
        print('  the CSV.')
        print(pd.concat({'ungrouped d=0': ung[narrow], 'grouped': gr[narrow],
                         'difference': gr[narrow] - ung[narrow]},
                        axis=1).round(4).to_string())
        o = pd.DataFrame({'ungrouped d=0': ung.mean(), 'grouped': gr.mean()})
        o['difference'] = o['grouped'] - o['ungrouped d=0']
        print('\n  Overall')
        print(o.round(4).to_string())
        print('\n  Reading it. A grouped score that falls close to where the')
        print('  buffered curve settles says the d = 0 advantage was mostly')
        print('  duplicate polygons on both sides of the split. One that')
        print('  barely moves says the short-range signal is real.')
        cc.save_table(GROUPED, TAG, f'T10_{TAG}_grouped.csv')

banner('READING THIS')
print('  The distance effect column is the contribution. It is the decline')
print('  that survives after matching how much data was lost.')
print('  None of this identifies a CAUSE. It bounds the range.')

if SAVE_PREDICTIONS and PRED_ROWS:
    p = cc.save_table(pd.concat(PRED_ROWS, ignore_index=True), TAG,
                      f'P_{TAG}_predictions_by_buffer.csv')
    print(f'\n  per-polygon predictions by buffer -> {p}')
    print('  Script 7 reads this to cut any class pair by distance.')
    print('  buffer_km of -1 marks the grouped-split rows.')
print(f'  tables -> {cc.out_dir(TAG)}')
print(f'\n  Total runtime {(time.time() - t0) / 60:.1f} min')
