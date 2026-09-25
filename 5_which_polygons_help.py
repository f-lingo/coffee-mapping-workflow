"""QUESTION 1. Does it matter where your reference polygons come from?

Four models trained on different polygons, scored on the SAME held-out
polygons, over 10 random splits. The training sets NEST on purpose, so each
comparison changes exactly one thing.

    T = the pooled training polygons for this seed
    E = the pooled test polygons
    C = the polygons in this cluster

    pooled     trains on T                 everywhere
    local      trains on T and C           inside the cluster only
    loro_out   trains on T not C           outside the cluster only
    ALL score on E and C, the identical held-out polygons of this cluster

Because T is exactly local plus loro_out, the rows decompose cleanly.

    pooled minus loro_out   what the nearby polygons added
    pooled minus local      what the distant polygons added
    local  minus loro_out   nearby against distant, same test set

A fourth model trains on every polygon outside the cluster, including ones
the pooled split held out elsewhere. It is fitted once and scored twice.

    loro_full  on EVERY polygon in the cluster. The classic transfer number.
    loro_all   the same fit on the held-out share only, so it joins the
               paired table. It trains on roughly as many polygons as pooled,
               which the other two do not, so pooled minus loro_all is the
               size-matched contrast.

WHAT THIS WRITES, AND WHO READS IT
    splits/S_*_splits.json          scripts 6, 7 and 8 verify against this
    P_*_polygon_predictions.csv     script 7 reads this and fits nothing
    T*_*.csv                        the tables behind figure F1
    T6c_*_pixels_by_cluster         pooled and loro_all fits, pixel level,
                                    per cluster on the held-out share, for
                                    tables T1 and T1a in script 9
    T6d_*_pooled_pixel_confusion    pooled pixel confusion counts, every
                                    seed summed, for figure F18

Run time about 15 minutes at 130 forest fits.
"""

import gc
import time

import numpy as np
import pandas as pd

import coffee_common as cc
from coffee_common import (ID_COL, CLUSTER_COL, CLASS_NAMES, ALL_CLASSES,
                           SEEDS, FINAL_PIXEL_CAP, TEST_PIXEL_CAP,
                           TAG_DESIGNS, banner)

# ---- method settings. Machine settings live in gee_local_config.py ---------
TAG = TAG_DESIGNS
POLY_VOTE = 'hard'          # 'hard' | 'soft'
RUN_LOCAL = True            # local and loro_out, the decomposition
RUN_LORO_FULL = True        # the transfer number, and loro_all with it
MIN_LOCAL_TRAIN_POLY = 30
MIN_LOCAL_TEST_POLY = 5
CM_SEEDS = 'first'          # 'first' | 'all', for the confusion matrix
SAVE_PREDICTIONS = True

DESIGN_ORDER = ['pooled', 'local', 'loro_out', 'loro_all']
t0 = time.time()

banner('SCRIPT 5. WHICH REFERENCE POLYGONS HELP')
print(f'  {len(cc.FEATURE_COLUMNS)} features, {len(SEEDS)} seeds, '
      f'{cc.FINAL_TREES} trees, polygon vote {POLY_VOTE}')
_n = len(SEEDS) * (1 + (2 * cc.N_REGIONS if RUN_LOCAL else 0))
_n += len(SEEDS) * cc.N_REGIONS if RUN_LORO_FULL else 0
print(f'  {_n} forest fits')

D = cc.load_data()
SPLITS = D.build_splits(SEEDS)

banner('SPLITS')
print(f'  {cc.save_splits(SPLITS, TAG)}')
rows = []
for seed, (tr, te) in SPLITS.items():
    for role, ids in (('train', tr), ('test', te)):
        for u in sorted(ids):
            rows.append({'seed': seed, ID_COL: u, 'role': role,
                         CLUSTER_COL: D.cluster_of[u],
                         'region': D.region_of[u],
                         'class': D.class_of[u],
                         'class_name': CLASS_NAMES[D.class_of[u]],
                         'n_pixels': D.npix_of[u]})
cc.save_table(pd.DataFrame(rows), TAG, f'S_{TAG}_splits_long.csv')
del rows

loc = []
for seed, (tr, te) in SPLITS.items():
    for c in D.clusters:
        ltr, lte = D.cluster_ids[c] & tr, D.cluster_ids[c] & te
        te_cls = {D.class_of[u] for u in lte}
        tr_cls = {D.class_of[u] for u in ltr}
        loc.append({'seed': seed, CLUSTER_COL: c,
                    'region': D.region_of_cluster[c],
                    'n_local_train': len(ltr), 'n_local_test': len(lte),
                    'n_outside_train': len(tr) - len(ltr),
                    'missing_in_train': ','.join(
                        CLASS_NAMES[k] for k in sorted(te_cls - tr_cls))})
LOCAL_SIZES = pd.DataFrame(loc)
del loc
print('\n  Training polygons available to each design, mean over seeds')
print(LOCAL_SIZES.groupby([CLUSTER_COL, 'region'])[
    ['n_local_train', 'n_outside_train', 'n_local_test']]
    .mean().round(1).to_string())
_bad = LOCAL_SIZES[LOCAL_SIZES['missing_in_train'] != '']
if len(_bad):
    print(f'\n  WARNING. {len(_bad)} of {len(LOCAL_SIZES)} cells have a class')
    print('  in the test set that is ABSENT from LOCAL training. Those')
    print('  classes cannot be predicted at all, which caps that cell before')
    print('  the model does anything.')
cc.save_table(LOCAL_SIZES, TAG, f'S_{TAG}_local_sizes.csv')

PRED_ROWS = []


def score(tr_ids, te_ids, seed, design, tag):
    """One fit. Returns the record and the per-polygon frame."""
    tr = D.capped(tr_ids, seed, FINAL_PIXEL_CAP)
    te = D.capped(te_ids, seed, TEST_PIXEL_CAP)
    ytr = tr['class'].values
    rf, sc = cc.fit_rf(tr, cc.FEATURE_COLUMNS, ytr, seed)
    yp, P = cc.predict_batched(rf, sc, te, cc.FEATURE_COLUMNS,
                               want_proba=(POLY_VOTE == 'soft'))
    rec = cc.metrics(te['class'].values, yp)
    rec.update({f'poly_{a}': b for a, b in
                cc.polygon_metrics(te, yp, P, POLY_VOTE).items()})
    pid, pyt, pyp = cc.polygon_labels(te, yp, P, POLY_VOTE)
    polys = pd.DataFrame({ID_COL: pid, 'y_true': pyt, 'y_pred': pyp})
    rec.update({'seed': seed, 'design': design,
                'n_train_poly': len(tr_ids), 'n_test_poly': len(te_ids),
                'n_train_px': len(tr), 'n_test_px': len(te),
                'n_train_classes': int(len(np.unique(ytr)))})
    if SAVE_PREDICTIONS:
        p = polys.copy()
        p['design'], p['seed'] = design, seed
        for k, v in tag.items():
            p[k] = v
        PRED_ROWS.append(p)
    yt_out = te['class'].values.copy()
    del rf, sc, tr, te, P
    gc.collect()
    return rec, yt_out, yp, polys


# =============================================================================
# POOLED, LOCAL AND LORO_OUT. The nested decomposition.
# =============================================================================
banner('POOLED, LOCAL AND LORO_OUT')
print('  Training sets nest exactly. pooled = local union loro_out.')

pooled_rows, nested_rows, skipped = [], [], []
# The pooled and loro_all fits scored cluster by cluster at PIXEL level, so
# script 9 can set them beside local and loro_out on the same test pixels.
# T6 already holds the polygon level. Adds no fit, changes no other number.
DESIGN_PX_BY_CLUSTER = []
# Pooled pixel confusion summed over EVERY seed, for script 9's aggregated
# confusion figure. The printed matrix above uses one seed only.
from sklearn.metrics import confusion_matrix as _cm
POOLED_PX_CM = np.zeros((cc.K, cc.K), dtype=np.int64)
p_true, p_pred = [], []
POOLED_POLYS, LOCAL_POLYS, OUT_POLYS, ALL_POLYS = {}, {}, {}, {}

for seed in SEEDS:
    tr_ids, te_ids = SPLITS[seed]
    rec, yt, yp, polys = score(tr_ids, te_ids, seed, 'pooled',
                               {CLUSTER_COL: -1})
    pooled_rows.append(rec)
    POOLED_POLYS[seed] = polys
    te_px_ids = D.capped(te_ids, seed, TEST_PIXEL_CAP)[ID_COL].to_numpy()
    for c in D.clusters:
        in_c = np.isin(te_px_ids, list(D.cluster_ids[c] & te_ids))
        if in_c.any():
            DESIGN_PX_BY_CLUSTER.append({
                'seed': seed, CLUSTER_COL: c,
                'region': D.region_of_cluster[c], 'design': 'pooled',
                'n_test_px': int(in_c.sum()), **cc.metrics(yt[in_c], yp[in_c])})
    del te_px_ids
    POOLED_PX_CM += _cm(yt, yp, labels=cc.ALL_CLASSES)
    if CM_SEEDS == 'all' or seed == SEEDS[0]:
        p_true.extend(yt)
        p_pred.extend(yp)

    if RUN_LOCAL:
        for c in D.clusters:
            ltr = D.cluster_ids[c] & tr_ids
            lte = D.cluster_ids[c] & te_ids
            otr = tr_ids - D.cluster_ids[c]
            why = None
            if len(ltr) < MIN_LOCAL_TRAIN_POLY:
                why = f'{len(ltr)} local training polygons'
            elif len(lte) < MIN_LOCAL_TEST_POLY:
                why = f'{len(lte)} test polygons'
            elif len({D.class_of[u] for u in ltr}) < 2:
                why = 'fewer than two classes in local training'
            if why:
                skipped.append({'seed': seed, CLUSTER_COL: c, 'reason': why})
                continue
            for name, trset, store in (('local', ltr, LOCAL_POLYS),
                                       ('loro_out', otr, OUT_POLYS)):
                r, _, _, pl = score(trset, lte, seed, name, {CLUSTER_COL: c})
                r.update({CLUSTER_COL: c, 'region': D.region_of_cluster[c]})
                nested_rows.append(r)
                store[(seed, c)] = pl
    print(f'    seed {seed:>4} done', flush=True)

POOLED = pd.DataFrame(pooled_rows)
NESTED = pd.DataFrame(nested_rows)
cc.print_confusion(p_true, p_pred, 'Pooled confusion, pixels, '
                   + ('all seeds' if CM_SEEDS == 'all' else f'seed {SEEDS[0]}'))
if skipped:
    print(f'\n  {len(skipped)} cells skipped for lack of data')
    print(pd.DataFrame(skipped)['reason'].value_counts().to_string())


# =============================================================================
# LORO_FULL, AND LORO_ALL FROM THE SAME FIT
# =============================================================================
LORO = pd.DataFrame()
if RUN_LORO_FULL:
    banner('LORO_FULL, THE TRANSFER NUMBER')
    print('  Trains on EVERY polygon outside the cluster and tests EVERY')
    print('  polygon inside it. Its test set is larger than everyone else s,')
    print('  so it is reported on its own. The same fit is scored a second')
    print('  time on the held-out share alone, as loro_all, which does join')
    print('  the paired table. No extra fit and no leakage, because every')
    print('  training polygon lies outside the cluster being tested.')
    loro_rows, l_true, l_pred = [], [], []
    for c in D.clusters:
        te_ids, tr_ids = D.cluster_ids[c], D.all_ids - D.cluster_ids[c]
        region = D.region_of_cluster[c]
        if len(tr_ids) < 50 or len(te_ids) < 10:
            print(f'    {region}, too few polygons, skipped')
            continue
        for seed in SEEDS:
            rec, yt, yp, lp = score(tr_ids, te_ids, seed, 'loro_full',
                                    {CLUSTER_COL: c})
            rec.update({'held_out': region, CLUSTER_COL: c, 'region': region})
            loro_rows.append(rec)
            lte = D.cluster_ids[c] & SPLITS[seed][1]
            if len(lte) >= MIN_LOCAL_TEST_POLY:
                sub = lp[lp[ID_COL].isin(lte)]
                if len(sub) == len(lte):
                    ALL_POLYS[(seed, c)] = sub.copy()
                    # The same fit at PIXEL level on the held-out share,
                    # so script 9 can pair loro_all with pooled and local.
                    px_ids = D.capped(te_ids, seed,
                                      TEST_PIXEL_CAP)[ID_COL].to_numpy()
                    in_l = np.isin(px_ids, list(lte))
                    DESIGN_PX_BY_CLUSTER.append({
                        'seed': seed, CLUSTER_COL: c, 'region': region,
                        'design': 'loro_all', 'n_test_px': int(in_l.sum()),
                        **cc.metrics(yt[in_l], yp[in_l])})
                    del px_ids
            if CM_SEEDS == 'all' or seed == SEEDS[0]:
                l_true.extend(yt)
                l_pred.extend(yp)
        m = [r for r in loro_rows if r.get(CLUSTER_COL) == c]
        print(f'    {region}  polygon OA '
              f'{np.mean([r["poly_oa"] for r in m]):.4f}  macro F1 '
              f'{np.mean([r["poly_macro_f1"] for r in m]):.4f}')
    LORO = pd.DataFrame(loro_rows)
    cc.print_confusion(l_true, l_pred, 'LORO_FULL confusion, pixels, '
                       'all regions pooled')


# =============================================================================
# THE PAIRED DECOMPOSITION. Identical test polygons.
# =============================================================================
PAIRED = pd.DataFrame()
if RUN_LOCAL and LOCAL_POLYS:
    banner('PAIRED DECOMPOSITION, IDENTICAL TEST POLYGONS')
    rows = []
    for (seed, c), lp in LOCAL_POLYS.items():
        op, pp = OUT_POLYS.get((seed, c)), POOLED_POLYS.get(seed)
        if op is None or pp is None:
            continue
        ids = set(lp[ID_COL])
        pp = pp[pp[ID_COL].isin(ids)]
        if not (len(pp) == len(op) == len(lp)):
            continue
        tr_ids = SPLITS[seed][0]
        n_tr = {'pooled': len(tr_ids),
                'local': len(D.cluster_ids[c] & tr_ids),
                'loro_out': len(tr_ids - D.cluster_ids[c]),
                'loro_all': len(D.all_ids - D.cluster_ids[c])}
        base = {'seed': seed, CLUSTER_COL: c,
                'region': D.region_of_cluster[c], 'n_test_poly': len(lp)}
        quartet = [('pooled', pp), ('local', lp), ('loro_out', op)]
        ap = ALL_POLYS.get((seed, c))
        if ap is not None and len(ap) == len(lp):
            quartet.append(('loro_all', ap))
        for name, frame in quartet:
            frame = frame.sort_values(ID_COL)
            rows.append({**base, 'design': name, 'n_train_poly': n_tr[name],
                         **cc.metrics(frame['y_true'].to_numpy(),
                                      frame['y_pred'].to_numpy())})
    PAIRED = pd.DataFrame(rows)
    del rows

if len(PAIRED):
    order = [d for d in DESIGN_ORDER if d in set(PAIRED['design'])]
    show = ['n_train_poly', 'oa', 'macro_f1', f'f1_{CLASS_NAMES[0]}',
            f'f1_{CLASS_NAMES[1]}', 'coffee_mean_f1']
    print(f'\n  {int(len(PAIRED) / len(order))} paired cells, '
          f'{PAIRED["seed"].nunique()} seeds, {len(order)} designs')
    print('  Every design is scored on the SAME held-out polygons of the')
    print('  same cluster, with the same pixels.')
    print('\n  Polygon-level scores by design, mean over seeds')
    print(PAIRED.groupby('design')[show].mean().reindex(order)
          .round(4).to_string())
    print('\n  f1 columns are one-vs-rest against all five classes.')
    print('  n_train_poly is the mean training size. Read it before any')
    print('  difference below, because two designs of unequal size differ in')
    print('  how much data they saw as well as where it came from.')

    if 'loro_all' in order:
        sz = (PAIRED.pivot_table(index=[CLUSTER_COL, 'region'],
                                 columns='design', values='n_train_poly')
              .reindex(columns=order))
        sz['pooled minus loro_all'] = sz['pooled'] - sz['loro_all']
        print('\n  Training polygons per cluster. pooled and loro_all are the')
        print('  pair to watch, they are the closest in size.')
        print(sz.round(1).to_string())

    bc = (PAIRED.pivot_table(index=[CLUSTER_COL, 'region'], columns='design',
                             values='macro_f1').reindex(columns=order))
    bc['local adds'] = bc['pooled'] - bc['loro_out']
    bc['outside adds'] = bc['pooled'] - bc['local']
    if 'loro_all' in bc.columns:
        bc['local adds, size matched'] = bc['pooled'] - bc['loro_all']
        bc['more distant adds'] = bc['loro_all'] - bc['loro_out']
    print('\n  By cluster, polygon macro F1')
    print(bc.round(4).to_string())

    print('\n  Paired differences, per cell')
    for metric in ['macro_f1', f'f1_{CLASS_NAMES[0]}', f'f1_{CLASS_NAMES[1]}']:
        w = PAIRED.pivot_table(index=['seed', CLUSTER_COL], columns='design',
                               values=metric)
        print(f'\n    {metric}')
        for a, b, label in (
                ('pooled', 'loro_out', 'what the nearby data adds'),
                ('pooled', 'local', 'what the distant data adds'),
                ('local', 'loro_out', 'nearby against distant'),
                ('pooled', 'loro_all', 'nearby data at a matched size'),
                ('loro_all', 'loro_out', 'more distant data, none nearby')):
            if a in w.columns and b in w.columns:
                d = (w[a] - w[b]).dropna()
                if len(d):
                    print(f'      {a:<9} minus {b:<9} {d.mean():+.4f} '
                          f'(SD {d.std(ddof=1):.4f}), {a} better in '
                          f'{int((d > 0).sum())}/{len(d)}, {label}')
    print('\n  Reading it. These differences hold for THIS feature set, these')
    print('  settings and these splits. A difference near zero with a win')
    print('  count near half the cells is noise, whatever the mean says.')
    print('  Note also that local trains on far fewer polygons, so parity is')
    print('  already informative about the distant data.')
    cc.save_table(PAIRED, TAG, f'T6_{TAG}_paired.csv')
    cc.save_table(bc, TAG, f'T6b_{TAG}_paired_by_cluster.csv', index=True)


# =============================================================================
# COVERAGE, VARIATION, AND WHAT MAY BE COMPARED WITH WHAT
# =============================================================================
ALL = pd.concat([f for f in (POOLED, NESTED, LORO) if len(f)],
                ignore_index=True)
banner('DESIGN COVERAGE')
scope = {'pooled': 'all clusters, held-out share',
         'local': 'one cluster, held-out share',
         'loro_out': 'one cluster, held-out share',
         'loro_full': 'one cluster, EVERY polygon'}
cov = (ALL.groupby('design')
       .agg(seeds_used=('seed', 'nunique'), cells=('seed', 'size'),
            mean_train_poly=('n_train_poly', 'mean'),
            mean_test_poly=('n_test_poly', 'mean')).round(1))
cov['test_scope'] = [scope.get(i, '') for i in cov.index]
print(cov.to_string())
short = [d for d, n in cov['seeds_used'].items() if n != len(SEEDS)]
print(f'\n  {"WARNING. " + str(short) + " missed seeds." if short else "All designs ran every one of the " + str(len(SEEDS)) + " seeds."}')
print('\n  COMPARABLE. The paired decomposition. Same polygons, nested')
print('    training sets, so each difference is attributable to one thing.')
print('  COMPARABLE. Any one design against itself across regions or seeds.')
print('  NOT COMPARABLE. The per-design means. Read test_scope.')
print('  loro_all is not a separate fit. It is loro_full scored on the')
print('    held-out share, so it appears in the paired table, not here.')

if len(LORO):
    banner('REGION TO REGION VARIATION, THE UNCERTAINTY THAT MATTERS')
    print('  A seed interval says how precisely the average is pinned down,')
    print('  not how well the model does in a region nobody surveyed. For')
    print('  that the spread ACROSS REGIONS is honest, and n is 4.')
    KEY = ['oa', 'macro_f1', 'poly_oa', 'poly_macro_f1', 'coffee_mean_f1']
    byreg = LORO.groupby('held_out')[KEY].mean()
    byreg.loc['mean'] = byreg.mean()
    byreg.loc['sd'] = LORO.groupby('held_out')[KEY].mean().std(ddof=1)
    print('\n' + byreg.round(4).to_string())
    print('\n  Seed to seed, for reference')
    for name, frame in (('pooled', POOLED), ('loro_full', LORO)):
        ps = frame.groupby('seed')[KEY].mean()
        print(f'\n  {name.upper()}')
        print(pd.DataFrame({'mean': ps.mean(), 'sd': ps.std(ddof=1),
                            'min': ps.min(), 'max': ps.max()})
              .round(4).to_string())

    lf1 = [f'poly_f1_{CLASS_NAMES[c]}' for c in ALL_CLASSES
           if f'poly_f1_{CLASS_NAMES[c]}' in LORO.columns]
    lt = LORO.groupby('held_out')[lf1].mean().T
    lt['mean'] = lt.mean(axis=1)
    lt = lt.rename(index=lambda s: s.replace('poly_f1_', ''))
    banner('LORO_FULL PER-CLASS F1 BY HELD-OUT REGION, POLYGON LEVEL')
    print(lt.round(3).to_string())

cc.save_table(ALL, TAG, f'T5_{TAG}_evaluation.csv')
cc.save_table(pd.DataFrame(POOLED_PX_CM, index=cc.ALL_CLASSES,
                           columns=cc.ALL_CLASSES).rename_axis('true'), TAG,
              f'T6d_{TAG}_pooled_pixel_confusion.csv', index=True)
if DESIGN_PX_BY_CLUSTER:
    cc.save_table(pd.DataFrame(DESIGN_PX_BY_CLUSTER), TAG,
                  f'T6c_{TAG}_pixels_by_cluster.csv')
cc.save_table(cov, TAG, f'T7_{TAG}_coverage.csv', index=True)
if SAVE_PREDICTIONS and PRED_ROWS:
    p = cc.save_table(pd.concat(PRED_ROWS, ignore_index=True), TAG,
                      f'P_{TAG}_polygon_predictions.csv')
    print(f'\n  per-polygon predictions -> {p}')
    print('  Script 7 reads this and fits nothing.')
print(f'  tables -> {cc.out_dir(TAG)}')
print(f'\n  Total runtime {(time.time() - t0) / 60:.1f} min')
