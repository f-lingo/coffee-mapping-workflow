"""THE MODEL ITSELF. The numbers scripts 5 to 8 go on to explain.

Everything after this is a diagnostic. This is the map.

FOUR THINGS, ALL FROM THE SAME FITS

  1. PIXEL LEVEL. Overall accuracy, kappa, per-class precision, recall and
     F1, and the confusion matrix. The number most papers report.

  2. POLYGON LEVEL. The same predictions voted to one label per polygon.
     A polygon is the unit a map user acts on, and voting is not cosmetic:
     the per-class change is reported so it can be read rather than assumed.

  3. HARD AGAINST SOFT VOTE. Majority of pixel labels against the mean of
     pixel probabilities. One fit, two ways of reading it.

  4. PER-REGION CLASSIFIERS. One model per cluster, trained and tested
     inside that cluster alone. What a regional mapping effort would get if
     it only ever used its own reference data, which is the realistic
     alternative to a national model.

READ THIS BEFORE QUOTING ANY NUMBER BELOW

The polygon accuracy here is the OPTIMISTIC one. It uses an ordinary
stratified holdout, so training polygons sit a median of 208 m from test
polygons and the score carries about 0.10 of spatial inflation. Script 6
measures that inflation and reports the honest figure two independent ways.

Quote this script for the confusion structure, the pixel against polygon
comparison, and the per-region spread. Quote script 6 for accuracy.

Optionally fits a final model on every polygon and pickles it, which is what
you would actually apply to imagery.

Run time about 10 minutes.
"""

import gc
import pickle
import time

import numpy as np
import pandas as pd

import coffee_common as cc
from coffee_common import (ID_COL, CLUSTER_COL, CLASS_NAMES, CLASS_SHORT,
                           ALL_CLASSES, SEEDS, FINAL_PIXEL_CAP,
                           TEST_PIXEL_CAP, TAG_DESIGNS, banner)

# ---- method settings -------------------------------------------------------
TAG = 'baseline'
N_SEEDS = 10
MIN_REGION_TRAIN = 30
MIN_REGION_TEST = 5
FIT_FINAL = True        # True to pickle a model fitted on every polygon
CM_SEED = 0              # index into SEEDS, for the printed confusion matrix

BASE_SEEDS = SEEDS[:N_SEEDS]
t0 = time.time()

banner('SCRIPT 4. THE BASELINE MODEL')
print(f'  {len(cc.FEATURE_COLUMNS)} features, {len(BASE_SEEDS)} seeds, '
      f'{cc.FINAL_TREES} trees, max depth {cc.RF_MAX_DEPTH}')
print(f'  class weight {cc.FINAL_WEIGHT}, '
      f'{FINAL_PIXEL_CAP} training pixels per polygon, test uncapped')
print(f'  {len(BASE_SEEDS) * (1 + cc.N_REGIONS)} forest fits')

D = cc.load_data()
SPLITS = D.build_splits(BASE_SEEDS)
banner('SPLITS')
cc.verify_splits(SPLITS, TAG_DESIGNS)


# =============================================================================
# THE POOLED MODEL, SCORED THREE WAYS FROM ONE FIT
# =============================================================================
banner('POOLED MODEL')
print('  Trained on every training polygon, tested on the held-out share.')

rows, cm_pixel, cm_poly = [], None, None
for i, seed in enumerate(BASE_SEEDS):
    tr_ids, te_ids = SPLITS[seed]
    tr = D.capped(tr_ids, seed, FINAL_PIXEL_CAP)
    te = D.capped(te_ids, seed, TEST_PIXEL_CAP)
    rf, sc = cc.fit_rf(tr, cc.FEATURE_COLUMNS, tr['class'].values, seed)
    yp, P = cc.predict_batched(rf, sc, te, cc.FEATURE_COLUMNS,
                               want_proba=True)
    yt = te['class'].values

    rows.append({'seed': seed, 'level': 'pixel', 'vote': '-',
                 'n_train_px': len(tr), 'n_test_px': len(te),
                 **cc.metrics(yt, yp)})
    for vote in ('hard', 'soft'):
        _, pyt, pyp = cc.polygon_labels(te, yp, P, vote)
        rows.append({'seed': seed, 'level': 'polygon', 'vote': vote,
                     'n_train_poly': len(tr_ids), 'n_test_poly': len(te_ids),
                     **cc.metrics(pyt, pyp)})
    if i == CM_SEED:
        cm_pixel = (yt.copy(), yp.copy())
        _, a, b = cc.polygon_labels(te, yp, P, 'hard')
        cm_poly = (a, b)
    del rf, sc, tr, te, P
    gc.collect()
    print(f'    seed {seed:>4} done', flush=True)

POOLED = pd.DataFrame(rows)
del rows

SHOW = ['oa', 'kappa', 'macro_f1'] + \
       [f'f1_{CLASS_NAMES[c]}' for c in ALL_CLASSES] + \
       ['coffee_mean_f1', 'sc_nsc_f1']
summary = (POOLED.assign(model=POOLED['level'] + ' ' + POOLED['vote'])
           .groupby('model')[SHOW].mean()
           .reindex(['pixel -', 'polygon hard', 'polygon soft']).round(4))
print('\n  Mean over seeds')
print(summary.to_string())

sd = (POOLED.assign(model=POOLED['level'] + ' ' + POOLED['vote'])
      .groupby('model')[['oa', 'macro_f1']].std(ddof=1)
      .reindex(['pixel -', 'polygon hard', 'polygon soft']).round(4))
print('\n  Seed to seed SD. Small, and that is the point. The variation that')
print('  matters is between REGIONS, which script 5 reports at about 0.043')
print('  for polygon macro F1 against 0.005 here.')
print(sd.to_string())

banner('WHAT VOTING TO THE POLYGON CHANGES, PER CLASS')
print('  Pixels are voted to one label per polygon. A polygon is the unit a')
print('  map user acts on, so this is the level to report, but the change is')
print('  not uniform across classes and is worth seeing.')
pxl = POOLED[POOLED['level'] == 'pixel'][SHOW].mean()
pol = POOLED[(POOLED['level'] == 'polygon') &
             (POOLED['vote'] == 'hard')][SHOW].mean()
delta = pd.DataFrame({'pixel': pxl, 'polygon': pol,
                      'change': pol - pxl}).round(4)
print(delta.to_string())
print('\n  A class that gains is one whose pixels are noisy but whose')
print('  polygons are mostly right. A class that LOSES is one where a few')
print('  confident wrong pixels outvote the rest.')

hard = POOLED[(POOLED['level'] == 'polygon') & (POOLED['vote'] == 'hard')]
soft = POOLED[(POOLED['level'] == 'polygon') & (POOLED['vote'] == 'soft')]
if len(hard) == len(soft):
    d = (soft.set_index('seed')['macro_f1'] -
         hard.set_index('seed')['macro_f1']).dropna()
    print(f'\n  Soft minus hard vote, paired by seed: {d.mean():+.4f} '
          f'(SD {d.std(ddof=1):.4f}), soft better in '
          f'{int((d > 0).sum())}/{len(d)} seeds')

cc.print_confusion(*cm_pixel, f'Pixel confusion, seed {BASE_SEEDS[CM_SEED]}')
cc.print_confusion(*cm_poly, f'Polygon confusion, seed {BASE_SEEDS[CM_SEED]}')


# =============================================================================
# PER-REGION CLASSIFIERS
# =============================================================================
banner('PER-REGION CLASSIFIERS')
print('  One model per cluster, trained AND tested inside that cluster.')
print('  What a regional effort would get using only its own reference data,')
print('  which is the realistic alternative to a national model.')

rows, skipped = [], []
for seed in BASE_SEEDS:
    tr_ids, te_ids = SPLITS[seed]
    for c in D.clusters:
        ltr, lte = D.cluster_ids[c] & tr_ids, D.cluster_ids[c] & te_ids
        if len(ltr) < MIN_REGION_TRAIN or len(lte) < MIN_REGION_TEST:
            skipped.append({'seed': seed, CLUSTER_COL: c,
                            'reason': f'{len(ltr)} train, {len(lte)} test'})
            continue
        if len({D.class_of[u] for u in ltr}) < 2:
            skipped.append({'seed': seed, CLUSTER_COL: c,
                            'reason': 'fewer than two classes'})
            continue
        tr = D.capped(ltr, seed, FINAL_PIXEL_CAP)
        te = D.capped(lte, seed, TEST_PIXEL_CAP)
        rf, sc = cc.fit_rf(tr, cc.FEATURE_COLUMNS, tr['class'].values, seed)
        yp, _ = cc.predict_batched(rf, sc, te, cc.FEATURE_COLUMNS)
        _, pyt, pyp = cc.polygon_labels(te, yp)
        rows.append({'seed': seed, CLUSTER_COL: c,
                     'region': D.region_of_cluster[c],
                     'n_train_poly': len(ltr), 'n_test_poly': len(lte),
                     'n_train_classes': int(len(np.unique(tr['class'].values))),
                     **cc.metrics(pyt, pyp)})
        del rf, sc, tr, te
        gc.collect()
    print(f'    seed {seed:>4} done', flush=True)

REGIONAL = pd.DataFrame(rows)
del rows
if skipped:
    print(f'\n  {len(skipped)} region cells skipped')
    print(pd.DataFrame(skipped)['reason'].value_counts().head().to_string())

if len(REGIONAL):
    RSHOW = ['n_train_poly', 'n_test_poly', 'oa', 'macro_f1'] + \
            [f'f1_{CLASS_NAMES[c]}' for c in ALL_CLASSES] + ['sc_nsc_f1']
    byreg = REGIONAL.groupby('region')[RSHOW].mean().round(4)
    print('\n  Polygon level, mean over seeds')
    print(byreg.to_string())
    print('\n  Spread across regions, which is the honest uncertainty')
    print(REGIONAL.groupby('region')[['oa', 'macro_f1', 'sc_nsc_f1']]
          .mean().agg(['mean', 'std', 'min', 'max']).round(4).to_string())
    _small = ', '.join(f'R{c} ({cc.REGION_NAME_HINT.get(c, "?")})'
                       for c in cc.SMALL_REGIONS)
    print(f'\n  {_small} carries too few coffee test polygons to support a')
    print('  per-region claim. Reported for completeness only.')

    nat = POOLED[(POOLED['level'] == 'polygon') &
                 (POOLED['vote'] == 'hard')]['macro_f1'].mean()
    reg = REGIONAL['macro_f1'].mean()
    print(f'\n  National model {nat:.4f} against per-region models '
          f'{reg:.4f}, difference {nat - reg:+.4f}')
    print('  Read this with the training sizes above. The per-region models')
    print('  train on a quarter of the polygons, so near-parity is itself')
    print('  informative. Script 5 takes that comparison apart properly with')
    print('  nested training sets and identical test polygons.')
    cc.save_table(REGIONAL, TAG, f'M2_{TAG}_per_region.csv')
    cc.save_table(byreg, TAG, f'M2b_{TAG}_per_region_mean.csv', index=True)


# =============================================================================
# FINAL FIT
# =============================================================================
if FIT_FINAL:
    banner('FITTING ON EVERY POLYGON')
    rows_all = D.capped(D.all_ids, BASE_SEEDS[0], FINAL_PIXEL_CAP)
    rf, sc = cc.fit_rf(rows_all, cc.FEATURE_COLUMNS,
                       rows_all['class'].values, BASE_SEEDS[0], quiet=False)
    bundle = {'model': rf, 'scaler': sc,
              'features': list(cc.FEATURE_COLUMNS),
              'class_names': CLASS_NAMES, 'raw_to_clean': cc.RAW_TO_CLEAN,
              'trained_on_polygons': len(D.all_ids),
              'pixel_cap': FINAL_PIXEL_CAP, 'seed': BASE_SEEDS[0]}
    p = cc.out_dir(TAG) / f'M3_{TAG}_model.pkl'
    with open(p, 'wb') as fh:
        pickle.dump(bundle, fh)
    with open(p, 'rb') as fh:
        assert pickle.load(fh)['features'] == list(cc.FEATURE_COLUMNS)
    print(f'  {len(rows_all):,} pixels from {len(D.all_ids)} polygons')
    print(f'  bundle -> {p}  (round trip checked)')
    print('  It carries the feature list and the class mapping, so applying')
    print('  it to imagery cannot silently use the wrong column order.')
    del rows_all, rf, sc
    gc.collect()
else:
    print('\n  Final fit skipped. Set FIT_FINAL = True to pickle a model.')


# =============================================================================
# WHAT TO QUOTE, AND WHAT NOT TO
# =============================================================================
banner('WHAT TO QUOTE FROM THIS SCRIPT')
print('  QUOTE  the confusion structure, which classes trade with which.')
print('  QUOTE  the pixel against polygon comparison.')
print('  QUOTE  the spread across per-region models.')
print()
print('  DO NOT quote the polygon accuracy above as the accuracy of the')
print('  map. It comes from an ordinary stratified holdout, so training')
print('  polygons sit a median of 208 m from test polygons and it carries')
print('  about 0.10 of spatial inflation. Script 6 measures that inflation')
print('  and reports the honest figure two independent ways. Quoting this')
print('  one is the exact mistake this project exists to document.')

cc.save_table(POOLED, TAG, f'M1_{TAG}_pooled.csv')
cc.save_table(summary, TAG, f'M1b_{TAG}_summary.csv', index=True)
cc.save_table(delta, TAG, f'M1c_{TAG}_pixel_vs_polygon.csv', index=True)
print(f'\n  tables -> {cc.out_dir(TAG)}')
print(f'  Total runtime {(time.time() - t0) / 60:.1f} min')
