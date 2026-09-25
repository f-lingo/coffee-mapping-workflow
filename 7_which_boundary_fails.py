"""QUESTION 3a. Which boundary fails, in which direction, and does distance
matter to it?

FITS NOTHING. Reads the per-polygon predictions scripts 3 and 4 already
saved, so it runs in seconds and is safe to rerun while drafting.

The macro F1 averages five classes and hides the fact that two of them fail
for different reasons. This takes the errors apart.

    A. SHADE COFFEE AGAINST FOREST
       A shade plot called forest and a forest called a shade plot are
       different mistakes with different consequences, so they are counted
       separately rather than netted off.

    B. NON-SHADE COFFEE
       Where its errors go, and whether the model is MISSING it or
       OVER-CALLING it. Those hurt a map user in opposite ways.

    C. CLASS SCHEMES
       The same predictions scored at 5, 4, 3 and 2 classes, so the decision
       about what the map can honestly deliver is made on numbers.

Everything is polygon level, because a polygon is the unit a map user acts
on. loro_all is reconstructed from loro_full restricted to the held-out
share, using the splits file.
"""

import json

import numpy as np
import pandas as pd

from sklearn.metrics import f1_score, accuracy_score, cohen_kappa_score

import coffee_common as cc
from coffee_common import (ID_COL, CLUSTER_COL, CLASS_NAMES, CLASS_SHORT,
                           ALL_CLASSES, NSC, SHADE_COFFEE, FOREST, OPEN,
                           URBAN, TAG_DESIGNS, TAG_DISTANCE, banner)

TAG = TAG_DESIGNS
DESIGN_ORDER = ['pooled', 'local', 'loro_out', 'loro_all', 'loro_full']
HAS_LOCAL = {'pooled': True, 'local': True, 'loro_out': False,
             'loro_all': False, 'loro_full': False}
SAVE = True

banner('SCRIPT 5. WHICH BOUNDARY FAILS')

pred_path = cc.out_dir(TAG) / f'P_{TAG}_polygon_predictions.csv'
if not pred_path.exists():
    raise FileNotFoundError(
        f'No predictions at {pred_path}. Run script 5 with SAVE_PREDICTIONS '
        'on first.')
P = pd.read_csv(pred_path)
print(f'  {len(P):,} per-polygon predictions')
print(f'  designs {sorted(P["design"].unique())}')

sp = cc.splits_path(TAG)
if sp.exists() and 'loro_full' in set(P['design']):
    disk = json.loads(sp.read_text())
    keep = []
    for seed, folds in disk.items():
        te = set()
        for f, d in folds.items():
            te |= set(d['test'])
        sub = P[(P['design'] == 'loro_full') & (P['seed'] == int(seed))]
        keep.append(sub[sub[ID_COL].isin(te)])
    if keep:
        LA = pd.concat(keep, ignore_index=True)
        LA['design'] = 'loro_all'
        P = pd.concat([P, LA], ignore_index=True)
        print(f'  reconstructed loro_all, {len(LA):,} rows, from loro_full')

P['region'] = P[CLUSTER_COL].map(
    lambda i: f'R{i} ({cc.REGION_NAME_HINT.get(i, "?")})' if i >= 0 else 'all')
DESIGNS = [d for d in DESIGN_ORDER if d in set(P['design'])]


def cm(frame):
    return pd.crosstab(frame['y_true'], frame['y_pred']).reindex(
        index=ALL_CLASSES, columns=ALL_CLASSES, fill_value=0)


# =============================================================================
# A. SHADE COFFEE AGAINST FOREST
# =============================================================================
banner('A. SHADE COFFEE AGAINST FOREST')
print('  A shade plot called forest, or a forest called a shade plot, are')
print('  different mistakes with different consequences.')

rows = []
for d in DESIGNS:
    f = P[P['design'] == d]
    M = cm(f)
    n_sc, n_fo = int(M.loc[SHADE_COFFEE].sum()), int(M.loc[FOREST].sum())
    sc_fo, fo_sc = int(M.loc[SHADE_COFFEE, FOREST]), int(M.loc[FOREST, SHADE_COFFEE])
    sc_err = n_sc - int(M.loc[SHADE_COFFEE, SHADE_COFFEE])
    fo_err = n_fo - int(M.loc[FOREST, FOREST])
    sub = f[f['y_true'].isin([SHADE_COFFEE, FOREST])]
    both = sub[sub['y_pred'].isin([SHADE_COFFEE, FOREST])]
    rows.append({
        'design': d, 'local_data': 'yes' if HAS_LOCAL.get(d) else 'no',
        'n_shade': n_sc, 'n_forest': n_fo,
        'shade_called_forest': sc_fo, 'forest_called_shade': fo_sc,
        'pct_shade_lost_to_forest': round(100 * sc_fo / max(n_sc, 1), 1),
        'pct_forest_lost_to_shade': round(100 * fo_sc / max(n_fo, 1), 1),
        'pct_shade_errors_that_are_forest':
            round(100 * sc_fo / sc_err, 1) if sc_err else np.nan,
        'pct_forest_errors_that_are_shade':
            round(100 * fo_sc / fo_err, 1) if fo_err else np.nan,
        'sc_vs_forest_f1': round(f1_score(
            both['y_true'] == SHADE_COFFEE, both['y_pred'] == SHADE_COFFEE,
            zero_division=0), 4) if len(both) else np.nan,
        'leaves_the_pair_pct': round(100 * (1 - len(both) / max(len(sub), 1)), 1),
    })
A1 = pd.DataFrame(rows).set_index('design').reindex(DESIGNS)
del rows
print('\n  Polygon counts and the exchange between the two classes')
print(A1[['local_data', 'n_shade', 'n_forest', 'shade_called_forest',
          'forest_called_shade', 'pct_shade_lost_to_forest',
          'pct_forest_lost_to_shade']].to_string())
print('\n  How much of each class s error is the other one')
print(A1[['pct_shade_errors_that_are_forest',
          'pct_forest_errors_that_are_shade', 'sc_vs_forest_f1',
          'leaves_the_pair_pct']].to_string())
print('\n  sc_vs_forest_f1 is scored only on polygons that truly are one of')
print('  the two AND were called one of the two. Read it with')
print('  leaves_the_pair_pct, which is how many escaped to a third class.')
print('  A high F1 with a high escape rate is not a good result.')

banner('  The same thing, as confusion matrices', '-')
for d in DESIGNS:
    M = cm(P[P['design'] == d]).loc[[SHADE_COFFEE, FOREST], ALL_CLASSES]
    M.index = [CLASS_SHORT[i] + ' (truth)' for i in M.index]
    M.columns = [CLASS_SHORT[c] for c in M.columns]
    print(f'\n  {d}, local training data '
          f'{"yes" if HAS_LOCAL.get(d) else "no"}')
    print(M.to_string())

banner('  By region, shade against forest', '-')
reg = []
for d in DESIGNS:
    for r, f in P[P['design'] == d].groupby('region'):
        if r == 'all':
            continue
        M = cm(f)
        reg.append({'design': d, 'region': r,
                    'n_shade': int(M.loc[SHADE_COFFEE].sum()),
                    'n_forest': int(M.loc[FOREST].sum()),
                    'shade_to_forest': int(M.loc[SHADE_COFFEE, FOREST]),
                    'forest_to_shade': int(M.loc[FOREST, SHADE_COFFEE])})
A2 = pd.DataFrame(reg)
del reg
if len(A2):
    print(A2.pivot_table(index='region', columns='design',
                         values=['shade_to_forest', 'forest_to_shade'],
                         aggfunc='sum').to_string())
    print('\n  A direction that FLIPS between regions means no single global')
    print('  correction can fix it.')
    _small = ', '.join(f'R{c} ({cc.REGION_NAME_HINT.get(c, "?")})'
                       for c in cc.SMALL_REGIONS)
    print(f'  {_small} is too small for a per-region claim.')


# =============================================================================
# A3. THE SAME EXCHANGE, BY DISTANCE
# =============================================================================
buf_path = cc.out_dir(TAG_DISTANCE) / \
    f'P_{TAG_DISTANCE}_predictions_by_buffer.csv'
A3 = pd.DataFrame()
if buf_path.exists():
    BUF = pd.read_csv(buf_path)
    banner('A3. THE EXCHANGE AS A FUNCTION OF DISTANCE')
    print('  Sliced by how far the training polygons were pushed away. The')
    print('  random arm removed the same number with the same class mix from')
    print('  anywhere, so the gap between arms is the part caused by distance.')
    rows = []
    for (d, arm), g in BUF.groupby(['buffer_km', 'arm']):
        M = cm(g)
        n_sc, n_fo = int(M.loc[SHADE_COFFEE].sum()), int(M.loc[FOREST].sum())
        n_ns = int(M.loc[NSC].sum())
        rows.append({
            'buffer_km': d, 'arm': arm,
            'forest_called_shade': int(M.loc[FOREST, SHADE_COFFEE]),
            'pct_forest_lost': round(
                100 * M.loc[FOREST, SHADE_COFFEE] / max(n_fo, 1), 1),
            'shade_called_forest': int(M.loc[SHADE_COFFEE, FOREST]),
            'pct_shade_lost': round(
                100 * M.loc[SHADE_COFFEE, FOREST] / max(n_sc, 1), 1),
            'nsc_called_open': int(M.loc[NSC, OPEN]),
            'pct_nsc_lost_to_open': round(
                100 * M.loc[NSC, OPEN] / max(n_ns, 1), 1)})
    A3 = pd.DataFrame(rows).sort_values(['buffer_km', 'arm'])
    del rows
    grp, A3 = A3[A3['buffer_km'] < 0], A3[A3['buffer_km'] >= 0]
    for col, label in (
            ('pct_shade_lost', 'shade coffee polygons called forest, pct'),
            ('pct_forest_lost', 'forest polygons called shade coffee, pct'),
            ('pct_nsc_lost_to_open', 'sun coffee polygons called Open, pct')):
        piv = A3.pivot_table(index='buffer_km', columns='arm', values=col)
        if {'buffer', 'random'} <= set(piv.columns):
            # at d = 0 nothing was removed, so both arms are one model and
            # script 6 stored one set of predictions. Copy it across so the
            # distance effect starts from a true zero.
            if 0.0 in piv.index and pd.isna(piv.loc[0.0, 'random']):
                piv.loc[0.0, 'random'] = piv.loc[0.0, 'buffer']
            piv['distance effect'] = piv['buffer'] - piv['random']
        print(f'\n  {label}')
        print(piv.round(2).to_string())
    print('\n  A distance effect that GROWS means the error is caused by')
    print('  losing NEARBY training data rather than by losing data.')
    if len(grp):
        print('\n  Grouped split for reference. Its test set differs.')
        print(grp.drop(columns=['buffer_km']).round(2).to_string(index=False))
    if SAVE:
        cc.save_table(A3, TAG, f'B5_{TAG}_exchange_by_dist.csv')
else:
    banner('A3. THE EXCHANGE BY DISTANCE, NOT AVAILABLE')
    print(f'  No file at {buf_path}')
    print('  Run script 6 with SAVE_PREDICTIONS on. Without it the confusion')
    print('  cannot be cut by buffer, and the design contrasts above are the')
    print('  only stand-in.')


# =============================================================================
# B. NON-SHADE COFFEE
# =============================================================================
banner('B. NON-SHADE COFFEE')
print('  A sun coffee plot called Open is a plot that reads as cleared')
print('  ground, the error with the sharpest consequence in the matrix.')
rows = []
for d in DESIGNS:
    f = P[P['design'] == d]
    M = cm(f)
    n, hit, called = (int(M.loc[NSC].sum()), int(M.loc[NSC, NSC]),
                      int(M[NSC].sum()))
    r = {'design': d, 'n_polygons': n,
         'recall': round(hit / max(n, 1), 4),
         'precision': round(hit / max(called, 1), 4),
         'f1': round(f1_score(f['y_true'] == NSC, f['y_pred'] == NSC,
                              zero_division=0), 4)}
    for c in ALL_CLASSES:
        if c != NSC:
            r[f'lost_to_{CLASS_SHORT[c]}'] = int(M.loc[NSC, c])
    r['false_pos_from_Open'] = int(M.loc[OPEN, NSC])
    r['false_pos_from_Shade'] = int(M.loc[SHADE_COFFEE, NSC])
    rows.append(r)
B1 = pd.DataFrame(rows).set_index('design').reindex(DESIGNS)
del rows
print('\n  Is the model missing it, or over-calling it')
print(B1[['n_polygons', 'recall', 'precision', 'f1']].to_string())
print('\n  Where the true sun coffee polygons go when they are missed')
print(B1[[c for c in B1.columns if c.startswith('lost_to_')]].to_string())
print('\n  And what gets wrongly called sun coffee')
print(B1[['false_pos_from_Open', 'false_pos_from_Shade']].to_string())
print('\n  Recall below precision means the model is MISSING real sun coffee,')
print('  so a map user under-counts. The reverse means over-calling.')


# =============================================================================
# C. CLASS SCHEMES
# =============================================================================
banner('C. THE SAME PREDICTIONS AT COARSER CLASS SCHEMES')
print('  Every row is the SAME model and the SAME polygons. Only the labels')
print('  are collapsed afterwards.')
SCHEMES = {
    '5 class, as run': {c: c for c in ALL_CLASSES},
    '4 class, coffee merged': {NSC: 0, SHADE_COFFEE: 0, FOREST: 1, OPEN: 2,
                               URBAN: 3},
    '3 class, coffee / forest / other': {NSC: 0, SHADE_COFFEE: 0, FOREST: 1,
                                         OPEN: 2, URBAN: 2},
    '2 class, coffee / not': {NSC: 0, SHADE_COFFEE: 0, FOREST: 1, OPEN: 1,
                              URBAN: 1},
}
rows = []
for name, mapping in SCHEMES.items():
    for d in DESIGNS:
        f = P[P['design'] == d]
        yt, yp = (f['y_true'].map(mapping).to_numpy(),
                  f['y_pred'].map(mapping).to_numpy())
        labs = sorted(set(mapping.values()))
        f1s = [f1_score(yt == c, yp == c, zero_division=0)
               for c in labs if (yt == c).sum()]
        rows.append({'scheme': name, 'design': d, 'classes': len(labs),
                     'oa': round(accuracy_score(yt, yp), 4),
                     'kappa': round(cohen_kappa_score(yt, yp), 4)
                     if len(set(yt)) > 1 else np.nan,
                     'macro_f1': round(float(np.mean(f1s)), 4) if f1s else np.nan,
                     'coffee_f1': round(f1_score(yt == 0, yp == 0,
                                                 zero_division=0), 4)})
C1 = pd.DataFrame(rows)
del rows
for v, note in (('macro_f1', 'macro F1'), ('coffee_f1', 'coffee F1')):
    print(f'\n  {note}')
    print(C1.pivot_table(index='scheme', columns='design', values=v,
                         sort=False).reindex(columns=DESIGNS).round(4)
          .to_string())
print('\n  In the 5-class row coffee_f1 is sun coffee alone. In every other')
print('  row it is the merged coffee class. It is identical across the 4, 3')
print('  and 2 class rows because collapsing the NON-coffee labels cannot')
print('  change a one-vs-rest score for coffee.')
print('\n  Reading it. If the merged class holds up while the split classes')
print('  do not, the map can honestly deliver COFFEE but not SHADE VERSUS')
print('  SUN, and the paper should say exactly that. If the merged class is')
print('  also weak, merging hides a problem instead of solving one.')

banner('WHAT THIS CANNOT ANSWER')
print('  Whether sun coffee is fixable by collecting more of it, whether the')
print('  pixel floor removed it selectively, and whether the five-class')
print('  setup handicaps it. Script 8 answers all three.')

if SAVE:
    cc.save_table(A1, TAG, f'B1_{TAG}_shade_vs_forest.csv', index=True)
    if len(A2):
        cc.save_table(A2, TAG, f'B2_{TAG}_sc_forest_region.csv')
    cc.save_table(B1, TAG, f'B3_{TAG}_non_shade_coffee.csv', index=True)
    cc.save_table(C1, TAG, f'B4_{TAG}_class_schemes.csv')
    print(f'\n  tables -> {cc.out_dir(TAG)}')
