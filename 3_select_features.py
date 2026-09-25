"""HOW THE FEATURES WERE CHOSEN.

Provenance for the 15 features every later script uses. Run it to reproduce
the selection, to check the 15 still hold up, or to reselect after a new
export.

THE METHOD, AND WHY IT IS NOT THE OBVIOUS ONE

  1. Candidates are every numeric annual or static column that is not
     metadata and not a coordinate. Coordinates are excluded on purpose. A
     model given easting and northing memorises where the polygons are and
     the score collapses the moment it is asked about somewhere new.
  2. Constant and all-NaN columns are dropped. An earlier export had 25 dead
     contrast features from a GLCM scaling bug, and a dead column is not
     harmless: it dilutes importance and wastes a slot.
  3. Correlated pairs are collapsed at CORR_THRESHOLD. Of each pair the one
     with the higher importance is kept. Texture at several radii on the same
     band is near-duplicated by construction, so without this the top of the
     ranking fills with five versions of one thing.
  4. Importance is computed under LEAVE-ONE-CLUSTER-OUT, not random folds.

Point 4 is the one that matters and it is the whole argument of this project
applied to its own feature selection. Reference polygons sit in tight clumps,
median nearest neighbour 208 m. Rank features on random folds and a feature
that merely identifies a neighbourhood scores well, because its neighbours
are in the training set. Rank them on held-out regions and only features that
travel survive. The script reports BOTH rankings so the difference is visible
rather than assumed.

Then a performance curve, top-k features against polygon macro F1, to show
where adding features stops paying. That is what a count like 15 should rest
on rather than a round number.

WHAT IT WRITES
    F1_*_candidates.csv     every candidate with its two importance ranks
    F2_*_selected.json      the selected list, ready to paste
    F3_*_curve.csv          polygon macro F1 against number of features

Run time about 15 minutes.
"""

import gc
import json
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

import coffee_common as cc
from coffee_common import ID_COL, CLUSTER_COL, CLASS_NAMES, banner

# ---- method settings -------------------------------------------------------
TAG = 'feature_selection'
CORR_THRESHOLD = 0.65
N_SELECT = 15                 # the count in use. The curve tests whether it holds
CURVE_K = [3, 5, 8, 10, 12, 15, 20, 25, 30] 
SEED = 93                     # the seed the original selection used
TREES = 300                   # importance is stable well below the final 500
PIXEL_CAP = 50

# Never candidates. Location leaks, identifiers, and labels.
EXCLUDE = {ID_COL, CLUSTER_COL, 'class', 'poly_source', 'original_label',
           'lon', 'lat', 'easting', 'northing', 'geo', 'region', 'group'}

t0 = time.time()
banner('SCRIPT 3. HOW THE FEATURES WERE CHOSEN')


# =============================================================================
# CANDIDATES
# =============================================================================
schema = pq.ParquetFile(cc.ANNUAL_FILE).schema_arrow
numeric = {n for n, t in zip(schema.names, schema.types)
           if pd.api.types.is_numeric_dtype(t.to_pandas_dtype())
           if True}
candidates = sorted(n for n in schema.names
                    if n in numeric and n not in EXCLUDE)
print(f'  {len(schema.names)} columns in the file')
print(f'  {len(candidates)} numeric candidates after excluding metadata and')
print('  coordinates. Coordinates are excluded on purpose: a model given')
print('  easting and northing memorises location and fails on new ground.')
if len(candidates) <= N_SELECT:
    print(f'\n  NOTE. Only {len(candidates)} candidates, at or below the '
          f'{N_SELECT} to select.')
    print('  This file has already been reduced to the modelling set, so the')
    print('  selection below is a check rather than a search. Point')
    print('  ANNUAL_OUTPUT_NAME at the full feature table to reselect.')

D = cc.load_data(features=candidates, verbose=True)
df, poly_df = D.df, D.poly_df

banner('DROPPING DEAD COLUMNS')
arr = df[candidates].to_numpy(dtype=np.float32)
finite = np.isfinite(arr)
all_nan = ~finite.any(axis=0)
with np.errstate(invalid='ignore'):
    sd = np.nanstd(np.where(finite, arr, np.nan), axis=0)
constant = (~all_nan) & (np.nan_to_num(sd) <= 0)
dead = [c for c, a, k in zip(candidates, all_nan, constant) if a or k]
alive = [c for c in candidates if c not in set(dead)]
print(f'  {len(dead)} dead, {len(alive)} alive')
for c in dead[:20]:
    print(f'    {c}')
if len(dead) > 20:
    print(f'    ... and {len(dead) - 20} more')
del arr, finite, sd
gc.collect()


# =============================================================================
# IMPORTANCE, TWO WAYS
# =============================================================================
def importance(fold_ids, label):
    """Mean RF importance over folds. fold_ids yields (train, test) id sets."""
    imps, scores = [], []
    for i, (tr_ids, te_ids) in enumerate(fold_ids):
        tr = D.capped(tr_ids, SEED, PIXEL_CAP)
        if len(np.unique(tr['class'].values)) < 2:
            continue
        rf, sc = cc.fit_rf(tr, alive, tr['class'].values, SEED,
                           n_trees=TREES)
        imps.append(rf.feature_importances_)
        te = D.capped(te_ids, SEED, None)
        yp, _ = cc.predict_batched(rf, sc, te, alive)
        _, pyt, pyp = cc.polygon_labels(te, yp)
        scores.append(cc.metrics(pyt, pyp)['macro_f1'])
        del rf, sc, tr, te
        gc.collect()
        print(f'    {label} fold {i + 1} done', flush=True)
    return (np.mean(imps, axis=0) if imps else np.zeros(len(alive)),
            float(np.mean(scores)) if scores else np.nan)


banner('IMPORTANCE UNDER TWO FOLD SCHEMES')
print('  spatial  leave one cluster out. Only features that TRAVEL score.')
print('  random   an ordinary stratified holdout, repeated. A feature that')
print('           merely identifies a neighbourhood scores well here,')
print('           because its neighbours are in the training set.')

spatial_folds = [(D.all_ids - D.cluster_ids[c], D.cluster_ids[c])
                 for c in D.clusters]
imp_sp, sc_sp = importance(spatial_folds, 'spatial')

rng = np.random.RandomState(SEED)
random_folds = []
for k in range(len(D.clusters)):
    te = D.strat_pick(D.all_ids, cc.FINAL_TEST_FRAC, int(rng.randint(1e6)))
    random_folds.append((D.all_ids - te, te))
imp_rn, sc_rn = importance(random_folds, 'random ')

IMP = pd.DataFrame({'feature': alive, 'imp_spatial': imp_sp,
                    'imp_random': imp_rn})
IMP['rank_spatial'] = IMP['imp_spatial'].rank(ascending=False).astype(int)
IMP['rank_random'] = IMP['imp_random'].rank(ascending=False).astype(int)
IMP['rank_shift'] = IMP['rank_random'] - IMP['rank_spatial']
IMP = IMP.sort_values('imp_spatial', ascending=False).reset_index(drop=True)

print(f'\n  polygon macro F1, spatial folds {sc_sp:.4f}, '
      f'random folds {sc_rn:.4f}')
print(f'  the gap, {sc_rn - sc_sp:+.4f}, is what random folds add for free.')
print('\n  Top 25 by spatial importance')
print(IMP.head(25).round(5).to_string(index=False))
_moved = IMP.reindex(IMP['rank_shift'].abs().sort_values(
    ascending=False).index).head(10)
print('\n  Biggest rank disagreements between the two schemes. A large')
print('  positive shift means random folds UNDER-rate it and it travels')
print('  better than it looks. A large negative means random folds')
print('  FLATTER it, which is the near-neighbour effect.')
print(_moved[['feature', 'rank_spatial', 'rank_random',
              'rank_shift']].to_string(index=False))


# =============================================================================
# CORRELATION FILTER
# =============================================================================
banner(f'COLLAPSING PAIRS CORRELATED ABOVE {CORR_THRESHOLD}')
print('  Texture at several radii on the same band is near-duplicated by')
print('  construction. Without this the top of the ranking fills with five')
print('  versions of one thing. Of each pair the more important survives.')
ordered = IMP['feature'].tolist()
sample = D.capped(D.all_ids, SEED, PIXEL_CAP)
corr = sample[ordered].corr().abs().to_numpy()
del sample
gc.collect()

kept, dropped_for = [], {}
for i, f in enumerate(ordered):
    clash = next((kept[j] for j, k in enumerate(kept)
                  if corr[i, ordered.index(k)] >= CORR_THRESHOLD), None)
    if clash is None:
        kept.append(f)
    else:
        dropped_for[f] = clash
print(f'  {len(ordered)} -> {len(kept)} after collapsing '
      f'{len(dropped_for)} correlated features')
if dropped_for:
    print('\n  First 15 dropped, and what they duplicated')
    for f, k in list(dropped_for.items())[:15]:
        print(f'    {f:<42} ~ {k}')

SELECTED = kept[:N_SELECT]
banner(f'SELECTED, TOP {N_SELECT} AFTER THE FILTER')
for i, f in enumerate(SELECTED, 1):
    print(f'  {i:>2}. {f}')

in_use = set(cc.FEATURE_COLUMNS)
both = in_use & set(SELECTED)
print(f'\n  Against the {len(in_use)} currently in coffee_common.py')
print(f'    {len(both)} in common')
if in_use - set(SELECTED):
    print(f'    in use but not selected here: '
          f'{sorted(in_use - set(SELECTED))}')
if set(SELECTED) - in_use:
    print(f'    selected here but not in use: '
          f'{sorted(set(SELECTED) - in_use)}')
print('\n  A difference is not automatically a problem. The list in use was')
print('  chosen by hand and has produced every number in the paper. Treat')
print('  this as documentation of the method, and change the list only')
print('  deliberately, because every downstream result moves with it.')


# =============================================================================
# HOW MANY FEATURES ARE ACTUALLY WORTH KEEPING
# =============================================================================
banner('POLYGON MACRO F1 AGAINST NUMBER OF FEATURES')
print('  Top k by spatial importance after the correlation filter, scored on')
print('  held-out clusters. A count should rest on where this flattens')
print('  rather than on a round number.')
rows = []
for k in [k for k in CURVE_K if k <= len(kept)]:
    feats = kept[:k]
    scores = []
    for c in D.clusters:
        tr = D.capped(D.all_ids - D.cluster_ids[c], SEED, PIXEL_CAP)
        if len(np.unique(tr['class'].values)) < 2:
            continue
        rf, sc = cc.fit_rf(tr, feats, tr['class'].values, SEED, n_trees=TREES)
        te = D.capped(D.cluster_ids[c], SEED, None)
        yp, _ = cc.predict_batched(rf, sc, te, feats)
        _, pyt, pyp = cc.polygon_labels(te, yp)
        m = cc.metrics(pyt, pyp)
        scores.append((m['macro_f1'], m['sc_nsc_f1']))
        del rf, sc, tr, te
        gc.collect()
    if scores:
        rows.append({'n_features': k,
                     'poly_macro_f1': round(float(np.mean(
                         [s[0] for s in scores])), 4),
                     'shade_vs_sun_f1': round(float(np.nanmean(
                         [s[1] for s in scores])), 4)})
    print(f'    k={k} done', flush=True)
CURVE = pd.DataFrame(rows)
del rows
if len(CURVE):
    CURVE['gain_from_previous'] = CURVE['poly_macro_f1'].diff().round(4)
    print('\n' + CURVE.to_string(index=False))
    print('\n  gain_from_previous at or below zero means those extra features')
    print('  bought nothing on held-out ground, whatever they add on random')
    print('  folds.')

cc.save_table(IMP, TAG, f'F1_{TAG}_candidates.csv')
if len(CURVE):
    cc.save_table(CURVE, TAG, f'F3_{TAG}_curve.csv')
out = cc.out_dir(TAG) / f'F2_{TAG}_selected.json'
out.write_text(json.dumps({
    'selected': SELECTED, 'n_selected': N_SELECT,
    'corr_threshold': CORR_THRESHOLD, 'seed': SEED,
    'importance_folds': 'leave-one-cluster-out',
    'n_candidates': len(candidates), 'n_alive': len(alive),
    'currently_in_use': sorted(in_use),
    'overlap_with_in_use': sorted(both)}, indent=2))
print(f'\n  selected list -> {out}')
print(f'  tables -> {cc.out_dir(TAG)}')
print(f'\n  Total runtime {(time.time() - t0) / 60:.1f} min')
