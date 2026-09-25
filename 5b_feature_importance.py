"""WHICH FEATURES CARRY THE MODEL INTO A REGION IT HAS NOT SEEN.

Permutation importance of the 15 features inside every leave-one-region-out
fold, under two objectives. This is the importance half of the LOROCV cell in
2_classification_lorocv_script_TOP_FINAL.ipynb, moved onto the new pipeline:
the same splits machinery, the same pixel sampler and the same forest as
scripts 4 to 8.

DESIGN
  * Folds. Train on every polygon outside one region, test on every polygon
    inside it. The same fit as script 5's loro_full.
  * Seeds. Each fold is repeated over SEEDS. Seeds change the training pixel
    sample and the forest, never the polygons.
  * Importance is measured on the HELD-OUT region only, so it describes
    what the model relies on when it transfers, not what it memorised.
  * Two objectives, both on the same fit.
        5-class        drop in macro F1 over all five classes
        shade coffee   drop in shade coffee F1 alone
  * Test pixels are subsampled to PERM_TEST_CAP per fold, stratified by
    class, because permutation predicts the test set many times.

WHAT IT CANNOT SAY
  Permutation measures reliance of this fitted model. Correlated features
  share credit, so a low score does not mean a feature is uninformative. The
  15 features were chosen with a |0.65| correlation cap, which limits this
  but does not remove it. Folds and seeds are not independent replicates.
  n is 4 regions.

WHAT THIS WRITES
  I1_*_perm_importance.csv     one row per seed, region, objective, feature
  I2_*_by_region.csv           mean over seeds, per region
  I3_*_summary.csv             mean over regions, CV across regions, and
                               rank agreement between regions

Run time about 30 minutes at 10 seeds. Set N_SEEDS lower to go faster.
"""

import gc
import time

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.inspection import permutation_importance
from sklearn.metrics import f1_score

import coffee_common as cc
from coffee_common import (ID_COL, CLUSTER_COL, SEEDS, FINAL_PIXEL_CAP,
                           TEST_PIXEL_CAP, SHADE_COFFEE, TAG_IMPORTANCE,
                           banner)

# ---- method settings -------------------------------------------------------
TAG = TAG_IMPORTANCE
N_SEEDS = 10
N_REPEATS = 5            # permutation repeats per feature, as in the notebook
PERM_TEST_CAP = 10000    # held-out pixels used for permutation, per fold
MIN_TEST_POLY = 10
CV_MIN_MEAN = 0.005      # CV is only reported for features at least this
                         # important on average, since CV of ~0 is noise

RUN_SEEDS = SEEDS[:N_SEEDS]
OBJECTIVES = ('5-class', 'shade coffee')
t0 = time.time()


def macro_f1(est, X, y):
    return f1_score(y, est.predict(X), average='macro', zero_division=0)


def shade_f1(est, X, y):
    return f1_score(y, est.predict(X), labels=[SHADE_COFFEE], average=None,
                    zero_division=0)[0]


SCORERS = {'5-class': macro_f1, 'shade coffee': shade_f1}


def stratified_cap(y, cap, seed):
    if cap is None or len(y) <= cap:
        return np.arange(len(y))
    rng = np.random.RandomState(seed)
    keep = []
    for c in np.unique(y):
        idx = np.flatnonzero(y == c)
        k = min(len(idx), max(1, int(round(cap * len(idx) / len(y)))))
        keep.extend(rng.choice(idx, size=k, replace=False))
    return np.sort(np.asarray(keep))


banner('SCRIPT 5b. FEATURE IMPORTANCE UNDER LEAVE-ONE-REGION-OUT')
feats = list(cc.FEATURE_COLUMNS)
print(f'  {len(feats)} features, {len(RUN_SEEDS)} seeds, {cc.N_REGIONS} folds')
print(f'  {N_REPEATS} repeats, up to {PERM_TEST_CAP:,} held-out pixels')
print(f'  objectives {", ".join(OBJECTIVES)}')

D = cc.load_data()

rows = []
for c in D.clusters:
    region = D.region_of_cluster[c]
    te_ids, tr_ids = D.cluster_ids[c], D.all_ids - D.cluster_ids[c]
    if len(te_ids) < MIN_TEST_POLY:
        print(f'  {region} skipped, {len(te_ids)} polygons')
        continue
    for seed in RUN_SEEDS:
        tr = D.capped(tr_ids, seed, FINAL_PIXEL_CAP)
        te = D.capped(te_ids, seed, TEST_PIXEL_CAP)
        rf, sc = cc.fit_rf(tr, feats, tr['class'].values, seed)
        yt = te['class'].to_numpy()
        keep = stratified_cap(yt, PERM_TEST_CAP, seed)
        X = sc.transform(cc.prep_X(te.iloc[keep], feats)).astype(np.float32)
        y = yt[keep]
        for obj in OBJECTIVES:
            base = SCORERS[obj](rf, X, y)
            r = permutation_importance(rf, X, y, scoring=SCORERS[obj],
                                       n_repeats=N_REPEATS,
                                       random_state=seed, n_jobs=1)
            for j, f in enumerate(feats):
                rows.append({'seed': seed, CLUSTER_COL: c, 'region': region,
                             'objective': obj, 'feature': f,
                             'importance_mean': r.importances_mean[j],
                             'importance_std': r.importances_std[j],
                             'baseline_score': base,
                             'n_perm_pixels': len(y)})
        del rf, sc, tr, te, X
        gc.collect()
        print(f'    {region}  seed {seed:>4} done', flush=True)

I1 = pd.DataFrame(rows)
del rows

# =============================================================================
# SUMMARIES
# =============================================================================
I2 = (I1.groupby(['objective', 'region', 'feature'])
      .agg(importance=('importance_mean', 'mean'),
           seed_sd=('importance_mean', 'std'),
           baseline_score=('baseline_score', 'mean'))
      .reset_index())
I2['rank'] = I2.groupby(['objective', 'region'])['importance'].rank(
    ascending=False, method='min').astype(int)

summ = []
for obj, g in I2.groupby('objective'):
    W = g.pivot(index='feature', columns='region', values='importance')
    mean, sd = W.mean(axis=1), W.std(axis=1, ddof=1)
    cv = (sd / mean).where(mean >= CV_MIN_MEAN)
    R = W.rank(ascending=False)
    rhos = [spearmanr(R[a], R[b])[0] for i, a in enumerate(R.columns)
            for b in R.columns[i + 1:]]
    s = pd.DataFrame({'objective': obj, 'mean_importance': mean,
                      'sd_across_regions': sd, 'cv_across_regions': cv,
                      'min_region': W.min(axis=1),
                      'positive_in_regions': (W > 0).sum(axis=1),
                      'mean_rank': R.mean(axis=1)})
    s['cross_region_rank_rho'] = float(np.nanmean(rhos))
    summ.append(s.reset_index())
I3 = pd.concat(summ, ignore_index=True).sort_values(
    ['objective', 'mean_importance'], ascending=[True, False])

for obj in OBJECTIVES:
    banner(f'{obj.upper()} OBJECTIVE, MEAN OVER SEEDS')
    W = (I2[I2['objective'].eq(obj)]
         .pivot(index='feature', columns='region', values='importance'))
    s = I3[I3['objective'].eq(obj)].set_index('feature')
    W = W.loc[s.index]
    W['mean'] = s['mean_importance']
    W['CV'] = s['cv_across_regions']
    W['positive in'] = s['positive_in_regions'].astype(str) + '/' + \
        str(I2['region'].nunique())
    print(W.round(4).to_string())
    print(f'\n  mean Spearman rho of feature ranks between regions '
          f'{s["cross_region_rank_rho"].iloc[0]:.2f}')

banner('READING THIS')
print('  A feature that is important in EVERY held-out region, with a low')
print('  CV, is one the model relies on when it transfers. A feature that')
print('  is important in one region only is consistent with a local')
print('  signal that does not carry. CV is left blank for features whose')
print(f'  mean importance is under {CV_MIN_MEAN}, where CV is noise.')
print('  Correlated features share credit, so read low scores with care.')

cc.save_table(I1, TAG, f'I1_{TAG}_perm_importance.csv')
cc.save_table(I2, TAG, f'I2_{TAG}_by_region.csv')
cc.save_table(I3, TAG, f'I3_{TAG}_summary.csv')
print(f'\n  tables -> {cc.out_dir(TAG)}')
print(f'  Total runtime {(time.time() - t0) / 60:.1f} min')
