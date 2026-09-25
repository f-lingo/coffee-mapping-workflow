"""QUESTION 3b. Is sun coffee fixable, without merging the coffee classes?

Merging is off the table. Shade against sun IS the research question, so the
class stays. That makes one question unavoidable.

    Is sun coffee broken because there is not enough of it, because the
    sample is biased, because the five-class setup handicaps it, or because
    it is not separable from Open at all?

Four diagnoses, four different fixes, and nothing earlier distinguishes them.
Three tasks do.

    TASK 1  PLOT SIZE AND THE PIXEL FLOOR. No fitting.
            If sun coffee plots are smaller than everything else, every pixel
            in them is mixed with whatever surrounds them, and the confusion
            with Open is a minimum-mapping-unit problem rather than a model
            problem. Urban is the control: it is the smallest class of all,
            so if it still maps well then size ALONE is survivable and it is
            size combined with looking like your neighbours that is fatal.

    TASK 2  LEARNING CURVE. Does more of it help?
            Thin that class's TRAINING polygons, hold every other class
            fixed, keep the test set identical. Shade coffee gets the same
            treatment as a control, so the shape reads as class-specific
            rather than general.

    TASK 3  DOES THE FIVE-CLASS SETUP HANDICAP IT?
            a. HIERARCHICAL. Coffee against not-coffee first, then shade
               against sun inside coffee, and the rest inside not-coffee.
               The output is still five classes. Nothing is merged, only the
               decision path changes.
            b. CEILING. A dedicated two-class model, sun coffee against Open
               alone. If a model built for nothing else still fails, the pair
               is not separable with these features.

What each outcome means is printed at the end so the reading cannot drift.
Run time about 15 minutes.
"""

import gc
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from sklearn.metrics import f1_score, accuracy_score

import coffee_common as cc
from coffee_common import (ID_COL, CLASS_NAMES, CLASS_SHORT, ALL_CLASSES,
                           SEEDS, FINAL_PIXEL_CAP, TEST_PIXEL_CAP, NSC,
                           SHADE_COFFEE, OPEN, COFFEE, TAG_NSC, TAG_DESIGNS,
                           banner)

# ---- method settings -------------------------------------------------------
TAG = TAG_NSC
N_SEEDS = 5
# Fractions of that class's OWN training polygons, so both classes get a
# curve of the same resolution despite very different totals. The absolute
# count is reported as n_focal_train, which is the number to quote.
CURVE_FRACTIONS = [0.1, 0.2, 0.35, 0.5, 0.7, 0.85, 1.0]
CURVE_CLASSES = [NSC, SHADE_COFFEE]
RUN_TASK1 = RUN_TASK2 = RUN_TASK3 = True

CURVE_SEEDS = SEEDS[:N_SEEDS]
t0 = time.time()

banner('SCRIPT 8. IS SUN COFFEE FIXABLE')
_n2 = len(CURVE_FRACTIONS) * len(CURVE_CLASSES) * len(CURVE_SEEDS) if RUN_TASK2 else 0
_n3 = 4 * len(CURVE_SEEDS) if RUN_TASK3 else 0
print(f'  {len(CURVE_SEEDS)} seeds, {_n2 + _n3} forest fits')

D = cc.load_data()
SPLITS = D.build_splits(CURVE_SEEDS)
banner('SPLITS')
cc.verify_splits(SPLITS, TAG_DESIGNS)

IDS_OF_CLASS = {c: np.array(sorted(
    D.poly_df.loc[D.poly_df['class'] == c, ID_COL])) for c in ALL_CLASSES}


# =============================================================================
# TASK 1. PLOT SIZE AND WHAT THE PIXEL FLOOR REMOVED. No fitting.
# =============================================================================
if RUN_TASK1:
    banner('TASK 1. PLOT SIZE AND WHAT THE PIXEL FLOOR REMOVED')
    print('  If sun coffee plots are smaller than everything else, every')
    print('  pixel in them is mixed with whatever surrounds them, and the')
    print('  confusion with Open is a minimum-mapping-unit problem rather')
    print('  than a model problem. That changes the whole diagnosis.')

    raw = pd.read_parquet(cc.ANNUAL_FILE, columns=[ID_COL, 'class'],
                          engine='pyarrow', use_threads=False)
    raw['class'] = raw['class'].map(cc.RAW_TO_CLEAN)
    pre_px = raw.groupby(ID_COL).size()
    pre_cls = raw.groupby(ID_COL)['class'].first()
    del raw
    gc.collect()

    kept = set(D.df[ID_COL].unique())
    rows = []
    for c in ALL_CLASSES:
        ids = pre_cls[pre_cls == c].index
        px = pre_px.loc[ids]
        k = [i for i in ids if i in kept]
        dropped = [i for i in ids if i not in kept]
        rows.append({
            'class': CLASS_NAMES[c], 'poly_before': len(ids),
            'poly_after': len(k),
            'pct_kept': round(100 * len(k) / max(len(ids), 1), 1),
            'median_px_all': int(px.median()),
            'median_px_kept': int(px.loc[k].median()) if k else np.nan,
            'median_px_dropped': int(px.loc[dropped].median())
            if dropped else np.nan,
            'median_ha': round(px.median() * 100 / 1e4, 2),
            'pct_under_25px': round(100 * (px < 25).mean(), 1)})
    T1 = pd.DataFrame(rows)
    del rows
    print('\n  Plot size by class, BEFORE any filtering. 1 pixel = 100 m2.')
    print(T1.to_string(index=False))
    nsc = T1[T1['class'] == CLASS_NAMES[NSC]].iloc[0]
    oth = T1[T1['class'] != CLASS_NAMES[NSC]]
    print(f'\n  Sun coffee median {nsc["median_ha"]:.2f} ha against '
          f'{oth["median_ha"].median():.2f} ha for the rest.')
    print(f'  {nsc["pct_under_25px"]}% of its plots hold under 25 pixels, '
          f'against {oth["pct_under_25px"].median():.1f}% elsewhere.')

    side = np.sqrt(nsc['median_px_all'] * 100)
    print(f'\n  A median sun coffee plot is about {side:.0f} m across. At')
    print('  each pixel size, the share that is pure interior once a')
    print('  one-pixel edge is eroded:')
    for res in (10, 5, 3):
        n = side / res
        print(f'    {res:>2} m   {100 * max(n - 2, 0) ** 2 / n ** 2:.0f}%')
    print('  The rest is edge, mixed with whatever is next door.')
    cc.save_table(T1, TAG, f'N1_{TAG}_size_before.csv')


# =============================================================================
# TASK 2. LEARNING CURVE
# =============================================================================
def score5(tr_ids, te_ids, seed):
    tr = D.capped(tr_ids, seed, FINAL_PIXEL_CAP)
    te = D.capped(te_ids, seed, TEST_PIXEL_CAP)
    if len(np.unique(tr['class'].values)) < len(ALL_CLASSES):
        return None
    rf, sc = cc.fit_rf(tr, cc.FEATURE_COLUMNS, tr['class'].values, seed)
    yp, _ = cc.predict_batched(rf, sc, te, cc.FEATURE_COLUMNS)
    _, pyt, pyp = cc.polygon_labels(te, yp)
    out = cc.metrics(pyt, pyp)
    del rf, sc, tr, te
    gc.collect()
    return out


CURVE = pd.DataFrame()
if RUN_TASK2:
    banner('TASK 2. LEARNING CURVE. DOES MORE OF IT HELP')
    print('  The focal class s TRAINING polygons are thinned. Every other')
    print('  class keeps all of its training data and the test set never')
    print('  changes. Levels are nested, so each is a subset of the next.')
    rows = []
    for focal in CURVE_CLASSES:
        print(f'\n    focal class {CLASS_NAMES[focal]}')
        for seed in CURVE_SEEDS:
            tr_ids, te_ids = SPLITS[seed]
            ftr = np.array(sorted(
                IDS_OF_CLASS[focal][np.isin(IDS_OF_CLASS[focal],
                                            list(tr_ids))]))
            order = np.random.RandomState(1000 + seed).permutation(len(ftr))
            other = tr_ids - set(ftr.tolist())
            seen = set()
            for frac in CURVE_FRACTIONS:
                n = min(max(2, int(round(frac * len(ftr)))), len(ftr))
                if n in seen:
                    continue
                seen.add(n)
                rec = score5(other | set(ftr[order[:n]].tolist()),
                             te_ids, seed)
                rows.append({'focal': CLASS_NAMES[focal], 'seed': seed,
                             'n_focal_train': n,
                             'incomplete': rec is None, **(rec or {})})
            print(f'      seed {seed:>4} done', flush=True)
    CURVE = pd.DataFrame(rows)
    del rows
    for focal in CURVE_CLASSES:
        name = CLASS_NAMES[focal]
        sub = CURVE[(CURVE['focal'] == name) & (~CURVE['incomplete'])]
        if not len(sub):
            continue
        col = f'f1_{name}'
        piv = sub.groupby('n_focal_train')[[col, 'macro_f1',
                                            'sc_nsc_f1']].mean().round(4)
        piv['gain_per_10_polys'] = (
            piv[col].diff() / (piv.index.to_series().diff() / 10)).round(4)
        print(f'\n  {CLASS_SHORT[focal]}, its own F1 as its training '
              f'polygons are added')
        print(piv.to_string())
    print('\n  gain_per_10_polys is the slope. Still clearly above zero at')
    print('  the last step means more polygons would help, and you can say')
    print('  how many. At zero means the class has all the data it can use.')
    print('  Compare the two. Both flat means the ceiling is general rather')
    print('  than specific to one class.')
    cc.save_table(CURVE, TAG, f'N2_{TAG}_curve.csv')


# =============================================================================
# TASK 3. DOES THE FIVE-CLASS SETUP HANDICAP IT
# =============================================================================
STRUCT = pd.DataFrame()
if RUN_TASK3:
    banner('TASK 3. DOES THE FIVE-CLASS SETUP HANDICAP IT')
    print('  a. HIERARCHICAL. Coffee against not-coffee first, then shade')
    print('     against sun inside coffee. Output is still five classes.')
    print('     Nothing is merged, only the decision path changes.')
    print('  b. CEILING. Sun coffee against Open alone. If a dedicated model')
    print('     cannot do it, the pair is not separable with these features')
    print('     and no restructuring will help.')
    rows = []
    for seed in CURVE_SEEDS:
        tr_ids, te_ids = SPLITS[seed]
        tr = D.capped(tr_ids, seed, FINAL_PIXEL_CAP)
        te = D.capped(te_ids, seed, TEST_PIXEL_CAP)
        ytr = tr['class'].values

        rf, sc = cc.fit_rf(tr, cc.FEATURE_COLUMNS, ytr, seed)
        yp, _ = cc.predict_batched(rf, sc, te, cc.FEATURE_COLUMNS)
        _, yt_p, yp_p = cc.polygon_labels(te, yp)
        rows.append({'seed': seed, 'model': 'flat 5-class',
                     **cc.metrics(yt_p, yp_p)})
        del rf, sc
        gc.collect()

        rf1, sc1 = cc.fit_rf(tr, cc.FEATURE_COLUMNS,
                             np.isin(ytr, COFFEE).astype(int), seed)
        cof = tr[np.isin(ytr, COFFEE)]
        rf2, sc2 = cc.fit_rf(cof, cc.FEATURE_COLUMNS, cof['class'].values, seed)
        non = tr[~np.isin(ytr, COFFEE)]
        rf3, sc3 = cc.fit_rf(non, cc.FEATURE_COLUMNS, non['class'].values, seed)
        s1, _ = cc.predict_batched(rf1, sc1, te, cc.FEATURE_COLUMNS)
        out = np.empty(len(te), dtype=int)
        m = s1 == 1
        if m.sum():
            out[m], _ = cc.predict_batched(
                rf2, sc2, te.iloc[np.flatnonzero(m)], cc.FEATURE_COLUMNS)
        if (~m).sum():
            out[~m], _ = cc.predict_batched(
                rf3, sc3, te.iloc[np.flatnonzero(~m)], cc.FEATURE_COLUMNS)
        _, yt_h, yp_h = cc.polygon_labels(te, out)
        rows.append({'seed': seed, 'model': 'hierarchical',
                     **cc.metrics(yt_h, yp_h)})
        del rf1, sc1, rf2, sc2, rf3, sc3
        gc.collect()

        ptr = tr[np.isin(ytr, [NSC, OPEN])]
        pte = te[np.isin(te['class'].values, [NSC, OPEN])]
        if len(ptr) and len(pte):
            rf4, sc4 = cc.fit_rf(ptr, cc.FEATURE_COLUMNS,
                                 ptr['class'].values, seed)
            yp4, _ = cc.predict_batched(rf4, sc4, pte, cc.FEATURE_COLUMNS)
            _, yt4, yp4p = cc.polygon_labels(pte, yp4)
            rows.append({'seed': seed, 'model': 'ceiling, sun vs Open only',
                         'nsc_vs_open_f1': f1_score(yt4 == NSC, yp4p == NSC,
                                                    zero_division=0),
                         'oa': accuracy_score(yt4, yp4p)})
            del rf4, sc4
            gc.collect()
        del tr, te
        gc.collect()
        print(f'    seed {seed:>4} done', flush=True)

    STRUCT = pd.DataFrame(rows)
    del rows
    SHOW = ['macro_f1', f'f1_{CLASS_NAMES[NSC]}',
            f'f1_{CLASS_NAMES[SHADE_COFFEE]}', 'sc_nsc_f1', 'nsc_vs_open_f1',
            'oa']
    print('\n  Mean over seeds, polygon level, identical test polygons')
    print(STRUCT.groupby('model')[[c for c in SHOW if c in STRUCT.columns]]
          .mean().reindex(['flat 5-class', 'hierarchical',
                           'ceiling, sun vs Open only']).round(4).to_string())
    if {'flat 5-class', 'hierarchical'} <= set(STRUCT['model']):
        a = STRUCT[STRUCT['model'] == 'flat 5-class'].set_index('seed')
        b = STRUCT[STRUCT['model'] == 'hierarchical'].set_index('seed')
        print('\n  Hierarchical minus flat, paired by seed')
        for c in [f'f1_{CLASS_NAMES[NSC]}', f'f1_{CLASS_NAMES[SHADE_COFFEE]}',
                  'macro_f1', 'sc_nsc_f1']:
            if c in a.columns and c in b.columns:
                d = (b[c] - a[c]).dropna()
                if len(d):
                    print(f'    {c:<26} {d.mean():+.4f} '
                          f'(SD {d.std(ddof=1):.4f}), better in '
                          f'{int((d > 0).sum())}/{len(d)} seeds')
    cc.save_table(STRUCT, TAG, f'N3_{TAG}_structure.csv')


# =============================================================================
# HOW TO READ ALL THREE
# =============================================================================
banner('WHAT EACH OUTCOME MEANS')
print('  TASK 1, size')
print('    Sun coffee plots much smaller than the rest means its pixels are')
print('    mixed, the confusion with Open is a minimum-mapping-unit problem,')
print('    and the fix is finer imagery rather than a better classifier.')
print('    Check Urban. If it is smaller still and maps well, size alone is')
print('    survivable and the real problem is size PLUS looking like your')
print('    neighbours. Sizes similar to other classes rules this out.')
print()
print('  TASK 2, learning curve')
print('    Slope still positive at the last level means the class is short')
print('    of examples. Say how many more and make it a recommendation.')
print('    Slope at zero means more polygons will not help and the ceiling')
print('    is the features. If BOTH classes are flat, the ceiling is general.')
print()
print('  TASK 3, structure')
print('    Hierarchical clearly better for sun coffee means the flat')
print('    five-class decision was handicapping it, and you keep both')
print('    classes AND get the number up, which solves the problem without')
print('    merging anything. No better means the decision path was not it.')
print('    Then read the ceiling row. A dedicated two-class model that also')
print('    fails means the pair is not separable at this resolution. One')
print('    that SUCCEEDS means the pair is separable and what kills it is')
print('    competing against the other classes at the same time.')
print()
print('  The honest worst case. Small plots, flat curve, no gain from')
print('  hierarchy, low ceiling. That says the class cannot be mapped from')
print('  this sensor in this landscape, which is a real result and belongs')
print('  in the paper stated plainly rather than buried.')
print(f'\n  tables -> {cc.out_dir(TAG)}')
print(f'  Total runtime {(time.time() - t0) / 60:.1f} min')
