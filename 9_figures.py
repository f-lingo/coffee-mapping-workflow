"""QUESTION-BY-QUESTION FIGURES. Reads the saved tables, fits nothing.

=============================================================================
MAKE FIGURES. Reads the saved tables, fits nothing, writes the paper's plots.
=============================================================================
Run this after scripts 4 to 8. It touches no model and no pixel data, so it is
fast, safe to rerun, and every figure traces back to a CSV you can open.

Twenty-three figures and three tables.

  F1  Which reference polygons help          Q1
  F2  Accuracy against distance, with the matched control   Q2
  F3  Two methods, one honest number         Q2
  F4  The two boundaries behave differently  Q3
  F5  Plot size by class, the mechanism      Q3
  F6  Learning curves, two ceilings          Q3
  F7  Confusion, with and without local data  supporting
  F8  Four training designs, pixel against polygon macro F1
  F9  Four training designs, every class, pixel against polygon
  F10 Distance at pixel and polygon level, and the distance effect
  T1  Four designs, F1 per class and region, pixel and polygon, paired
  T2  Buffer sweep at pixel and polygon level

From the LOROCV notebook, rebuilt on the new data:
  F11 Feature hierarchy                 F12 Feature table (and T3 as CSV)
  F13 Pixels per polygon, by class      F14 Correlation between features
  F15 The four regions                  F16 Class mix in each region
  F17 Study area map                    F18 Global model confusion, pixel
                                            and polygon, every split
  F19 Per-class F1 when each region is held out, pixel and polygon
  F20 Feature importance in each held-out region, two objectives
  F21 Importance rank stability across regions
  F22 Importance per feature, 5-class against shade coffee
  F23 Mapped class area by region, from a table exported from the map

F11 to F17 read the pixel table once through coffee_common.load_data, so
they take about a minute. F17 needs geopandas and downloads country outlines
once, then caches them. F18 needs script 5 rerun once for its pixel counts.
F20 to F22 need script 5b. F23 needs region_class_area_km2.csv in
ANALYSIS_DIR. The JM separability figures are not rebuilt, because JM was
removed from the paper.

F8, F9 and T1 need script 5's T6c file. F10 and T2 need script 6's pixel
columns. Rerun those two scripts once if these are skipped.

Every figure is written twice, as a 300 dpi PNG for drafts and a PDF for
submission. Colours come from a palette checked for colour-vision deficiency
and print, and every series also carries its own marker shape, so nothing
depends on colour alone.

A missing input is reported and skipped, not fatal. Run whichever Parts you
have and this will make whatever it can.
=============================================================================
"""

import os
import warnings

import numpy as np
import pandas as pd

import matplotlib

# SHOW controls whether figures are displayed as well as written.
#   None   work it out. On inside Jupyter, off when run as a script.
#   True   always display. Use this to force it in an odd environment.
#   False  never display, which is what you want on a headless server.
# Set it before importing this module, or edit it here.
SHOW = None


def _in_notebook():
    try:
        from IPython import get_ipython
        ip = get_ipython()
        if ip is None:
            return False
        return 'IPKernelApp' in getattr(ip, 'config', {}) or \
            type(ip).__name__ == 'ZMQInteractiveShell'
    except Exception:
        return False


_SHOW = _in_notebook() if SHOW is None else bool(SHOW)
if not _SHOW:
    # Must be set before pyplot is imported, and only when not displaying.
    matplotlib.use('Agg')

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import MultipleLocator  # noqa: E402

warnings.filterwarnings('ignore', category=FutureWarning)


# =============================================================================
# CONFIG. Paths come from gee_local_config.py through coffee_common.
# =============================================================================
import coffee_common as cc                                      # noqa: E402
from coffee_common import (CLASS_NAMES, CLASS_SHORT, ALL_CLASSES,  # noqa: E402
                           TAG_DESIGNS, TAG_DISTANCE, TAG_NSC,
                           TAG_IMPORTANCE)

ROOT = cc.ANALYSIS_DIR
NESTED_TAG, DIST_TAG, NSC_TAG = TAG_DESIGNS, TAG_DISTANCE, TAG_NSC
FIG_DIR = str(cc.out_dir('figures'))

SAVE_PDF = True
SAVE_PNG = True
DPI = 1000

# Journal column widths in inches.
W1, W2 = 3.5, 7.2

CLASS_ORDER = [CLASS_NAMES[c] for c in ALL_CLASSES]
DESIGN_ORDER = ['pooled', 'local', 'loro_out', 'loro_all', 'loro_full']
DESIGN_LABEL = {'pooled': 'Global', 'local': 'Local',
                'loro_out': 'Outside only', 'loro_all': 'LORO',
                'loro_full': 'LORO, every polygon'}

# Validated categorical palette. Checked for lightness band, chroma floor,
# adjacent-pair separation under colour-vision deficiency, and contrast.
# Worst adjacent pair is dE 22.9 normal, 9.1 protan. Every series also gets a
# marker, so identity never rests on colour.
C = {'blue': '#2a78d6', 'orange': '#eb6834', 'aqua': '#1baf7a',
     'yellow': '#eda100', 'violet': '#4a3aa7', 'magenta': '#e87ba4'}
INK = '#0b0b0b'
INK2 = '#52514e'
GRID = '#d8d7d2'
MARKERS = ['o', 's', '^', 'D', 'v', 'P']




def _short(name):
    """Short display label from a full class name."""
    for code, full in CLASS_NAMES.items():
        if full == name:
            return CLASS_SHORT[code]
    return name


def style():
    plt.rcParams.update({
        'figure.dpi': 110,
        'savefig.dpi': DPI,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.02,
        'font.size': 8,
        'axes.titlesize': 9,
        'axes.labelsize': 8,
        'legend.fontsize': 7.5,
        'xtick.labelsize': 7.5,
        'ytick.labelsize': 7.5,
        'axes.edgecolor': INK2,
        'axes.linewidth': 0.6,
        'axes.labelcolor': INK,
        'text.color': INK,
        'xtick.color': INK2,
        'ytick.color': INK2,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'legend.frameon': False,
        'lines.linewidth': 1.6,
        'lines.markersize': 4.5,
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
    })


def grid(ax, axis='y'):
    ax.grid(True, axis=axis, color=GRID, linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)


def save(fig, name):
    """Write the figure, and display it too when SHOW is on.

    Saving happens BEFORE showing, because some backends detach the canvas
    once a figure has been drawn and a later savefig then writes a blank."""
    made = []
    if SAVE_PNG:
        p = os.path.join(FIG_DIR, f'{name}.png')
        fig.savefig(p)
        made.append(p)
    if SAVE_PDF:
        p = os.path.join(FIG_DIR, f'{name}.pdf')
        fig.savefig(p)
        made.append(p)
    if _SHOW:
        plt.show()
    else:
        plt.close(fig)
    return made


def read(path, what):
    if not os.path.exists(path):
        raise FileNotFoundError(f'{what} not found at {path}')
    return pd.read_csv(path)


# =============================================================================
# F1. WHICH REFERENCE POLYGONS HELP
# =============================================================================
def fig01():
    P = read(os.path.join(str(ROOT), NESTED_TAG, f'T6_{NESTED_TAG}_paired.csv'),
             'paired decomposition (script 5)')
    g = (P.groupby('design')
         .agg(f1=('macro_f1', 'mean'), sd=('macro_f1', 'std'),
              n_train=('n_train_poly', 'mean'), cells=('macro_f1', 'size')))
    order = [d for d in DESIGN_ORDER if d in g.index]
    g = g.loc[order]
    g['se'] = g['sd'] / np.sqrt(g['cells'])

    fig, ax = plt.subplots(figsize=(W1, 2.5))
    grid(ax, 'x')
    ypos = np.arange(len(g))[::-1]
    has_local = {'pooled': True, 'local': True, 'loro_out': False,
                 'loro_all': False, 'loro_full': False}
    for i, (d, row) in enumerate(g.iterrows()):
        col = C['blue'] if has_local.get(d) else C['orange']
        ax.errorbar(row['f1'], ypos[i], xerr=1.96 * row['se'], fmt=MARKERS[i],
                    color=col, ecolor=col, elinewidth=1.1, capsize=2.2,
                    markersize=5, zorder=3)
        ax.annotate(f'{row["f1"]:.3f}', (row['f1'], ypos[i]),
                    textcoords='offset points', xytext=(0, 7),
                    ha='center', fontsize=7, color=INK)
    ax.set_yticks(ypos)
    ax.set_yticklabels([f'{DESIGN_LABEL.get(d, d)}\n{int(g.loc[d, "n_train"])} polygons'
                        for d in g.index])
    ax.set_xlabel('Polygon macro F1 (95% CI)')
    ax.set_title('Polygon macro F1 by training design', loc='left', pad=8)
    h = [plt.Line2D([], [], color=C['blue'], marker='o', ls='', ms=5,
                    label='Has local training data'),
         plt.Line2D([], [], color=C['orange'], marker='s', ls='', ms=5,
                    label='No local training data')]
    ax.legend(handles=h, loc='lower right', handletextpad=0.4)
    ax.margins(x=0.18)
    return save(fig, 'F1_which_polygons_help')


# =============================================================================
# F2. ACCURACY AGAINST DISTANCE, WITH THE MATCHED CONTROL
# =============================================================================
def fig02():
    S = read(os.path.join(str(ROOT), DIST_TAG, f'T8_{DIST_TAG}_sweep.csv'),
             'buffer sweep (script 6)')
    S = S[S['buffer_km'] >= 0]
    piv = S.pivot_table(index='buffer_km', columns='arm',
                        values='poly_macro_f1')
    rm = S[S['arm'] == 'buffer'].groupby('buffer_km')['n_removed'].mean()
    x = np.arange(len(piv))
    labels = [f'{v:g}' for v in piv.index]

    fig, (ax, ax2) = plt.subplots(
        2, 1, figsize=(W1, 3.6), sharex=True,
        gridspec_kw={'height_ratios': [3, 1], 'hspace': 0.12})
    grid(ax)
    for i, arm in enumerate([a for a in ('random', 'buffer')
                             if a in piv.columns]):
        lab = ('Removed at random, same count\nand class mix'
               if arm == 'random' else 'Removed by distance')
        ax.plot(x, piv[arm], marker=MARKERS[i],
                color=C['aqua'] if arm == 'random' else C['blue'],
                label=lab, zorder=3)
    ax.set_ylabel('Polygon macro F1')
    ax.set_title('Accuracy against buffer distance,\nwith a matched random '
                 'control', loc='left', pad=8)
    ax.legend(loc='best', handletextpad=0.5)

    grid(ax2)
    ax2.bar(x, rm.values, color=GRID, edgecolor=INK2, linewidth=0.4, zorder=3)
    ax2.set_ylabel('Polygons\nremoved', fontsize=7)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)
    ax2.set_xlabel('Buffer radius (km)')
    return save(fig, 'F2_accuracy_vs_distance')


# =============================================================================
# F3. TWO METHODS, ONE HONEST NUMBER
# =============================================================================
def fig03():
    S = read(os.path.join(str(ROOT), DIST_TAG, f'T8_{DIST_TAG}_sweep.csv'),
             'buffer sweep (script 6)')
    S = S[(S['buffer_km'] >= 0) & (S['arm'] == 'buffer')]
    curve = S.groupby('buffer_km')['poly_sc_nsc_f1'].mean()
    gpath = os.path.join(str(ROOT), DIST_TAG, f'T10_{DIST_TAG}_grouped.csv')
    grouped = (pd.read_csv(gpath)['poly_sc_nsc_f1'].mean()
               if os.path.exists(gpath) else np.nan)

    x = np.arange(len(curve))
    fig, ax = plt.subplots(figsize=(W1, 2.5))
    grid(ax)
    ax.plot(x, curve.values, marker='o', color=C['blue'],
            label='Buffered curve', zorder=3)
    if np.isfinite(grouped):
        ax.axhline(grouped, color=C['orange'], linestyle=(0, (4, 2)),
                   linewidth=1.6, zorder=2,
                   label='Grouped split (independent method)')
        ax.annotate(f'{grouped:.3f}', (len(x) - 1, grouped),
                    textcoords='offset points', xytext=(-4, -12),
                    ha='right', fontsize=7.5, color=C['orange'])
    ax.annotate(f'{curve.iloc[0]:.3f}', (0, curve.iloc[0]),
                textcoords='offset points', xytext=(4, 4), fontsize=7.5,
                color=INK)
    ax.set_xticks(x)
    ax.set_xticklabels([f'{v:g}' for v in curve.index])
    ax.set_xlabel('Buffer radius (km)')
    ax.set_ylabel('Shade vs sun coffee, polygon F1')
    ax.set_title('Shade vs sun coffee, buffered curve\nagainst the grouped '
                 'split', loc='left', pad=8)
    ax.legend(loc='lower right', handletextpad=0.5)
    return save(fig, 'F3_two_methods_one_number')


# =============================================================================
# F4. THE TWO BOUNDARIES BEHAVE DIFFERENTLY
# =============================================================================
def fig04():
    B = read(os.path.join(str(ROOT), NESTED_TAG,
                          f'B5_{NESTED_TAG}_exchange_by_dist.csv'),
             'exchange by distance (script 7 section A3)')
    B = B[(B['buffer_km'] >= 0) & (B['arm'] == 'buffer')].sort_values(
        'buffer_km')
    x = np.arange(len(B))
    series = [('pct_shade_lost', 'Shade coffee called forest', C['blue'], 'o'),
              ('pct_forest_lost', 'Forest called shade coffee', C['violet'],
               '^'),
              ('pct_nsc_lost_to_open', 'Sun coffee called Open', C['orange'],
               's')]

    fig, ax = plt.subplots(figsize=(W1, 2.7))
    grid(ax)
    for col, lab, colour, mk in series:
        if col not in B.columns:
            continue
        ax.plot(x, B[col].values, marker=mk, color=colour, label=lab, zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels([f'{v:g}' for v in B['buffer_km']])
    ax.set_xlabel('Buffer radius (km)')
    ax.set_ylabel('Percent of that class misassigned')
    ax.set_title('Class confusions against buffer distance',
                 loc='left', pad=8)
    ax.legend(loc='upper left', handletextpad=0.5)
    ax.yaxis.set_major_locator(MultipleLocator(5))
    return save(fig, 'F4_two_boundaries')


# =============================================================================
# F5. PLOT SIZE BY CLASS
# =============================================================================
def fig05():
    N = read(os.path.join(str(ROOT), NSC_TAG, f'N1_{NSC_TAG}_size_before.csv'),
             'polygon size (script 8 task 1)')
    N = N.set_index('class').reindex(CLASS_ORDER).dropna(how='all')
    ha = N['median_px_all'] * 100 / 1e4

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(W2, 2.5))
    y = np.arange(len(N))[::-1]
    grid(ax, 'x')
    cols = [C['orange'] if c == 'Non-Shade Coffee' else C['blue']
            for c in N.index]
    ax.barh(y, ha.values, color=cols, height=0.62, zorder=3)
    for i, v in enumerate(ha.values):
        ax.annotate(f'{v:.2f} ha', (v, y[i]), textcoords='offset points',
                    xytext=(4, 0), va='center', fontsize=7, color=INK)
    ax.set_yticks(y)
    ax.set_yticklabels([_short(c) for c in N.index])
    ax.set_xlabel('Median plot size (hectares)')
    ax.set_title('Median plot size by class', loc='left', pad=8)
    ax.margins(x=0.22)

    grid(ax2, 'x')
    ax2.barh(y, N['pct_under_25px'].values, color=cols, height=0.62, zorder=3)
    for i, v in enumerate(N['pct_under_25px'].values):
        ax2.annotate(f'{v:.0f}%', (v, y[i]), textcoords='offset points',
                     xytext=(4, 0), va='center', fontsize=7, color=INK)
    ax2.set_yticks(y)
    ax2.set_yticklabels([])
    ax2.set_xlabel('Percent of plots under 0.25 ha')
    ax2.set_title('Share of plots under 0.25 ha', loc='left', pad=8)
    ax2.margins(x=0.22)
    return save(fig, 'F5_plot_size_by_class')


# =============================================================================
# F6. LEARNING CURVES
# =============================================================================
def fig06():
    N = read(os.path.join(str(ROOT), NSC_TAG, f'N2_{NSC_TAG}_curve.csv'),
             'learning curves (script 8 task 2)')
    if 'incomplete' in N.columns:
        N = N[~N['incomplete'].astype(bool)]

    fig, ax = plt.subplots(figsize=(W1, 2.6))
    grid(ax)
    for i, (focal, colour, mk) in enumerate(
            [('Shade Coffee', C['blue'], 'o'),
             ('Non-Shade Coffee', C['orange'], 's')]):
        sub = N[N['focal'] == focal]
        if not len(sub):
            continue
        col = f'f1_{focal}'
        g = sub.groupby('n_focal_train')[col].mean()
        ax.plot(g.index, g.values, marker=mk, color=colour,
                label=CLASS_SHORT.get(focal, focal), zorder=3)
        ax.annotate(f'{g.iloc[-1]:.2f}', (g.index[-1], g.iloc[-1]),
                    textcoords='offset points', xytext=(5, -1),
                    fontsize=7.5, color=colour, va='center')
    ax.set_xlabel('Training polygons of that class')
    ax.set_ylabel('Polygon F1 for that class')
    ax.set_title('Class F1 against that class s training size',
                 loc='left', pad=8)
    ax.legend(loc='upper left', handletextpad=0.5)
    ax.margins(x=0.12)
    return save(fig, 'F6_learning_curves')


# =============================================================================
# F7. CONFUSION, WITH AND WITHOUT LOCAL DATA
# =============================================================================
def fig07():
    P = read(os.path.join(str(ROOT), NESTED_TAG,
                          f'P_{NESTED_TAG}_polygon_predictions.csv'),
             'per-polygon predictions (script 5)')
    names = {i: n for i, n in enumerate(CLASS_ORDER)}
    panels = [('pooled', 'With local training data'),
              ('loro_out', 'Without local training data')]
    panels = [(d, t) for d, t in panels if d in set(P['design'])]
    if not panels:
        raise ValueError('neither pooled nor loro_out found in predictions')

    fig, axes = plt.subplots(1, len(panels), figsize=(W2, 3.1))
    axes = np.atleast_1d(axes)
    for ax, (design, title) in zip(axes, panels):
        f = P[P['design'] == design]
        M = pd.crosstab(f['y_true'], f['y_pred']).reindex(
            index=range(5), columns=range(5), fill_value=0).to_numpy(float)
        row = M.sum(axis=1, keepdims=True)
        pct = 100 * M / np.where(row == 0, 1, row)
        # sequential single hue, light to dark. Never a rainbow.
        ax.imshow(pct, cmap='Blues', vmin=0, vmax=100, aspect='equal')
        for r in range(5):
            for c in range(5):
                ax.text(c, r, f'{pct[r, c]:.0f}', ha='center', va='center',
                        fontsize=6.5,
                        color='white' if pct[r, c] > 55 else INK)
        ax.set_xticks(range(5))
        ax.set_yticks(range(5))
        ax.set_xticklabels([_short(names[i]) for i in range(5)],
                           rotation=45, ha='right')
        ax.set_yticklabels([_short(names[i]) for i in range(5)])
        ax.set_title(title, loc='left', pad=6, fontsize=8.5)
        ax.set_xlabel('Predicted')
        for s in ax.spines.values():
            s.set_visible(False)
        ax.tick_params(length=0)
    axes[0].set_ylabel('Reference')
    fig.suptitle('Polygon confusion, row percentages', x=0.01, ha='left',
                 fontsize=9, y=1.02)
    return save(fig, 'F7_confusion_matrices')


# =============================================================================
# THE FOUR TRAINING DESIGNS, PIXEL AND POLYGON. Shared by T1, F8 and F9.
# =============================================================================
# Four models, all scored on the SAME held-out pixels and polygons of each
# region, for every seed, so every difference is paired.
#   Global        pooled. Every training polygon, from all four regions.
#   Local         the target region's training polygons only.
#   Outside only  loro_out. The training polygons of the other three regions.
#   LORO          loro_all. Every polygon of the other three regions, none
#                 from the target. About the same size as Global, so Global
#                 minus LORO is what the region's own data adds at equal size.
# Global minus Local is what the other regions add on top of local data.
T1_DESIGNS = [('pooled', 'global'), ('local', 'local'),
              ('loro_out', 'outside'), ('loro_all', 'loro')]
T1_MODELS = [m for _, m in T1_DESIGNS]
T1_MODEL_LABEL = {'global': 'Global', 'local': 'Local',
                  'outside': 'Outside only', 'loro': 'LORO'}
T1_HAS_LOCAL = {'global': True, 'local': True, 'outside': False,
                'loro': False}
T1_METRICS = ['macro_f1'] + [f'f1_{n}' for n in CLASS_ORDER]
T1_LABEL = {'macro_f1': 'Macro F1',
            **{f'f1_{n}': n for n in CLASS_ORDER}}
T1_CONTRASTS = [('global', 'local'), ('global', 'loro'), ('local', 'loro'),
                ('loro', 'outside')]
LEVEL_STYLE = {'pixel': dict(color=C['yellow'], marker='s', label='Pixel'),
               'polygon': dict(color=C['blue'], marker='o',
                               label='Polygon')}


def design_scores():
    """Long table. One row per seed, region, model, level and metric.

    Only seed-region cells where all four models exist at BOTH levels are
    kept, so every model and both levels are scored on one test set."""
    nest = os.path.join(str(ROOT), NESTED_TAG)
    poly = read(os.path.join(nest, f'T6_{NESTED_TAG}_paired.csv'),
                'paired polygon table (script 5)')
    t5 = read(os.path.join(nest, f'T5_{NESTED_TAG}_evaluation.csv'),
              'evaluation table (script 5)')
    pxp = os.path.join(nest, f'T6c_{NESTED_TAG}_pixels_by_cluster.csv')
    if not os.path.exists(pxp):
        raise FileNotFoundError(
            f'{os.path.basename(pxp)} not found. Rerun script 5, which now '
            'writes the pixel scores per region that this needs.')
    t6c = pd.read_csv(pxp)
    model_of = dict(T1_DESIGNS)
    cols = ['seed', 'cluster_id', 'region', 'design'] + T1_METRICS
    # Pixel. Local and outside-only from script 5's own records, global and
    # LORO from T6c, all on the held-out share of each region.
    own = t5[t5['design'].isin(['local', 'loro_out'])].copy()
    own['cluster_id'] = own['cluster_id'].astype(int)
    px = pd.concat([t6c[cols], own[cols]], ignore_index=True)
    px = px[px['design'].isin(model_of)]
    pg = poly[poly['design'].isin(model_of)][cols + ['n_train_poly',
                                                    'n_test_poly']]
    key = ['seed', 'cluster_id']
    full = None
    for f in (px, pg):
        ok = f.groupby(key)['design'].nunique().eq(len(T1_DESIGNS))
        ok = ok[ok].reset_index()[key]
        full = ok if full is None else full.merge(ok, on=key)
    px, pg = px.merge(full, on=key), pg.merge(full, on=key)
    out = []
    for f, level in ((px, 'pixel'), (pg, 'polygon')):
        m = f.melt(id_vars=['seed', 'cluster_id', 'region', 'design'],
                   value_vars=T1_METRICS, var_name='metric', value_name='f1')
        m['level'] = level
        out.append(m)
    L = pd.concat(out, ignore_index=True)
    L['model'] = L['design'].map(model_of)
    sizes = pg.groupby('design')['n_train_poly'].mean().rename(model_of)
    n_test = pg.groupby('region')['n_test_poly'].mean()
    return L, sizes, n_test


def _cell_mean_ci(L):
    """Mean over regions of the per-region seed mean, with a 95% interval
    from the seed-region cells. Regions weigh equally."""
    g = L.groupby(['level', 'metric', 'model', 'region'])['f1'].mean()
    mean = g.groupby(['level', 'metric', 'model']).mean()
    cells = L.groupby(['level', 'metric', 'model'])['f1']
    se = cells.std(ddof=1) / np.sqrt(cells.count())
    return pd.DataFrame({'mean': mean, 'ci': 1.96 * se}).reset_index()


# =============================================================================
# T1. FOUR DESIGNS, PER CLASS, PIXEL AND POLYGON
# =============================================================================
def table_pooled_vs_local():
    L, sizes, n_test = design_scores()
    W = L.pivot_table(index=['level', 'metric', 'region', 'seed'],
                      columns='model', values='f1').reset_index()
    for a, b in T1_CONTRASTS:
        W[f'{a}_minus_{b}'] = W[a] - W[b]
        W[f'{a}_beats_{b}'] = (W[a] > W[b]).astype(int)
        W[f'{a}_vs_{b}_n'] = W[[a, b]].notna().all(axis=1).astype(int)

    val = T1_MODELS + [f'{a}_minus_{b}' for a, b in T1_CONTRASTS]
    cnt = [f'{a}_beats_{b}' for a, b in T1_CONTRASTS] + \
          [f'{a}_vs_{b}_n' for a, b in T1_CONTRASTS]
    how = {**{c: 'mean' for c in val}, **{c: 'sum' for c in cnt}}
    R = W.groupby(['level', 'metric', 'region'])[val + cnt].agg(how)
    R = R.reset_index()
    A = R.groupby(['level', 'metric'])[val + cnt].agg(how).reset_index()
    A['region'] = 'All regions'
    R = pd.concat([R, A], ignore_index=True)
    for a, b in T1_CONTRASTS:
        R[f'{a}_better_than_{b}'] = (R[f'{a}_beats_{b}'].astype(int)
                                     .astype(str) + '/' +
                                     R[f'{a}_vs_{b}_n'].astype(int)
                                     .astype(str))

    keep = val + [f'{a}_better_than_{b}' for a, b in T1_CONTRASTS]
    T = R.pivot_table(index=['region', 'metric'], columns='level',
                      values=keep, aggfunc='first')
    T.columns = [f'{lv}_{c}' for c, lv in T.columns]
    order = [f'{lv}_{c}' for lv in ('pixel', 'polygon') for c in keep]
    T = T[order].reset_index()
    regions = sorted(L['region'].unique()) + ['All regions']
    T['region'] = pd.Categorical(T['region'], regions, ordered=True)
    T['metric'] = pd.Categorical(T['metric'].map(T1_LABEL),
                                 [T1_LABEL[m] for m in T1_METRICS],
                                 ordered=True)
    T = T.sort_values(['region', 'metric']).reset_index(drop=True)
    num = [c for c in T.columns
           if c.endswith(tuple(val)) and '_better_than_' not in c]
    T[num] = T[num].astype(float).round(3)
    n_test = n_test.copy()
    n_test['All regions'] = n_test.sum()
    T.insert(2, 'mean_test_polygons',
             T['region'].astype(str).map(n_test.round(1)))

    p = os.path.join(FIG_DIR, 'T1_global_local_loro.csv')
    T.to_csv(p, index=False)
    S = (T[T['region'].astype(str).eq('All regions')]
         .drop(columns=['region', 'mean_test_polygons'])
         .rename(columns={'metric': 'class'}))
    ps = os.path.join(FIG_DIR, 'T1a_global_local_loro_summary.csv')
    S.to_csv(ps, index=False)

    show = ['class'] + [f'{lv}_{m}' for lv in ('pixel', 'polygon')
                        for m in T1_MODELS]
    print('\n  T1a. F1 by class and design, all regions')
    print('  Mean training polygons  ' + ', '.join(
        f'{T1_MODEL_LABEL[m]} {sizes.get(m, 0):.0f}' for m in T1_MODELS)
        + '\n')
    with pd.option_context('display.width', 220,
                           'display.max_columns', None):
        print(S[show].to_string(index=False))
        for lv in ('polygon', 'pixel'):
            print(f'\n  Paired differences, {lv} level')
            d = ['class'] + [f'{lv}_{a}_minus_{b}' for a, b in T1_CONTRASTS]
            d += [f'{lv}_{a}_better_than_{b}' for a, b in T1_CONTRASTS]
            print(S[d].to_string(index=False))

    small = ', '.join(f'R{c} ({cc.REGION_NAME_HINT.get(c, "?")})'
                      for c in cc.SMALL_REGIONS)
    print('\n  global_minus_local   what the other regions add on top of the')
    print('                       target region\'s own data.')
    print('  global_minus_loro    what the target region\'s own data adds, at')
    print('                       about the same training size.')
    print('  loro_minus_outside   what MORE distant data adds, none local.')
    print('  better_than counts region-seed pairs, ties count as not better.')
    print(f'  {small} holds few coffee test polygons.')
    print('  Levels carry the random-holdout inflation script 6 measures. The')
    print('  CONTRASTS are fair because all four designs share that holdout.')
    print(f'\n  per-region detail -> {os.path.basename(p)}')
    return [ps, p]


# =============================================================================
# F8. FOUR DESIGNS, MACRO F1, PIXEL AGAINST POLYGON
# =============================================================================
def fig08():
    L, sizes, _ = design_scores()
    M = _cell_mean_ci(L[L['metric'].eq('macro_f1')])
    fig, ax = plt.subplots(figsize=(W1, 2.7))
    grid(ax, 'x')
    ypos = np.arange(len(T1_MODELS))[::-1]
    off = {'pixel': -0.14, 'polygon': 0.14}
    for lv, st in LEVEL_STYLE.items():
        sub = M[M['level'].eq(lv)].set_index('model').reindex(T1_MODELS)
        ax.errorbar(sub['mean'], ypos + off[lv], xerr=sub['ci'],
                    fmt=st['marker'], color=st['color'], ecolor=st['color'],
                    elinewidth=1.1, capsize=2.2, markersize=5, zorder=3,
                    label=st['label'])
        for y, v in zip(ypos + off[lv], sub['mean']):
            ax.annotate(f'{v:.3f}', (v, y), textcoords='offset points',
                        xytext=(0, 5), ha='center', fontsize=6.5,
                        color=st['color'])
    for i, m in enumerate(T1_MODELS):
        if not T1_HAS_LOCAL[m]:
            ax.axhspan(ypos[i] - 0.45, ypos[i] + 0.45, color=GRID,
                       alpha=0.35, zorder=0, linewidth=0)
    ax.set_yticks(ypos)
    ax.set_yticklabels([f'{T1_MODEL_LABEL[m]}\n{sizes.get(m, 0):.0f} polygons'
                        for m in T1_MODELS])
    ax.set_xlabel('Macro F1 (95% CI over region-splits)')
    ax.set_title('Four training designs, pixel and polygon',
                 loc='left', pad=8)
    ax.legend(loc='lower right', handletextpad=0.4)
    ax.margins(x=0.15)
    ax.annotate('shaded: no training data\nfrom the target region',
                xy=(0.01, 0.01), xycoords='axes fraction', fontsize=6.5,
                color=INK2, va='bottom')
    return save(fig, 'F8_designs_pixel_polygon')


# =============================================================================
# F9. FOUR DESIGNS, EVERY CLASS, PIXEL AGAINST POLYGON
# =============================================================================
def fig09():
    L, _, _ = design_scores()
    M = _cell_mean_ci(L)
    fig, axes = plt.subplots(2, 3, figsize=(W2, 4.3), sharey=True)
    x = np.arange(len(T1_MODELS))
    off = {'pixel': -0.12, 'polygon': 0.12}
    for ax, metric in zip(axes.ravel(), T1_METRICS):
        grid(ax)
        for lv, st in LEVEL_STYLE.items():
            sub = (M[M['level'].eq(lv) & M['metric'].eq(metric)]
                   .set_index('model').reindex(T1_MODELS))
            ax.errorbar(x + off[lv], sub['mean'], yerr=sub['ci'],
                        fmt=st['marker'], color=st['color'],
                        ecolor=st['color'], elinewidth=1.0, capsize=2,
                        markersize=4.5, zorder=3, label=st['label'],
                        linestyle='-', linewidth=0.9)
        ax.axvspan(1.5, 3.5, color=GRID, alpha=0.35, zorder=0, linewidth=0)
        ax.set_xticks(x)
        ax.set_xticklabels([T1_MODEL_LABEL[m].replace(' only', '\nonly')
                            for m in T1_MODELS], fontsize=6.8)
        name = T1_LABEL[metric]
        ax.set_title(name if metric == 'macro_f1' else _short(name),
                     loc='left', pad=5, fontsize=8.5)
    for ax in axes[:, 0]:
        ax.set_ylabel('F1')
    axes[0, 0].legend(loc='lower left', handletextpad=0.4)
    fig.suptitle('F1 by class and training design, pixel and polygon. '
                 'Shaded designs hold no target-region data.',
                 x=0.01, ha='left', fontsize=9, y=1.01)
    fig.tight_layout()
    return save(fig, 'F9_designs_by_class')


# =============================================================================
# F10 AND T2. DISTANCE, PIXEL AGAINST POLYGON
# =============================================================================
DIST_METRICS = ['macro_f1', f'f1_{CLASS_NAMES[1]}', f'f1_{CLASS_NAMES[0]}',
                'sc_nsc_f1']
DIST_LABEL = {'macro_f1': 'Macro F1', f'f1_{CLASS_NAMES[1]}': 'Shade coffee',
              f'f1_{CLASS_NAMES[0]}': 'Sun coffee',
              'sc_nsc_f1': 'Shade vs sun'}


def _sweep_levels():
    S = read(os.path.join(str(ROOT), DIST_TAG, f'T8_{DIST_TAG}_sweep.csv'),
             'buffer sweep (script 6)')
    S = S[S['buffer_km'] >= 0]
    if not any(c.startswith('px_') for c in S.columns):
        raise FileNotFoundError(
            'T8 sweep has no pixel columns. Rerun script 6, which now '
            'scores every fit at pixel level too.')
    return S


def fig10():
    S = _sweep_levels()
    x_vals = sorted(S['buffer_km'].unique())
    x = np.arange(len(x_vals))
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(W2, 2.8))
    grid(ax)
    grid(ax2)
    for lv, pre in (('pixel', 'px_'), ('polygon', 'poly_')):
        st = LEVEL_STYLE[lv]
        pv = S.pivot_table(index='buffer_km', columns='arm',
                           values=pre + 'macro_f1').reindex(x_vals)
        ax.plot(x, pv['buffer'], marker=st['marker'], color=st['color'],
                label=f'{st["label"]}, removed by distance', zorder=3)
        if 'random' in pv.columns:
            ax.plot(x, pv['random'], marker=st['marker'], color=st['color'],
                    linestyle=(0, (3, 2)), markerfacecolor='white',
                    label=f'{st["label"]}, removed at random', zorder=3)
            ax2.plot(x, pv['random'] - pv['buffer'], marker=st['marker'],
                     color=st['color'], label=st['label'], zorder=3)
    for a in (ax, ax2):
        a.set_xticks(x)
        a.set_xticklabels([f'{v:g}' for v in x_vals], fontsize=6.8)
        a.set_xlabel('Buffer radius (km)')
    ax.set_ylabel('Macro F1')
    ax.set_title('Macro F1 against buffer distance', loc='left', pad=8)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=2,
              fontsize=6.5, handletextpad=0.4)
    ax2.axhline(0, color=INK2, linewidth=0.6)
    ax2.set_ylabel('Distance effect (F1)')
    ax2.set_title('Distance effect, random minus buffer',
                  loc='left', pad=8)
    ax2.legend(loc='upper left', handletextpad=0.4)
    fig.tight_layout()
    return save(fig, 'F10_distance_pixel_polygon')


def table_distance_levels():
    S = _sweep_levels()
    rows = []
    for m in DIST_METRICS:
        r = {}
        for lv, pre in (('pixel', 'px_'), ('polygon', 'poly_')):
            if pre + m not in S.columns:
                continue
            pv = S.pivot_table(index='buffer_km', columns='arm',
                               values=pre + m)
            r[f'{lv}_buffer'] = pv['buffer']
            if 'random' in pv.columns:
                r[f'{lv}_random'] = pv['random']
                r[f'{lv}_distance_effect'] = pv['random'] - pv['buffer']
        t = pd.DataFrame(r)
        if {'pixel_distance_effect', 'polygon_distance_effect'} <= \
                set(t.columns):
            t['polygon_minus_pixel_effect'] = (t['polygon_distance_effect']
                                               - t['pixel_distance_effect'])
        t = t.reset_index()
        t.insert(0, 'metric', DIST_LABEL[m])
        rows.append(t)
    T = pd.concat(rows, ignore_index=True).round(4)
    p = os.path.join(FIG_DIR, 'T2_distance_pixel_polygon.csv')
    T.to_csv(p, index=False)
    print('\n  T2. Buffer sweep at pixel and polygon level')
    print('  distance_effect is random minus buffer. polygon_minus_pixel_effect')
    print('  above zero means the majority vote amplifies the distance loss.\n')
    with pd.option_context('display.width', 220,
                           'display.max_columns', None):
        print(T.to_string(index=False))
    return [p]


# =============================================================================
# FROM THE LOROCV NOTEBOOK. Data description, the global model, transfer, and
# feature importance, rebuilt on the new pipeline's data and tables.
# =============================================================================
# F11 to F17 describe the data. They read the pixel table once, through
# coffee_common.load_data, with the same filters every model sees. Everything
# else reads saved tables.
CLASS_COLOR = {'Non-Shade Coffee': '#d95f02', 'Shade Coffee': '#1b9e77',
               'Forest': '#386641', 'Open': '#bcbd22', 'Urban': '#7f7f7f'}
REGION_COLOR = {0: C['orange'], 1: C['blue'], 2: C['violet'], 3: C['aqua']}
GROUP_COLOR = {'Spectral': '#2a78d6', 'Structural': '#1b9e77',
               'Topographic': '#c0392b'}
GROUP_FILL = {'Spectral': '#dbe8f8', 'Structural': '#d6efe4',
              'Topographic': '#f6dcd8'}

_BANDS = {'B2': 'blue', 'B3': 'green', 'B4': 'red', 'B5': 'red edge 1',
          'B6': 'red edge 2', 'B7': 'red edge 3', 'B8': 'near infrared',
          'B8A': 'narrow near infrared', 'B11': 'shortwave infrared 1',
          'B12': 'shortwave infrared 2'}
_INDICES = {'NDVI': 'NDVI, (B8 - B4) / (B8 + B4)',
            'NDI45': 'red-edge NDI, (B5 - B4) / (B5 + B4)',
            'GCVI': 'green chlorophyll index, B8 / B3 - 1',
            'NDTI': 'tillage index, (B11 - B12) / (B11 + B12)',
            'VARI': 'visible atmospherically resistant index',
            'RGB': 'visible bands combined'}
_STAT = {'mean': 'Local mean', 'variance': 'Local variance',
         'contrast': 'GLCM contrast'}
GROUP_ORDER = ['Spectral', 'Structural', 'Topographic']
SUB_ORDER = ['Spectral bands', 'Spectral texture', 'Spectral indices',
             'Spectral index texture', 'Canopy height',
             'Canopy height texture', 'Topography', 'Topography texture']

_DATA = {}
CLUSTER_COL = cc.CLUSTER_COL
_DARK = {'#386641', '#1b9e77', '#d95f02', '#7f7f7f'}


def data():
    """The filtered pixel table, loaded once per session."""
    if 'D' not in _DATA:
        _DATA['D'] = cc.load_data(verbose=False)
    return _DATA['D']


def parse_feature(name):
    """Group, sub-group, source and a plain description, from the name.

    Texture suffix _kR is a (2R+1) x (2R+1) pixel window at 10 m, as built
    in script 1 (ee.Kernel.square(R), glcmTexture(size=R))."""
    base, stat, r = name, None, None
    parts = name.rsplit('_', 2)
    if len(parts) == 3 and parts[1] in _STAT and parts[2].startswith('k'):
        base, stat, r = parts[0], parts[1], int(parts[2][1:])
    if base in _BANDS or base == 'RGB':
        group, kind = 'Spectral', 'Spectral bands'
        what = (f'{_BANDS[base]} reflectance' if base in _BANDS
                else 'visible reflectance')
        source = 'Sentinel-2 L2A'
    elif base in _INDICES:
        group, kind = 'Spectral', 'Spectral indices'
        what, source = _INDICES[base], 'Sentinel-2 L2A'
    elif base.startswith('canopy_height'):
        group, kind = 'Structural', 'Canopy height'
        what, source = 'canopy height', 'Meta canopy height, 1 m'
    elif base in ('elevation', 'slope'):
        group, kind = 'Topographic', 'Topography'
        what, source = base, 'SRTM, 30 m'
    else:
        group, kind, what, source = 'Other', 'Other', base, '?'
    if stat is None:
        sub = kind
        desc = what[0].upper() + what[1:]
    else:
        sub = {'Spectral bands': 'Spectral texture',
               'Spectral indices': 'Spectral index texture',
               'Canopy height': 'Canopy height texture',
               'Topography': 'Topography texture'}.get(kind, kind)
        w = 2 * r + 1
        desc = f'{_STAT[stat]} of {what}, {w} x {w} window ({w * 10} m)'
    return {'feature': name, 'group': group, 'subgroup': sub,
            'source': source, 'description': desc}


def feature_frame(features=None):
    F = pd.DataFrame([parse_feature(f)
                      for f in (features or cc.FEATURE_COLUMNS)])
    F['g'] = F['group'].map({g: i for i, g in enumerate(GROUP_ORDER)})
    F['s'] = F['subgroup'].map({s: i for i, s in enumerate(SUB_ORDER)})
    return F.sort_values(['g', 's', 'feature']).drop(columns=['g', 's'])


# =============================================================================
# F11. FEATURE HIERARCHY
# =============================================================================
def fig11():
    F = feature_frame()
    n = len(F)
    fig, ax = plt.subplots(figsize=(W2, 0.28 * n + 0.6))
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.8, n - 0.2)
    ax.axis('off')
    y_feat = {f: n - 1 - i for i, f in enumerate(F['feature'])}
    x_g, x_s, x_f = 0.02, 0.26, 0.56
    for g, gF in F.groupby('group', sort=False):
        col = GROUP_COLOR.get(g, INK2)
        gy = np.mean([y_feat[f] for f in gF['feature']])
        ax.text(x_g, gy, g, ha='left', va='center', fontsize=8.5,
                color='white', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.35', fc=col, ec=col))
        for s, sF in gF.groupby('subgroup', sort=False):
            sy = np.mean([y_feat[f] for f in sF['feature']])
            ax.plot([x_g + 0.12, x_s - 0.01], [gy, sy], color=col, lw=0.9)
            ax.text(x_s, sy, s, ha='left', va='center', fontsize=7.5,
                    color=INK, bbox=dict(boxstyle='round,pad=0.3',
                                         fc='white', ec=col, lw=0.8))
            for f in sF['feature']:
                ax.plot([x_s + 0.2, x_f - 0.01], [sy, y_feat[f]],
                        color=GRID, lw=0.8)
                ax.text(x_f, y_feat[f], f, ha='left', va='center',
                        fontsize=7, color=INK, family='monospace')
    ax.set_title(f'The {n} features, by group', loc='left', pad=4)
    return save(fig, 'F11_feature_hierarchy')


# =============================================================================
# F12 AND T3. FEATURE TABLE
# =============================================================================
def table_features():
    F = feature_frame()
    p = os.path.join(FIG_DIR, 'T3_feature_table.csv')
    F.to_csv(p, index=False)
    print('\n  T3. The features in use')
    with pd.option_context('display.width', 200, 'display.max_colwidth', 70):
        print(F.to_string(index=False))
    return [p]


def fig12():
    """Drawn cell by cell, because matplotlib's table pads unevenly and
    cannot wrap."""
    import textwrap
    F = feature_frame()
    cols = ['group', 'subgroup', 'feature', 'source', 'description']
    heads = ['Group', 'Sub-group', 'Feature', 'Source', 'Description']
    # left edge of each column, as a fraction of the width, and wrap width
    x0 = [0.0, 0.11, 0.25, 0.53, 0.66, 1.0]
    wrap = [12, 15, 34, 14, 44]
    rows = [[textwrap.fill(str(v), w, break_long_words=True)
             for v, w in zip(r, wrap)] for r in F[cols].values]
    nl = [max(t.count('\n') + 1 for t in r) for r in rows]
    line_h, pad = 0.155, 0.07          # inches
    heights = [0.26] + [n * line_h + 2 * pad for n in nl]
    H = sum(heights)
    fig = plt.figure(figsize=(W2, H + 0.35))
    ax = fig.add_axes([0, 0, 1, H / (H + 0.35)])
    ax.set_xlim(0, 1)
    ax.set_ylim(H, 0)
    ax.axis('off')
    y = 0.0
    for i, h in enumerate(heights):
        if i == 0:
            fc, texts, colr, weight = INK, heads, 'white', 'bold'
        else:
            g = F.iloc[i - 1]['group']
            fc, texts = GROUP_FILL.get(g, '#f2f2f2'), rows[i - 1]
            colr, weight = INK, 'normal'
        ax.add_patch(plt.Rectangle((0, y), 1, h, fc=fc, ec='white', lw=1.2))
        for j, t in enumerate(texts):
            ax.text(x0[j] + 0.008, y + h / 2, t, ha='left', va='center',
                    fontsize=6.4, color=colr, fontweight=weight,
                    family='monospace' if (j == 2 and i > 0) else None,
                    linespacing=1.25)
        y += h
    for xv in x0[1:-1]:
        ax.plot([xv, xv], [0, H], color='white', lw=1.2)
    fig.text(0.0, 1.0, 'Feature definitions', ha='left', va='top',
             fontsize=9)
    return save(fig, 'F12_feature_table')


# =============================================================================
# F13. PIXELS PER POLYGON, BY CLASS
# =============================================================================
def fig13(top=10):
    D = data()
    P = D.poly_df
    fig, ax = plt.subplots(figsize=(W2, 2.9))
    grid(ax, 'x')
    y = np.arange(len(CLASS_ORDER))[::-1]
    for yi, (code, name) in zip(y, [(c, CLASS_NAMES[c])
                                    for c in ALL_CLASSES]):
        px = P.loc[P['class'].eq(code), 'n_pixels'].sort_values(
            ascending=False).to_numpy()
        left, col = 0, CLASS_COLOR[name]
        for v in px[:top]:
            ax.barh(yi, v, left=left, color=col, edgecolor='white',
                    linewidth=0.5, height=0.62, zorder=3)
            left += v
        rest = px[top:].sum()
        ax.barh(yi, rest, left=left, color=col, alpha=0.35, hatch='///',
                edgecolor='white', linewidth=0.3, height=0.62, zorder=3)
        ax.annotate(f'{len(px)} polygons\n{px.sum():,} pixels',
                    (left + rest, yi), textcoords='offset points',
                    xytext=(5, 0), va='center', fontsize=6.8, color=INK)
    ax.set_yticks(y)
    ax.set_yticklabels([_short(n) for n in CLASS_ORDER])
    ax.set_xlabel('Pixels (10 m) in the modelling table')
    ax.set_title('Pixels per class, split by polygon', loc='left', pad=8)
    h = [plt.Rectangle((0, 0), 1, 1, fc=INK2, ec='white'),
         plt.Rectangle((0, 0), 1, 1, fc=INK2, alpha=0.35, hatch='///',
                       ec='white')]
    ax.legend(h, [f'Largest {top} polygons, one block each',
                  'All remaining polygons'], loc='lower right')
    ax.margins(x=0.16)
    return save(fig, 'F13_pixels_per_polygon')


# =============================================================================
# F14. CORRELATION BETWEEN THE FEATURES
# =============================================================================
def fig14():
    D = data()
    F = feature_frame()
    feats = F['feature'].tolist()
    X = D.capped(D.all_ids, cc.SEEDS[0], cc.FINAL_PIXEL_CAP)[feats]
    pr = X.corr(method='pearson')
    sp = X.corr(method='spearman')
    iu = np.triu_indices(len(feats), 1)
    mp, ms = np.abs(pr.to_numpy()[iu]).max(), np.abs(sp.to_numpy()[iu]).max()
    M = pr.to_numpy().copy()
    M[np.triu_indices(len(feats), 1)] = np.nan
    fig, ax = plt.subplots(figsize=(W2 * 0.8, W2 * 0.72))
    im = ax.imshow(M, cmap='RdBu_r', vmin=-1, vmax=1)
    for i in range(len(feats)):
        for j in range(i + 1):
            v = M[i, j]
            ax.text(j, i, f'{v:.2f}', ha='center', va='center', fontsize=5.6,
                    color='white' if abs(v) > 0.6 else INK)
    ax.set_xticks(range(len(feats)))
    ax.set_yticks(range(len(feats)))
    ax.set_xticklabels(feats, rotation=60, ha='right', fontsize=6.3)
    ax.set_yticklabels(feats, fontsize=6.3)
    for lab, f in zip(ax.get_yticklabels(), feats):
        lab.set_color(GROUP_COLOR.get(parse_feature(f)['group'], INK))
    for lab, f in zip(ax.get_xticklabels(), feats):
        lab.set_color(GROUP_COLOR.get(parse_feature(f)['group'], INK))
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cb.set_label('Pearson r')
    ax.set_title(f'Correlation between features. Largest |r|, Pearson {mp:.2f}'
                 f', Spearman {ms:.2f}', loc='left', pad=8)
    return save(fig, 'F14_feature_correlation')


# =============================================================================
# F15. THE FOUR REGIONS
# =============================================================================
def _region_label(c):
    return cc.REGION_NAME_HINT.get(c, f'R{c}')


def fig15():
    D = data()
    P = D.poly_df
    fig, ax = plt.subplots(figsize=(W1 * 1.25, 3.2))
    grid(ax, 'both')
    for c in sorted(P[CLUSTER_COL].unique()):
        s = P[P[CLUSTER_COL].eq(c)]
        ax.scatter(s['x'] / 1000, s['y'] / 1000, s=6,
                   color=REGION_COLOR.get(c, INK2), alpha=0.75,
                   linewidths=0, zorder=3,
                   label=f'{_region_label(c)} ({len(s)})')
        cx, cy = s['x'].mean() / 1000, s['y'].mean() / 1000
        ax.scatter(cx, cy, marker='X', s=45, color=INK, zorder=4)
        ax.annotate(_region_label(c), (cx, cy), textcoords='offset points',
                    xytext=(7, 3), fontsize=7.5, fontweight='bold')
    ax.set_xlabel('Easting (km)')
    ax.set_ylabel('Northing (km)')
    ax.set_aspect('equal', adjustable='datalim')
    ax.set_title('Reference polygons in the four regions', loc='left', pad=8)
    ax.legend(loc='upper left', title='Region (polygons)', fontsize=6.8,
              title_fontsize=7, markerscale=2)
    return save(fig, 'F15_regions')



# =============================================================================
# F16. CLASS MIX IN EACH REGION
# =============================================================================
def fig16():
    D = data()
    P = D.poly_df
    regs = sorted(P[CLUSTER_COL].unique())
    fig, axes = plt.subplots(1, len(regs), figsize=(W2, 2.3))
    for ax, c in zip(np.atleast_1d(axes), regs):
        s = P[P[CLUSTER_COL].eq(c)]['class'].value_counts().reindex(
            ALL_CLASSES, fill_value=0)
        wedges, _ = ax.pie(s.values,
                           colors=[CLASS_COLOR[CLASS_NAMES[k]] for k in
                                   ALL_CLASSES],
                           startangle=90, counterclock=False,
                           wedgeprops=dict(edgecolor='white', linewidth=1))
        for w, v, k in zip(wedges, s.values, ALL_CLASSES):
            if v == 0:
                continue
            a = np.deg2rad((w.theta1 + w.theta2) / 2)
            ax.text(0.66 * np.cos(a), 0.66 * np.sin(a), str(v),
                    ha='center', va='center', fontsize=7,
                    color='white' if CLASS_COLOR[CLASS_NAMES[k]] in _DARK
                    else INK)
        ax.set_title(f'{_region_label(c)}\n{int(s.sum())} polygons',
                     fontsize=8)
    h = [plt.Rectangle((0, 0), 1, 1, fc=CLASS_COLOR[n]) for n in CLASS_ORDER]
    fig.legend(h, [_short(n) for n in CLASS_ORDER], loc='lower center',
               ncol=5, bbox_to_anchor=(0.5, -0.06))
    fig.suptitle('Reference polygons by class and region', x=0.01,
                 ha='left', fontsize=9, y=1.04)
    return save(fig, 'F16_class_mix_by_region')


# =============================================================================
# F17. WHERE THE STUDY IS. Needs geopandas and one download, cached after.
# =============================================================================
WORLD_URLS = [
    'https://raw.githubusercontent.com/datasets/geo-countries/master/'
    'data/countries.geojson',
    'https://raw.githubusercontent.com/holtzy/The-Python-Graph-Gallery/'
    'main/static/data/world.geojson']


def _world():
    try:
        import geopandas as gpd
    except ImportError:
        raise FileNotFoundError('geopandas is not installed')
    cache = os.path.join(str(ROOT), '_cache', 'countries.geojson')
    if os.path.exists(cache):
        return gpd.read_file(cache)
    last = None
    for url in WORLD_URLS:
        try:
            w = gpd.read_file(url)
            os.makedirs(os.path.dirname(cache), exist_ok=True)
            w.to_file(cache, driver='GeoJSON')
            return w
        except Exception as exc:
            last = exc
    raise FileNotFoundError(f'country outlines could not be downloaded '
                            f'({last}). Put a countries GeoJSON at {cache}.')


def fig17():
    from shapely.geometry import box
    W = _world()
    namecol = next((c for c in W.columns
                    if c.lower() in ('admin', 'name', 'name_en', 'country')),
                   None)
    if namecol is None:
        raise ValueError('no country name column in the outlines file')
    col = W[W[namecol].astype(str).str.lower().eq('colombia')]
    sa = W[W.intersects(box(-95, -60, -30, 15))]
    D = data()
    P = D.poly_df
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(W2, 3.6),
                                 gridspec_kw={'width_ratios': [1, 1.25]})
    sa.plot(ax=a1, color='#d9d9d6', edgecolor='white', linewidth=0.4)
    col.plot(ax=a1, color=C['orange'], edgecolor=INK, linewidth=0.6)
    a1.set_xlim(-92, -32)
    a1.set_ylim(-57, 14)
    a1.set_title('Colombia in South America', loc='left', fontsize=8.5)
    col.plot(ax=a2, color='#f1f1ee', edgecolor=INK2, linewidth=0.6)
    for c in sorted(P[CLUSTER_COL].unique()):
        s = P[P[CLUSTER_COL].eq(c)]
        a2.scatter(s['lon'], s['lat'], s=5, color=REGION_COLOR.get(c, INK2),
                   linewidths=0, zorder=3, label=f'{_region_label(c)}')
    a2.set_title('Reference polygons by region', loc='left', fontsize=8.5)
    a2.legend(loc='lower left', markerscale=2.5, fontsize=6.8)
    for a in (a1, a2):
        a.set_xticks([])
        a.set_yticks([])
        for s in a.spines.values():
            s.set_visible(False)
    return save(fig, 'F17_study_area')


# =============================================================================
# F18. THE GLOBAL MODEL, CONFUSION AT PIXEL AND POLYGON LEVEL, EVERY SEED
# =============================================================================
def _cm_panel(ax, M, title):
    row = M.sum(axis=1, keepdims=True)
    pct = 100 * M / np.where(row == 0, 1, row)
    im = ax.imshow(pct, cmap='Blues', vmin=0, vmax=100)
    for r in range(M.shape[0]):
        for c in range(M.shape[1]):
            ax.text(c, r, f'{pct[r, c]:.1f}\nn={int(M[r, c]):,}',
                    ha='center', va='center', fontsize=5.8,
                    color='white' if pct[r, c] > 55 else INK)
    lab = [_short(CLASS_NAMES[k]) for k in ALL_CLASSES]
    ax.set_xticks(range(len(lab)))
    ax.set_yticks(range(len(lab)))
    ax.set_xticklabels(lab, rotation=40, ha='right')
    ax.set_yticklabels(lab)
    ax.set_xlabel('Predicted')
    oa = np.trace(M) / M.sum()
    ax.set_title(f'{title}\noverall accuracy {oa:.3f}, '
                 f'n = {int(M.sum()):,}', loc='left', fontsize=8.3)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    return im


def fig18():
    nest = os.path.join(str(ROOT), NESTED_TAG)
    cm_px = read(os.path.join(nest, f'T6d_{NESTED_TAG}_pooled_pixel_'
                                    'confusion.csv'),
                 'pooled pixel confusion (rerun script 5)').set_index('true')
    P = read(os.path.join(nest, f'P_{NESTED_TAG}_polygon_predictions.csv'),
             'per-polygon predictions (script 5)')
    P = P[P['design'].eq('pooled')]
    cm_pg = pd.crosstab(P['y_true'], P['y_pred']).reindex(
        index=ALL_CLASSES, columns=ALL_CLASSES, fill_value=0).to_numpy(float)
    fig, axes = plt.subplots(1, 2, figsize=(W2, 3.5))
    _cm_panel(axes[0], cm_px.to_numpy(float), 'Pixels')
    im = _cm_panel(axes[1], cm_pg,
                   f'Polygons, majority vote ({P["seed"].nunique()} splits)')
    axes[0].set_ylabel('Reference')
    cb = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02)
    cb.set_label('Percent of reference class')
    fig.suptitle('Global model confusion, every random split summed. Rows '
                 'sum to 100.', x=0.01, ha='left', fontsize=9, y=1.03)
    return save(fig, 'F18_global_confusion')


# =============================================================================
# F19. TRANSFER TO A REGION THE MODEL HAS NOT SEEN, PER CLASS
# =============================================================================
def fig19():
    t5 = read(os.path.join(str(ROOT), NESTED_TAG,
                           f'T5_{NESTED_TAG}_evaluation.csv'),
              'evaluation table (script 5)')
    L = t5[t5['design'].eq('loro_full')].copy()
    regs = sorted(L['region'].unique())
    fig, axes = plt.subplots(1, 2, figsize=(W2, 2.9), sharey=True)
    oas = {}
    for ax, (pre, title) in zip(axes, [('', 'Pixels'),
                                       ('poly_', 'Polygons')]):
        cols = [f'{pre}f1_{n}' for n in CLASS_ORDER]
        mu = L.groupby('region')[cols].mean().reindex(regs)
        sd = L.groupby('region')[cols].std(ddof=1).reindex(regs)
        oa = L.groupby('region')[f'{pre}oa'].agg(['mean', 'std']).reindex(
            regs)
        im = ax.imshow(mu.to_numpy(), cmap='Blues', vmin=0, vmax=1,
                       aspect='auto')
        for i in range(len(regs)):
            for j in range(len(cols)):
                v = mu.iloc[i, j]
                ax.text(j, i, f'{v:.2f}\n({sd.iloc[i, j]:.2f})',
                        ha='center', va='center', fontsize=6.5,
                        color='white' if v > 0.6 else INK)
        ax.set_xticks(range(len(cols)))
        ax.set_xticklabels([_short(n) for n in CLASS_ORDER], rotation=35,
                           ha='right')
        oas[pre] = oa
        ax.set_title(title, loc='left', fontsize=8.5)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.tick_params(length=0)
    axes[0].set_yticks(range(len(regs)))
    axes[0].set_yticklabels(
        [f'{r.split("(")[-1].rstrip(")")}\nOA {oas[""].loc[r, "mean"]:.2f} '
         f'px, {oas["poly_"].loc[r, "mean"]:.2f} poly' for r in regs],
        fontsize=6.8)
    axes[0].set_ylabel('Held-out region')
    cb = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02)
    cb.set_label('F1')
    fig.suptitle(f'Per-class F1 when each region is held out, mean (SD) '
                 f'over {L["seed"].nunique()} seeds', x=0.01, ha='left',
                 fontsize=9, y=1.03)
    return save(fig, 'F19_loro_f1_by_region')


# =============================================================================
# F20 TO F22. FEATURE IMPORTANCE ACROSS HELD-OUT REGIONS (script 5b)
# =============================================================================
OBJ_LABEL = {'5-class': '5-class objective (macro F1)',
             'shade coffee': 'Shade coffee objective (shade F1)'}


def _importance():
    base = os.path.join(str(ROOT), TAG_IMPORTANCE)
    I2 = read(os.path.join(base, f'I2_{TAG_IMPORTANCE}_by_region.csv'),
              'importance by region (script 5b)')
    I3 = read(os.path.join(base, f'I3_{TAG_IMPORTANCE}_summary.csv'),
              'importance summary (script 5b)')
    return I2, I3


def _feature_order(I3):
    """Features grouped Spectral, Structural, Topographic, and by mean
    5-class importance inside each group."""
    F = feature_frame()
    imp = (I3[I3['objective'].eq('5-class')]
           .set_index('feature')['mean_importance'])
    F['imp'] = F['feature'].map(imp).fillna(0)
    F['g'] = F['group'].map({g: i for i, g in enumerate(GROUP_ORDER)})
    F = F.sort_values(['g', 'imp'], ascending=[True, False])
    return F['feature'].tolist(), F.set_index('feature')['group']


def _group_brackets(ax, feats, groups, x=None):
    """Colour each feature name by its group, and rule between groups."""
    prev = None
    for i, f in enumerate(feats):
        g = groups[f]
        if prev is not None and g != prev:
            ax.axhline(i - 0.5, color=INK, linewidth=0.8)
        prev = g
    for lab in ax.get_yticklabels():
        lab.set_color(GROUP_COLOR.get(groups.get(lab.get_text(), ''), INK))


def _group_key(fig, y=0.0):
    h = [plt.Rectangle((0, 0), 1, 1, fc=GROUP_COLOR[g]) for g in GROUP_ORDER]
    fig.legend(h, GROUP_ORDER, loc='lower left', ncol=3, fontsize=6.8,
               bbox_to_anchor=(0.0, y), title='Feature group',
               title_fontsize=6.8, handlelength=1, frameon=False)


def fig20():
    I2, I3 = _importance()
    feats, groups = _feature_order(I3)
    regs = sorted(I2['region'].unique())
    vmax = I2['importance'].max()
    vmin = min(0, I2['importance'].min())
    fig, axes = plt.subplots(1, 2, figsize=(W2, 0.27 * len(feats) + 1.2),
                             sharey=True)
    for ax, obj in zip(axes, ['5-class', 'shade coffee']):
        W = (I2[I2['objective'].eq(obj)]
             .pivot(index='feature', columns='region', values='importance')
             .reindex(index=feats, columns=regs))
        cv = (I3[I3['objective'].eq(obj)].set_index('feature')
              ['cv_across_regions'].reindex(feats))
        im = ax.imshow(W.to_numpy(), cmap='viridis', vmin=vmin, vmax=vmax,
                       aspect='auto')
        for i in range(len(feats)):
            for j in range(len(regs)):
                v = W.iloc[i, j]
                ax.text(j, i, f'{v:.3f}', ha='center', va='center',
                        fontsize=5.8,
                        color=INK if v > 0.6 * vmax else 'white')
            c = cv.iloc[i]
            ax.text(len(regs) - 0.35, i, '-' if pd.isna(c) else f'{c:.2f}',
                    va='center', fontsize=6.3,
                    color=INK2 if pd.isna(c) else
                    (C['aqua'] if c < 0.5 else C['orange']))
        ax.text(len(regs) - 0.35, -0.9, 'CV', fontsize=6.8,
                fontweight='bold')
        ax.set_xticks(range(len(regs)))
        ax.set_xticklabels([r.split('(')[-1].rstrip(')') for r in regs],
                           rotation=35, ha='right')
        ax.set_yticks(range(len(feats)))
        ax.set_yticklabels(feats, fontsize=6.5)
        ax.set_xlabel('Held-out region')
        ax.set_title(OBJ_LABEL[obj], loc='left', fontsize=8.3, pad=14)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.tick_params(length=0)
        ax.set_xlim(-0.5, len(regs) + 0.3)
    for a in axes:
        _group_brackets(a, feats, groups)
    cb = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.07)
    cb.set_label('Permutation importance, mean over seeds')
    _group_key(fig, -0.13)
    fig.suptitle('Feature importance in each held-out region', x=0.01,
                 ha='left', fontsize=9, y=1.0)
    return save(fig, 'F20_importance_by_region')


def fig21():
    I2, I3 = _importance()
    feats, groups = _feature_order(I3)
    regs = sorted(I2['region'].unique())
    fig, axes = plt.subplots(1, 2, figsize=(W2, 0.27 * len(feats) + 1.0),
                             sharey=True)
    n = len(feats)
    for ax, obj in zip(axes, ['5-class', 'shade coffee']):
        R = (I2[I2['objective'].eq(obj)]
             .pivot(index='feature', columns='region', values='rank')
             .reindex(index=feats, columns=regs))
        rho = I3[I3['objective'].eq(obj)]['cross_region_rank_rho'].iloc[0]
        im = ax.imshow(R.to_numpy(), cmap='viridis_r', vmin=1, vmax=n,
                       aspect='auto')
        for i in range(n):
            for j in range(len(regs)):
                v = R.iloc[i, j]
                ax.text(j, i, f'{int(v)}', ha='center', va='center',
                        fontsize=6.3, color='white' if v > n * 0.45 else INK)
        ax.set_xticks(range(len(regs)))
        ax.set_xticklabels([r.split('(')[-1].rstrip(')') for r in regs],
                           rotation=35, ha='right')
        ax.set_yticks(range(n))
        ax.set_yticklabels(feats, fontsize=6.5)
        ax.set_xlabel('Held-out region')
        ax.set_title(f'{OBJ_LABEL[obj]}\nmean Spearman rho between regions '
                     f'{rho:.2f}', loc='left', fontsize=8)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.tick_params(length=0)
    for a in axes:
        _group_brackets(a, feats, groups)
    _group_key(fig, -0.14)
    cb = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02)
    cb.set_label('Rank, 1 = most important')
    cb.ax.invert_yaxis()
    fig.suptitle('Does the importance ranking hold across regions?',
                 x=0.01, ha='left', fontsize=9, y=1.02)
    return save(fig, 'F21_importance_rank_stability')


def fig22():
    I2, I3 = _importance()
    feats, groups = _feature_order(I3)
    fig, ax = plt.subplots(figsize=(W1 * 1.35, 0.27 * len(feats) + 0.9))
    grid(ax, 'x')
    y = np.arange(len(feats))
    off = {'5-class': -0.15, 'shade coffee': 0.15}
    sty = {'5-class': dict(color=C['blue'], marker='o'),
           'shade coffee': dict(color=C['magenta'], marker='D')}
    for obj in ('5-class', 'shade coffee'):
        W = (I2[I2['objective'].eq(obj)]
             .pivot(index='feature', columns='region', values='importance')
             .reindex(feats))
        m = W.mean(axis=1)
        ax.hlines(y + off[obj], W.min(axis=1), W.max(axis=1),
                  color=sty[obj]['color'], linewidth=1.0, alpha=0.6,
                  zorder=2)
        ax.plot(m, y + off[obj], linestyle='', markersize=5, zorder=3,
                label=OBJ_LABEL[obj], **sty[obj])
    ax.axvline(0, color=INK2, linewidth=0.6)
    ax.set_yticks(y)
    ax.set_yticklabels(feats, fontsize=6.5)
    ax.invert_yaxis()
    _group_brackets(ax, feats, groups)
    _group_key(fig, -0.1)
    ax.set_xlabel('Permutation importance, mean over held-out regions\n'
                  '(line spans the lowest to highest region)')
    ax.set_title('Importance per feature, 5-class against shade coffee',
                 loc='left', pad=8)
    ax.legend(loc='lower right', fontsize=6.5)
    return save(fig, 'F22_importance_two_objectives')


# =============================================================================
# F23. MAPPED CLASS AREA BY REGION. Needs a table exported from the map.
# =============================================================================
# The wall-to-wall map is made in Earth Engine, not here, so its areas cannot
# be recomputed from the pixel table. Export them to ANALYSIS_DIR as
#     region_class_area_km2.csv
# with columns region, class, area_km2, and optionally total_km2 per region
# (the region's full area, so unclassified area shows as Missing).
AREA_CSV = 'region_class_area_km2.csv'


def fig23():
    p = os.path.join(str(ROOT), AREA_CSV)
    A = read(p, f'{AREA_CSV}, exported from the classified map')
    W = A.pivot_table(index='region', columns='class', values='area_km2',
                      aggfunc='sum').reindex(columns=CLASS_ORDER).fillna(0)
    if 'total_km2' in A.columns:
        tot = A.groupby('region')['total_km2'].first()
        W['Missing'] = (tot - W.sum(axis=1)).clip(lower=0)
    tot = W.sum(axis=1)
    W = W.loc[tot.sort_values().index]
    pct = W.div(W.sum(axis=1), axis=0) * 100
    colors = {**CLASS_COLOR, 'Missing': '#000000'}
    fig, ax = plt.subplots(figsize=(W2, 0.55 * len(W) + 1.1))
    left = np.zeros(len(W))
    y = np.arange(len(W))
    for c in W.columns:
        ax.barh(y, pct[c], left=left, color=colors.get(c, INK2),
                edgecolor='white', linewidth=0.5, height=0.62, zorder=3,
                label=_short(c) if c in CLASS_ORDER else c)
        for i in range(len(W)):
            if pct[c].iloc[i] >= 7:
                ax.text(left[i] + pct[c].iloc[i] / 2, y[i],
                        f'{pct[c].iloc[i]:.0f}%\n{W[c].iloc[i]:,.0f}',
                        ha='center', va='center', fontsize=6.3,
                        color='white' if c in ('Shade Coffee', 'Forest',
                                               'Non-Shade Coffee', 'Urban',
                                               'Missing') else INK)
        left += pct[c].to_numpy()
    for i, t in enumerate(tot.loc[W.index]):
        ax.text(101, y[i], f'{t:,.0f} km$^2$', va='center', fontsize=7)
    ax.set_yticks(y)
    ax.set_yticklabels(W.index)
    ax.set_xlim(0, 115)
    ax.set_xlabel('Percent of mapped area (labels give km$^2$)')
    ax.set_title('Mapped class area by region', loc='left', pad=8)
    ax.legend(loc='upper center', bbox_to_anchor=(0.45, -0.25), ncol=6,
              fontsize=6.8)
    return save(fig, 'F23_mapped_area_by_region')


TABLES = [
    ('T1  four designs, per class and region, pixel and polygon',
     table_pooled_vs_local),
    ('T2  distance sweep, pixel and polygon', table_distance_levels),
    ('T3  feature table', table_features),
]


# =============================================================================
# RUN
# =============================================================================
FIGURES = [
    ('F1  which polygons help', fig01),
    ('F2  accuracy vs distance', fig02),
    ('F3  two methods one number', fig03),
    ('F4  the two boundaries', fig04),
    ('F5  plot size by class', fig05),
    ('F6  learning curves', fig06),
    ('F7  confusion matrices', fig07),
    ('F8  four designs, pixel and polygon', fig08),
    ('F9  four designs by class', fig09),
    ('F10 distance, pixel and polygon', fig10),
    ('F11 feature hierarchy', fig11),
    ('F12 feature table', fig12),
    ('F13 pixels per polygon', fig13),
    ('F14 feature correlation', fig14),
    ('F15 regions', fig15),
    ('F16 class mix by region', fig16),
    ('F17 study area map', fig17),
    ('F18 global model confusion', fig18),
    ('F19 LORO F1 by region', fig19),
    ('F20 importance by region', fig20),
    ('F21 importance rank stability', fig21),
    ('F22 importance, two objectives', fig22),
    ('F23 mapped area by region', fig23),
]

def captions():
    """Draft captions with the numbers filled in FROM THE TABLES, so they
    cannot drift out of step with the figures the way a typed-in number can.
    These are a starting point. The wording and the claims are yours."""
    out = {}
    try:
        P = pd.read_csv(os.path.join(str(ROOT), NESTED_TAG,
                                     f'T6_{NESTED_TAG}_paired.csv'))
        g = P.groupby('design')[['macro_f1', 'n_train_poly']].mean()
        out['F1'] = (
            'Polygon macro F1 for four training designs scored on identical '
            'held-out polygons, averaged over '
            f'{P["seed"].nunique()} random splits. Training sets nest, so '
            'each contrast isolates one thing. '
            + '; '.join(f'{DESIGN_LABEL.get(d, d)} {g.loc[d, "macro_f1"]:.3f} '
                        f'on {g.loc[d, "n_train_poly"]:.0f} polygons'
                        for d in DESIGN_ORDER if d in g.index) + '.')
    except Exception:
        pass
    try:
        S = pd.read_csv(os.path.join(str(ROOT), DIST_TAG,
                                     f'T8_{DIST_TAG}_sweep.csv'))
        S = S[S['buffer_km'] >= 0]
        pv = S.pivot_table(index='buffer_km', columns='arm',
                           values='poly_macro_f1')
        if {'buffer', 'random'} <= set(pv.columns):
            db = pv['buffer'].iloc[0] - pv['buffer'].iloc[-1]
            dr = pv['random'].iloc[0] - pv['random'].iloc[-1]
            out['F2'] = (
                'Training polygons within the buffer are removed (blue). The '
                'control (green) removes the same number with the same class '
                'mix from any distance. Bars give the polygons removed at '
                'each step. Removing by distance costs '
                f'{db:.3f} macro F1 against {dr:.3f} for the control'
                + (f', a factor of {db / dr:.1f}' if dr > 0 else '')
                + '. The x axis is ordinal, not linear.')
        cur = (S[S['arm'] == 'buffer'].groupby('buffer_km')['poly_sc_nsc_f1']
               .mean())
        gp = os.path.join(str(ROOT), DIST_TAG, f'T10_{DIST_TAG}_grouped.csv')
        if os.path.exists(gp):
            gv = pd.read_csv(gp)['poly_sc_nsc_f1'].mean()
            plateau = cur[cur.index >= 1].mean()
            out['F3'] = (
                'Shade against sun coffee as training polygons are pushed '
                f'away. The curve starts at {cur.iloc[0]:.3f} and settles at '
                f'{plateau:.3f} beyond 1 km. The dashed line is an '
                'independent check in which polygons within 200 m sharing a '
                f'class are treated as one unit before splitting, at {gv:.3f}. '
                + ('The two methods agree to within 0.02. '
                   if abs(gv - plateau) < 0.02 else
                   f'The two methods differ by {abs(gv - plateau):.3f}, so '
                   'they bound the honest figure rather than fix it. ')
                + 'The x axis is ordinal, not linear.')
    except Exception:
        pass
    try:
        B = pd.read_csv(os.path.join(str(ROOT), NESTED_TAG,
                                     f'B5_{NESTED_TAG}_exchange_by_dist.csv'))
        B = B[(B['buffer_km'] >= 0) & (B['arm'] == 'buffer')].sort_values(
            'buffer_km')
        ds = B['pct_shade_lost'].iloc[-1] - B['pct_shade_lost'].iloc[0]
        dn = (B['pct_nsc_lost_to_open'].iloc[-1]
              - B['pct_nsc_lost_to_open'].iloc[0])
        if ds > 2 * max(dn, 0.1):
            cmp_ = ('Sun coffee called Open rises much less with distance '
                    'than shade coffee called forest.')
        elif ds > dn:
            cmp_ = ('Both confusions rise with distance, shade coffee called '
                    'forest somewhat more.')
        else:
            cmp_ = ('Sun coffee called Open rises at least as much with '
                    'distance as shade coffee called forest.')
        out['F4'] = (
            'Percent of each class assigned to the named other class, as '
            'training polygons are pushed away. Shade coffee called forest '
            f'moves from {B["pct_shade_lost"].iloc[0]:.1f} to '
            f'{B["pct_shade_lost"].iloc[-1]:.1f} percent ({ds:+.1f} points), '
            'while sun coffee called Open moves from '
            f'{B["pct_nsc_lost_to_open"].iloc[0]:.1f} to '
            f'{B["pct_nsc_lost_to_open"].iloc[-1]:.1f} percent '
            f'({dn:+.1f} points). ' + cmp_ + ' The x axis is ordinal, '
            'not linear.')
    except Exception:
        pass
    try:
        N = pd.read_csv(os.path.join(str(ROOT), NSC_TAG,
                                     f'N1_{NSC_TAG}_size_before.csv')
                        ).set_index('class')
        nsc = N.loc['Non-Shade Coffee']
        rest = N.drop('Non-Shade Coffee')
        out['F5'] = (
            'Plot size by class before any filtering. Sun coffee has a median '
            f'of {nsc["median_px_all"] * 100 / 1e4:.2f} ha against '
            f'{rest["median_px_all"].median() * 100 / 1e4:.2f} ha for the '
            f'other classes, and {nsc["pct_under_25px"]:.0f} percent of its '
            'plots fall under 0.25 ha. Urban is smaller still yet maps well, '
            'so size alone is not the constraint.')
    except Exception:
        pass
    try:
        Cv = pd.read_csv(os.path.join(str(ROOT), NSC_TAG, f'N2_{NSC_TAG}_curve.csv'))
        if 'incomplete' in Cv.columns:
            Cv = Cv[~Cv['incomplete'].astype(bool)]
        bits = []
        for f in ('Shade Coffee', 'Non-Shade Coffee'):
            s = Cv[Cv['focal'] == f]
            if not len(s):
                continue
            g = s.groupby('n_focal_train')[f'f1_{f}'].mean()
            sl = ((g.iloc[-1] - g.iloc[-2])
                  / (g.index[-1] - g.index[-2]) * 10)
            bits.append(f'{_short(f)} reaches {g.iloc[-1]:.3f} at '
                        f'{g.index[-1]} polygons, final slope {sl:.4f} per 10')
        if bits:
            out['F6'] = ('Each class\'s own F1 as its training polygons are '
                         'added, with every other class held fixed and the '
                         'test set unchanged. ' + '; '.join(bits) + '.')
    except Exception:
        pass
    out.setdefault('F7', (
        'Polygon confusion as row percentages, with and without training '
        'polygons from the target region. Rows are reference, columns are '
        'predicted.'))
    try:
        S = pd.read_csv(os.path.join(FIG_DIR,
                                     'T1a_global_local_loro_summary.csv'))
        m = S.set_index('class').loc['Macro F1']
        out['T1'] = (
            'F1 by class at pixel and polygon level for four training designs '
            'scored on the same held-out pixels and polygons of each region, '
            'averaged over regions and random splits. Global trains on every '
            'training polygon, Local on the target region only, Outside only '
            'on the other regions\' training polygons, and LORO on every '
            'polygon outside the target region. Polygon macro F1 is '
            f'{m["polygon_global"]:.3f} global, {m["polygon_local"]:.3f} '
            f'local, {m["polygon_outside"]:.3f} outside only and '
            f'{m["polygon_loro"]:.3f} LORO. Global minus local is '
            f'{m["polygon_global_minus_local"]:+.3f} (global higher in '
            f'{m["polygon_global_better_than_local"]}); global minus LORO is '
            f'{m["polygon_global_minus_loro"]:+.3f} (global higher in '
            f'{m["polygon_global_better_than_loro"]}). Magdalena holds few '
            'coffee test polygons.')
    except Exception:
        pass
    try:
        L, sizes, _ = design_scores()
        M = _cell_mean_ci(L[L['metric'].eq('macro_f1')]).set_index(
            ['level', 'model'])['mean']
        bits = '; '.join(
            f'{T1_MODEL_LABEL[m]} {M[("pixel", m)]:.3f} pixel, '
            f'{M[("polygon", m)]:.3f} polygon' for m in T1_MODELS)
        out['F8'] = (
            'Macro F1 for four training designs scored on the same held-out '
            'pixels and polygons of each region. Global trains on every '
            'training polygon, Local on the target region only, Outside only '
            'on the other regions\' training polygons, and LORO on every '
            'polygon of the other regions. Shaded designs hold no data from '
            'the target region. Bars are 95% intervals over region-splits. '
            + bits + '.')
        out['F9'] = (
            'F1 by class for the same four designs, at pixel and polygon '
            'level. Shaded designs hold no data from the target region. '
            'Bars are 95% intervals over region-splits.')
    except Exception:
        pass
    try:
        S = _sweep_levels()
        eff = {}
        for lv, pre in (('pixel', 'px_'), ('polygon', 'poly_')):
            pv = S.pivot_table(index='buffer_km', columns='arm',
                               values=pre + 'macro_f1')
            eff[lv] = (pv['random'] - pv['buffer']).iloc[-1]
        out['F10'] = (
            'Macro F1 against buffer distance at pixel and polygon level. '
            'Solid lines remove training polygons by distance, dashed lines '
            'remove the same number with the same class mix at random. At '
            f'the widest buffer the distance effect is {eff["pixel"]:.3f} at '
            f'pixel level and {eff["polygon"]:.3f} at polygon level. The x '
            'axis is ordinal, not linear.')
    except Exception:
        pass
    out['F11'] = ('The features in use, grouped as spectral, structural and '
                  'topographic, with texture separated from the base layers.')
    out['F12'] = ('Feature definitions. A texture suffix _kR is a '
                  '(2R+1) x (2R+1) pixel window at 10 m. Mean and variance '
                  'are neighbourhood statistics, contrast is GLCM contrast.')
    try:
        D = data()
        P = D.poly_df
        n = P.groupby('class').agg(p=('n_pixels', 'size'),
                                   px=('n_pixels', 'sum'))
        bits = '; '.join(f'{_short(CLASS_NAMES[k])} {int(n.loc[k, "p"])} '
                         f'polygons, {int(n.loc[k, "px"]):,} pixels'
                         for k in ALL_CLASSES if k in n.index)
        out['F13'] = ('Pixels in the modelling table by class, each block '
                      'one of the ten largest polygons, hatching the rest. '
                      + bits + '. Training uses at most '
                      f'{cc.FINAL_PIXEL_CAP} pixels per polygon, so large '
                      'polygons do not dominate the fit.')
        out['F15'] = ('Reference polygon centroids in projected coordinates, '
                      'coloured by the four k-means regions, crosses at '
                      'region centres. ' + ', '.join(
                          f'{_region_label(c)} {len(D.cluster_ids[c])}'
                          for c in D.clusters) + ' polygons.')
        out['F16'] = ('Reference polygons by class in each region.')
    except Exception:
        pass
    out['F14'] = ('Pearson correlation between the features on the training '
                  'pixel sample, 50 pixels per polygon. The title gives the '
                  'largest absolute Pearson and Spearman correlation.')
    out['F17'] = ('Study area. Left, Colombia in South America. Right, '
                  'reference polygon locations by region.')
    try:
        t5 = pd.read_csv(os.path.join(str(ROOT), NESTED_TAG,
                                      f'T5_{NESTED_TAG}_evaluation.csv'))
        L = t5[t5['design'].eq('loro_full')]
        out['F19'] = (
            'Per-class F1 when each region is held out, the model trained on '
            'every polygon of the other three. Mean (SD) over '
            f'{L["seed"].nunique()} seeds, pixel and polygon level. Mean '
            f'polygon macro F1 across held-out regions '
            f'{L.groupby("region")["poly_macro_f1"].mean().mean():.3f}.')
    except Exception:
        pass
    out['F18'] = ('Confusion of the global model on held-out data, summed '
                  'over every random split, as row percentages with counts. '
                  'Left, pixels. Right, polygons after majority vote. '
                  'Forest dominates the pixel counts, so read per-class rows.')
    try:
        I2, I3 = _importance()
        bits = []
        for obj in ('5-class', 'shade coffee'):
            s3 = I3[I3['objective'].eq(obj)].sort_values(
                'mean_importance', ascending=False).iloc[0]
            bits.append(f'{obj}, {s3["feature"]} '
                        f'({s3["mean_importance"]:.3f}, positive in '
                        f'{int(s3["positive_in_regions"])} of '
                        f'{I2["region"].nunique()} regions)')
        out['F20'] = ('Permutation importance of each feature in each '
                      'held-out region, under the 5-class and shade coffee '
                      'objectives. CV across regions is shown only where '
                      'mean importance is at least 0.005. Most important: '
                      + '; '.join(bits) + '.')
        out['F21'] = ('Importance rank of each feature in each held-out '
                      'region. The title gives the mean Spearman correlation '
                      'of ranks between regions.')
        out['F22'] = ('Mean permutation importance over held-out regions, '
                      'lines spanning the lowest to highest region, under '
                      'the two objectives.')
    except Exception:
        pass
    if os.path.exists(os.path.join(str(ROOT), AREA_CSV)):
        out['F23'] = ('Mapped area by class in each region, as percent of the '
                      'region and km2. Missing is area the map left '
                      'unclassified.')
    return out


def run_all(show_captions=True):
    """Make every figure. Call this from a notebook cell.

        from make_figures import run_all
        run_all()

    Or one at a time, which is what you want while fiddling with a single
    panel.

        import make_figures as mf
        mf.style()
        mf.fig04()

    To display without writing anything, set mf.SAVE_PNG = mf.SAVE_PDF =
    False first."""
    style()
    print('=' * 78)
    print('MAKING FIGURES')
    print('=' * 78)
    print(f'  into {FIG_DIR}')
    print(f'  displaying inline: {_SHOW}')
    ok, skipped = 0, []
    for label, fn in FIGURES:
        try:
            made = fn()
            ok += 1
            print(f'  [done] {label}')
            for p in made:
                print(f'           {os.path.basename(p)}')
        except FileNotFoundError as exc:
            skipped.append((label, str(exc)))
            print(f'  [skip] {label}')
            print(f'           {exc}')
        except Exception as exc:
            skipped.append((label, f'{type(exc).__name__}: {exc}'))
            print(f'  [FAIL] {label}')
            print(f'           {type(exc).__name__}: {exc}')
    print(f'\n  {ok} of {len(FIGURES)} figures written')
    for label, fn in TABLES:
        try:
            made = fn()
            print(f'\n  [done] {label}')
            for p in made:
                print(f'           {os.path.basename(p)}')
        except FileNotFoundError as exc:
            skipped.append((label, str(exc)))
            print(f'  [skip] {label}')
            print(f'           {exc}')
        except Exception as exc:
            skipped.append((label, f'{type(exc).__name__}: {exc}'))
            print(f'  [FAIL] {label}')
            print(f'           {type(exc).__name__}: {exc}')
    if skipped:
        print('\n  Not made, and why')
        for label, why in skipped:
            print(f'    {label}')
            print(f'      {why}')
        print('\n  A skip usually means that script has not been run yet.')
    if show_captions:
        _print_captions()
    return ok


def _cap_key(k):
    """F2 before F10, figures before tables."""
    return (k[0] != 'F', int(''.join(ch for ch in k if ch.isdigit()) or 0))


def _print_captions():
    caps = captions()
    if not caps:
        return
    print('\n' + '=' * 78)
    print('DRAFT CAPTIONS, NUMBERS READ FROM THE TABLES')
    print('=' * 78)
    print('  Titles on the figures say what is shown. The claim belongs')
    print('  in the caption, which is yours to check and to word. These')
    print('  are a starting point with the numbers already filled in, so')
    print('  they cannot drift out of step the way a typed number can.')
    for k in sorted(caps, key=_cap_key):
        print(f'\n  {k}. {caps[k]}')
    cp = os.path.join(FIG_DIR, 'captions_draft.txt')
    with open(cp, 'w') as fh:
        for k in sorted(caps, key=_cap_key):
            fh.write(f'{k}. {caps[k]}\n\n')
    print(f'\n  also written to {cp}')


def show_saved(which=None, width=760):
    """Display the PNGs already on disk, in a notebook.

    This is the reliable route. It reads the files rather than re-drawing,
    so it does not care which matplotlib backend is active, and it works
    after a plain `%run make_figures.py` where the figures were written but
    never shown.

        import make_figures as mf
        mf.show_saved()            # all of them
        mf.show_saved('F4')        # just one
        mf.show_saved(['F2','F5']) # a couple

    Returns the paths it displayed, so it also works as a check that the
    files are where you think they are."""
    try:
        from IPython.display import display, Image, Markdown
    except ImportError:
        print('  Not in a notebook. The PNGs are in:')
        print(f'    {FIG_DIR}')
        return []
    if isinstance(which, str):
        which = [which]
    names = [n for n, _ in FIGURES]
    keys = [n.split()[0] for n in names]
    pngs = sorted(p for p in os.listdir(FIG_DIR) if p.endswith('.png'))
    if which:
        want = {w.upper() for w in which}
        pngs = [p for p in pngs if p.split('_')[0].upper() in want]
        missing = want - {p.split('_')[0].upper() for p in pngs}
        for m in sorted(missing):
            print(f'  {m} has no PNG in {FIG_DIR}')
    if not pngs:
        if which:
            print('  Nothing matched. Names are F1 to F23.')
        else:
            print(f'  No figures found in {FIG_DIR}. Run run_all() first.')
        return []
    caps = captions()
    shown = []
    for p in pngs:
        key = p.split('_')[0].upper()
        display(Markdown(f'**{p}**'))
        display(Image(filename=os.path.join(FIG_DIR, p), width=width))
        if key in caps:
            display(Markdown(f'*{caps[key]}*'))
        shown.append(os.path.join(FIG_DIR, p))
    return shown


if __name__ == '__main__':
    # Run as a script. In a notebook, import run_all instead so the figures
    # display as well as being written, or call show_saved() afterwards.
    run_all()