"""QUESTION-BY-QUESTION FIGURES. Reads the saved tables, fits nothing.

=============================================================================
MAKE FIGURES. Reads the saved tables, fits nothing, writes the paper's plots.
=============================================================================
Run this after scripts 4 to 8. It touches no model and no pixel data, so it is
fast, safe to rerun, and every figure traces back to a CSV you can open.

Seven figures, one per claim the paper makes.

  F1  Which reference polygons help          Q1
  F2  Accuracy against distance, with the matched control   Q2
  F3  Two methods, one honest number         Q2
  F4  The two boundaries behave differently  Q3
  F5  Plot size by class, the mechanism      Q3
  F6  Learning curves, two ceilings          Q3
  F7  Confusion, with and without local data  supporting

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
                           TAG_DESIGNS, TAG_DISTANCE, TAG_NSC)

ROOT = cc.ANALYSIS_DIR
NESTED_TAG, DIST_TAG, NSC_TAG = TAG_DESIGNS, TAG_DISTANCE, TAG_NSC
FIG_DIR = str(cc.out_dir('figures'))

SAVE_PDF = True
SAVE_PNG = True
DPI = 300

# Journal column widths in inches.
W1, W2 = 3.5, 7.2

CLASS_ORDER = [CLASS_NAMES[c] for c in ALL_CLASSES]
DESIGN_ORDER = ['pooled', 'local', 'loro_out', 'loro_all', 'loro_full']
DESIGN_LABEL = {'pooled': 'Pooled', 'local': 'Local only',
                'loro_out': 'Outside only', 'loro_all': 'All outside',
                'loro_full': 'Full transfer'}

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
                'Two different methods, the same number. The x axis is '
                'ordinal, not linear.')
    except Exception:
        pass
    try:
        B = pd.read_csv(os.path.join(str(ROOT), NESTED_TAG,
                                     f'B5_{NESTED_TAG}_exchange_by_dist.csv'))
        B = B[(B['buffer_km'] >= 0) & (B['arm'] == 'buffer')].sort_values(
            'buffer_km')
        out['F4'] = (
            'Percent of each class assigned to the named other class, as '
            'training polygons are pushed away. Shade coffee called forest '
            f'moves from {B["pct_shade_lost"].iloc[0]:.1f} to '
            f'{B["pct_shade_lost"].iloc[-1]:.1f} percent, while sun coffee '
            f'called Open moves from {B["pct_nsc_lost_to_open"].iloc[0]:.1f} '
            f'to {B["pct_nsc_lost_to_open"].iloc[-1]:.1f} percent. Sun coffee called Open depends much less '
            'on distance than shade coffee called forest, although the '
            'distance effect varies across buffers. The x axis '
            'is ordinal, not linear.')
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
    if skipped:
        print('\n  Not made, and why')
        for label, why in skipped:
            print(f'    {label}')
            print(f'      {why}')
        print('\n  A skip usually means that script has not been run yet.')
    if show_captions:
        _print_captions()
    return ok


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
    for k in sorted(caps):
        print(f'\n  {k}. {caps[k]}')
    cp = os.path.join(FIG_DIR, 'captions_draft.txt')
    with open(cp, 'w') as fh:
        for k in sorted(caps):
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
            print('  Nothing matched. Names are F1 to F7.')
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
