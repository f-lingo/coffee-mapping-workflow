# Shade coffee agroforestry classification, Colombia

Spatially honest accuracy assessment for Sentinel-2 classification of shade
coffee agroforestry across four Colombian departments, from 923 field
reference polygons.

The headline result is not the map. It is that the accuracy usually reported
for this kind of map is about 0.10 too high, that the target region's own
reference data matters far more than data from anywhere else, and that the
two coffee boundaries fail for different reasons.

## Pipeline

Run in order. Scripts 6, 7 and 8 refuse to continue unless they reproduce
script 5's splits exactly, so the chain cannot silently drift apart.

| Script | What it answers | Fits | Time |
|---|---|---|---|
| `1_gee.py` | Build the pixel table from Earth Engine | none | hours |
| `2_clean_define_annual.py` | Drop seasonal columns, keep annual | none | seconds |
| `selection_procedure.py` | **Where the 15 features came from.** The nested feature-selection experiment, see Feature selection below. | many | ~40 min |
| `3_select_features.py` | A later check on the feature list. Correlation filter, importance under leave-one-cluster-out, and a performance curve against feature count. It does not reproduce the list. | ~40 | ~15 min |
| `4_baseline_model.py` | **The model itself.** Pixel against polygon, hard against soft vote, per-region classifiers, confusion matrices. | 50 | ~10 min |
| `5_which_polygons_help.py` | **Q1.** Does it matter where reference polygons come from? | 130 | ~15 min |
| `5b_feature_importance.py` | Which features carry the model into a region it has not seen. | 40 | ~30 min |
| `6_how_far_does_it_reach.py` | **Q2.** How accurate is it really, and how far does a training polygon reach? | 480 | ~45 min |
| `7_which_boundary_fails.py` | **Q3a.** Which boundary fails, in which direction, does distance matter? | 0 | seconds |
| `8_is_sun_coffee_fixable.py` | **Q3b.** Is sun coffee fixable without merging the classes? | 90 | ~15 min |
| `9_figures.py` | 23 figures, 3 tables and draft captions | 0 | ~1 min |

Scripts 7 and 9 fit nothing, so they are cheap to rerun while drafting.
Script 9 reads the pixel table once for the data-description figures, which
is why it takes about a minute rather than seconds.

Script 4 is the map. Everything after it is a diagnostic explaining why that
map has the weaknesses it has.

**One warning about script 4.** Its polygon accuracy is the optimistic one.
It uses an ordinary stratified holdout, so training polygons sit a median of
208 m from test polygons and the score carries spatial inflation. Quote
script 4 for the confusion structure, the pixel against polygon comparison
and the per-region spread. Quote script 6 for accuracy. The script prints
this warning itself so it cannot be missed.

`coffee_common.py` holds the feature list, the class codes, the region names,
the pixel sampler, the model helpers, the split machinery and the output
tags. It exists because separate copies of those constants drifted twice,
and both times silently. One script still held an older experiment while
later scripts read its output, and the region names disagreed so one script
labelled a cluster Magdalena while the next called it Santander. Neither
raised an error.

## Setup

```bash
pip install -r requirements.txt
cp gee_config.example.py gee_local_config.py   # then edit it
```

`gee_local_config.py` holds only settings that depend on where your files
are. Everything that is method, the seeds, tree counts, pixel caps and buffer
ladders, lives in the script that uses it, because those belong with the
experiment rather than with your machine.

In Jupyter, restart the kernel after changing `coffee_common.py`. Python
imports it once per session, so `%run` keeps the old copy otherwise.

## The four training designs

Scripts 5 and 9 compare four ways of training a model for one region. All
four are scored on the same held-out pixels and polygons of that region, for
every seed, so every difference is paired.

| Design | Trains on | Mean training polygons |
|---|---|---|
| Global | every training polygon, all four regions | 691 |
| Local | the target region's training polygons only | 173 |
| Outside only | the other three regions' training polygons | 518 |
| LORO | every polygon of the other three regions | 692 |

Global and LORO train on almost the same number of polygons. The only
difference between them is whether the target region is included.

## The three answers

**Q1. The target region's own data matters, other regions add little.**
Polygon macro F1 is 0.720 for Global, 0.703 for Local, 0.590 for Outside only
and 0.596 for LORO.

| Paired difference, polygon macro F1 | Mean | First higher in |
|---|---|---|
| Global minus LORO, what the region's own data adds | +0.125 | 37 of 40 |
| Global minus Local, what the other regions add | +0.017 | 25 of 40 |
| LORO minus Outside only, what more distant data adds | +0.005 | 18 of 40 |

At about equal training size, leaving out the target region costs about 0.12
macro F1, and does so in 37 of 40 region-seed pairs. Adding the other three
regions to the local data gives a small gain that is not consistent across
pairs, and adding 174 more distant polygons changes nothing. At pixel level
the same three contrasts are +0.069, +0.000 and +0.005. With four regions,
"no consistent gain" is as far as the evidence goes.

Two classes depart from this. Sun coffee gains most from the other regions,
+0.083 polygon F1 for Global over Local, though Global is higher in only 25
of 40 pairs. Forest goes the other way, Local 0.043 above Global.

**Q2. The usual accuracy number is about 0.10 too high.** Shade against sun
coffee scores 0.813 polygon F1 under a random holdout. Removing training
polygons near the test polygons brings it to 0.708 beyond 1 km, and a grouped
split that keeps near-duplicate polygons on one side gives 0.709. Two
independent methods agree. For macro F1, removing nearby polygons costs
0.125, against 0.013 when the same number is removed at random, so the loss
comes from losing nearby data, not from losing data. The gap is consistent
with reference polygons sitting in tight clumps, median nearest neighbour
208 m.

The distance effect on macro F1 keeps growing across the whole ladder, from
0.03 at 100 m to 0.07 at 1 km, 0.09 at 10 km and 0.11 at 50 km. It shows no
plateau within 50 km. It is larger at polygon level than at pixel level at
every distance past 50 m, by up to 0.04, which is consistent with the
majority vote amplifying the loss.

**Q3. Two failures, two causes.**

- *Shade coffee against forest is a distance problem.* Shade coffee called
  forest rises from 10.7% to 25.6% as training data is pushed away, +14.9
  points. The direction of the error flips between regions. Without local
  data, forest called shade coffee dominates in Cauca and Magdalena, and
  shade coffee called forest dominates in Cundinamarca and Santander, so no
  single global correction will fix it. Closer reference data is the likely
  fix.
- *Sun coffee against open ground is mainly a size problem.* Sun coffee
  called Open moves only from 25.4% to 27.1%, +1.7 points, so distance
  barely affects it. Sun coffee plots have a median area of 0.30 ha, and 46%
  of them are under 0.25 ha, so at 10 m only about 40% of their pixels are
  unmixed. The pair is separable, 0.81 F1, when it is the only decision
  being made, and falls to 0.55 inside the five-class problem. We propose
  that finer pixels are the fix. A resolution degradation experiment would
  test this directly.
- *More sun coffee reference data may still help.* Its F1 reaches 0.488 at
  142 training polygons and is still rising 0.012 per 10 polygons at the last
  step, about five times the shade coffee slope of 0.0025.

Urban is the control for the size argument. Its plots are small too, yet it
maps well. Small alone is survivable. Small *and* spectrally similar to its
surroundings is not.

## Feature importance

Script 5b measures permutation importance of the 15 features inside every
leave-one-region-out fold, on the held-out region only.

- For the 5-class objective, `B2_variance_k4` (0.144) and `B11` (0.140)
  lead, both positive in all four regions. `canopy_height_tolan_variance_k5`
  is smaller (0.039) but the most consistent across regions, with a CV of
  0.15.
- For shade coffee alone, `elevation` leads (0.100), then
  `canopy_height_tolan_variance_k5` (0.052), both positive in all four
  regions.
- Rankings agree only moderately between regions, mean Spearman rho 0.57
  for the 5-class objective and 0.64 for shade coffee.

This is consistent with canopy structure and topography carrying the shade
coffee signal into new regions. Correlated features share credit, so a low
score does not mean a feature is uninformative.

## Feature selection

The 15 features come from a nested selection experiment,
`selection_procedure.py`. In each leave-one-region-out fold it groups
features so that no pair exceeds |Pearson r| or |Spearman ρ| of 0.65, ranks
the groups by permutation importance for coffee F1, and lets inner
validation choose between six candidate sets. The list in use is the set
chosen in the fold that held out Santander, seed 93.

Two things about this are worth stating plainly.

- Other folds chose other lists. Across the four held-out regions the
  candidate sets score within about 0.025 coffee F1 of each other, so the
  results are unlikely to depend strongly on which list was used.
- Rerunning the procedure in a different software environment reproduced 14
  of the 15 features. The difference is a near tie inside one correlated
  group.

`3_select_features.py` is a separate, later check. It ranks by macro F1 and
does not reproduce the list, and it is not how the list was made.

## Method notes that matter

- **Polygon level, not pixel level.** Pixels are voted to a polygon label
  before scoring, because a polygon is the unit a map user acts on. Scripts
  5, 6 and 9 also report pixel level, because the two can disagree. Forest
  holds about 60% of all pixels, so pixel overall accuracy is mostly a
  forest score. Read per-class F1.
- **Training pixels capped at 50 per polygon, test polygons uncapped.** The
  per-polygon sample depends on the seed and the polygon alone, never on
  which polygons were requested, so every design sees identical pixels for
  any polygon they share.
- **Training sets nest by construction.** Global is exactly the union of the
  Local and Outside only sets, so each contrast isolates one variable.
- **Report the regional spread, not the seed interval.** Under
  leave-one-region-out, region to region SD is 0.044 for polygon macro F1
  against 0.004 across seeds. n is 4.
- **Forest size.** Every model uses the settings in `coffee_common.py`. The
  numbers above come from `FINAL_TREES = 1000`, max depth 30, balanced class
  weights.
- **`cf_f1`, `sc_nsc_f1` and `nsc_vs_open_f1` are restricted metrics.** They
  are scored only on polygons whose truth is one of the two classes named, so
  they exclude false positives arriving from the other three. Report them
  beside the one-vs-rest F1 scores, never instead of them.
- **Coordinates.** Imagery is processed and sampled on a 10 m grid in the
  projected CRS set by `TARGET_CRS`. Polygon coordinates and all distances
  use the `easting` and `northing` columns, in metres.
- **Software versions change the numbers slightly.** Seeded random forests
  are exactly reproducible only within one set of library versions. Across
  environments, scores moved in the second decimal place and one feature
  swapped in the selection experiment. Pin versions before publishing
  numbers.

## Known limitations

- Four regions. Every regional claim rests on n = 4.
- Magdalena carries about 8 coffee test polygons per split, as few as 5. It
  is excluded from per-region claims and the exclusion is stated wherever it
  applies.
- Sun coffee is the weakest class. Its polygon F1 is 0.49 under a random
  holdout, about 0.42 once nearby training polygons are removed, and 0.35
  under the grouped split. It is reported at those values rather than merged
  away.
- The scale mechanism is a proposed explanation. The plot-size measurement
  supports it. A resolution degradation experiment would test it directly and
  has not been run.
- The feature list is one fold's selection, as described above. Seasonal
  composites and Sentinel-1 were tested separately and did not help.
- Feature selection used data from every region, so leave-one-region-out
  scores in scripts 5 and 6 may carry a small optimistic bias. The outer
  folds of `selection_procedure.py` are the leak-free transfer estimate.

## Outputs

Everything lands under `ANALYSIS_DIR`, one folder per experiment tag plus
`figures/`. Tables are CSV and carry the numbers behind every claim above,
so any figure traces back to the table that produced it. `9_figures.py` also
writes `captions_draft.txt`, with the numbers read from the tables rather
than typed, so a caption cannot drift out of step with its figure.

Figure F23, mapped area by region, needs one table exported from the
classified map in Earth Engine, saved as
`ANALYSIS_DIR/region_class_area_km2.csv` with columns `region`, `class`,
`area_km2`, and optionally `total_km2`. Without it F23 is skipped.

## Data

The Sentinel-2 pixel table and the field reference polygons are not in this
repository. Reference polygon locations are field-collected and may need
aggregation before release.