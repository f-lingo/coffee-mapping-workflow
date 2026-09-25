# Shade coffee agroforestry classification, Colombia

Spatially honest accuracy assessment for Sentinel-2 classification of shade
coffee agroforestry across four Colombian departments, from 923 field
reference polygons.

The headline result is not the map. It is that accuracy usually reported for
this kind of map is about 0.10 too high, that accuracy depends on how far you
are from field data, and that the two coffee boundaries fail for two
different reasons needing two different fixes.

## Pipeline

Run in order. Scripts 6, 7 and 8 refuse to continue unless they reproduce
script 5's splits exactly, so the chain cannot silently drift apart.

| Script | What it answers | Fits | Time |
|---|---|---|---|
| `1_gee.py` | Build the pixel table from Earth Engine | — | hours |
| `2_clean_define_annual.py` | Drop seasonal columns, keep annual | — | seconds |
| `3_select_features.py` | **How the 15 features were chosen.** Correlation filter, importance under leave-one-cluster-out, and a performance curve against feature count. | ~40 | ~15 min |
| `4_baseline_model.py` | **The model itself.** Pixel against polygon, hard against soft vote, per-region classifiers, confusion matrices. | 30 | ~10 min |
| `5_which_polygons_help.py` | **Q1.** Does it matter where reference polygons come from? | 130 | ~15 min |
| `6_how_far_does_it_reach.py` | **Q2.** How accurate is it really, and how far does a training polygon reach? | 480 | ~45 min |
| `7_which_boundary_fails.py` | **Q3a.** Which boundary fails, in which direction, does distance matter? | 0 | seconds |
| `8_is_sun_coffee_fixable.py` | **Q3b.** Is sun coffee fixable without merging the classes? | 90 | ~15 min |
| `9_figures.py` | Seven figures and draft captions | 0 | seconds |

Scripts 7 and 9 fit nothing, so they are cheap to rerun while drafting.

Script 4 is the map. Everything after it is a diagnostic explaining why that
map has the weaknesses it has.

**One warning about script 4.** Its polygon accuracy is the optimistic one.
It uses an ordinary stratified holdout, so training polygons sit a median of
208 m from test polygons and the score carries about 0.10 of spatial
inflation. Quote script 4 for the confusion structure, the pixel against
polygon comparison and the per-region spread. Quote script 6 for accuracy.
The script prints this warning itself so it cannot be missed.

`coffee_common.py` holds the feature list, the class codes, the region names,
the pixel sampler, the model helpers and the split machinery. It exists
because separate copies of those constants drifted twice, and both times
silently: one script still held an older experiment while later scripts read
its output, and the region names disagreed so one script labelled a cluster
Magdalena while the next called it Santander. Neither raised an error.

## Setup

```bash
pip install -r requirements.txt
cp gee_config.example.py gee_local_config.py   # then edit it
```

`gee_local_config.py` holds only settings that depend on where your files
are. Everything that is method, the seeds, tree counts, pixel caps and buffer
ladders, lives in the script that uses it, because those belong with the
experiment rather than with your machine.

## The three answers

**Q1. Only nearby polygons help.** 173 local training polygons beat 692
distant ones. Swapping 173 nearby for 174 distant at a fixed total costs
0.124 macro F1 and loses in 37 of 40 tests. Adding 174 *more* distant
polygons on top produced little average improvement: +0.002 macro F1,
with better performance in 21 of 40 seed–region comparisons.

**Q2. The usual accuracy number is about 0.10 too high.** Shade against sun
coffee scores 0.709 polygon F1, not the 0.805 a random holdout reports. Two
independent methods agree on 0.709: a buffered training sweep and a grouped
split that keeps near-duplicate polygons on one side. The gap comes from
reference polygons sitting in tight clumps, median nearest neighbour 208 m.

**Q3. Two failures, two causes.**

- *Shade coffee against forest is a distance problem.* Shade coffee called
  forest rises from 10.9% to 25.1% as training data is pushed away. Rates of
  23.5% at 5 km, 25.6% at 10 km and 25.1% at 50 km are consistent with a
  plateau beyond about 5 km. Nearby reference data reduces this error. The direction of the
  error flips between regions, so no single global correction will.
- *Sun coffee against bare ground is a size problem.* Sun coffee plots have a
  median area of 0.30 ha, so at 10 m only about 40% of their pixels are
  unmixed. The pair is separable at 0.82 when it is the only decision being
  made, and falls to 0.55 inside the five-class problem. It depends much less
  on distance: sun coffee called Open rises from 25.0% to 27.1%, although
  the distance effect reaches 4–6 percentage points at some buffers.
  More reference data improves F1 from 0.37 at 99 polygons to 0.47 at 121
  and 0.48 at 142; gains slow after about 120 polygons. The final slope is
  0.009 F1 per 10 polygons, compared with 0.004 for shade coffee. Finer
  pixels remain a proposed improvement, not a tested solution.

Urban is the control that makes the size argument work: it has the smallest
plots of any class and maps best of any class under transfer. Small alone is
survivable. Small *and* spectrally similar to its surroundings is not.

## Method notes that matter

- **Polygon level, not pixel level.** Pixels are voted to a polygon label
  before scoring, because a polygon is the unit a map user acts on.
- **Training pixels capped at 50 per polygon, test polygons uncapped.** The
  per-polygon sample depends on the seed and the polygon alone, never on
  which polygons were requested, so every design sees identical pixels for
  any polygon they share.
- **Training sets nest by construction.** Pooled is exactly the union of the
  local and outside-cluster sets, so each contrast isolates one variable.
- **Report the regional spread, not the seed interval.** Region to region SD
  is 0.043 for polygon macro F1 against 0.005 across seeds. n is 4.
- **`cf_f1`, `sc_nsc_f1` and `nsc_vs_open_f1` are restricted metrics.** They
  are scored only on polygons whose truth is one of the two classes named, so
  they exclude false positives arriving from the other three. Report them
  beside the one-vs-rest F1 scores, never instead of them.
- **Coordinates.** Imagery is processed and sampled on a 10 m grid in
  EPSG:3116. Polygon coordinates and all distances use the `easting` and
  `northing` columns. Scale distortion across the study area is under 0.15%,
  which is 0.3 m on the 200 m grouping threshold.

## Known limitations

- Four regions. Every regional claim rests on n = 4.
- Magdalena carries about 8 coffee test polygons per split, as few as 5. It
  is excluded from per-region claims and the exclusion is stated wherever it
  applies.
- Sun coffee does not work as a standalone class, 0.34 under the grouped
  split. It is reported at that value rather than merged away.
- The scale mechanism is a proposed explanation. The plot-size measurement
  supports it; a resolution degradation experiment would test it directly and
  has not been run.
- Features are 15 hand-selected annual Sentinel-2 and terrain variables.
  Seasonal composites and Sentinel-1 were tested separately and did not help.

## Outputs

Everything lands under `ANALYSIS_DIR`, one folder per script plus
`figures/`. Tables are CSV and carry the numbers behind every claim above, so
any figure traces back to the table that produced it. `9_figures.py` also
writes `captions_draft.txt`, with the numbers read from the tables rather
than typed, so a caption cannot drift out of step with its figure.

## Data

The Sentinel-2 pixel table and the field reference polygons are not in this
repository. Reference polygon locations are field-collected and may need
aggregation before release.
