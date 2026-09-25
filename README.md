# Coffee mapping workflow

Python workflow for extracting annual and quarterly Sentinel-2 features from reference polygons in Google Earth Engine and preparing an annual feature table for modeling. This repository contains the extraction and annual preparation stages; model training and research figures will be added separately.

## Files

- `1_gee.py`: builds spectral composites, indices, textures, topography, and canopy-height features; samples reference polygons; streams results to Parquet.
- `2_clean_define_annual.py`: reads annual/static columns, optionally applies local preparation, and writes a separate annual Parquet.
- `gee_config.example.py`: configuration template to copy and customize.
- `requirements.txt`: Python dependencies. Versions are not pinned to a validated environment yet.

## Setup

Install the dependencies in your Python environment:

```bash
python -m pip install -r requirements.txt
cp gee_config.example.py gee_local_config.py
```

Edit `gee_local_config.py` to set your Earth Engine project, polygon assets, output paths and filenames, class names, and analysis CRS. Both scripts require this file. It is excluded from Git so personal asset references and data corrections stay local.

The example uses `EPSG:3116` for the Colombia analysis grid. Choose an appropriate projected CRS for a different study area. Incoming longitude/latitude coordinates are interpreted as `EPSG:4326` before transformation to the configured CRS for easting/northing.

Authenticate with Earth Engine if needed:

```bash
python -c "import ee; ee.Authenticate()"
```

Your account needs access to the configured Earth Engine project, polygon assets, and canopy-height collection referenced in Script 1.

## Reference polygons

Each configured source needs polygon geometry and numeric `class` labels matching `CLASS_NAMES`. The workflow also reads the `original_label` property for provenance; supply it on reference features. The example class codes are 0 non-shade coffee, 1 shade coffee, 2 forest, 4 open, and 5 urban.

The script tags each source with `poly_source`, merges sources in configuration order, and assigns sequential `unique_id` values. Changing source order or asset contents can change IDs; check them before reusing existing train/test splits. Polygon overlaps are not automatically resolved.

Optional functions in the local configuration can customize preparation:

- `prepare_source(collection, source)` returns an Earth Engine FeatureCollection before merging.
- `prepare_annual_frame(df)` returns a pandas DataFrame before the annual table is saved.

Neither hook is required. Without hooks, source labels are used as supplied. Personal correction rules are not included in this repository.

## Run

Keep the local configuration beside the scripts, then run:

```bash
python 1_gee.py
python 2_clean_define_annual.py
```

For notebook use, keep the configuration in the notebook's working directory. Relative output paths resolve from the working directory.

Script 1 replaces the configured extraction output on rerun and writes a JSON record of skipped polygons. Review its completeness and missing-data reports before modeling. Script 2 leaves the extraction file untouched and replaces the configured annual output.

## Feature definitions

The current extraction uses the date settings retained from the research script: start `2020-01-01`, end `2020-12-30` (exclusive). This does not include the final two days of 2020. Those processing settings and cloud thresholds remain in Script 1.

Annual and quarterly spectral bands are temporal medians after cloud/shadow masking. Indices and spatial textures are derived from those medians. Quarters are JFM, AMJ, JAS, and OND; no solar-zenith-angle filtering or normalization is applied. Topography and canopy height are included once, without seasonal duplication. These ancillary products are not necessarily contemporaneous with the 2020 spectral imagery.

Missing values are retained in the saved tables. Script 2 selects annual/static columns; it does not perform feature selection, row filtering, or coordinate reprojection on existing downloads. Coordinates and provenance columns must be excluded from predictor lists later.

## Validation status

Python syntax and configuration behavior have been checked. Full Earth Engine extraction has not been rerun after cleanup and CRS standardization. The repository is a working research pipeline, not a fully validated software release.
