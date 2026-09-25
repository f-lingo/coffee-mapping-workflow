"""Extract annual and quarterly Sentinel-2 features to a local Parquet file.

Run this script before 2_clean_define_annual.py. Configure gee_local_config.py
beside the scripts using gee_config.example.py as a template. Dataset-specific
preparation hooks are optional. Only spectral features receive seasonal copies.
Dependencies: earthengine-api, numpy, pandas, geopandas, pyarrow.
Authenticate with ee.Authenticate() once if needed before running.
"""
from pathlib import Path
import gc
import importlib.util
import json
import os
import time

import ee
import geopandas as gpd
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# User settings live only in gee_local_config.py.
def load_local_config(path, required):
    """Load required settings; dataset-specific preparation hooks are optional."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing {path}. Copy gee_config.example.py to gee_local_config.py "
            "in the same folder as the scripts and fill in your settings."
        )
    spec = importlib.util.spec_from_file_location('gee_local_config', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    missing = [name for name in required if not hasattr(module, name)]
    if missing:
        raise ValueError(f"Missing settings in {path.name}: {', '.join(missing)}")
    return module


# For notebook use, keep the configuration in the current working directory.
SCRIPT_DIR = Path(__file__).resolve().parent if '__file__' in globals() else Path.cwd()
local_config = load_local_config(
    SCRIPT_DIR / 'gee_local_config.py',
    ('EE_PROJECT', 'OUTPUT_DIR', 'TARGET_CRS', 'OUTPUT_NAME', 'SKIPPED_NAME', 'POLYGON_SOURCES', 'CLASS_NAMES'),
)
EE_PROJECT = local_config.EE_PROJECT
OUTPUT_DIR = Path(local_config.OUTPUT_DIR)
TARGET_CRS = local_config.TARGET_CRS
OUTPUT_NAME = local_config.OUTPUT_NAME
SKIPPED_NAME = local_config.SKIPPED_NAME
POLYGON_SOURCES = local_config.POLYGON_SOURCES
CLASS_NAMES = local_config.CLASS_NAMES
prepare_source = getattr(local_config, 'prepare_source', None)

OUT_BASE = str(OUTPUT_DIR) + '/'
FINAL_PARQUET = str(OUTPUT_DIR / OUTPUT_NAME)
SKIPPED_JSON = str(OUTPUT_DIR / SKIPPED_NAME)

CHUNK = 5

# A polygon that still fails one at a time is retried with the band list
# split into this many groups, each sampled separately and joined back on
# pixel coordinates. This is safe only because the composite is
# unmask(NODATA) before sampling, so every group returns the identical
# pixel set and the join is exact rather than an inner join that quietly
# drops rows. The row count is asserted after every join.
RESCUE_BY_BAND_SPLIT = True
RESCUE_GROUPS = 4
COORD_DECIMALS = 9

POLYGON_ID_COL = 'unique_id'
CLASS_COL = 'class'
SOURCE_COL = 'poly_source'
PROPS = [CLASS_COL, POLYGON_ID_COL, SOURCE_COL]


# Date configuration
seasons = {
    'dry_early': {
        'start_dates': ['2020-01-01'],
        'end_dates': ['2020-12-30']
    }
}

# Cloud processing parameters
CLOUD_FILTER = 20
CLOUD_PROB_THRESH = 30
NIR_DRK_THRESH = 0.10
CLD_PRJ_DIST = 5
BUFFER = 200

ee.Initialize(project=EE_PROJECT)

# Apply optional dataset preparation, then tag provenance and merge.
# Without a local hook, the source's class labels are used as supplied.
polygons = None
for source in POLYGON_SOURCES:
    collection = ee.FeatureCollection(source['asset'])
    if prepare_source is not None:
        collection = prepare_source(collection, source)
    name = source['name']
    collection = collection.map(lambda feature: feature.set('poly_source', name))
    print(f"{name}: {collection.size().getInfo()} polygons")
    polygons = collection if polygons is None else polygons.merge(collection)
if polygons is None:
    raise ValueError('Configure at least one polygon source.')
print('merged:', polygons.size().getInfo())

# ============================================================
# ASSIGN unique_id TO EVERY FEATURE
# ============================================================
# Assign IDs to every polygon, including sources with missing IDs.
_n = polygons.size()
_lst = polygons.toList(_n)
polygons = ee.FeatureCollection(
    ee.List.sequence(0, _n.subtract(1)).map(
        lambda i: ee.Feature(_lst.get(i)).set('unique_id', ee.Number(i))
    )
)

# polys and polygons are the same collection from here on, so any
# downstream cell using either name gets the ids
polys = polygons

_n_null = polys.filter(ee.Filter.eq('unique_id', None)).size().getInfo()
print('features:', polys.size().getInfo())
print('null ids:', _n_null)
if _n_null:
    raise RuntimeError(f"{_n_null} features still have a null unique_id")

# geometry must survive, or ROI is empty and the composite fails
_bounds = polys.geometry().bounds().getInfo()['coordinates'][0]
_lons = [c[0] for c in _bounds]
_lats = [c[1] for c in _bounds]
print(f'bounds  : lon {min(_lons):.2f} to {max(_lons):.2f}, '
      f'lat {min(_lats):.2f} to {max(_lats):.2f}')
if max(_lons) - min(_lons) < 0.1:
    raise RuntimeError('geometry appears to be missing from the features')

print('classes  :', polys.aggregate_histogram('class').getInfo())
print('by source:', polys.aggregate_histogram('poly_source').getInfo())
# ============================================================
# HOW THE ORIGINAL LABELS MAP INTO THE FIVE CLASSES
# ============================================================
# Report original labels by source and standardized class.
print('\noriginal_label by class and source:')
for _src in [source['name'] for source in POLYGON_SOURCES]:
    _sub = polys.filter(ee.Filter.eq('poly_source', _src))
    for _cls in CLASS_NAMES:
        _h = _sub.filter(ee.Filter.eq('class', _cls)) \
                 .aggregate_histogram('original_label').getInfo() or {}
        if not _h:
            continue
        if len(_h) > 6:
            print(f"  {_src} class {_cls}: {len(_h)} distinct labels "
                  f"(systematic naming)")
        else:
            print(f"  {_src} class {_cls}: {_h}")


ROI = polygons.geometry().bounds().buffer(1000)      # for image collection filtering


# ============================================================
# BAND NAMES AND GLCM SCALING BOUNDS
# ============================================================
# unitScale(lo, hi) maps lo->0 and hi->1, and the result is then
# multiplied by 255 and cast to uint8 for glcmTexture. Using
# (0, 10000) for everything was correct for raw reflectance and
# wrong for every index: NDVI at 0.999 became 0.025, which
# truncates to 0. A constant image has zero GLCM contrast, which
# is exactly what the 25 dead *_contrast_k* columns were.
#
# clamp() is applied before unitScale because VARI has an
# unstable denominator (green + red - blue) and spikes well
# outside its nominal range, which would otherwise compress
# everything else into the bottom of the 8-bit space.
GLCM_DEFAULT_RANGE = (0, 10000)          # retained raw reflectance bands
INDEX_RANGE = {
    'NDVI':   (-1.0, 1.0),
    'NDI45':  (-1.0, 1.0),
    'NDTI':   (-1.0, 1.0),
    'VARI':   (-1.0, 1.0),
    'GCVI':   (-1.0, 25.0),
}

# Excluded from computation and output: B1, B9, WVP (including WVP_ND),
# and canopy_height_lang, including their textures and seasonal versions.
# B11 and B12 are retained; they are distinct bands from B1.
S2_BANDS = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A',
            'B11', 'B12']
INDEX_BANDS = ['NDVI', 'NDI45', 'GCVI', 'NDTI', 'VARI']
TOPO_BANDS = ['elevation', 'slope', 'canopy_height_tolan']
RADII = [1, 2, 3, 4, 5]
STATS = ['mean', 'variance', 'contrast']


def _texture_names(sources):
    return [f'{s}_{stat}_k{r}'
            for s in sources for r in RADII for stat in STATS]


# Built rather than written out, so the list cannot drift from what
# the texture functions actually produce. Same naming as before.
all_features = (
    S2_BANDS + INDEX_BANDS
    + _texture_names(S2_BANDS)
    + _texture_names(['RGB'])
    + _texture_names(INDEX_BANDS)
    + TOPO_BANDS
    + _texture_names(TOPO_BANDS)
)
all_features = list(dict.fromkeys(all_features))   # keep order, drop repeats

# Additional calendar-quarter versions of Sentinel-2-derived features only.
# These use the existing date range and cloud/shadow processing, with no SZA
# filtering or normalization. Existing unsuffixed features remain unchanged.
SPECTRAL_SEASONS = {
    'JFM': (1, 3),
    'AMJ': (4, 6),
    'JAS': (7, 9),
    'OND': (10, 12),
}
SPECTRAL_FEATURES = (
    S2_BANDS + INDEX_BANDS
    + _texture_names(S2_BANDS)
    + _texture_names(['RGB'])
    + _texture_names(INDEX_BANDS)
)
all_features += [
    f'{feature}_{quarter}'
    for quarter in SPECTRAL_SEASONS
    for feature in SPECTRAL_FEATURES
]

print(f"Total features to extract: {len(all_features)}")


def get_s2_sr_cld_col(aoi, start_date, end_date):
    s2_sr_col = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                 .filterBounds(aoi)
                 .filterDate(start_date, end_date)
                 .filter(ee.Filter.lte('CLOUDY_PIXEL_PERCENTAGE', CLOUD_FILTER)))
    s2_cloudless_col = (ee.ImageCollection('COPERNICUS/S2_CLOUD_PROBABILITY')
                        .filterBounds(aoi)
                        .filterDate(start_date, end_date))
    joined = ee.Join.saveFirst('s2cloudless').apply(
        primary=s2_sr_col,
        secondary=s2_cloudless_col,
        condition=ee.Filter.equals(leftField='system:index', rightField='system:index')
    )
    return ee.ImageCollection(joined)

def add_cloud_bands(img):
    cld_prb = ee.Image(img.get('s2cloudless')).select('probability')
    is_cloud = cld_prb.gt(CLOUD_PROB_THRESH).rename('clouds')
    return img.addBands(ee.Image([cld_prb, is_cloud]))

def add_shadow_bands(img):
    not_water = img.select('SCL').neq(6)
    SR_BAND_SCALE = 1e4
    dark_pixels = img.select('B8').lt(NIR_DRK_THRESH * SR_BAND_SCALE) \
        .multiply(not_water).rename('dark_pixels')
    shadow_azimuth = ee.Number(90).subtract(ee.Number(img.get('MEAN_SOLAR_AZIMUTH_ANGLE')))
    cld_proj = (img.select('clouds')
                .directionalDistanceTransform(shadow_azimuth, CLD_PRJ_DIST * 10)
                .reproject(**{'crs': TARGET_CRS, 'scale': 100})
                .select('distance')
                .mask()
                .rename('cloud_transform'))
    shadows = cld_proj.multiply(dark_pixels).rename('shadows')
    return img.addBands(ee.Image([dark_pixels, cld_proj, shadows]))

def add_cld_shdw_mask(img):
    img_cloud = add_cloud_bands(img)
    img_cloud_shadow = add_shadow_bands(img_cloud)
    is_cld_shdw = img_cloud_shadow.select('clouds') \
        .add(img_cloud_shadow.select('shadows')).gt(0)
    is_cld_shdw = (is_cld_shdw.focalMin(2)
                   .focalMax(BUFFER * 2 / 20)
                   .reproject(**{'crs': TARGET_CRS, 'scale': 20})
                   .rename('cloudmask'))
    return img_cloud_shadow.addBands(is_cld_shdw)

def apply_cld_shdw_mask(img):
    not_cld_shdw = img.select('cloudmask').Not()
    selected_img = img.select(S2_BANDS)
    return selected_img.updateMask(not_cld_shdw)


def reproject_image(image, crs=TARGET_CRS, scale=10):
    """Set interpolation before reprojecting to the analysis grid."""
    return image.resample('bicubic').reproject(crs=crs, scale=scale)


def add_indices(image):
    image = image.toFloat()
    NDVI = image.normalizedDifference(['B8', 'B4']).rename('NDVI')
    NDI45 = image.normalizedDifference(['B5', 'B4']).rename('NDI45')
    GCVI = image.select('B8').divide(image.select('B3')).subtract(1).rename('GCVI')
    NDTI = image.normalizedDifference(['B11', 'B12']).rename('NDTI')
    VARI = image.select('B3')\
        .subtract(image.select('B4'))\
        .divide(image.select('B3').add(image.select('B4')).subtract(image.select('B2')))\
        .rename('VARI')
    return image.addBands([NDVI, NDI45, GCVI, NDTI, VARI])


def _scale_for_glcm(band_img, band_name):
    """Scale each band to its specified 8-bit range for GLCM contrast."""
    lo, hi = INDEX_RANGE.get(band_name, GLCM_DEFAULT_RANGE)
    return band_img.clamp(lo, hi).unitScale(lo, hi).multiply(255).toUint8()


def compute_texture_s2(image, crs=TARGET_CRS, scale=10):
    image = image.toFloat()

    # kernel radius r -> (2r+1) pixel window at 10 m
    # 1 -> 3x3 (30 m), 2 -> 5x5 (50 m), 3 -> 7x7 (70 m),
    # 4 -> 9x9 (90 m), 5 -> 11x11 (110 m)

    def compute_band_texture(band):
        band_img = image.select(band)

        # For contrast: scale to 8-bit using this band's own range
        band_scaled = _scale_for_glcm(band_img, band)

        out = []
        for r in RADII:
            sfx = f'_k{r}'
            kernel = ee.Kernel.square(r)

            out.append(band_img.reduceNeighborhood(ee.Reducer.mean(), kernel)
                       .rename(band + '_mean' + sfx))
            out.append(band_img.reduceNeighborhood(ee.Reducer.variance(), kernel)
                       .rename(band + '_variance' + sfx))
            out.append(band_scaled.glcmTexture(size=r)
                       .select(band + '_contrast')
                       .rename(band + '_contrast' + sfx))
        return ee.Image.cat(out)

    # Compute textures for individual Sentinel-2 bands
    textures_s2 = ee.Image.cat([compute_band_texture(band) for band in S2_BANDS])

    # Compute texture for an RGB composite
    rgb = image.select(['B4', 'B3', 'B2']).reduce(ee.Reducer.mean()).rename('gray')
    rgb_scaled = _scale_for_glcm(rgb, 'gray')      # reflectance range

    rgb_out = []
    for r in RADII:
        sfx = f'_k{r}'
        kernel = ee.Kernel.square(r)

        rgb_out.append(rgb.reduceNeighborhood(ee.Reducer.mean(), kernel)
                       .rename('RGB_mean' + sfx))
        rgb_out.append(rgb.reduceNeighborhood(ee.Reducer.variance(), kernel)
                       .rename('RGB_variance' + sfx))
        rgb_out.append(rgb_scaled.glcmTexture(size=r)
                       .select('gray_contrast')
                       .rename('RGB_contrast' + sfx))

    texture_rgb = ee.Image.cat(rgb_out)

    # Compute textures for the spectral indices
    textures_indices = ee.Image.cat(
        [compute_band_texture(index) for index in INDEX_BANDS])

    # Combine all texture features
    all_textures = ee.Image.cat([textures_s2, texture_rgb, textures_indices])

    # Ensure consistent projection and resolution
    all_textures = all_textures.reproject(crs=crs, scale=scale)

    return image.addBands(all_textures)


def add_topography(image, crs=TARGET_CRS, scale=10):
    image = image.toFloat()

    # kernel radius r -> (2r+1) pixel window at 10 m
    # 1 -> 3x3 (30 m), 2 -> 5x5 (50 m), 3 -> 7x7 (70 m),
    # 4 -> 9x9 (90 m), 5 -> 11x11 (110 m)

    # FIX 7. SRTMGL1 is 30 m, the previous CGIAR/SRTM90_V4 was 90 m.
    # Even at 30 m, elevation_*_k1 measures a 30 m window over one
    # native cell, so k1 and k2 elevation texture describe the
    # resampling rather than the terrain. Read them accordingly.
    elevation = ee.Image('USGS/SRTMGL1_003').select('elevation').rename('elevation')
    elevation = elevation.reproject(crs=crs, scale=scale)

    # Get slope data (derived from elevation)
    slope = ee.Terrain.slope(elevation).rename('slope')
    slope = slope.reproject(crs=crs, scale=scale)

    # Get Meta Forest canopy height data (1m)
    canopy_meta_collection = ee.ImageCollection("projects/meta-forest-monitoring-okw37/assets/CanopyHeight")
    canopy_height_meta_raw = canopy_meta_collection.mosaic()

    first_band = canopy_height_meta_raw.bandNames().get(0)
    canopy_height_meta = canopy_height_meta_raw.select([first_band])

    # Set a default projection before using reduceResolution
    canopy_height_meta = canopy_height_meta.setDefaultProjection(crs, scale=1)

    # Now use reduceResolution to aggregate from 1m to 10m
    canopy_height_meta = canopy_height_meta.reduceResolution(
        reducer=ee.Reducer.mean(),
        bestEffort=True
    )

    # After reduction, reproject to the target projection and scale
    canopy_height_meta = canopy_height_meta.reproject(crs=crs, scale=scale).rename('canopy_height_tolan')

    # ------------------------------------------------------------
    # Multi-scale texture: (layer, output_name, glcm_band, lo, hi)
    # lo/hi are the 8-bit scaling bounds for the GLCM computation.
    # These were already per source here, which is why the topography
    # contrast features worked while the index ones did not.
    # ------------------------------------------------------------
    LAYERS = [
        (elevation,          'elevation',           'elevation',           0, 5000),
        (slope,              'slope',               'slope',               0,   90),
        (canopy_height_meta, 'canopy_height_tolan', 'canopy_height_tolan', 0,   50),
    ]

    texture_bands = []
    for img, name, glcm_band, lo, hi in LAYERS:
        img_scaled = img.clamp(lo, hi).unitScale(lo, hi).multiply(255).toByte()

        for r in RADII:
            sfx = f'_k{r}'
            kernel = ee.Kernel.square(r)

            texture_bands.append(
                img.reduceNeighborhood(ee.Reducer.mean(), kernel)
                   .rename(name + '_mean' + sfx))
            texture_bands.append(
                img.reduceNeighborhood(ee.Reducer.variance(), kernel)
                   .rename(name + '_variance' + sfx))
            texture_bands.append(
                img_scaled.glcmTexture(size=r)
                          .select(glcm_band + '_contrast')
                          .rename(name + '_contrast' + sfx))

    # Reproject all images to ensure consistent 10m resolution
    def reproject_all(images):
        return [reproject_image(img, crs=crs, scale=scale) for img in images]

    base_layers = reproject_all([elevation, slope, canopy_height_meta])
    texture_layers = reproject_all(texture_bands)

    # Combine all bands
    topo_bands = ee.Image.cat(base_layers + texture_layers)

    return image.addBands(topo_bands)


def add_seasonal_spectral_features(image, s2_period_images,
                                   crs=TARGET_CRS, scale=10):
    """Append quarterly spectral medians and their derived indices/textures."""
    raw_bands = S2_BANDS

    # A fully masked template keeps the expected bands even if a quarter has
    # no scenes. It contributes no observations to the median. Missing pixels
    # are handled by the existing NODATA/unmask processing downstream.
    empty_raw = (ee.Image(s2_period_images.first()).select(raw_bands)
                 .updateMask(ee.Image.constant(0)))

    for quarter, (first_month, last_month) in SPECTRAL_SEASONS.items():
        quarter_images = (s2_period_images
                          .filter(ee.Filter.calendarRange(
                              first_month, last_month, 'month'))
                          .select(raw_bands))
        quarter_composite = (quarter_images
                             .merge(ee.ImageCollection([empty_raw]))
                             .median()
                             .setDefaultProjection(crs, scale=scale))

        # Match the existing order: temporal median, then indices, then
        # spatial textures. Do not duplicate topography or canopy features.
        quarter_composite = add_indices(quarter_composite)
        quarter_composite = compute_texture_s2(
            quarter_composite, crs=crs, scale=scale)
        quarter_composite = quarter_composite.select(SPECTRAL_FEATURES).rename(
            [f'{feature}_{quarter}' for feature in SPECTRAL_FEATURES])
        image = image.addBands(quarter_composite)

    return image


def create_season_composite(season, region_geometry, keep_bands=None):
    crs_projection = TARGET_CRS
    start_dates = seasons[season]['start_dates']
    end_dates   = seasons[season]['end_dates']

    s2_period_images = ee.ImageCollection([])

    for start_date, end_date in zip(start_dates, end_dates):
        sentinel2_images = (get_s2_sr_cld_col(region_geometry, start_date, end_date)
                            .map(add_cld_shdw_mask)
                            .map(apply_cld_shdw_mask)
                            .map(reproject_image))
        s2_period_images = s2_period_images.merge(sentinel2_images)

    if s2_period_images.size().getInfo() == 0:
        return None

    # 1. clean composite FIRST
    s2_season_composite = s2_period_images.median()
    s2_season_composite = s2_season_composite.setDefaultProjection(
        crs_projection, scale=10)

    # 2. then derive everything from it
    s2_season_composite = add_indices(s2_season_composite)
    s2_season_composite = compute_texture_s2(s2_season_composite)
    s2_season_composite = add_topography(s2_season_composite)

    # Append all four quarterly versions without changing existing bands.
    s2_season_composite = add_seasonal_spectral_features(
        s2_season_composite, s2_period_images, crs=crs_projection, scale=10)

    combined_composite = s2_season_composite.toFloat()
    combined_composite = combined_composite.clip(region_geometry)

    if keep_bands is not None:
        present = combined_composite.bandNames()
        combined_composite = combined_composite.select(
            present.filter(ee.Filter.inList('item', keep_bands)))

    # unmask the whole image at once — no per-band getInfo, no ee.Image.cat
    # sampleRegions drops a pixel entirely if any band is masked, so the
    # unmask stays. FIX 8 converts the sentinel back to NaN in pandas.
    combined_composite = combined_composite.unmask(NODATA)
    return combined_composite


NODATA = -9999

# Build the composite
print("Creating composite...")
composite = create_season_composite('dry_early', ROI)

# Check which target features are available
if composite is not None:
    available_bands = composite.bandNames().getInfo()
    bands_to_use = [b for b in all_features if b in available_bands]
    missing = [b for b in all_features if b not in available_bands]

    print(f"Available: {len(bands_to_use)}/{len(all_features)} target features")
    if missing:
        print(f"Missing: {missing}")
    print(f"Bands: {bands_to_use}")
else:
    raise RuntimeError("No Sentinel-2 images found for the configured period and ROI")


# ============================================================
# CONTRAST SANITY CHECK
# ============================================================
# The whole point of FIX 4. A contrast band that comes back with one
# distinct value over a sample of pixels is dead, exactly as the index
# contrast bands were in the previous export. Check before committing
# to a full run rather than discovering it in the correlation matrix.
def check_contrast_bands(n=500):
    contrast = [b for b in bands_to_use if '_contrast_k' in b]
    sample = composite.select(contrast).sample(
        region=polys.geometry(), scale=10, numPixels=n, tileScale=16)
    stats = sample.reduceColumns(
        ee.Reducer.minMax().repeat(len(contrast)),
        contrast).getInfo()
    dead = [b for b, lo, hi in zip(contrast, stats['min'], stats['max'])
            if lo == hi]
    print(f"{len(contrast)} contrast bands checked, {len(dead)} constant")
    if dead:
        print("  constant, check the scaling range for these sources:")
        for b in dead:
            print(f"    {b}")
    else:
        print("  all contrast bands vary")


# check_contrast_bands()


# ============================================================
# SAMPLING AND OUTPUT
# ============================================================
# ------------------------------------------------------------
# OUTPUT CONFIG
# ------------------------------------------------------------
# ------------------------------------------------------------
# SETUP
# ------------------------------------------------------------
comp_sel = composite.select(bands_to_use)
ids = [int(i) for i in polys.aggregate_array('unique_id').getInfo()]
print(f"\n{len(ids)} polygons, {len(bands_to_use)} bands")

# Pull labels once and map them per batch to avoid a whole-table merge.
_attrs = ee.data.computeFeatures({
    'expression': polys.select(['unique_id', 'original_label'], None, False),
    'fileFormat': 'PANDAS_DATAFRAME'})
_attrs['unique_id'] = pd.to_numeric(_attrs['unique_id']).astype('int32')
LABEL_OF = dict(zip(_attrs['unique_id'], _attrs['original_label']))
del _attrs
gc.collect()

# Fixed output schema, so the Parquet schema cannot drift between batches
FEAT_COLS = list(bands_to_use)
META_COLS = [POLYGON_ID_COL, CLASS_COL, SOURCE_COL, 'original_label',
             'lon', 'lat', 'easting', 'northing']
OUT_COLS = META_COLS + FEAT_COLS
print(f"output schema, {len(OUT_COLS)} columns "
      f"({len(META_COLS)} metadata + {len(FEAT_COLS)} features)")


# ------------------------------------------------------------
# PER-BATCH CLEANING
# ------------------------------------------------------------
def tidy_batch(b):
    """Clean one sampled batch into the fixed output schema.

    Everything the old tail did to the whole 4.7 GB frame happens here,
    on a few thousand rows instead.
    """
    b = b[b['geo'].notna()].copy()
    if b.empty:
        return None

    # geo is a dict from computeFeatures. Pull the coordinates out and
    # drop the column.
    coords = np.asarray([g['coordinates'][:2] for g in b['geo']],
                        dtype='float64')
    b = b.drop(columns=['geo'])
    b['lon'] = coords[:, 0]
    b['lat'] = coords[:, 1]

    # Earth Engine point coordinates are longitude/latitude (EPSG:4326).
    # Declare their source CRS correctly, then transform to the analysis CRS.
    g = gpd.GeoSeries(gpd.points_from_xy(coords[:, 0], coords[:, 1]),
                      crs='EPSG:4326').to_crs(TARGET_CRS)
    b['easting'] = g.x.to_numpy()
    b['northing'] = g.y.to_numpy()
    del coords, g

    b[POLYGON_ID_COL] = pd.to_numeric(b[POLYGON_ID_COL]).astype('int32')
    b[CLASS_COL] = pd.to_numeric(b[CLASS_COL]).astype('int16')
    b[SOURCE_COL] = b[SOURCE_COL].astype(str)
    b['original_label'] = b[POLYGON_ID_COL].map(LABEL_OF).astype('object')

    # A band absent from this batch becomes an all-NaN column rather than
    # a schema change.
    missing_here = [c for c in FEAT_COLS if c not in b.columns]
    for c in missing_here:
        b[c] = np.nan
    if missing_here:
        print(f"      note, {len(missing_here)} bands absent from this batch")

    # Convert NODATA to NaN and float64 to float32 in
    # one pass, with no second copy of the frame.
    vals = b[FEAT_COLS].to_numpy(dtype='float32', copy=True)
    vals[vals == np.float32(NODATA)] = np.nan
    out = pd.DataFrame(vals, columns=FEAT_COLS, index=b.index)
    del vals

    for c in META_COLS:
        out[c] = b[c].values
    del b
    return out[OUT_COLS]


# ------------------------------------------------------------
# SAMPLING, WITH A BAND-SPLIT RESCUE
# ------------------------------------------------------------
def sample_batch(id_batch, bands=None):
    subset = polys.filter(ee.Filter.inList('unique_id', id_batch))
    img = comp_sel if bands is None else composite.select(bands)
    fc = img.sampleRegions(
        collection=subset,
        properties=PROPS,
        projection=ee.Projection(TARGET_CRS).atScale(10),
        scale=10,
        tileScale=16,
        geometries=True
    )
    return ee.data.computeFeatures({
        'expression': fc,
        'fileFormat': 'PANDAS_DATAFRAME'
    })


def sample_band_split(pid, n_groups=RESCUE_GROUPS):
    """Sample one polygon in band groups and join them on pixel coords.

    "Computed value is too large" is a response-size limit, rows times
    bands. Fewer bands per request brings the response under the limit
    without dropping any pixels.

    The join is exact because the composite is unmask(NODATA) before
    sampling, so every group returns the identical pixel set. Row counts
    are asserted before and after each join, so a silent inner-join loss
    cannot happen.
    """
    groups = np.array_split(np.array(FEAT_COLS, dtype=object), n_groups)
    merged = None
    n_expected = None

    for gi, grp in enumerate(groups):
        grp = [str(x) for x in grp]
        part = sample_batch([pid], bands=grp)
        part = part[part['geo'].notna()].copy()
        if part.empty:
            raise RuntimeError(f'polygon {pid} group {gi} returned no pixels')

        xy = np.asarray([g['coordinates'][:2] for g in part['geo']],
                        dtype='float64')
        part = part.drop(columns=['geo'])
        part['_x'] = np.round(xy[:, 0], COORD_DECIMALS)
        part['_y'] = np.round(xy[:, 1], COORD_DECIMALS)
        del xy

        if n_expected is None:
            n_expected = len(part)
        elif len(part) != n_expected:
            raise RuntimeError(
                f'polygon {pid} group {gi} returned {len(part)} pixels, '
                f'group 0 returned {n_expected}. Not joinable.')

        if merged is None:
            merged = part
        else:
            part = part.drop(columns=[c for c in PROPS if c in part.columns])
            merged = merged.merge(part, on=['_x', '_y'], how='inner')
            if len(merged) != n_expected:
                raise RuntimeError(
                    f'polygon {pid} lost rows joining group {gi}, '
                    f'{len(merged)} of {n_expected}')
        del part
        gc.collect()

    # Rebuild a geo column so tidy_batch can treat this like any batch
    merged['geo'] = [{'coordinates': [x, y]}
                     for x, y in zip(merged['_x'], merged['_y'])]
    return merged.drop(columns=['_x', '_y'])


# ------------------------------------------------------------
# STREAMING LOOP
# ------------------------------------------------------------
os.makedirs(OUT_BASE, exist_ok=True)
if os.path.exists(FINAL_PARQUET):
    os.remove(FINAL_PARQUET)

writer = None
schema = None
n_rows_written = 0
polys_written = set()
failed_ids = []
nan_counts = pd.Series(0, index=FEAT_COLS, dtype='int64')
_t0 = time.time()


def write_batch(out):
    """Append one cleaned batch to the Parquet file."""
    global writer, schema, n_rows_written
    tbl = pa.Table.from_pandas(out, preserve_index=False)
    if writer is None:
        schema = tbl.schema
        writer = pq.ParquetWriter(FINAL_PARQUET, schema, compression='snappy')
    else:
        tbl = tbl.cast(schema)
    writer.write_table(tbl)
    n_rows_written += len(out)
    del tbl


def handle(out):
    """Accumulate the running NaN tally, then append the batch.

    The NaN rate is accumulated here rather than computed at the end,
    because computing it at the end would mean reading all the feature
    columns back into memory, which is the thing this rewrite avoids.
    """
    global nan_counts
    if out is None or out.empty:
        return
    nan_counts = nan_counts.add(out[FEAT_COLS].isna().sum().astype('int64'),
                                fill_value=0)
    polys_written.update(out[POLYGON_ID_COL].unique().tolist())
    write_batch(out)


print("\n" + "=" * 60)
print("SAMPLING, STREAMED TO PARQUET")
print("=" * 60)

# EDIT C. Failed batches are retried and then counted. The original loop
# printed the exception and moved on with nothing appended, so a
# timed-out batch disappeared from the frame with no error.
for i in range(0, len(ids), CHUNK):
    batch = ids[i:i + CHUNK]
    try:
        out = tidy_batch(sample_batch(batch))
        handle(out)
        del out
        gc.collect()
        print(f"ok {i}-{i + len(batch)}  rows so far {n_rows_written}")
    except Exception as e:
        print(f"failed {batch}: {str(e)[:120]}")
        failed_ids.extend(batch)

if failed_ids:
    print(f"\nretrying {len(failed_ids)} ids one at a time")
    time.sleep(5)
    still_failed = []
    for pid in failed_ids:
        try:
            out = tidy_batch(sample_batch([pid]))
            handle(out)
            del out
            gc.collect()
            print(f"  recovered {pid}")
        except Exception as e:
            print(f"  still failing {pid}: {str(e)[:100]}")
            still_failed.append(pid)
    failed_ids = still_failed

if failed_ids and RESCUE_BY_BAND_SPLIT:
    print(f"\nband-split rescue for {len(failed_ids)} ids, "
          f"{RESCUE_GROUPS} groups each")
    still_failed = []
    for pid in failed_ids:
        try:
            out = tidy_batch(sample_band_split(pid))
            handle(out)
            del out
            gc.collect()
            print(f"  rescued {pid}")
        except Exception as e:
            print(f"  rescue failed {pid}: {str(e)[:120]}")
            still_failed.append(pid)
    failed_ids = still_failed

if writer is not None:
    writer.close()
else:
    raise RuntimeError('nothing was written, every batch failed')

print(f"\nwrote {n_rows_written} rows to {FINAL_PARQUET}")
print(f"file size {os.path.getsize(FINAL_PARQUET) / 1e9:.2f} GB")
print(f"polygons in file {len(polys_written)} of {len(ids)}")
print(f"elapsed {(time.time() - _t0) / 60:.1f} min")

failed_ids = sorted(set(ids) - polys_written)
with open(SKIPPED_JSON, 'w') as fh:
    json.dump({'skipped_unique_ids': sorted(int(x) for x in failed_ids),
               'n_polygons_expected': len(ids),
               'n_polygons_written': len(polys_written),
               'n_rows': int(n_rows_written),
               'reason': 'Sampling failed or returned no usable pixels; see run log.'}, fh, indent=2)

if failed_ids:
    print(f"\nSKIPPED {len(failed_ids)} polygons: {sorted(failed_ids)}")
    print(f"recorded in {SKIPPED_JSON}")
else:
    print("\nno polygons skipped")


# ============================================================
# COORDINATES + POLYGON INVENTORY BY SOURCE
# ============================================================
# Same reports as before. Only the metadata columns are read back.
# Parquet is columnar, so this costs a few MB rather than reloading the
# whole table.
meta = pd.read_parquet(FINAL_PARQUET, columns=META_COLS)

print("\n" + "=" * 60)
print("COORDINATES")
print("=" * 60)
print(meta[['lon', 'lat', 'easting', 'northing']].describe().round(3).to_string())
print("\nlon, lat, easting and northing are location, not predictors.")
print("Keep them out of any feature list, or the model memorises where")
print("the polygons are and the score collapses under LOROCV.")

print(f"Projected easting/northing CRS: {TARGET_CRS}")

print("\n" + "=" * 60)
print("INVENTORY")
print("=" * 60)
print(f"rows     : {len(meta)}")
print(f"polygons : {meta[POLYGON_ID_COL].nunique()}")
print(f"columns  : {len(OUT_COLS)}")

# polygons, not pixels, is the unit that matters here
poly = (meta.groupby(POLYGON_ID_COL)
        .agg(**{CLASS_COL: (CLASS_COL, 'first'),
                SOURCE_COL: (SOURCE_COL, 'first'),
                'n_pixels': (CLASS_COL, 'size')})
        .reset_index())

print("\nPolygons by source:")
print(poly[SOURCE_COL].value_counts().to_string())

print("\nPolygons by class and source:")
ct = pd.crosstab(poly[CLASS_COL].map(CLASS_NAMES), poly[SOURCE_COL],
                 margins=True, margins_name='total')
print(ct.to_string())

print("\nPixels by class and source:")
ctp = pd.crosstab(meta[CLASS_COL].map(CLASS_NAMES), meta[SOURCE_COL],
                  margins=True, margins_name='total')
print(ctp.to_string())

print("\nPixels per polygon, by class and source:")
pp = (poly.groupby([CLASS_COL, SOURCE_COL])['n_pixels']
      .agg(polygons='size', median='median', max='max')
      .reset_index())
pp[CLASS_COL] = pp[CLASS_COL].map(CLASS_NAMES)
print(pp.to_string(index=False))

print("\nPixels per polygon by class, the size skew that makes")
print("pixel-level scores disagree with polygon-level ones:")
print(poly.groupby(poly[CLASS_COL].map(CLASS_NAMES))['n_pixels']
      .agg(polygons='size', total='sum', median='median', max='max')
      .to_string())

print("\nunique_id range per source:")
print(poly.groupby(SOURCE_COL)[POLYGON_ID_COL]
      .agg(['min', 'max', 'nunique']).to_string())

# A class present in only one source cannot be checked for
# cross-source consistency, and one drawn from several may carry
# the same split seen in non shade coffee.
both = ct.drop(index='total', errors='ignore').drop(
    columns='total', errors='ignore')
single = both.index[(both > 0).sum(axis=1) == 1].tolist()
mixed = both.index[(both > 0).sum(axis=1) > 1].tolist()
print(f"\nClasses drawn from one source only : {single}")
print(f"Classes drawn from more than one   : {mixed}")


# ============================================================
# MISSING DATA
# ============================================================
# FIX 8's report, accumulated during the loop rather than computed on
# the whole frame at the end.
print("\n" + "=" * 60)
print("MISSING DATA")
print("=" * 60)
_rate = (nan_counts / max(n_rows_written, 1) * 100).sort_values(ascending=False)
_bad = _rate[_rate > 0]
print(f"{len(_bad)} of {len(FEAT_COLS)} feature columns have any NaN")
if len(_bad):
    print("columns with the most missing:")
    print(_bad.head(15).round(2).to_string())
print("\nDecide on imputation before modelling. Leaving NaN is fine for")
print("a correlation matrix, which skips it pairwise, but scikit-learn")
print("estimators will reject it.")


# ============================================================
# CONSTANT FEATURE CHECK
# ============================================================
# The pandas-side counterpart to check_contrast_bands(). If the GLCM
# scaling fix worked this list should be empty. Columns are checked in
# blocks so the whole table is never in memory.
print("\n" + "=" * 60)
print("CONSTANT FEATURES")
print("=" * 60)
_const = []
_BLOCK = 100
for _s in range(0, len(FEAT_COLS), _BLOCK):
    _cols = FEAT_COLS[_s:_s + _BLOCK]
    _chunk = pd.read_parquet(FINAL_PARQUET, columns=_cols)
    _nun = _chunk.nunique(dropna=True)
    _const.extend(_nun[_nun < 2].index.tolist())
    del _chunk, _nun
    gc.collect()
_const = sorted(_const)
print(f"{len(_const)} of {len(FEAT_COLS)} feature columns are constant")
for c in _const:
    print(f"  {c}")
if not _const:
    print("  none, the GLCM scaling fix is working")


# ============================================================
# COMPLETENESS CHECK
# ============================================================
# Compare the file against the merged collection so a partial dataset
# does not get modelled months later. Skipped polygons are reported
# rather than raised, since the run is meant to finish.
_exp_source = polys.aggregate_histogram('poly_source').getInfo() or {}
_exp_total = int(polys.size().getInfo())
_got_source = (meta.groupby(POLYGON_ID_COL)[SOURCE_COL].first()
               .value_counts().to_dict())

print("\n" + "=" * 60)
print("COMPLETENESS")
print("=" * 60)
print(f"polygons expected {_exp_total}, in file "
      f"{meta[POLYGON_ID_COL].nunique()}")
for _src in [source['name'] for source in POLYGON_SOURCES]:
    print(f"  {_src:<10} expected {int(_exp_source.get(_src, 0)):>4}, "
          f"in file {int(_got_source.get(_src, 0)):>4}")

if failed_ids:
    print("\nSkipped polygons, by source and class:")
    _sk = ee.data.computeFeatures({
        'expression': polys.filter(
            ee.Filter.inList('unique_id', failed_ids)).select(
            ['unique_id', 'class', 'poly_source'], None, False),
        'fileFormat': 'PANDAS_DATAFRAME'})
    print(_sk.to_string(index=False))

for source in POLYGON_SOURCES:
    name = source['name']
    if int(_exp_source.get(name, 0)) > 0 and int(_got_source.get(name, 0)) == 0:
        raise RuntimeError(f"No polygons from {name!r} were written. Check the sampling log.")


# ============================================================
# HOW TO READ THE FILE
# ============================================================
print(f"""
=============================================================
Read only the columns you need. This is the point of Parquet
and it is why the modelling script stops being memory bound.

    FEATURES = ['B11_mean_k5', 'B11_variance_k5', ...]
    df = pd.read_parquet(
        '{FINAL_PARQUET}',
        columns=FEATURES + ['class', 'unique_id', 'poly_source',
                            'easting', 'northing', 'elevation'])

Point the transfer script at this file and swap pd.read_csv for
pd.read_parquet with a columns= list. Feature columns are
float32, class is int16, unique_id is int32.
=============================================================
""")