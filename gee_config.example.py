"""Copy to gee_local_config.py beside the scripts and enter your settings.

Publish this example, not your completed local configuration. Every script reads
that same local file; their processing code does not need editing for these settings.
"""
from pathlib import Path

EE_PROJECT = 'your-ee-project'
OUTPUT_DIR = Path('data')  # Relative paths resolve from the working directory.
TARGET_CRS = 'EPSG:3116'  # Colombia analysis CRS; choose one for your study area.
OUTPUT_NAME = 'pixel_data_2020_seasons.parquet'
ANNUAL_OUTPUT_NAME = 'pixel_data_2020_annual.parquet'
SKIPPED_NAME = 'pixel_data_2020_seasons_skipped.json'
# Optional: where scripts 3 to 9 write. Defaults to OUTPUT_DIR / 'analysis'.
# ANALYSIS_DIR = Path('data/analysis')

# Raw class codes as stored in the polygon assets (note the gap at 3).
# coffee_common.py remaps these to 0..4 (RAW_TO_CLEAN) for scripts 3 to 9.
CLASS_NAMES = {0: 'Non-Shade Coffee', 1: 'Shade Coffee', 2: 'Forest',
               4: 'Open', 5: 'Urban'}

# Each source must use your numeric class codes in its 'class' property.
# Keep source names unique. Changing source order/content can change polygon IDs.
POLYGON_SOURCES = [
    {'name': 'reference',
     'asset': 'projects/your-ee-project/assets/reference_polygons'},
]

# Optional hooks may be defined in your local configuration when needed:
#   prepare_source(collection, source) -> Earth Engine FeatureCollection
#   prepare_annual_frame(df) -> pandas DataFrame
# If omitted, source labels are used unchanged and annual preparation only
# selects annual/static columns. No dataset-specific corrections are required.
