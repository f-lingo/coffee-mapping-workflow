"""Prepare annual features using the same required local settings as Script 1.

Reads annual/static features and metadata only; leaves the input file untouched.
Rows, missing values, and coordinates are preserved. Dataset-specific corrections
belong in gee_local_config.py's optional prepare_annual_frame() hook.
Dependencies: pandas, pyarrow.
"""
from pathlib import Path
import importlib.util

import pandas as pd
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
    ('OUTPUT_DIR', 'OUTPUT_NAME', 'ANNUAL_OUTPUT_NAME'),
)
OUTPUT_DIR = Path(local_config.OUTPUT_DIR)
OUTPUT_NAME = local_config.OUTPUT_NAME
ANNUAL_OUTPUT_NAME = local_config.ANNUAL_OUTPUT_NAME
SEASON_SUFFIXES = ('_JFM', '_AMJ', '_JAS', '_OND')
prepare_annual_frame = getattr(local_config, 'prepare_annual_frame', None)
INPUT_FILE = OUTPUT_DIR / OUTPUT_NAME
OUTPUT_FILE = OUTPUT_DIR / ANNUAL_OUTPUT_NAME


def prepare_annual(input_file=INPUT_FILE, output_file=OUTPUT_FILE):
    """Select annual columns, apply optional local preparation, and save."""
    input_file, output_file = Path(input_file), Path(output_file)
    if input_file.resolve() == output_file.resolve():
        raise ValueError('Input and output paths must differ.')

    columns = pq.ParquetFile(input_file).schema_arrow.names
    keep_columns = [c for c in columns if not c.endswith(SEASON_SUFFIXES)]
    df = pd.read_parquet(input_file, columns=keep_columns)
    if prepare_annual_frame is not None:
        df = prepare_annual_frame(df)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_file, index=False)
    print(f'Dropped {len(columns) - len(keep_columns)} seasonal columns.')
    print(f'Saved {len(df):,} rows and {len(df.columns)} columns to {output_file}')
    return df


if __name__ == '__main__':
    prepare_annual()
