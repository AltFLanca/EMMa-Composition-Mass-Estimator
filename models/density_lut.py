import csv
import json
from pathlib import Path

import numpy as np


def load_density_lut(path, material_names, unit='g_cm3'):
    """Load densities in EMMa material order and return kg/m^3 values."""
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix == '.npy':
        densities = np.load(path, allow_pickle=False)
    elif suffix == '.json':
        with path.open(encoding='utf-8') as handle:
            values = json.load(handle)
        densities = _ordered_values(values, material_names)
    elif suffix == '.csv':
        with path.open(newline='', encoding='utf-8') as handle:
            rows = list(csv.DictReader(handle))
        values = {row['material']: float(row['density']) for row in rows}
        densities = _ordered_values(values, material_names)
    else:
        raise ValueError('Density LUT must be .npy, .json, or .csv')

    densities = np.asarray(densities, dtype=np.float32).reshape(-1)
    if densities.size != len(material_names):
        raise ValueError(
            'Density LUT has %d entries; EMMa has %d materials'
            % (densities.size, len(material_names))
        )
    if unit == 'g_cm3':
        densities = densities * 1000.0
    return densities


def _ordered_values(values, material_names):
    if isinstance(values, list):
        return values
    return [values[name] for name in material_names]
