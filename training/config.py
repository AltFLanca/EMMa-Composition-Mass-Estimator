import copy
import json
from pathlib import Path


def load_json(path):
    path = Path(path)
    with path.open(encoding='utf-8') as handle:
        return json.load(handle)


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as handle:
        json.dump(value, handle, indent=2)
        handle.write('\n')


def prepare_experiment(experiment_path, data_path, overrides=None):
    experiment = load_json(experiment_path)
    data = load_json(data_path)
    resolved = copy.deepcopy(experiment)
    resolved['_experiment_file'] = str(Path(experiment_path).resolve())
    resolved['_data_file'] = str(Path(data_path).resolve())

    for key, value in (overrides or {}).items():
        if value is None:
            continue

        if '.' in key:
            section, field = key.split('.', 1)
            resolved.setdefault(section, {})[field] = value
        else:
            resolved[key] = value

    density_file = resolved.get('model', {}).get('density_activation_file')
    if density_file:
        path = Path(density_file)
        if not path.is_absolute():
            path = Path(experiment_path).resolve().parent.parent / density_file
            if not path.exists():
                path = Path(experiment_path).resolve().parent / density_file
        resolved['model']['density_activation'] = load_json(path)

    return resolved, data
