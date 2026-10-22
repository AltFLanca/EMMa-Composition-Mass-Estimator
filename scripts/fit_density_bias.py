from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from emma import EmmaDataset
from scipy.optimize import least_squares


DENSITY_SCALE = 1000.0
POUNDS_TO_KG = 0.45359237
INCHES_TO_METRES = 0.0254
DIMENSION_PADDING_METRES = 0.005

ARTIFACT_DIR = (
    Path(__file__).resolve().parents[1]
    / "artifacts"
    / "density_bias"
)
ARTIFACT_FILE = ARTIFACT_DIR / "density_bias_fit.npz"
DEFAULT_JSON_OUTPUT = ARTIFACT_DIR / "density_bias_fit.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit the DensityBlock exponential prior to EMMa densities."
    )
    parser.add_argument(
        "--emma-root",
        type=Path,
        required=True,
        help="Directory containing the EMMa dataset files.",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=50,
        help="Number of equal-width density histogram bins (default: 50).",
    )
    parser.add_argument(
        "--max-density",
        type=float,
        default=1000.0,
        help="Upper histogram limit in kg/m^3 (default: 1000).",
    )
    parser.add_argument(
        "--density-at-one",
        type=float,
        default=1000.0,
        help=(
            "Required activation output at decoder input x=1 "
            "(default: 1000 kg/m^3)."
        ),
    )
    parser.add_argument(
        "--asins",
        type=Path,
        required=True,
        help="ASIN .npy file to fit.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=DEFAULT_JSON_OUTPUT,
        help=(
            "JSON file containing the fitted activation parameters "
            f"(default: {DEFAULT_JSON_OUTPUT})."
        ),
    )
    return parser.parse_args()


def record_density(dataset: EmmaDataset, asin: str) -> float | None:
    record = dataset.get_record(asin)
    dimensions = np.squeeze(
        np.asarray(dataset.get_package_dimensions(record), dtype=np.float64)
    )

    dimensions_metres = dimensions * INCHES_TO_METRES
    mass_kg = float(dataset.get_mass(record)) * POUNDS_TO_KG

    volume_m3 = float(
        np.prod(dimensions_metres + DIMENSION_PADDING_METRES)
    )
    density = mass_kg / volume_m3
    return density if np.isfinite(density) and density > 0 else None


def load_densities(dataset: EmmaDataset, asins: list) -> np.ndarray:
    densities = []
    for asin in asins:
        density = record_density(dataset, str(asin))
        densities.append(density)
    return np.asarray(densities, dtype=np.float64)


def fit_density_distribution(
    densities,
    bins=50,
    max_density=1000.0,
    density_at_one=1000.0,
):
    def constrained_function(x, fitting_parameters):
        weight, b_slow, difference = fitting_parameters
        b_fast = b_slow + difference

        a_slow = weight * np.exp(b_slow) / density_at_one
        a_fast = (1.0 - weight) * np.exp(b_fast) / density_at_one

        return (
            a_slow * np.exp(-b_slow * x)
            + a_fast * np.exp(-b_fast * x)
        )

    def residuals(fitting_parameters):
        predicted = constrained_function(
            x[mask],
            fitting_parameters,
        )
        return np.log(predicted) - np.log(y[mask])

    valid = densities[
        np.isfinite(densities)
        & (densities > 0)
        & (densities <= max_density)
    ]

    excluded = int(densities.size - valid.size)

    counts, edges = np.histogram(
        valid / DENSITY_SCALE,
        bins=bins,
        range=(0.0, max_density / DENSITY_SCALE),
    )

    x = (edges[:-1] + edges[1:]) / 2.0
    y = counts / counts.sum()

    mode = np.argmax(y)
    mask = (np.arange(bins) >= mode) & (counts > 0)

    result = least_squares(
        residuals,
        x0=[0.5, 2.0, 5.0],
        bounds=(
            [1e-5, 1e-5, 1e-5],
            [1.0 - 1e-5, 100.0, 100.0],
        ),
    )

    weight, b_slow, difference = result.x
    b_fast = b_slow + difference

    a_slow = weight * np.exp(b_slow) / density_at_one
    a_fast = (1.0 - weight) * np.exp(b_fast) / density_at_one

    parameters = np.array(
        [a_slow, b_slow, a_fast, b_fast],
        dtype=np.float64,
    )

    return x, y, mask, parameters, excluded


def fitted_function(x, parameters):
    a_slow, b_slow, a_fast, b_fast = parameters
    return (
        a_slow * np.exp(-b_slow * x)
        + a_fast * np.exp(-b_fast * x)
    )


def activation(x, parameters):
    return 1.0 / fitted_function(x, parameters)


def main() -> None:
    args = parse_args()
    dataset = EmmaDataset(args.emma_root)
    asins = np.load(args.asins, allow_pickle=False).astype(str).reshape(-1)
    densities = load_densities(dataset, asins)

    x, y, mask, parameters, excluded = fit_density_distribution(
        densities,
        bins=args.bins,
        max_density=args.max_density,
        density_at_one=args.density_at_one,
    )

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    np.savez(
        ARTIFACT_FILE,
        x=x,
        y=y,
        mask=mask,
        parameters=parameters,
        excluded=excluded,
        max_density=args.max_density,
        density_at_one=args.density_at_one,
    )

    a_slow, b_slow, a_fast, b_fast = parameters

    payload = {
        "type": "double_exponential",
        "a_slow": float(a_slow),
        "b_slow": float(b_slow),
        "a_fast": float(a_fast),
        "b_fast": float(b_fast),
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2) + "\n")

    print("Density records: %d" % densities.size)
    print(
        "Density range: %.3f to %.3f kg/m^3"
        % (float(np.nanmin(densities)), float(np.nanmax(densities)))
    )
    print(
        "Records above %.3f kg/m^3 excluded from fit: %d"
        % (args.max_density, excluded)
    )
    print("\nDensityBlock parameters:")
    print(f"a_slow = {a_slow:.17g}")
    print(f"b_slow = {b_slow:.17g}")
    print(f"a_fast = {a_fast:.17g}")
    print(f"b_fast = {b_fast:.17g}")
    print(f"activation(1) = {activation(1.0, parameters):.17g}")
    print("\nFit artifact saved to: %s" % ARTIFACT_FILE.resolve())
    print("Parameters saved to: %s" % args.output_json.resolve())


if __name__ == "__main__":
    main()