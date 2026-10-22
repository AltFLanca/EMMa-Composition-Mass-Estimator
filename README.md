# EMMa-Composition-Mass-Estimator

Research code for category-agnostic object mass estimation on EMMa, including scalar-density baselines and the proposed bulk-material-composition models.

The repository is intentionally organized around **explicit model/configuration choices**, not hard-coded paper variant names. Labels such as `B1`, `B2`, `P2`, etc. are metadata on the experiment presets. A checkpoint stores the complete model configuration needed for inference and evaluation.

## Repository layout

```text
.
├── train_baseline.py            # scalar-density family
├── train_composition.py         # composition/fill-factor family
├── inference.py                 # checkpoint-driven inference
├── configs/
│   ├── data.example.json        # local paths (copy to data.json)
│   ├── density/                 # density activation fits
│   └── ablations/               # thesis ablations / model presets
├── dataset/                     # EMMa wrapper used by the model
├── models/                      # architecture and physical aggregation
├── training/                    # shared train/validation/checkpoint engine
├── scripts/                     # standalone research diagnostics
└── notebooks/
    ├── data/                    # dataset filtering/splitting/curation
    └── evaluation/              # one checkpoint-agnostic evaluation notebook
```

A flat source layout is kept deliberately so the existing `uv run ...` workflow works directly from a checkout without requiring the project to be installed first.

## Environment

This cleaned archive does **not** invent dependency versions. Before publishing, keep the `pyproject.toml` and `uv.lock` from the environment used for the final experiments (or add equivalent exact dependency metadata).

The code expects the EMMa Python wrapper (`emma`) and the packages used by the existing repository, including PyTorch, torchvision, NumPy, SciPy, pandas, OpenCV, h5py and Albumentations.

## Local data configuration

Copy the example once:

```bash
cp configs/data.example.json configs/data.json
```

Then edit `configs/data.json` with local paths. It contains the paths to:

- the final filtered train / validation / test ASIN lists;
- GLP and DepthAnythingV3 depth files;
- Watershed and SAM3 segmentation files;
- optional text embeddings;
- density LUT and material hierarchy;
- curated single-material files used by the evaluation notebook.

`configs/data.json` is ignored by Git because it is machine-specific.

## Training

### Scalar-density baseline

```bash
uv run train_baseline.py \
  --experiment configs/ablations/b1_glp_watershed_softplus.json \
  --data configs/data.json
```

### Composition model

```bash
uv run train_composition.py \
  --experiment configs/ablations/p2_frontier_rms.json \
  --data configs/data.json
```

The command-line interface intentionally exposes only common run-level overrides. Architecture and ablation choices live in JSON presets rather than becoming dozens of command-line flags.

Useful overrides include:

```bash
--epochs 24
--seed 1
--lr 1e-4
```

For composition models:

```bash
--lambda-material 0.05
--beta-entropy 0.0
```

To continue a run, use a `*_training.ckpt` checkpoint:

```bash
--resume logs/.../epoch_018_training.ckpt
```

### New follow-up experiments

For a new architecture/ablation, copy the closest JSON preset and change the explicit model/data/loss fields. `models/depth2mass.py` does not know what `B1` or `P3` means.

The important model switches are:

```text
use_material_prior    false / true
use_text              false / true
composition           false / true
fill_factor           false / true
geometry              original / rms
volume_activation     relu / softplus
reciprocal_scale      numeric c
```

Composition supervision is configured separately:

```text
material_target       presence / frontier
lambda_material       auxiliary material loss weight
beta_entropy          entropy loss weight
```

`frontier` means **most-specific positive frontier supervision**: an internal taxonomy node remains a valid target only when no more-specific positive descendant is available for that object.

## Checkpoints

Every new model checkpoint is self-describing. It stores:

- model `state_dict`;
- explicit model configuration;
- paper label and experiment name;
- selected depth/mask inputs;
- material-name order.

Training checkpoints additionally store optimizer and LR-scheduler state.

This is intentional: downstream inference/evaluation should not have to know whether a checkpoint came from `B2`, `P2`, an RMS ablation, or another future model.

## Inference

```bash
uv run inference.py \
  --checkpoint logs/.../best_model.ckpt \
  --data configs/data.json \
  --asins /path/to/asins.npy \
  --output predictions.csv
```

If `--asins` is omitted, `configs/data.json`'s test split is used.

The output contains mass, density and volume. Composition checkpoints additionally report fill factor (when present) and the top material.

## Evaluation

Use:

```text
notebooks/evaluation/evaluate_checkpoint.ipynb
```

Only the first configuration cell should normally need editing. The notebook loads the checkpoint's embedded model configuration, so it works across baseline/composition models and across original/RMS geometry without variant-specific monkey-patching.

The notebook includes the final research diagnostics developed for the thesis:

- overall mass performance;
- log-spaced mass-bin performance and signed bias;
- leaf-category metric distributions;
- curated single-material hierarchy-aware analysis;
- density-range-span analysis;
- volume / package-box scale diagnostics;
- point-cloud scale-equivariance diagnostics;
- fill-factor category studies;
- named-material and proportion-only category profiles;
- train/validation profile consistency and inter-category similarity;
- empirical single-material material-prior counterfactuals;
- text ablations when the checkpoint is text-conditioned.

Keep `SPLITS=("train", "val")` while developing models. Add `"test"` only after the final model/training recipe is locked.

## Thesis ablation presets

The presets are deliberately explicit. The principal sequence is:

| Preset | Purpose |
|---|---|
| `b1_cardoso_inherited.json` | Cardoso-style inherited B1 behavior / density prior |
| `b1_emma_density_fit.json` | same inherited setup, EMMa-refitted density activation |
| `b1_glp_watershed_relu.json` | stabilized EMMa B1, original ReLU volume head |
| `b1_glp_watershed_softplus.json` | isolates Softplus volume head |
| `b1_glp_sam3_softplus.json` | GLP + SAM3 |
| `b1_depth_anything_v3_watershed_softplus.json` | DA3 + Watershed |
| `b1_depth_anything_v3_sam3_softplus.json` | DA3 + SAM3 |
| `b2.json` | scalar-density baseline with material prior |
| `b3.json` | B2 + text |
| `p2_original_material_original_geometry.json` | P2 with original material-presence loss / original geometry |
| `p2_frontier_original_geometry.json` | isolates hierarchy-frontier supervision |
| `p2_frontier_rms.json` | isolates RMS-normalized, cubically equivariant geometry |
| `p1_frontier_rms.json` | P1 using the selected proposed-model formulation |
| `p3_frontier_rms.json` | P3 + text using the selected proposed-model formulation |

The four B1 depth/mask presets form the full `2 x 2` GLP/DA3 × Watershed/SAM3 study. Once the winning perception pair is selected, copy/update the B2/B3/P presets' `data` block to that same pair before final training. No Python changes are required.

For structural ablations, use the fixed development budget (planned: 18 epochs). After architecture and loss weights are locked, use the equal final budget (planned: 24 epochs) for B1/B2/B3/P1/P2/P3 and repeat the headline models across the final seeds.

## Density activation fitting

Fit the EMMa density prior from the **filtered training ASIN list**:

```bash
uv run scripts/fit_density_bias.py \
  --emma-root /path/to/EMMa \
  --asins /path/to/final_filtered_train_asins.npy \
  --output density_fit.png \
  --output-json configs/density/emma_refit.json
```

The stored density JSON is then selected through `density_activation_file` in a baseline preset.

## Reciprocal `c` diagnostic

The reciprocal density/volume scale can be checked without a full training sweep:

```bash
uv run scripts/verify_c_invariance.py \
  --experiment configs/ablations/b1_glp_watershed_softplus.json \
  --data configs/data.json \
  --c-values 1 0.060106 0.60106 6.0106 \
  --device cpu
```

It reports density scale, volume scale, mass differences, gradients and first optimizer updates. The production presets use `reciprocal_scale=1.0`; the historical inherited preset retains Cardoso's original reciprocal value for reproduction.

## Training details worth documenting

The stable presets preserve the choices reached during development:

- Adam optimizer;
- log-L1 mass loss;
- gradient clipping;
- stable log-domain density activation with configurable log clipping;
- pretrained DenseNet core frozen while the trainable projection/head remains in training mode;
- validation datasets constructed with `is_train=False`;
- deterministic worker seeding for repeated final runs.

`b1_cardoso_inherited.json` intentionally retains the inherited historical `model.eval()`-during-training behavior and legacy LR schedule so that the adaptation can be reproduced. The stabilized presets use the corrected training mode.

## Reproducibility

Each run writes the fully resolved experiment/data configuration alongside `history.csv` and checkpoints. Keep the final Git commit hash, environment lockfile, final data-file hashes and seeds with the thesis results.

GPU execution can still vary across platforms/PyTorch releases even when seeds are fixed, so final headline models should be reported over repeated seeds rather than relying on bitwise reproducibility.
