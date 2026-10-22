from __future__ import annotations

import argparse
import csv
import math
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

EPOCH_RE = re.compile(r"epoch_(\d+)_")
OUTPUT_NAMES = ("density", "volume", "mass", "composition", "fill_factor")
COMPONENT_PREFIXES = {
    "densityBlock": "density_tower",
    "volumeBlock": "volume_tower",
    "compositionBlock": "composition_density_path",
    "fillFactorBlock": "fill_factor_density_path",
    "model_densenet": "image_encoder",
    "model_pc": "point_encoder",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Probe saved checkpoints on fixed batches without taking optimizer steps.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--run-config", help="Defaults to <checkpoint-dir>/run_config.json")
    parser.add_argument("--pattern", help="Checkpoint glob; defaults to training ckpts, then model ckpts")
    parser.add_argument("--include-best", action="store_true")
    parser.add_argument("--start-epoch", type=int)
    parser.add_argument("--end-epoch", type=int)
    parser.add_argument("--every", type=int, default=1)
    parser.add_argument("--max-checkpoints", type=int)

    parser.add_argument("--split", choices=["train", "val_train", "val", "test"], default="train")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--max-batches", type=int, default=1)
    parser.add_argument(
        "--train-augmentation",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use training augmentation when probing the train split; batches are cached once.",
    )
    parser.add_argument("--model-mode", choices=["config", "train", "eval"], default="config")
    parser.add_argument("--clip-grad-norm", type=float)
    parser.add_argument("--device", choices=["auto", "cpu", "gpu"], default="auto")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--workers", type=int, default=0)

    parser.add_argument("--output-json")
    parser.add_argument("--output-csv")
    parser.add_argument("--summary-only", action="store_true")
    return parser.parse_args()


def choose_device(name: str) -> torch.device:
    if name == "gpu" and not torch.cuda.is_available():
        raise RuntimeError("--device gpu requested, but CUDA is unavailable")
    if name == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def epoch_of(path: Path) -> int | None:
    match = EPOCH_RE.search(path.name)
    return int(match.group(1)) if match else None


def discover_checkpoints(args: argparse.Namespace, directory: Path) -> list[Path]:
    pattern = args.pattern
    if pattern:
        paths = list(directory.glob(pattern))
    else:
        paths = list(directory.glob("epoch_*_training.ckpt"))
        if not paths:
            paths = list(directory.glob("epoch_*_model.ckpt"))

    numbered = []
    extras = []
    for path in paths:
        epoch = epoch_of(path)
        if epoch is None:
            extras.append(path)
        elif (args.start_epoch is None or epoch >= args.start_epoch) and (
            args.end_epoch is None or epoch <= args.end_epoch
        ):
            numbered.append(path)

    if args.every < 1:
        raise ValueError("--every must be at least 1")
    numbered.sort(key=lambda p: (epoch_of(p), p.name))
    numbered = numbered[:: args.every]

    if args.max_checkpoints is not None:
        if args.max_checkpoints < 1:
            raise ValueError("--max-checkpoints must be at least 1")
        if len(numbered) > args.max_checkpoints:
            indices = np.linspace(0, len(numbered) - 1, args.max_checkpoints).round().astype(int)
            numbered = [numbered[i] for i in sorted(set(indices.tolist()))]

    if args.include_best:
        best = directory / "best_model.ckpt"
        if best.exists() and best not in extras:
            extras.append(best)

    result = numbered + sorted(extras, key=lambda p: p.name)
    if not result:
        raise FileNotFoundError(f"No checkpoints found in {directory}")
    return result


def read_history(path: Path) -> dict[int, dict[str, Any]]:
    if not path.exists():
        return {}
    history = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            try:
                epoch = int(row["epoch"])
            except (KeyError, TypeError, ValueError):
                continue
            parsed = {}
            for key, value in row.items():
                try:
                    parsed[key] = float(value) if value not in (None, "") else value
                except ValueError:
                    parsed[key] = value
            history[epoch] = parsed
    return history


def component(name: str) -> str:
    for prefix, label in COMPONENT_PREFIXES.items():
        if name.startswith(prefix):
            return label
    return "other"


def tensor_stats(tensors: list[torch.Tensor]) -> dict[str, float | int | None]:
    """Full descriptive stats for small outputs/local gradients."""
    if not tensors:
        return {}
    values = torch.cat([x.detach().reshape(-1).cpu().double() for x in tensors])
    squared = torch.sum(values.square())
    return {
        "count": values.numel(),
        "mean": values.mean().item(),
        "median": values.median().item(),
        "mean_abs": values.abs().mean().item(),
        "rms": math.sqrt(squared.item() / values.numel()),
        "l2_norm": math.sqrt(squared.item()),
        "min": values.min().item(),
        "max": values.max().item(),
        "max_abs": values.abs().max().item(),
    }


def norm_stats(tensors) -> dict[str, float | int]:
    """Streaming norm stats for potentially large parameter/state collections."""
    count = 0
    squared = 0.0
    max_abs = 0.0
    for tensor in tensors:
        if tensor is None or tensor.numel() == 0:
            continue
        value = tensor.detach().double()
        count += value.numel()
        squared += value.square().sum().item()
        max_abs = max(max_abs, value.abs().max().item())
    return {
        "count": count,
        "l2_norm": math.sqrt(squared),
        "rms": math.sqrt(squared / count) if count else 0.0,
        "max_abs": max_abs,
    }


def grouped_stats(named_tensors) -> dict[str, dict[str, float | int]]:
    groups = defaultdict(list)
    for name, tensor in named_tensors:
        groups[component(name)].append(tensor)
    return {name: norm_stats(values) for name, values in sorted(groups.items())}


def mean_group_stats(per_batch: list[dict[str, dict[str, float | int]]]) -> dict[str, dict[str, float]]:
    names = sorted({name for batch in per_batch for name in batch})
    result = {}
    for name in names:
        values = [batch[name] for batch in per_batch if name in batch]
        result[name] = {
            "l2_norm": float(np.mean([x["l2_norm"] for x in values])),
            "rms": float(np.mean([x["rms"] for x in values])),
            "max_abs": float(np.mean([x["max_abs"] for x in values])),
        }
    return result


def optimizer_summary(optimizer: torch.optim.Optimizer, saved: bool) -> dict[str, Any]:
    states = defaultdict(list)
    steps = []
    for state in optimizer.state.values():
        for name, value in state.items():
            if name == "step":
                steps.append(float(value.item() if torch.is_tensor(value) else value))
            elif torch.is_tensor(value) and value.is_floating_point():
                states[name].append(value)
    return {
        "saved_state_loaded": saved,
        "optimizer_class": optimizer.__class__.__name__,
        "learning_rates": [float(group["lr"]) for group in optimizer.param_groups],
        "state_steps": {"min": min(steps) if steps else None, "max": max(steps) if steps else None},
        "state_tensors": {name: norm_stats(values) for name, values in sorted(states.items())},
    }


def cache_batches(args, experiment, data_config, device, seed):
    from training.data import build_dataset, build_loader
    from training.engine import seed_everything

    if args.max_batches < 1:
        raise ValueError("--max-batches must be at least 1")
    batch_size = args.batch_size if args.batch_size is not None else int(experiment["train"]["batch_size"])
    if batch_size < 1:
        raise ValueError("--batch-size must be at least 1")

    seed_everything(seed)
    dataset = build_dataset(
        data_config,
        experiment,
        experiment["model"],
        args.split,
        is_train=args.split == "train" and args.train_augmentation,
    )
    loader = build_loader(
        dataset,
        batch_size=batch_size,
        workers=args.workers,
        device_type=device.type,
        shuffle=False,
        drop_last=False,
        seed=seed,
        prefetch_factor=1,
        persistent_workers=False,
    )
    batches = []
    for batch in loader:
        batches.append(batch)
        if len(batches) == args.max_batches:
            break
    if not batches:
        raise RuntimeError("Selected split produced no probe batches")
    return dataset, batches, batch_size


def build_hierarchy(experiment, dataset):
    if experiment.get("loss", {}).get("material_target", "presence") != "frontier":
        return None
    from models.hierarchy import MaterialFrontier

    return MaterialFrontier(dataset.material_hierarchy, dataset.material_names)


def diagnostic_loss(output, batch, device, experiment, hierarchy):
    from models.losses import composition_losses

    target = batch["mass"].float().reshape(-1, 1).to(device)
    mass = torch.mean(torch.abs(torch.log(output.mass) - torch.log(target)))
    material = mass.new_zeros(())
    entropy = mass.new_zeros(())
    loss_cfg = experiment.get("loss", {})
    if output.composition is not None:
        material, entropy = composition_losses(
            output.composition,
            batch["material_target"].to(device),
            target_mode=loss_cfg.get("material_target", "presence"),
            hierarchy=hierarchy,
        )
    total = mass + float(loss_cfg.get("lambda_material", 0.0)) * material
    total += float(loss_cfg.get("beta_entropy", 0.0)) * entropy
    return total, {"mass": mass, "material": material, "entropy": entropy, "total": total}, target


def inspect_checkpoint(path, experiment, dataset, batches, device, seed, mode, clip_norm, history_row):
    from training.checkpoints import load_model
    from training.data import model_inputs
    from training.engine import build_optimizer, seed_everything, set_training_mode

    model, payload = load_model(path, map_location="cpu")
    if payload.get("material_names") is not None and list(payload["material_names"]) != list(dataset.material_names):
        raise ValueError(f"Material ordering in {path.name} does not match the run dataset")

    model.to(device)
    train_cfg = experiment["train"]
    optimizer = build_optimizer(model, train_cfg)
    saved_state = payload.get("optimizer_state_dict")
    if saved_state is not None:
        optimizer.load_state_dict(saved_state)

    set_training_mode(model, bool(train_cfg.get("freeze_image_encoder", True)), mode=mode)
    parameter_stats = grouped_stats((name, p) for name, p in model.named_parameters())
    optimizer_stats = optimizer_summary(optimizer, saved_state is not None)
    hierarchy = build_hierarchy(experiment, dataset)
    buffers = {name: value.detach().clone() for name, value in model.named_buffers()}

    outputs = defaultdict(list)
    output_grads = defaultdict(list)
    losses = defaultdict(list)
    pre_norms, post_norms, clip_scales, clipped = [], [], [], []
    gradient_groups = []
    total_samples = 0

    for batch_index, batch in enumerate(batches):
        with torch.no_grad():
            for name, value in model.named_buffers():
                if name in buffers:
                    value.copy_(buffers[name])
        seed_everything(seed + 100_000 + batch_index)
        optimizer.zero_grad(set_to_none=True)

        output = model(model_inputs(batch, model, device))
        for name in (*OUTPUT_NAMES, "concentration"):
            value = getattr(output, name, None)
            if torch.is_tensor(value) and value.requires_grad:
                value.retain_grad()

        loss, loss_parts, target = diagnostic_loss(output, batch, device, experiment, hierarchy)
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite diagnostic loss in {path.name}, batch {batch_index}")

        n = target.shape[0]
        total_samples += n
        for name in OUTPUT_NAMES:
            value = getattr(output, name, None)
            if torch.is_tensor(value):
                outputs[name].append(value.detach().cpu())
        outputs["target_mass"].append(target.detach().cpu())

        loss.backward()
        for name in (*OUTPUT_NAMES, "concentration"):
            value = getattr(output, name, None)
            if torch.is_tensor(value) and value.grad is not None:
                output_grads[name].append(value.grad.detach().cpu())

        named_grads = [(name, p.grad) for name, p in model.named_parameters() if p.grad is not None]
        pre = norm_stats(grad for _, grad in named_grads)["l2_norm"]
        gradient_groups.append(grouped_stats(named_grads))

        trainable = [p for p in model.parameters() if p.requires_grad]
        if clip_norm > 0:
            returned = float(clip_grad_norm_(trainable, clip_norm, error_if_nonfinite=True).item())
            was_clipped = returned > clip_norm
        else:
            returned = pre
            was_clipped = False
        post = norm_stats(p.grad for p in trainable if p.grad is not None)["l2_norm"]

        pre_norms.append(pre)
        post_norms.append(post)
        clip_scales.append(post / pre if pre else 1.0)
        clipped.append(was_clipped)
        for name, value in loss_parts.items():
            losses[name].append((float(value.detach()), n))

    with torch.no_grad():
        for name, value in model.named_buffers():
            if name in buffers:
                value.copy_(buffers[name])
    optimizer.zero_grad(set_to_none=True)

    def mean_loss(name):
        values = losses[name]
        count = sum(n for _, n in values)
        return sum(value * n for value, n in values) / count

    epoch = payload.get("epoch", epoch_of(path))
    return {
        "checkpoint": str(path),
        "checkpoint_name": path.name,
        "epoch": int(epoch) if epoch is not None else None,
        "checkpoint_kind": "training" if saved_state is not None else "model",
        "reciprocal_scale": float(model.reciprocal_scale),
        "model_mode": mode,
        "probe_batches": len(batches),
        "probe_samples": total_samples,
        "losses": {name: mean_loss(name) for name in losses},
        "output_summaries": {name: tensor_stats(values) for name, values in outputs.items()},
        "local_gradient_summaries": {name: tensor_stats(values) for name, values in output_grads.items()},
        "parameter_stats": {"by_component": parameter_stats},
        "gradient_clipping": {
            "max_norm": clip_norm,
            "preclip_norm_mean": float(np.mean(pre_norms)),
            "preclip_norm_max": float(np.max(pre_norms)),
            "postclip_norm_mean": float(np.mean(post_norms)),
            "clip_scale_mean": float(np.mean(clip_scales)),
            "diagnostic_clip_fraction": float(np.mean(clipped)),
            "preclip_by_component_mean": mean_group_stats(gradient_groups),
        },
        "optimizer": optimizer_stats,
        "recorded_history": dict(history_row) if history_row else None,
    }


def fmt(value: Any) -> str:
    if value is None:
        return "-"
    try:
        value = float(value)
    except (TypeError, ValueError):
        return str(value)
    return f"{value:.3e}" if math.isfinite(value) else str(value)


def learning_rate(report) -> float | None:
    rates = report["optimizer"]["learning_rates"]
    return rates[-1] if rates else None


def print_summary(reports):
    headers = ("checkpoint", "epoch", "density", "volume", "mass", "loss", "grad-pre", "grad-post", "clip-x", "clip%", "lr", "hist-clip")
    print("\n" + " ".join(f"{h:>11}" for h in headers))
    for report in reports:
        out = report["output_summaries"]
        clip = report["gradient_clipping"]
        hist = report.get("recorded_history") or {}
        values = (
            report["checkpoint_name"][:24], report["epoch"], out["density"]["mean"], out["volume"]["mean"],
            out["mass"]["mean"], report["losses"]["total"], clip["preclip_norm_mean"],
            clip["postclip_norm_mean"], clip["clip_scale_mean"], clip["diagnostic_clip_fraction"],
            learning_rate(report), hist.get("clip_fraction"),
        )
        print(f"{str(values[0]):>24} {str(values[1]):>5} " + " ".join(f"{fmt(v):>11}" for v in values[2:]))


def print_details(report):
    print(f"\n{report['checkpoint_name']} | epoch {report['epoch']} | {report['checkpoint_kind']}")
    print("outputs:")
    for name, stats in report["output_summaries"].items():
        print(f"  {name:12} mean={stats['mean']:.4e} rms={stats['rms']:.4e} min={stats['min']:.4e} max={stats['max']:.4e}")
    print("dL/d(output):")
    for name, stats in report["local_gradient_summaries"].items():
        print(f"  {name:12} rms={stats['rms']:.4e} l2={stats['l2_norm']:.4e} max={stats['max_abs']:.4e}")

    clip = report["gradient_clipping"]
    print(
        "clipping: max=%.4g pre=%.4e post=%.4e scale=%.4e fraction=%.3f"
        % (clip["max_norm"], clip["preclip_norm_mean"], clip["postclip_norm_mean"], clip["clip_scale_mean"], clip["diagnostic_clip_fraction"])
    )
    print("parameter / preclip-gradient L2 by component:")
    params = report["parameter_stats"]["by_component"]
    grads = clip["preclip_by_component_mean"]
    for name in sorted(set(params) | set(grads)):
        p = params.get(name, {}).get("l2_norm", float("nan"))
        g = grads.get(name, {}).get("l2_norm", float("nan"))
        ratio = g / p if p and math.isfinite(g) else float("nan")
        print(f"  {name:26} param={p:.4e} grad={g:.4e} grad/param={ratio:.4e}")

    opt = report["optimizer"]
    steps = opt["state_steps"]
    print(f"optimizer: {opt['optimizer_class']} saved={opt['saved_state_loaded']} lr={', '.join(fmt(x) for x in opt['learning_rates']) or '-'}")
    if steps["min"] is not None:
        print(f"  steps: {steps['min']:.0f} .. {steps['max']:.0f}")
    for name, stats in opt["state_tensors"].items():
        print(f"  {name:12} rms={stats['rms']:.4e} l2={stats['l2_norm']:.4e} max={stats['max_abs']:.4e}")

    hist = report.get("recorded_history") or {}
    if hist:
        print(f"history: clip_fraction={fmt(hist.get('clip_fraction'))} lr={fmt(hist.get('lr'))}")


def csv_row(report) -> dict[str, Any]:
    out = report["output_summaries"]
    grads = report["local_gradient_summaries"]
    clip = report["gradient_clipping"]
    hist = report.get("recorded_history") or {}
    row = {
        "checkpoint": report["checkpoint_name"], "epoch": report["epoch"], "checkpoint_kind": report["checkpoint_kind"],
        "reciprocal_scale": report["reciprocal_scale"], "model_mode": report["model_mode"],
        "probe_batches": report["probe_batches"], "probe_samples": report["probe_samples"],
        **{f"loss_{name}": value for name, value in report["losses"].items()},
        "grad_norm_preclip_mean": clip["preclip_norm_mean"], "grad_norm_preclip_max": clip["preclip_norm_max"],
        "grad_norm_postclip_mean": clip["postclip_norm_mean"], "clip_scale_mean": clip["clip_scale_mean"],
        "diagnostic_clip_fraction": clip["diagnostic_clip_fraction"], "clip_max_norm": clip["max_norm"],
        "optimizer_lr": learning_rate(report), "recorded_epoch_clip_fraction": hist.get("clip_fraction"),
        "recorded_epoch_lr": hist.get("lr"), "recorded_train_total_loss": hist.get("train_total_loss"),
        "recorded_train_mass_loss": hist.get("train_mass_loss"),
    }
    for name in ("density", "volume", "mass", "target_mass", "fill_factor"):
        if name in out:
            for field in ("mean", "median", "rms", "min", "max", "mean_abs"):
                row[f"{name}_{field}"] = out[name][field]
    for name in OUTPUT_NAMES:
        if name in grads:
            for field in ("mean_abs", "rms", "l2_norm", "max_abs"):
                row[f"{name}_grad_{field}"] = grads[name][field]
    for name in sorted(set(report["parameter_stats"]["by_component"]) | set(clip["preclip_by_component_mean"])):
        key = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
        p = report["parameter_stats"]["by_component"].get(name, {}).get("l2_norm")
        g = clip["preclip_by_component_mean"].get(name, {}).get("l2_norm")
        row[f"{key}_param_l2"] = p
        row[f"{key}_grad_pre_l2_mean"] = g
        row[f"{key}_grad_to_param_l2"] = g / p if p and g is not None else None
        row[f"{key}_grad_pre_max_abs_mean"] = clip["preclip_by_component_mean"].get(name, {}).get("max_abs")
    return row


def write_csv(path: Path, reports):
    rows = [csv_row(report) for report in reports]
    fields = list(dict.fromkeys(key for row in rows for key in row))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    from training.config import load_json, save_json

    args = parse_args()
    directory = Path(args.checkpoint_dir).expanduser().resolve()
    if not directory.is_dir():
        raise NotADirectoryError(directory)

    config_path = Path(args.run_config).expanduser().resolve() if args.run_config else directory / "run_config.json"
    config = load_json(config_path)
    experiment, data_config = config["experiment"], config["data"]
    train_cfg = experiment["train"]

    seed = int(args.seed if args.seed is not None else train_cfg.get("seed", 0))
    device = choose_device(args.device)
    clip_norm = float(args.clip_grad_norm if args.clip_grad_norm is not None else train_cfg.get("clip_grad_norm", 5.0))
    mode = train_cfg.get("training_mode", "train") if args.model_mode == "config" else args.model_mode
    if mode not in ("train", "eval"):
        raise ValueError(f"Resolved model mode must be train or eval, got {mode!r}")

    checkpoints = discover_checkpoints(args, directory)
    history = read_history(directory / "history.csv")
    dataset, batches, batch_size = cache_batches(args, experiment, data_config, device, seed)

    print(f"Probing {len(checkpoints)} checkpoint(s) on {len(batches)} cached {args.split} batch(es) ({batch_size=}, {device=}, mode={mode}, clip={clip_norm:g}); optimizer.step() is never called.")
    reports = []
    for index, path in enumerate(checkpoints, 1):
        print(f"[{index}/{len(checkpoints)}] {path.name}")
        epoch = epoch_of(path)
        reports.append(inspect_checkpoint(path, experiment, dataset, batches, device, seed, mode, clip_norm, history.get(epoch)))
        if device.type == "cuda":
            torch.cuda.empty_cache()

    print_summary(reports)
    if not args.summary_only:
        for report in reports:
            print_details(report)

    output_dir = REPO_ROOT / "artifacts" / "checkpoint_diagnosis" / directory.name
    json_path = Path(args.output_json).expanduser() if args.output_json else output_dir / "diagnosis.json"
    csv_path = Path(args.output_csv).expanduser() if args.output_csv else output_dir / "diagnosis.csv"
    metadata = {
        "checkpoint_dir": str(directory), "run_config": str(config_path), "device": str(device),
        "split": args.split, "batch_size": batch_size, "max_batches": args.max_batches,
        "train_augmentation": bool(args.split == "train" and args.train_augmentation),
        "model_mode": mode, "clip_grad_norm": clip_norm, "seed": seed, "torch_version": torch.__version__,
        "note": "Gradients are fixed-batch checkpoint-state probes, not historical training gradients; no optimizer step is taken.",
    }
    save_json(json_path, {"metadata": metadata, "checkpoints": reports})
    write_csv(csv_path, reports)
    print(f"\nWrote {json_path}\nWrote {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
