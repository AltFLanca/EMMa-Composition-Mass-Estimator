import argparse
import copy
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import torch
from torch.nn.utils import clip_grad_norm_

from models.hierarchy import MaterialFrontier
from models.losses import composition_losses
from training.config import prepare_experiment
from training.data import build_dataset, build_loader, model_inputs
from training.engine import build_model


def max_diff(left, right):
    common = left.keys() & right.keys()
    return max((left[k] - right[k]).abs().max().item() for k in common)


def run_once(base, batch, device, experiment, material_hierarchy, c):
    model = copy.deepcopy(base).to(device)
    model.reciprocal_scale = float(c)
    model.eval()
    optimizer = torch.optim.Adam(model.parameters(), lr=float(experiment['train']['lr']))
    before = {n: p.detach().clone() for n, p in model.named_parameters()}

    output = model(model_inputs(batch, model, device))
    target = batch['mass'].float().reshape(-1, 1).to(device)
    loss = torch.mean(torch.abs(torch.log(output.mass) - torch.log(target)))

    if output.composition is not None:
        loss_cfg = experiment.get('loss', {})
        hierarchy = None
        if loss_cfg.get('material_target') == 'frontier':
            hierarchy = MaterialFrontier(material_hierarchy, base.material_names_for_diagnostic)
        mat, ent = composition_losses(
            output.composition,
            batch['material_target'].to(device),
            target_mode=loss_cfg.get('material_target', 'presence'),
            hierarchy=hierarchy,
        )
        loss = loss + float(loss_cfg.get('lambda_material', 0.0)) * mat + float(loss_cfg.get('beta_entropy', 0.0)) * ent

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    grad_pre = {n: p.grad.detach().clone() for n, p in model.named_parameters() if p.grad is not None}
    clip_grad_norm_(model.parameters(), max_norm=float(experiment['train'].get('clip_grad_norm', 5.0)))
    grad_post = {n: p.grad.detach().clone() for n, p in model.named_parameters() if p.grad is not None}
    optimizer.step()
    updates = {n: p.detach() - before[n] for n, p in model.named_parameters()}

    return {
        'density': output.density.detach().cpu(),
        'volume': output.volume.detach().cpu(),
        'mass': output.mass.detach().cpu(),
        'loss': float(loss.detach()),
        'grad_pre': {k: v.cpu() for k, v in grad_pre.items()},
        'grad_post': {k: v.cpu() for k, v in grad_post.items()},
        'updates': {k: v.cpu() for k, v in updates.items()},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', required=True)
    parser.add_argument('--data', required=True)
    parser.add_argument('--c-values', type=float, nargs='+', required=True)
    parser.add_argument('--device', choices=['cpu', 'gpu'], default='cpu')
    args = parser.parse_args()

    experiment, data_config = prepare_experiment(args.experiment, args.data)
    device = torch.device('cuda' if args.device == 'gpu' and torch.cuda.is_available() else 'cpu')
    dataset = build_dataset(data_config, experiment, experiment['model'], 'val', is_train=False)
    loader = build_loader(dataset, int(experiment['train'].get('batch_size', 68)), 0, device.type, False, False, 0)
    batch = next(iter(loader))
    model = build_model(experiment, data_config, dataset)
    model.material_names_for_diagnostic = list(dataset.material_names)
    model.eval()

    reference_c = float(args.c_values[0])
    results = [(c, run_once(model, batch, device, experiment, dataset.material_hierarchy, c)) for c in args.c_values]
    ref = results[0][1]

    report = []

    print('c          density-scale volume-scale  mass-diff    loss-diff    grad-diff    clipped-grad  update-diff')
    for c, result in results:
        row = {
            'c': float(c),
            'density_scale': result['density'].abs().mean().item() / ref['density'].abs().mean().item(),
            'volume_scale': result['volume'].abs().mean().item() / ref['volume'].abs().mean().item(),
            'mass_diff': (result['mass'] - ref['mass']).abs().max().item(),
            'loss_diff': abs(result['loss'] - ref['loss']),
            'grad_diff': max_diff(result['grad_pre'], ref['grad_pre']),
            'clipped_grad_diff': max_diff(result['grad_post'], ref['grad_post']),
            'update_diff': max_diff(result['updates'], ref['updates']),
        }
        report.append(row)

        print(
            '%-10.5g %-13.5g %-13.5g %-12.3e %-12.3e %-12.3e %-12.3e %.3e'
            % (
                row['c'],
                row['density_scale'],
                row['volume_scale'],
                row['mass_diff'],
                row['loss_diff'],
                row['grad_diff'],
                row['clipped_grad_diff'],
                row['update_diff'],
            )
        )

    output_path = REPO_ROOT / 'artifacts' / 'c_invariance' / 'diagnosis.json'
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open('w', encoding='utf-8') as handle:
        json.dump(
            {
                'reference_c': reference_c,
                'results': report,
            },
            handle,
            indent=2,
        )
        handle.write('\n')

    print(f'Wrote JSON report: {output_path}')


if __name__ == '__main__':
    main()