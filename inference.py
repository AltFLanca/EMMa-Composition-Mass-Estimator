import argparse
import csv

import torch

from training.checkpoints import load_model
from training.config import load_json
from training.data import build_dataset, build_loader, model_inputs
from utils.evaluation import mass_metrics


def main():
    parser = argparse.ArgumentParser(description='Run inference from a self-describing checkpoint.')
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--data', required=True, help='Local EMMa paths JSON.')
    parser.add_argument('--asins', help='Optional ASIN .npy; defaults to the test split from data config.')
    parser.add_argument('--output', default='predictions.csv')
    parser.add_argument('--device', choices=['gpu', 'cpu'], default='gpu')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--workers', type=int, default=0)
    args = parser.parse_args()

    device = torch.device('cuda' if args.device == 'gpu' and torch.cuda.is_available() else 'cpu')
    data_config = load_json(args.data)
    model, payload = load_model(args.checkpoint)
    experiment = {
        'data': payload['data_selection'],
        'model': payload['model_config'],
    }
    dataset = build_dataset(
        data_config, experiment, payload['model_config'], 'test', is_train=False,
        asins_path=args.asins,
    )
    loader = build_loader(
        dataset, args.batch_size, args.workers, device.type,
        shuffle=False, drop_last=False, seed=0,
    )

    model.to(device).eval()
    rows = []
    predicted_mass, target_mass = [], []
    material_names = payload.get('material_names', [])

    with torch.inference_mode():
        for batch in loader:
            output = model(model_inputs(batch, model, device, non_blocking=True))
            pred = output.mass.detach().cpu().reshape(-1)
            truth = batch['mass'].float().reshape(-1)
            predicted_mass.append(pred)
            target_mass.append(truth)

            top_material = None
            if output.composition is not None:
                top_material = output.composition.detach().cpu().argmax(dim=1)

            for i, asin in enumerate(batch['filename']):
                row = {
                    'asin': str(asin),
                    'target_mass_kg': float(truth[i]),
                    'predicted_mass_kg': float(pred[i]),
                    'predicted_density_kg_m3': float(output.density.detach().cpu().reshape(-1)[i]),
                    'predicted_volume_m3': float(output.volume.detach().cpu().reshape(-1)[i]),
                    'fill_factor': '' if output.fill_factor is None else float(output.fill_factor.detach().cpu().reshape(-1)[i]),
                }
                if top_material is not None and material_names:
                    row['top_material'] = material_names[int(top_material[i])]
                rows.append(row)

    metrics = mass_metrics(torch.cat(predicted_mass), torch.cat(target_mass))
    print('Checkpoint:', args.checkpoint)
    print('Experiment:', payload.get('experiment_name'))
    for name, value in metrics.items():
        print('%-20s %.6f' % (name, value))

    with open(args.output, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print('Saved:', args.output)


if __name__ == '__main__':
    main()
