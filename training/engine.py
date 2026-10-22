import csv
import math
import os
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.nn.utils import clip_grad_norm_

from models.density_lut import load_density_lut
from models.depth2mass import Depth2Mass
from models.hierarchy import MaterialFrontier
from models.losses import composition_losses
from training.checkpoints import (
    load_checkpoint_payload,
    save_model_checkpoint,
    save_training_checkpoint,
)
from training.config import prepare_experiment, save_json
from training.data import build_dataset, build_loader, model_inputs
from utils.evaluation import composition_metrics, mass_metrics


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def unwrap(model):
    return model.module if isinstance(model, torch.nn.DataParallel) else model


def set_training_mode(model, freeze_image_encoder, mode='train'):
    if mode == 'eval':
        model.eval()
        return
    model.train()
    if freeze_image_encoder:
        unwrap(model).model_densenet.model.eval()


def build_model(experiment, data_config, dataset):
    config = dict(experiment['model'])
    config['num_materials'] = dataset.num_materials
    if config.get('use_text'):
        config['text_embedding_dims'] = dataset.text_embedding_dims

    densities = None
    if config.get('composition'):
        densities = load_density_lut(
            data_config['density_lut'], dataset.material_names,
            unit=data_config.get('density_lut_unit', 'g_cm3'),
        )
    return Depth2Mass(config, material_densities=densities, pretrained_image_encoder=True)


def build_optimizer(model, train_config):
    core = unwrap(model)
    freeze = bool(train_config.get('freeze_image_encoder', True))
    if freeze:
        for parameter in core.model_densenet.model.parameters():
            parameter.requires_grad = False
        return torch.optim.Adam(
            [p for p in model.parameters() if p.requires_grad],
            lr=float(train_config['lr']),
        )

    backbone = list(core.model_densenet.model.parameters())
    backbone_ids = {id(p) for p in backbone}
    others = [p for p in model.parameters() if p.requires_grad and id(p) not in backbone_ids]
    encoder_lr = float(train_config.get('encoder_lr', float(train_config['lr']) * 0.1))
    return torch.optim.Adam(
        [
            {'params': backbone, 'lr': encoder_lr},
            {'params': others, 'lr': float(train_config['lr'])},
        ]
    )


def build_scheduler(optimizer, train_config, total_steps):
    schedule = train_config.get('lr_schedule', 'constant')
    if schedule == 'constant' or schedule == 'legacy_poly':
        return None
    if schedule == 'cosine':
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=total_steps,
            eta_min=float(train_config.get('min_lr', 1e-6)),
        )
    raise ValueError('Unknown lr_schedule: %s' % schedule)


def apply_legacy_lr(optimizer, train_config, global_step, steps_per_epoch, epochs):
    half_epoch = epochs // 2
    max_lr = float(train_config.get('legacy_max_lr', 3.5e-5))
    min_lr = float(train_config.get('legacy_min_lr', 8e-6))
    power = float(train_config.get('legacy_lr_power', 0.9))
    half_steps = steps_per_epoch * half_epoch
    if half_steps <= 0:
        return
    if global_step < half_steps:
        current_lr = (max_lr - min_lr) * (global_step / half_steps) ** power + min_lr
    else:
        current_lr = (min_lr - max_lr) * (global_step / half_steps - 1) ** power + max_lr
    for group in optimizer.param_groups:
        group['lr'] = current_lr


def append_csv(path, row):
    path = Path(path)
    exists = path.exists()
    with path.open('a', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def validate(loader, model, device):
    model.eval()
    predictions, targets = [], []
    compositions, material_targets, fills = [], [], []

    with torch.inference_mode():
        for batch in loader:
            output = model(model_inputs(batch, unwrap(model), device, non_blocking=True))
            predictions.append(output.mass.detach().cpu())
            targets.append(batch['mass'].float().reshape(-1, 1))
            if output.composition is not None:
                compositions.append(output.composition.detach().cpu())
                material_targets.append(batch['material_target'].detach().cpu())
            if output.fill_factor is not None:
                fills.append(output.fill_factor.detach().cpu())

    prediction = torch.cat(predictions)
    target = torch.cat(targets)
    metrics = mass_metrics(prediction, target)

    comp = {}
    if compositions:
        core = unwrap(model)
        comp = composition_metrics(
            torch.cat(compositions),
            torch.cat(material_targets),
            core.material_densities.detach().cpu(),
            torch.cat(fills) if fills else None,
        )
    return metrics, comp


def run_training(experiment_path, data_path, expected_family, overrides=None, output_root='logs', resume=None):
    experiment, data_config = prepare_experiment(experiment_path, data_path, overrides)
    if experiment['family'] != expected_family:
        raise ValueError('Expected a %s config, got %s' % (expected_family, experiment['family']))

    train_config = experiment['train']
    seed = int(train_config.get('seed', 0))
    seed_everything(seed)

    if output_root is None:
        output_root = 'logs'
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path(output_root) / ('%s_%s' % (stamp, experiment['name']))
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(output_dir / 'run_config.json', {'experiment': experiment, 'data': data_config})

    device = torch.device(
        'cuda' if train_config.get('device', 'gpu') == 'gpu' and torch.cuda.is_available() else 'cpu'
    )
    if device.type == 'cuda':
        cudnn.benchmark = bool(train_config.get('cudnn_benchmark', True))

    train_dataset = build_dataset(data_config, experiment, experiment['model'], 'train', is_train=True)
    val_dataset = build_dataset(data_config, experiment, experiment['model'], 'val_train', is_train=False)

    model = build_model(experiment, data_config, train_dataset)
    freeze = bool(train_config.get('freeze_image_encoder', True))
    if freeze:
        for parameter in model.model_densenet.model.parameters():
            parameter.requires_grad = False

    if device.type == 'cuda' and torch.cuda.device_count() > 1 and train_config.get('data_parallel', True):
        model = torch.nn.DataParallel(model)
    model.to(device)

    batch_size = int(train_config.get('batch_size', 68))
    workers = int(train_config.get('workers', 8))
    val_workers = int(train_config.get('val_workers', 0))
    train_loader = build_loader(
        train_dataset, batch_size, workers, device.type, True, True, seed,
        train_config.get('prefetch_factor', 1), train_config.get('persistent_workers', False),
    )
    val_loader = build_loader(
        val_dataset, 1, val_workers, device.type, False, False, seed,
        train_config.get('prefetch_factor', 1), False,
    )

    epochs = int(train_config['epochs'])
    optimizer = build_optimizer(model, train_config)
    scheduler = build_scheduler(optimizer, train_config, epochs * len(train_loader))

    start_epoch = 1
    global_step = 0
    if resume:
        payload = load_checkpoint_payload(resume)
        unwrap(model).load_state_dict(payload['model_state_dict'])
        optimizer.load_state_dict(payload['optimizer_state_dict'])
        if scheduler is not None and payload.get('scheduler_state_dict') is not None:
            scheduler.load_state_dict(payload['scheduler_state_dict'])
        start_epoch = int(payload['epoch']) + 1
        global_step = (start_epoch - 1) * len(train_loader)

    hierarchy = None
    loss_config = experiment.get('loss', {})
    if loss_config.get('material_target', 'presence') == 'frontier':
        hierarchy = MaterialFrontier(train_dataset.material_hierarchy, train_dataset.material_names)

    print('Experiment:', experiment['name'])
    print('Paper label:', experiment.get('paper_label', '-'))
    print('Train size:', len(train_dataset), '| Val size:', len(val_dataset))
    print('Device:', device, '| Epochs:', epochs, '| Batch size:', batch_size)
    print('LR:', train_config['lr'], '| Schedule:', train_config.get('lr_schedule', 'constant'))

    clip_norm = float(train_config.get('clip_grad_norm', 5.0))
    training_mode = train_config.get('training_mode', 'train')
    material_weight = float(loss_config.get('lambda_material', 0.0))
    entropy_weight = float(loss_config.get('beta_entropy', 0.0))
    target_mode = loss_config.get('material_target', 'presence')
    history_path = output_dir / 'history.csv'
    best_alde = math.inf

    for epoch in range(start_epoch, epochs + 1):
        set_training_mode(model, freeze, training_mode)
        sum_total = sum_mass = sum_material = sum_entropy = 0.0
        seen = clipped = 0

        for batch_idx, batch in enumerate(train_loader):
            global_step += 1
            if train_config.get('lr_schedule', 'constant') == 'legacy_poly':
                apply_legacy_lr(optimizer, train_config, global_step, len(train_loader), epochs)

            core = unwrap(model)
            inputs = model_inputs(batch, core, device, non_blocking=True)
            mass = batch['mass'].float().reshape(-1, 1).to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            output = model(inputs)
            loss_mass = torch.mean(torch.abs(torch.log(output.mass) - torch.log(mass)))
            loss_material = loss_mass.new_zeros(())
            loss_entropy = loss_mass.new_zeros(())
            if output.composition is not None:
                loss_material, loss_entropy = composition_losses(
                    output.composition,
                    batch['material_target'].to(device, non_blocking=True),
                    target_mode=target_mode,
                    hierarchy=hierarchy,
                )
            loss = loss_mass + material_weight * loss_material + entropy_weight * loss_entropy
            if not torch.isfinite(loss):
                raise RuntimeError('Non-finite loss at epoch %d batch %d' % (epoch, batch_idx))

            loss.backward()
            norm = clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                max_norm=clip_norm,
                error_if_nonfinite=True,
            )
            clipped += int(float(norm.item()) > clip_norm)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            n = mass.shape[0]
            seen += n
            sum_total += float(loss.detach()) * n
            sum_mass += float(loss_mass.detach()) * n
            sum_material += float(loss_material.detach()) * n
            sum_entropy += float(loss_entropy.detach()) * n

            if (batch_idx + 1) % 100 == 0 or batch_idx + 1 == len(train_loader):
                print(
                    'epoch %02d/%02d batch %d/%d | mass %.4f | total %.4f | lr %.2e'
                    % (
                        epoch, epochs, batch_idx + 1, len(train_loader),
                        sum_mass / seen, sum_total / seen, optimizer.param_groups[-1]['lr'],
                    )
                )

        val_mass, val_comp = validate(val_loader, model, device)
        row = {
            'epoch': epoch,
            'optimizer_step': global_step,
            'train_total_loss': sum_total / seen,
            'train_mass_loss': sum_mass / seen,
            'train_material_loss': sum_material / seen,
            'train_entropy_loss': sum_entropy / seen,
            'clip_fraction': clipped / max(len(train_loader), 1),
            'lr': optimizer.param_groups[-1]['lr'],
            **{'val_' + k: v for k, v in val_mass.items()},
            **{'comp_' + k: v for k, v in val_comp.items()},
        }
        append_csv(history_path, row)
        print('Validation: ALDE %.4f | q %.4f' % (val_mass['alde'], val_mass['q']))
        if val_comp:
            print(
                'Composition: R@1 %.3f | labelled %.3f | mass-labelled %.3f'
                % (
                    val_comp['recall_at_1'], val_comp['labelled_fraction'],
                    val_comp['labelled_mass_fraction'],
                )
            )

        core = unwrap(model)
        if bool(train_config.get('save_all_checkpoints', True)):
            save_model_checkpoint(
                output_dir / ('epoch_%03d_model.ckpt' % epoch),
                core, experiment, train_dataset.material_names,
            )
            save_training_checkpoint(
                output_dir / ('epoch_%03d_training.ckpt' % epoch),
                core, optimizer, scheduler, epoch, experiment, train_dataset.material_names,
            )

        if val_mass['alde'] < best_alde:
            best_alde = val_mass['alde']
            save_model_checkpoint(output_dir / 'best_model.ckpt', core, experiment, train_dataset.material_names)

    print('Finished. Best validation ALDE: %.6f' % best_alde)
    print('Output:', output_dir)
    return output_dir
