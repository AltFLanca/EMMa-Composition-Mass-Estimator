from pathlib import Path

import torch

from models.depth2mass import Depth2Mass


def model_checkpoint_payload(model, experiment, material_names):
    return {
        'model_state_dict': model.state_dict(),
        'model_config': dict(model.config),
        'paper_label': experiment.get('paper_label'),
        'experiment_name': experiment.get('name'),
        'data_selection': dict(experiment.get('data', {})),
        'material_names': list(material_names),
    }


def save_model_checkpoint(path, model, experiment, material_names):
    torch.save(model_checkpoint_payload(model, experiment, material_names), path)


def save_training_checkpoint(path, model, optimizer, scheduler, epoch, experiment, material_names):
    payload = model_checkpoint_payload(model, experiment, material_names)
    payload.update(
        {
            'epoch': int(epoch),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': None if scheduler is None else scheduler.state_dict(),
        }
    )
    torch.save(payload, path)


def load_checkpoint_payload(path, map_location='cpu'):
    return torch.load(Path(path), map_location=map_location)


def load_model(path, map_location='cpu'):
    payload = load_checkpoint_payload(path, map_location=map_location)
    state = payload['model_state_dict']
    config = dict(payload['model_config'])
    densities = state.get('material_densities') if config.get('composition') else None
    model = Depth2Mass(config, material_densities=densities, pretrained_image_encoder=False)
    model.load_state_dict(state)
    return model, payload
