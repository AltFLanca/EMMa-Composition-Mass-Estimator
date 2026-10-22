import random

import numpy as np
import torch

from dataset.base_dataset import get_dataset


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def dataset_kwargs(data_config, experiment, model_config):
    selection = experiment['data']
    depth_name = selection['depth']
    segmentation_name = selection.get('segmentation')

    return {
        'pc_dims': int(selection['pc_dims']),
        'pointcloud_input_divisor': float(model_config.get('pointcloud_input_divisor', 5.0)),
        'emma_root': data_config['emma_root'],
        'emma_depth_path': data_config['depth'][depth_name],
        'emma_depth_model': depth_name,
        'emma_segmentation_path': (
            None if segmentation_name in (None, 'none')
            else data_config['segmentation'][segmentation_name]
        ),
        'emma_text_embedding_path': (
            data_config.get('text_embeddings') if model_config.get('use_text') else None
        ),
    }


def build_dataset(data_config, experiment, model_config, split, is_train=False, asins_path=None):
    if asins_path is None:
        asins_path = data_config['asins'][split]
    return get_dataset(
        emma_asins_npy=asins_path,
        is_train=is_train,
        **dataset_kwargs(data_config, experiment, model_config),
    )


def build_loader(dataset, batch_size, workers, device_type, shuffle, drop_last, seed,
                 prefetch_factor=1, persistent_workers=False):
    generator = torch.Generator()
    generator.manual_seed(seed)
    kwargs = {
        'batch_size': batch_size,
        'shuffle': shuffle,
        'drop_last': drop_last,
        'num_workers': workers,
        'pin_memory': device_type == 'cuda',
        'worker_init_fn': seed_worker,
        'generator': generator,
    }
    if workers > 0:
        kwargs['prefetch_factor'] = max(1, int(prefetch_factor))
        kwargs['persistent_workers'] = bool(persistent_workers)
    return torch.utils.data.DataLoader(dataset, **kwargs)


def model_inputs(batch, model, device, non_blocking=False):
    values = [
        batch['pc_incomplete'].to(device, non_blocking=non_blocking),
        batch['image'].to(device, non_blocking=non_blocking),
    ]
    if model.uses_material:
        values.append(batch['material_prior'].to(device, non_blocking=non_blocking))
    if model.uses_text:
        values.append(batch['text_embedding'].to(device, non_blocking=non_blocking))
    return tuple(values)
