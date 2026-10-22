from typing import NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.geometry import robust_rms_normalize


class MassPrediction(NamedTuple):
    mass: torch.Tensor
    density: torch.Tensor
    volume: torch.Tensor
    composition: torch.Tensor | None
    fill_factor: torch.Tensor | None
    concentration: torch.Tensor | None


class DensityBlock(nn.Module):
    def __init__(self, input_dims, activation, log_clip=None):
        super().__init__()
        self.activation = dict(activation)
        self.log_clip = log_clip
        self.linear1 = nn.Linear(input_dims, 512, bias=False)
        self.bn1 = nn.BatchNorm1d(512)
        self.linear2 = nn.Linear(512, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.linear3 = nn.Linear(64, 1)
        self.dp = nn.Dropout()

    def forward(self, x):
        x = F.relu(self.bn1(self.linear1(x)))
        x = self.dp(x)
        x = F.relu(self.bn2(self.linear2(x)))
        x = self.dp(x)
        return self.density_inductive_bias(self.linear3(x))

    def density_inductive_bias(self, x):
        kind = self.activation['type']

        if kind == 'single_exponential_shift':
            a = float(self.activation['a'])
            b = float(self.activation['b'])
            shift = float(self.activation['shift'])
            if self.log_clip is None:
                return 1.0 / (a * torch.exp(-b * (x - shift)))
            log_density = -torch.log(x.new_tensor(a)) + b * (x - shift)

        elif kind == 'double_exponential':
            a_slow = float(self.activation['a_slow'])
            b_slow = float(self.activation['b_slow'])
            a_fast = float(self.activation['a_fast'])
            b_fast = float(self.activation['b_fast'])
            if self.log_clip is None:
                return 1.0 / (
                    a_slow * torch.exp(-b_slow * x)
                    + a_fast * torch.exp(-b_fast * x)
                )
            log_terms = torch.stack(
                (
                    torch.log(x.new_tensor(a_slow)) - b_slow * x,
                    torch.log(x.new_tensor(a_fast)) - b_fast * x,
                ),
                dim=0,
            )
            log_density = -torch.logsumexp(log_terms, dim=0)
        else:
            raise ValueError('Unknown density activation: %s' % kind)

        clip = float(self.log_clip)
        return torch.exp(torch.clamp(log_density, -clip, clip))


class VolumeBlock(nn.Module):
    def __init__(self, input_dims, activation='relu'):
        super().__init__()
        self.activation = activation
        self.linear1 = nn.Linear(input_dims, 512, bias=False)
        self.bn1 = nn.BatchNorm1d(512)
        self.linear2 = nn.Linear(512, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.linear3 = nn.Linear(64, 1)
        self.dp = nn.Dropout()

    def forward(self, x):
        x = F.relu(self.bn1(self.linear1(x)))
        x = self.dp(x)
        x = F.relu(self.bn2(self.linear2(x)))
        x = self.dp(x)
        x = self.linear3(x)
        if self.activation == 'softplus':
            return F.softplus(x) + 1e-6
        if self.activation == 'relu':
            return F.relu(x) + 1e-6
        raise ValueError('Unknown volume activation: %s' % self.activation)


class CompositionBlock(nn.Module):
    def __init__(self, input_dims, num_materials):
        super().__init__()
        self.linear1 = nn.Linear(input_dims, 512, bias=False)
        self.bn1 = nn.BatchNorm1d(512)
        self.linear2 = nn.Linear(512, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.linear3 = nn.Linear(64, num_materials)
        self.dp = nn.Dropout()

    def forward(self, x):
        x = F.relu(self.bn1(self.linear1(x)))
        x = self.dp(x)
        x = F.relu(self.bn2(self.linear2(x)))
        logits = self.linear3(self.dp(x))
        return F.softmax(logits, dim=1), None


class FillFactorBlock(nn.Module):
    def __init__(self, input_dims):
        super().__init__()
        self.linear1 = nn.Linear(input_dims, 512, bias=False)
        self.bn1 = nn.BatchNorm1d(512)
        self.linear2 = nn.Linear(512, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.linear3 = nn.Linear(64, 1)
        self.dp = nn.Dropout()

    def forward(self, x):
        x = F.relu(self.bn1(self.linear1(x)))
        x = self.dp(x)
        x = F.relu(self.bn2(self.linear2(x)))
        return torch.sigmoid(self.linear3(self.dp(x)))


class Depth2Mass(nn.Module):

    def __init__(
        self,
        config,
        material_densities=None,
        pretrained_image_encoder=True,
    ):
        super().__init__()
        self.config = dict(config)
        emb_dims = int(self.config.get('emb_dims', 512))
        num_materials = int(self.config.get('num_materials', 182))
        text_embedding_dims = self.config.get('text_embedding_dims', 512)

        self.uses_material = bool(self.config.get('use_material_prior', False))
        self.uses_text = bool(self.config.get('use_text', False))
        self.uses_composition = bool(self.config.get('composition', False))
        self.uses_fill_factor = bool(self.config.get('fill_factor', False))
        self.geometry_mode = self.config.get('geometry', 'original')
        self.reciprocal_scale = float(self.config.get('reciprocal_scale', 1.0))

        from models.pointnet import PointNet
        from models.densenet import DenseNet

        self.model_pc = PointNet(emb_dims)
        self.model_densenet = DenseNet(emb_dims, pretrained=pretrained_image_encoder)
        fused_dims = 2 * emb_dims

        conditioned_dims = fused_dims
        if self.uses_material:
            conditioned_dims += num_materials
        if self.uses_text:
            conditioned_dims += int(text_embedding_dims)

        volume_dims = fused_dims
        if self.uses_text:
            volume_dims += int(text_embedding_dims)

        if self.uses_composition:
            self.compositionBlock = CompositionBlock(conditioned_dims, num_materials)
            if self.uses_fill_factor:
                self.fillFactorBlock = FillFactorBlock(volume_dims)
            densities = torch.as_tensor(material_densities, dtype=torch.float32).reshape(-1)
            self.register_buffer('material_densities', densities)
        else:
            self.densityBlock = DensityBlock(
                conditioned_dims,
                activation=self.config['density_activation'],
                log_clip=self.config.get('density_log_clip'),
            )
            self.register_buffer('material_densities', None)

        self.volumeBlock = VolumeBlock(
            volume_dims, activation=self.config.get('volume_activation', 'relu')
        )

    def forward(self, inputs):
        pc, image, *conditioning = inputs

        physical_radius = None
        if self.geometry_mode == 'rms':
            pc, physical_radius = robust_rms_normalize(
                pc,
                trim_quantile=float(self.config.get('rms_trim_quantile', 0.95)),
                input_divisor=float(self.config.get('pointcloud_input_divisor', 5.0)),
            )
        elif self.geometry_mode != 'original':
            raise ValueError('Unknown geometry mode: %s' % self.geometry_mode)

        image = image.squeeze(1)
        image_embedding = self.model_densenet(image)
        point_embedding = self.model_pc(pc)
        fused = torch.cat((image_embedding, point_embedding), dim=1)

        conditioned = [fused]
        conditioning_index = 0
        if self.uses_material:
            conditioned.append(conditioning[conditioning_index].float())
            conditioning_index += 1

        if self.uses_text:
            text_embedding = conditioning[conditioning_index].float()
            conditioned.append(text_embedding)
            volume_features = torch.cat((fused, text_embedding), dim=1)
        else:
            volume_features = fused

        density_features = torch.cat(conditioned, dim=1)
        composition = None
        fill_factor = None
        concentration = None
        c = self.reciprocal_scale

        if self.uses_composition:
            composition, concentration = self.compositionBlock(density_features)
            density = c * (composition @ self.material_densities.unsqueeze(1))
            if self.uses_fill_factor:
                fill_factor = self.fillFactorBlock(volume_features)
                density = density * fill_factor
        else:
            density = c * self.densityBlock(density_features)

        volume_factor = self.volumeBlock(volume_features)
        if self.geometry_mode == 'rms':
            volume = (1.0 / c) * physical_radius.pow(3) * volume_factor
        else:
            volume = (1.0 / c) * volume_factor

        mass = density * volume
        return MassPrediction(
            mass=mass,
            density=density,
            volume=volume,
            composition=composition,
            fill_factor=fill_factor,
            concentration=concentration,
        )
