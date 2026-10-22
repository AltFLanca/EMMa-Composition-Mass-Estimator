import albumentations as A
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset

from utils.transforms import CustomHorizontalFlip, CustomVerticalFlip


def get_dataset(dataset_name=None, **kwargs):
    if dataset_name is not None and dataset_name.lower() != "emma":
        raise ValueError("Only the EMMa dataset is supported")
    from dataset.emma import EMMaDataset

    return EMMaDataset(**kwargs)


class BaseDataset(Dataset):
    def __init__(self, colored_pc, pc_dims, pointcloud_input_divisor):
        self.colored_pc = colored_pc
        self.pc_dims = pc_dims
        self.pointcloud_input_divisor = pointcloud_input_divisor

        self.horizontal_transform = CustomHorizontalFlip()
        self.vertical_transform = CustomVerticalFlip()
        self.basic_transform = [
            self.horizontal_transform,
            self.vertical_transform,
            A.RandomBrightnessContrast(),
            A.RandomGamma(),
            A.HueSaturationValue(),
        ]
        self.preprocess_densenet = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )
        self.to_tensor = transforms.ToTensor()
        self.augment = A.Compose(
            transforms=self.basic_transform,
            additional_targets={"depth": "mask"},
        )

    def augment_training_data(self, image, depth, pc, intrinsics=None):
        augmented = self.augment(image=image, depth=depth)
        image = augmented["image"]
        depth = augmented["depth"]

        self.horizontal_transform.clear()
        self.vertical_transform.clear()

        if self.colored_pc:
            point_cloud = self.get_pointcloud(depth, image, intrinsics)
        else:
            point_cloud = self.get_pointcloud(depth, intrinsics=intrinsics)
        return image, point_cloud, None

    def convert_to_densenet_input(self, image):
        input_tensor = self.preprocess_densenet(Image.fromarray(image))
        return input_tensor.unsqueeze(0)

    def get_pointcloud(self, depth, image=None, intrinsics=None):
        height, width = depth.shape

        fx = intrinsics.fx
        fy = intrinsics.fy
        cx = intrinsics.cx
        cy = intrinsics.cy

        valid_indices = np.flatnonzero(depth.reshape(-1) != 0)

        if valid_indices.shape[0] > self.pc_dims:
            indices = np.random.choice(
                valid_indices.shape[0], self.pc_dims, replace=False
            )
            valid_indices = valid_indices[indices]

        y, x = np.divmod(valid_indices, width)
        depth = depth.reshape(-1)[valid_indices]

        normalized_x = (x - cx) / fx * depth
        normalized_y = (y - cy) / fy * depth

        point_cloud = np.column_stack(
            (normalized_x, normalized_y, depth)
        ).astype(np.float32)

        if image is not None:
            image = image.reshape(-1, 3)[valid_indices]
            point_cloud = np.concatenate((point_cloud, image), axis=1)

        point_cloud -= np.mean(point_cloud, axis=0)
        point_cloud /= self.pointcloud_input_divisor

        if point_cloud.shape[0] < self.pc_dims:
            point_cloud = np.pad(
                point_cloud,
                ((0, self.pc_dims - point_cloud.shape[0]), (0, 0)),
                "constant",
            )

        return point_cloud
