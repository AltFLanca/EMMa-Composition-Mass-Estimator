from io import BytesIO

import numpy as np
from PIL import Image
from emma import EmmaDataset

from dataset.base_dataset import BaseDataset
from dataset.preprocessing import (
    decode_foreground_mask,
    decode_text_embedding,
    make_depth_adapter,
    open_hdf5_artifact,
)


LB_TO_KG = 0.45359237
IN_TO_M = 0.0254
_CACHE = {}


def _get_dataset(root):
    if root not in _CACHE:
        _CACHE[root] = EmmaDataset(root)
    return _CACHE[root]


class EMMaDataset(BaseDataset):
    def __init__(
        self,
        emma_root,
        emma_depth_path,
        is_train=True,
        colored_pc=False,
        pc_dims=1024,
        pointcloud_input_divisor=5.0,
        emma_depth_model="glp",
        emma_segmentation_path=None,
        emma_text_embedding_path=None,
        emma_asins_npy=None,
        emma_max_items=None,
        **_,
    ):
        super().__init__(colored_pc, pc_dims, pointcloud_input_divisor)

        self.emma_dataset = _get_dataset(emma_root)
        self.is_train = is_train

        self.depth_source = open_hdf5_artifact(
            "depth",
            emma_depth_path,
        )
        self.depth_adapter = make_depth_adapter(emma_depth_model)
        
        self.mask_source = (
            open_hdf5_artifact(
                "masks",
                emma_segmentation_path,
            )
            if emma_segmentation_path
            else None
        )

        self.text_embedding_source = (
            open_hdf5_artifact(
                "text_embeddings",
                emma_text_embedding_path,
            )
            if emma_text_embedding_path
            else None
        )

        self.filenames_list = np.load(
                emma_asins_npy,
                allow_pickle=False,
            ).reshape(-1)

        if emma_max_items is not None:
            self.filenames_list = self.filenames_list[:emma_max_items]

        self._text_embedding_dims = (
            decode_text_embedding(
                self.text_embedding_source[
                    self.filenames_list[0]
                ]
            ).size
            if self.text_embedding_source is not None
            else None
        )

        print("Dataset: EMMa")
        print("Root: %s" % emma_root)
        print("Depth: %s (%s)" % (emma_depth_path, emma_depth_model))
        print("Segmentation: %s" % (emma_segmentation_path))
        print("ASIN list: %s" % emma_asins_npy)
        print("# of images: %d" % len(self.filenames_list))
        # print(self.text_embedding_dims)

    def __len__(self):
        return len(self.filenames_list)

    @property
    def num_materials(self):
        return self.emma_dataset.num_materials

    @property
    def material_names(self):
        return self.emma_dataset.material_names

    @property
    def material_hierarchy(self):
        return self.emma_dataset.material_hierarchy

    @property
    def text_embedding_dims(self):
        return self._text_embedding_dims

    def __getitem__(self, idx):
        asin = self.filenames_list[idx]
        record = self.emma_dataset.get_record(asin)

        image_hash = next(iter(self.emma_dataset.get_image_hashes(record)))
        image_bytes = self.emma_dataset.get_images(image_hashes=[image_hash])[image_hash]
        image = np.asarray(Image.open(BytesIO(image_bytes)).convert("RGB"))

        mask = (
            decode_foreground_mask(self.mask_source[asin])
            if self.mask_source is not None
            else None
        )

        depth_payload = self.depth_source[asin]

        prepared = self.depth_adapter.prepare(
            image,
            mask,
            depth_payload,
        )

        if self.is_train:
            image, point_cloud, _ = self.augment_training_data(
                prepared.image,
                prepared.depth,
                None,
                intrinsics=prepared.intrinsics,
            )
        else:
            image = prepared.image
            point_cloud = self.get_pointcloud(
                prepared.depth,
                intrinsics=prepared.intrinsics,
            )

        row = self.emma_dataset.asin_to_index[asin]

        material_prior = np.asarray(
            self.emma_dataset.get_material_probs(record),
            dtype=np.float32,
        ).reshape(-1)

        material_target = np.asarray(
            self.emma_dataset.material_bits[row],
            dtype=np.float32,
        ).reshape(-1)

        dimensions = (
            np.asarray(
                self.emma_dataset.get_package_dimensions(
                    record
                ),
                dtype=np.float32,
            ).squeeze()
            * IN_TO_M
        )

        volume = float(np.prod(dimensions + 0.005))
        
        mass = (
            self.emma_dataset.get_mass(record)
            * LB_TO_KG
        )

        sample = {
            "image": self.convert_to_densenet_input(image),
            "pc_incomplete": self.to_tensor(point_cloud),
            "mass": mass,
            "material_prior": material_prior,
            "material_target": material_target,
            "volume": volume,
            "density": mass / volume,
            "filename": asin,
        }

        if self.text_embedding_source is not None:
            sample["text_embedding"] = (
                decode_text_embedding(
                    self.text_embedding_source[asin]
                )
            )

        return sample


emma = EMMaDataset