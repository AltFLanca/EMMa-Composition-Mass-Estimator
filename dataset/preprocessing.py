from collections.abc import Mapping
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
from emma import HDF5ArraySource, HDF5Source
from PIL import Image


@dataclass(frozen=True)
class CameraIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float


@dataclass(frozen=True)
class PreparedDepth:
    image: np.ndarray
    depth: np.ndarray
    foreground_mask: np.ndarray
    intrinsics: CameraIntrinsics


def ensure_hdf5plugin():
    try:
        import hdf5plugin  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "Reading compressed EMMa artifacts requires hdf5plugin."
        ) from exc


def open_hdf5_artifact(name, path):
    ensure_hdf5plugin()
    path = Path(path)

    import h5py

    with h5py.File(path, "r") as handle:
        is_array_source = bool(handle.attrs.get("emma_hdf5_array_source")) or (
            "_index" in handle and "data" in handle
        )

    source_type = HDF5ArraySource if is_array_source else HDF5Source
    return source_type(name, path, mode="r")


def decode_foreground_mask(payload):
    mask = _payload_value(
        payload,
        "foreground_mask",
        "mask",
        "masks",
        "segmentation",
        "pred",
    )
    mask = np.squeeze(np.asarray(mask))
    if mask.ndim > 2:
        mask = np.any(mask > 0, axis=tuple(range(mask.ndim - 2)))
    return mask > 0


def decode_text_embedding(payload):
    embedding = _payload_value(
        payload,
        "text_embedding",
        "embedding",
        "embeddings",
        "features",
    )
    return np.asarray(embedding, dtype=np.float32).reshape(-1)


def make_depth_adapter(name):
    normalized = name.lower().replace("-", "_")
    if normalized in {"glp", "glpdepth", "glp_depth", "legacy"}:
        return GLPDepthAdapter()
    if normalized in {"glp_corrected", "glpdepth_corrected", "glp_depth_corrected"}:
        return CorrectedGLPDepthAdapter()
    if normalized in {"dav3", "depthanythingv3", "depth_anything_v3"}:
        return DepthAnythingV3Adapter()
    raise ValueError("Unknown depth model: %r" % name)


class GLPDepthAdapter:
    def prepare(self, image, foreground_mask, payload):
        depth = self.decode(payload)
        height, width = depth.shape

        aligned_image = self._legacy_glp_resize(image, (height, width), is_mask=False)

        aligned_mask = self._legacy_glp_resize(
            foreground_mask, (height, width), is_mask=True
        )

        valid_region = np.zeros((height, width), dtype=bool)
        valid_region[45:471, 41:601] = True
        aligned_mask &= valid_region

        if np.count_nonzero(aligned_mask) == 0:
            aligned_mask = valid_region

        masked_depth = _masked_depth(depth, aligned_mask)
        intrinsics = self._camera_intrinsics(height, width)

        return PreparedDepth(aligned_image, masked_depth, aligned_mask, intrinsics)

    @staticmethod
    def _camera_intrinsics(height, width):
        return CameraIntrinsics(
            fx=588.0,
            fy=588.0,
            cx=width / 2.0,
            cy=height / 2.0,
        )

    @staticmethod
    def _legacy_glp_resize(value, out_hw, is_mask):
        height, width = out_hw
        interpolation = cv2.INTER_NEAREST if is_mask else cv2.INTER_LINEAR
        resized = cv2.resize(value.astype(np.uint8) if is_mask else value,
                                (height, height), interpolation=interpolation)
        horizontal_padding = width - height
        if horizontal_padding < 0:
            resized = cv2.resize(resized, (width, height), interpolation=interpolation)
        else:
            left = horizontal_padding // 2
            right = horizontal_padding - left
            border_value = 0 if is_mask else [255, 255, 255]
            resized = cv2.copyMakeBorder(
                resized, 0, 0, left, right, cv2.BORDER_CONSTANT, value=border_value
            )
        return resized.astype(bool) if is_mask else resized

    @staticmethod
    def decode(payload):
        fields = _payload_fields(payload)
        if "pred_u16" in fields:
            depth = _array(fields["pred_u16"]) * _scalar(fields["u16_scale"])
            depth *= _scalar(fields["normalization"])
        elif "pred_norm" in fields:
            depth = _array(fields["pred_norm"])
            depth *= _scalar(fields["normalization"])
        elif "pred" in fields:
            depth = _array(fields["pred"])
            if "scale" in fields:
                depth /= _scalar(fields["scale"])
        else:
            depth = _array(payload)
        return _depth_map(depth)


class CorrectedGLPDepthAdapter(GLPDepthAdapter):
    @staticmethod
    def _camera_intrinsics(height, width):
        if (height, width) != (480, 640):
            raise ValueError(
                "Corrected GLP intrinsics expect a 480x640 depth map; "
                "got %dx%d" % (height, width)
            )
        return CameraIntrinsics(
            fx=700.0,
            fy=700.0,
            cx=319.5,
            cy=239.5,
        )


class DepthAnythingV3Adapter:
    def prepare(self, image, foreground_mask, payload):
        depth = self.decode(payload)
        height, width = depth.shape
        aligned_image = cv2.resize(image, (width, height))

        if foreground_mask is None:
            aligned_mask = np.ones((height, width), dtype=bool)
        else:
            aligned_mask = cv2.resize(
                foreground_mask.astype(np.uint8),
                (width, height),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)

        if np.count_nonzero(aligned_mask) == 0:
            aligned_mask = np.ones((height, width), dtype=bool)

        masked_depth = _masked_depth(depth, aligned_mask)
        intrinsics = self.decode_intrinsics(payload)
        return PreparedDepth(aligned_image, masked_depth, aligned_mask, intrinsics)

    @staticmethod
    def decode(payload):
        fields = _payload_fields(payload)
        depth = _array(_first_present(fields, "pred"))
        if "scale" in fields:
            depth /= _scalar(fields["scale"])
        return _depth_map(depth)

    @staticmethod
    def decode_intrinsics(payload):
        fields = _payload_fields(payload)
        intrinsics = np.squeeze(
            np.asarray(_first_present(fields, "intrinsics"), dtype=np.float32)
        )
        return CameraIntrinsics(
            fx=float(intrinsics[0, 0]),
            fy=float(intrinsics[1, 1]),
            cx=float(intrinsics[0, 2]),
            cy=float(intrinsics[1, 2]),
        )


def _masked_depth(depth, mask):
    depth = depth.astype(np.float32, copy=True)
    depth[~np.isfinite(depth) | (depth <= 0) | ~mask] = 0.0
    return depth


def _depth_map(value):
    depth = np.squeeze(np.asarray(value, dtype=np.float32))
    return depth


def _array(value):
    return np.squeeze(np.asarray(value, dtype=np.float32))


def _scalar(value):
    return float(np.asarray(value).reshape(-1)[0])


def _payload_fields(payload):
    if isinstance(payload, Mapping):
        return payload
    if isinstance(payload, np.void) and payload.dtype.names:
        return {name: payload[name] for name in payload.dtype.names}
    array = np.asarray(payload)
    if array.dtype.names:
        return {name: array[name] for name in array.dtype.names}
    return {}


def _payload_value(payload, *names):
    fields = _payload_fields(payload)
    value = _first_present(fields, *names, default=None)
    if value is not None:
        return value
    if len(fields) == 1:
        return next(iter(fields.values()))
    if fields:
        raise KeyError("Artifact has none of the expected fields: %s" % (names,))
    return payload


def _first_present(fields, *names, default=...):
    for name in names:
        if name in fields:
            return fields[name]
    if default is not ...:
        return default
    raise KeyError("Artifact has none of the expected fields: %s" % (names,))
