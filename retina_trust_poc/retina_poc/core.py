from __future__ import annotations

import io
import math
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Iterable

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
from scipy import ndimage


ANALYSIS_SIZE = 256
CANONICAL_WORKING_SIZE = 1024
SUPPORTED_DEGRADATIONS = {
    "none",
    "underexposure",
    "overexposure",
    "gamma",
    "color_shift",
    "blur",
    "jpeg",
    "combined",
}


@dataclass(frozen=True)
class PreparedImage:
    image: Image.Image
    array: np.ndarray
    mask: np.ndarray


def load_rgb(source: str | Path | bytes | BinaryIO | Image.Image) -> Image.Image:
    """Load an image safely and return an EXIF-corrected RGB PIL image."""
    if isinstance(source, Image.Image):
        image = source.copy()
    elif isinstance(source, (str, Path)):
        image = Image.open(source)
    elif isinstance(source, bytes):
        image = Image.open(io.BytesIO(source))
    else:
        image = Image.open(source)
    image = ImageOps.exif_transpose(image).convert("RGB")
    image.load()
    return image


def canonicalize_image(
    image: Image.Image,
    max_size: int = CANONICAL_WORKING_SIZE,
) -> Image.Image:
    """Return the shared pre-perturbation image used by every model path.

    v0.1 resized only degraded inputs through a 1024 px working space while
    clean inputs went directly to feature extraction. The canonical path makes
    training, clean evaluation and robustness evaluation use the same first
    resampling operation.
    """
    if max_size < ANALYSIS_SIZE:
        raise ValueError(f"max_size must be at least {ANALYSIS_SIZE}")
    working = image.convert("RGB").copy()
    working.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
    return working


def _largest_component(mask: np.ndarray) -> np.ndarray:
    labels, count = ndimage.label(mask)
    if count == 0:
        return mask
    sizes = ndimage.sum(mask, labels, range(1, count + 1))
    return labels == (int(np.argmax(sizes)) + 1)


def prepare_image(image: Image.Image, size: int = ANALYSIS_SIZE) -> PreparedImage:
    """Crop the retinal field and create a stable circular-field mask."""
    preview = image.copy()
    preview.thumbnail((768, 768), Image.Resampling.LANCZOS)
    arr = np.asarray(preview, dtype=np.float32) / 255.0
    luminance = 0.2126 * arr[..., 0] + 0.7152 * arr[..., 1] + 0.0722 * arr[..., 2]
    rough = luminance > 0.025
    rough = ndimage.binary_closing(rough, iterations=3)
    rough = ndimage.binary_fill_holes(_largest_component(rough))

    if rough.sum() < rough.size * 0.08:
        rough = np.ones_like(rough, dtype=bool)

    ys, xs = np.where(rough)
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    pad_y = max(2, int((y1 - y0) * 0.015))
    pad_x = max(2, int((x1 - x0) * 0.015))
    y0, y1 = max(0, y0 - pad_y), min(arr.shape[0], y1 + pad_y)
    x0, x1 = max(0, x0 - pad_x), min(arr.shape[1], x1 + pad_x)

    cropped = preview.crop((x0, y0, x1, y1)).resize((size, size), Image.Resampling.LANCZOS)
    crop_arr = np.asarray(cropped, dtype=np.float32) / 255.0
    crop_lum = 0.2126 * crop_arr[..., 0] + 0.7152 * crop_arr[..., 1] + 0.0722 * crop_arr[..., 2]
    mask = crop_lum > 0.025
    mask = ndimage.binary_closing(mask, iterations=2)
    mask = ndimage.binary_fill_holes(_largest_component(mask))
    if mask.sum() < mask.size * 0.2:
        mask = np.ones_like(mask, dtype=bool)
    return PreparedImage(cropped, crop_arr, mask)


def _safe_values(array: np.ndarray, mask: np.ndarray) -> np.ndarray:
    values = np.asarray(array)[mask]
    if values.size == 0:
        return np.asarray(array).reshape(-1)
    return values


def quality_metrics(prepared: PreparedImage) -> dict[str, float]:
    arr, mask = prepared.array, prepared.mask
    lum = 0.2126 * arr[..., 0] + 0.7152 * arr[..., 1] + 0.0722 * arr[..., 2]
    values = _safe_values(lum, mask)
    gx = ndimage.sobel(lum, axis=1, mode="reflect")
    gy = ndimage.sobel(lum, axis=0, mode="reflect")
    gradient = np.hypot(gx, gy)
    smooth = ndimage.gaussian_filter(lum, sigma=18)
    channel_means = np.array([_safe_values(arr[..., i], mask).mean() for i in range(3)])
    mean_channel = float(channel_means.mean()) + 1e-8

    return {
        "brightness": float(values.mean()),
        "contrast": float(values.std()),
        "sharpness": float(_safe_values(gradient, mask).mean()),
        "field_coverage": float(mask.mean()),
        "dark_clipping": float(np.mean(values < 0.045)),
        "bright_clipping": float(np.mean(values > 0.94)),
        "illumination_nonuniformity": float(_safe_values(smooth, mask).std()),
        "color_cast": float(channel_means.std() / mean_channel),
        "saturation": float(np.mean((arr.max(axis=2) - arr.min(axis=2))[mask])),
    }


def _entropy(values: np.ndarray, bins: int = 32) -> float:
    hist, _ = np.histogram(values, bins=bins, range=(0.0, 1.0), density=False)
    p = hist.astype(np.float64)
    p /= max(p.sum(), 1.0)
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def _append_stats(
    names: list[str], values: list[float], prefix: str, data: np.ndarray, percentiles: Iterable[int]
) -> None:
    data = np.asarray(data, dtype=np.float64)
    for p, v in zip(percentiles, np.percentile(data, list(percentiles))):
        names.append(f"{prefix}_p{p}")
        values.append(float(v))
    names.extend([f"{prefix}_mean", f"{prefix}_std", f"{prefix}_entropy"])
    values.extend([float(data.mean()), float(data.std()), _entropy(np.clip(data, 0, 1))])


def extract_features(image: Image.Image) -> tuple[np.ndarray, list[str], dict[str, float]]:
    """Extract deterministic colour, texture, radial and lesion-proxy features."""
    prepared = prepare_image(image)
    arr, mask = prepared.array, prepared.mask
    lum = 0.2126 * arr[..., 0] + 0.7152 * arr[..., 1] + 0.0722 * arr[..., 2]
    names: list[str] = []
    values: list[float] = []
    percentiles = (2, 5, 10, 25, 50, 75, 90, 95, 98)

    for index, channel in enumerate(("r", "g", "b")):
        _append_stats(names, values, channel, _safe_values(arr[..., index], mask), percentiles)
        hist, _ = np.histogram(_safe_values(arr[..., index], mask), bins=12, range=(0, 1))
        hist = hist / max(hist.sum(), 1)
        for bin_index, value in enumerate(hist):
            names.append(f"{channel}_hist_{bin_index:02d}")
            values.append(float(value))

    _append_stats(names, values, "lum", _safe_values(lum, mask), percentiles)
    saturation = arr.max(axis=2) - arr.min(axis=2)
    _append_stats(names, values, "sat", _safe_values(saturation, mask), (5, 25, 50, 75, 95))

    gx = ndimage.sobel(lum, axis=1, mode="reflect")
    gy = ndimage.sobel(lum, axis=0, mode="reflect")
    magnitude = np.hypot(gx, gy)
    _append_stats(names, values, "grad", _safe_values(magnitude, mask), (25, 50, 75, 90, 95, 99))
    angles = (np.arctan2(gy, gx) + math.pi) % math.pi
    hist, _ = np.histogram(angles[mask], bins=8, range=(0, math.pi), weights=magnitude[mask])
    hist = hist / max(hist.sum(), 1e-8)
    for index, value in enumerate(hist):
        names.append(f"grad_orientation_{index}")
        values.append(float(value))

    green = arr[..., 1]
    for sigma in (1.2, 3.0, 7.0):
        residual = green - ndimage.gaussian_filter(green, sigma=sigma)
        rvals = _safe_values(residual, mask)
        for label, value in (
            ("neg_p01", np.percentile(rvals, 1)),
            ("neg_p05", np.percentile(rvals, 5)),
            ("abs_p95", np.percentile(np.abs(rvals), 95)),
            ("pos_p95", np.percentile(rvals, 95)),
            ("pos_p99", np.percentile(rvals, 99)),
            ("std", rvals.std()),
        ):
            names.append(f"green_dog_{sigma}_{label}")
            values.append(float(value))

    yy, xx = np.indices(mask.shape)
    cy, cx = (np.array(mask.shape) - 1) / 2
    radius = np.hypot((yy - cy) / max(cy, 1), (xx - cx) / max(cx, 1))
    for index, (low, high) in enumerate(zip((0, .2, .4, .6, .8), (.2, .4, .6, .8, 1.05))):
        ring = mask & (radius >= low) & (radius < high)
        ring_values = _safe_values(lum, ring)
        names.extend([f"radial_{index}_mean", f"radial_{index}_std"])
        values.extend([float(ring_values.mean()), float(ring_values.std())])

    block_edges = np.linspace(0, ANALYSIS_SIZE, 5, dtype=int)
    for row in range(4):
        for col in range(4):
            sl = np.s_[block_edges[row]:block_edges[row + 1], block_edges[col]:block_edges[col + 1]]
            block_mask = mask[sl]
            block = _safe_values(lum[sl], block_mask)
            names.extend([f"block_{row}_{col}_mean", f"block_{row}_{col}_std"])
            values.extend([float(block.mean()), float(block.std())])

    qmetrics = quality_metrics(prepared)
    for name, value in qmetrics.items():
        names.append(f"quality_{name}")
        values.append(float(value))

    vector = np.asarray(values, dtype=np.float32)
    vector = np.nan_to_num(vector, nan=0.0, posinf=1e4, neginf=-1e4)
    return vector, names, qmetrics


def apply_degradation(image: Image.Image, kind: str, level: int = 1) -> Image.Image:
    kind = kind.lower().strip()
    if kind not in SUPPORTED_DEGRADATIONS:
        raise ValueError(f"Unsupported degradation: {kind}")
    if level not in (1, 2, 3):
        raise ValueError("level must be 1, 2 or 3")
    # Every path starts from the same deterministic working image. This keeps
    # severity comparable across cameras without confounding clean/degraded
    # comparisons with an extra resize in only one branch.
    working = canonicalize_image(image)
    if kind == "none":
        return working.copy()
    if kind == "underexposure":
        return ImageEnhance.Brightness(working).enhance((0.72, 0.48, 0.28)[level - 1])
    if kind == "overexposure":
        return ImageEnhance.Brightness(working).enhance((1.28, 1.65, 2.15)[level - 1])
    if kind == "gamma":
        gamma = (1.35, 1.8, 2.4)[level - 1]
        array = np.asarray(working, dtype=np.float32) / 255.0
        return Image.fromarray(np.uint8(np.clip(array ** gamma, 0, 1) * 255), "RGB")
    if kind == "color_shift":
        factors = ((1.08, .96, .90), (1.18, .88, .76), (1.32, .76, .58))[level - 1]
        array = np.asarray(working, dtype=np.float32) / 255.0
        array *= np.asarray(factors, dtype=np.float32)[None, None, :]
        return Image.fromarray(np.uint8(np.clip(array, 0, 1) * 255), "RGB")
    if kind == "blur":
        return working.filter(ImageFilter.GaussianBlur(radius=(1.5, 4.0, 8.0)[level - 1]))
    if kind == "jpeg":
        quality = (55, 28, 10)[level - 1]
        buffer = io.BytesIO()
        working.save(buffer, format="JPEG", quality=quality, optimize=False)
        buffer.seek(0)
        result = Image.open(buffer).convert("RGB")
        result.load()
        return result
    if kind == "combined":
        result = ImageEnhance.Brightness(working).enhance((0.72, 0.48, 0.28)[level - 1])
        result = result.filter(ImageFilter.GaussianBlur(radius=(1.5, 4.0, 8.0)[level - 1]))
        quality = (55, 28, 10)[level - 1]
        buffer = io.BytesIO()
        result.save(buffer, format="JPEG", quality=quality, optimize=False)
        buffer.seek(0)
        result = Image.open(buffer).convert("RGB")
        result.load()
        return result
    raise AssertionError("unreachable")


def normalize_image(image: Image.Image) -> Image.Image:
    """Conservative demonstration-only photometric normalization."""
    image = image.convert("RGB")
    array = np.asarray(image, dtype=np.float32) / 255.0
    lum = 0.2126 * array[..., 0] + 0.7152 * array[..., 1] + 0.0722 * array[..., 2]
    active = lum > 0.025
    if active.sum() < 100:
        return image.copy()
    p3, p97 = np.percentile(lum[active], (3, 97))
    if p97 <= p3 + 1e-4:
        return image.copy()
    target = np.clip((lum - p3) / (p97 - p3), 0, 1)
    target = np.power(target, 0.92)
    ratio = target / np.maximum(lum, 0.025)
    corrected = np.clip(array * ratio[..., None], 0, 1)
    corrected[~active] = array[~active]
    return Image.fromarray(np.uint8(corrected * 255), "RGB")


def image_to_data_url(image: Image.Image, max_size: tuple[int, int] = (1100, 800)) -> str:
    import base64

    preview = image.copy()
    preview.thumbnail(max_size, Image.Resampling.LANCZOS)
    buffer = io.BytesIO()
    preview.save(buffer, format="JPEG", quality=88, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def analyze_image(image: Image.Image) -> dict[str, object]:
    vector, names, metrics = extract_features(image)
    return {"features": vector, "feature_names": names, "quality_metrics": metrics}
