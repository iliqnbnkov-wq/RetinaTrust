from __future__ import annotations

import base64
import binascii
import io
import warnings
from dataclasses import dataclass
from typing import Any

from PIL import Image, ImageOps

from .core import SUPPORTED_DEGRADATIONS


ALLOWED_DATA_URL_PREFIXES = {
    "data:image/jpeg;base64,": "image/jpeg",
    "data:image/png;base64,": "image/png",
}
MAX_REQUEST_BYTES = 28 * 1024 * 1024
MAX_DECODED_BYTES = 20 * 1024 * 1024
MIN_IMAGE_DIMENSION = 256
MAX_IMAGE_PIXELS = 50_000_000


@dataclass(frozen=True)
class AnalysisRequest:
    degradation: str
    level: int
    normalize: bool


class ValidationError(Exception):
    status_code = 400


class UnsupportedMediaTypeError(ValidationError):
    status_code = 415


class RequestTooLargeError(ValidationError):
    status_code = 413


def validate_content_type(content_type: str | None) -> None:
    media_type = (content_type or "").split(";", 1)[0].strip().lower()
    if media_type != "application/json":
        raise UnsupportedMediaTypeError("/api/analyze приема само application/json.")


def parse_content_length(value: str | None) -> int:
    if value is None:
        raise ValidationError("Липсва Content-Length.")
    try:
        length = int(value)
    except (TypeError, ValueError) as exc:
        raise ValidationError("Невалиден Content-Length.") from exc
    if length <= 0:
        raise ValidationError("Заявката е празна.")
    if length > MAX_REQUEST_BYTES:
        raise RequestTooLargeError("Заявката е твърде голяма; максималният файл е 20 MB.")
    return length


def _validate_json_object(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValidationError("Заявката трябва да е JSON обект.")
    return payload


def _decode_data_url(payload: dict[str, Any]) -> tuple[bytes, str]:
    data_url = payload.get("image_data_url")
    if not isinstance(data_url, str):
        raise ValidationError("Липсва валидно поле image_data_url.")

    declared_mime = None
    encoded = None
    for prefix, mime in ALLOWED_DATA_URL_PREFIXES.items():
        if data_url.startswith(prefix):
            declared_mime = mime
            encoded = data_url[len(prefix):]
            break
    if declared_mime is None or encoded is None:
        raise ValidationError("Поддържат се само точни JPEG и PNG Base64 Data URL формати.")

    try:
        raw = base64.b64decode(encoded, validate=True)
    except (ValueError, binascii.Error) as exc:
        raise ValidationError("Невалидни Base64 данни.") from exc
    if not raw:
        raise ValidationError("Изображението е празно.")
    if len(raw) > MAX_DECODED_BYTES:
        raise RequestTooLargeError("Декодираното изображение е по-голямо от 20 MB.")
    return raw, declared_mime


def _load_validated_image(raw: bytes, declared_mime: str) -> Image.Image:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            image = Image.open(io.BytesIO(raw))
            detected_format = (image.format or "").upper()
            width, height = image.size

            if width < MIN_IMAGE_DIMENSION or height < MIN_IMAGE_DIMENSION:
                raise ValidationError("Минималният размер е 256 x 256 пиксела.")
            if width * height > MAX_IMAGE_PIXELS:
                raise ValidationError("Изображението съдържа твърде много пиксели.")

            expected_format = "JPEG" if declared_mime == "image/jpeg" else "PNG"
            if detected_format != expected_format:
                raise ValidationError(
                    f"Декларираният формат ({declared_mime}) не съответства на реалния ({detected_format or 'неизвестен'})."
                )
            if int(getattr(image, "n_frames", 1)) != 1:
                raise ValidationError("Анимирани изображения не се поддържат.")

            image.load()
    except ValidationError:
        raise
    except (OSError, Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
        raise ValidationError("Изображението не може да бъде декодирано безопасно.") from exc

    return ImageOps.exif_transpose(image).convert("RGB")


def _validate_degradation(payload: dict[str, Any]) -> str:
    degradation = payload.get("degradation", "none")
    if not isinstance(degradation, str):
        raise ValidationError("Нарушението трябва да е текстов низ.")
    if degradation not in SUPPORTED_DEGRADATIONS:
        raise ValidationError(f"Неподдържано нарушение: {degradation}")
    return degradation


def _validate_level(payload: dict[str, Any]) -> int:
    level = payload.get("level", 1)
    if type(level) is not int:
        raise ValidationError("Нивото трябва да е цяло число.")
    if level not in (1, 2, 3):
        raise ValidationError("Нивото трябва да е 1, 2 или 3.")
    return level


def _validate_normalize(payload: dict[str, Any]) -> bool:
    normalize = payload.get("normalize", False)
    if type(normalize) is not bool:
        raise ValidationError("normalize трябва да е true или false.")
    return normalize


def parse_and_validate_request(
    payload: Any, content_length: int
) -> tuple[AnalysisRequest, Image.Image]:
    if content_length <= 0:
        raise ValidationError("Заявката е празна.")
    if content_length > MAX_REQUEST_BYTES:
        raise RequestTooLargeError("Заявката е твърде голяма; максималният файл е 20 MB.")

    body = _validate_json_object(payload)
    raw, declared_mime = _decode_data_url(body)
    image = _load_validated_image(raw, declared_mime)
    request = AnalysisRequest(
        degradation=_validate_degradation(body),
        level=_validate_level(body),
        normalize=_validate_normalize(body),
    )
    return request, image
