"""Create crop comparisons with amplified difference maps."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def _parse_crop(value: str) -> tuple[int, int, int, int]:
    parts = [int(part) for part in value.split(",")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("Crop must be x,y,width,height.")
    x, y, width, height = parts
    if width <= 0 or height <= 0:
        raise argparse.ArgumentTypeError("Crop width and height must be positive.")
    return x, y, width, height


def _parse_item(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Items must use label=path.")
    label, path = value.split("=", 1)
    label = label.strip()
    if not label:
        raise argparse.ArgumentTypeError("Item label cannot be empty.")
    return label, Path(path)


def _load_crop(path: Path, crop: tuple[int, int, int, int]) -> Image.Image:
    with Image.open(path) as image:
        image = image.convert("RGB")
        x, y, width, height = crop
        return image.crop((x, y, x + width, y + height))


def _resize_for_display(image: Image.Image, scale: int) -> Image.Image:
    if scale == 1:
        return image
    return image.resize((image.width * scale, image.height * scale), Image.Resampling.NEAREST)


def _error_map(reference: Image.Image, image: Image.Image, factor: float) -> Image.Image:
    ref = np.asarray(reference, dtype=np.float32)
    cur = np.asarray(image, dtype=np.float32)
    diff = np.abs(cur - ref).mean(axis=2)
    diff = np.clip(diff * factor, 0.0, 255.0).astype(np.uint8)
    # Black means no difference; brighter pixels indicate larger absolute error.
    return Image.fromarray(diff, mode="L").convert("RGB")


def _load_font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("arial.ttf", size)
    except OSError:
        return ImageFont.load_default()


def _text_size(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> tuple[int, int]:
    bbox = draw.textbbox((0, 0), text, font=font)
    return bbox[2] - bbox[0], bbox[3] - bbox[1]


def _draw_centered_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    x: int,
    y: int,
    width: int,
    height: int,
    font: ImageFont.ImageFont,
) -> None:
    text_width, text_height = _text_size(draw, text, font)
    draw.text(
        (x + (width - text_width) // 2, y + (height - text_height) // 2),
        text,
        fill="black",
        font=font,
    )


def make_error_comparison(
    reference: Path,
    items: list[tuple[str, Path]],
    crop: tuple[int, int, int, int],
    output: Path,
    error_factor: float,
    scale: int,
    padding: int,
    title_height: int,
    row_label_width: int,
) -> None:
    reference_crop = _load_crop(reference, crop)
    loaded = [(label, _load_crop(path, crop)) for label, path in items]
    if not loaded:
        raise ValueError("At least one item is required.")

    crop_tiles = [(label, _resize_for_display(image, scale)) for label, image in loaded]
    error_tiles = [
        (label, _resize_for_display(_error_map(reference_crop, image, error_factor), scale))
        for label, image in loaded
    ]

    tile_width = max(image.width for _, image in crop_tiles)
    tile_height = max(image.height for _, image in crop_tiles)
    columns = len(crop_tiles)
    canvas_width = (
        padding
        + row_label_width
        + padding
        + columns * tile_width
        + (columns - 1) * padding
        + padding
    )
    canvas_height = padding + title_height + tile_height + padding + title_height + tile_height + padding
    canvas = Image.new("RGB", (canvas_width, canvas_height), "white")
    draw = ImageDraw.Draw(canvas)
    title_font = _load_font(18)
    label_font = _load_font(16)

    x0 = padding + row_label_width + padding
    y_title = padding
    y_image = y_title + title_height
    y_error_title = y_image + tile_height + padding
    y_error = y_error_title + title_height

    _draw_centered_text(draw, "Crop", padding, y_image, row_label_width, tile_height, label_font)
    _draw_centered_text(
        draw,
        f"|x-FP32| x{error_factor:g}",
        padding,
        y_error,
        row_label_width,
        tile_height,
        label_font,
    )

    x = x0
    for label, image in crop_tiles:
        _draw_centered_text(draw, label, x, y_title, tile_width, title_height, title_font)
        canvas.paste(image, (x + (tile_width - image.width) // 2, y_image))
        x += tile_width + padding

    x = x0
    for _label, image in error_tiles:
        canvas.paste(image, (x + (tile_width - image.width) // 2, y_error))
        x += tile_width + padding

    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True, help="FP32 reference reconstruction path.")
    parser.add_argument(
        "--item",
        action="append",
        required=True,
        type=_parse_item,
        help="Comparison item in label=path format. Repeat this argument for each column.",
    )
    parser.add_argument("--crop", type=_parse_crop, required=True, help="Crop rectangle as x,y,width,height.")
    parser.add_argument("--output", required=True, help="Output PNG path.")
    parser.add_argument("--error-factor", type=float, default=20.0, help="Multiplier for absolute error maps.")
    parser.add_argument("--scale", type=int, default=2, help="Nearest-neighbor display scale.")
    parser.add_argument("--padding", type=int, default=12, help="Padding in pixels.")
    parser.add_argument("--title-height", type=int, default=32, help="Column title area height.")
    parser.add_argument("--row-label-width", type=int, default=110, help="Width for row labels.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    make_error_comparison(
        reference=Path(args.reference),
        items=args.item,
        crop=args.crop,
        output=Path(args.output),
        error_factor=args.error_factor,
        scale=args.scale,
        padding=args.padding,
        title_height=args.title_height,
        row_label_width=args.row_label_width,
    )
    print(f"Saved error-map comparison figure to: {args.output}")


if __name__ == "__main__":
    main()
