"""Create cropped reconstruction comparison figures."""

from __future__ import annotations

import argparse
from pathlib import Path

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


def _load_crop(path: Path, crop: tuple[int, int, int, int], scale: int) -> Image.Image:
    with Image.open(path) as image:
        image = image.convert("RGB")
        x, y, width, height = crop
        cropped = image.crop((x, y, x + width, y + height))
    if scale != 1:
        cropped = cropped.resize((cropped.width * scale, cropped.height * scale), Image.Resampling.NEAREST)
    return cropped


def _load_font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("arial.ttf", size)
    except OSError:
        return ImageFont.load_default()


def _text_size(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> tuple[int, int]:
    bbox = draw.textbbox((0, 0), text, font=font)
    return bbox[2] - bbox[0], bbox[3] - bbox[1]


def make_comparison(
    items: list[tuple[str, Path]],
    crop: tuple[int, int, int, int],
    output: Path,
    scale: int,
    padding: int,
    title_height: int,
) -> None:
    crops = [(label, _load_crop(path, crop, scale)) for label, path in items]
    if not crops:
        raise ValueError("At least one item is required.")

    tile_width = max(image.width for _, image in crops)
    tile_height = max(image.height for _, image in crops)
    canvas_width = padding + len(crops) * tile_width + (len(crops) - 1) * padding + padding
    canvas_height = padding + title_height + tile_height + padding
    canvas = Image.new("RGB", (canvas_width, canvas_height), "white")
    draw = ImageDraw.Draw(canvas)
    font = _load_font(18)

    x = padding
    for label, image in crops:
        text_width, text_height = _text_size(draw, label, font)
        draw.text((x + (tile_width - text_width) // 2, padding + (title_height - text_height) // 2), label, fill="black", font=font)
        canvas.paste(image, (x + (tile_width - image.width) // 2, padding + title_height))
        x += tile_width + padding

    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--item",
        action="append",
        required=True,
        type=_parse_item,
        help="Comparison item in label=path format. Repeat this argument for each column.",
    )
    parser.add_argument(
        "--crop",
        type=_parse_crop,
        required=True,
        help="Crop rectangle as x,y,width,height.",
    )
    parser.add_argument("--output", required=True, help="Output PNG path.")
    parser.add_argument("--scale", type=int, default=2, help="Nearest-neighbor crop enlargement factor.")
    parser.add_argument("--padding", type=int, default=12, help="Padding in pixels.")
    parser.add_argument("--title-height", type=int, default=32, help="Column title area height.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    make_comparison(
        items=args.item,
        crop=args.crop,
        output=Path(args.output),
        scale=args.scale,
        padding=args.padding,
        title_height=args.title_height,
    )
    print(f"Saved comparison figure to: {args.output}")


if __name__ == "__main__":
    main()
