"""Online ROI crops ported from multi_roi/ConvNeXt/training/ConvNeXt_27view.py.

One random scale/direction per training access, fixed medium-center elsewhere.
No image replication or pixel cache; input images remain caller-owned.
"""

from __future__ import annotations

import random
from contextlib import closing
from dataclasses import dataclass

from PIL import Image

from setting.bbox import BBox
from setting.config import ROICropMethod


OFFSET_DIRECTIONS = (
    (-1, -1), (0, -1), (1, -1),
    (-1, 0), (0, 0), (1, 0),
    (-1, 1), (0, 1), (1, 1),
)
LETTERBOX_FILL = (124, 116, 104)  # Same RGB padding as multi_roi.


def resize_roi(image: Image.Image, size: int, method: ROICropMethod) -> Image.Image:
    """Return an owned RGB image; preserve aspect ratio only for letterbox."""
    if method == "roi_resize" or image.width == image.height:
        return image.resize((size, size), Image.Resampling.BICUBIC)
    scale = size / max(image.size)
    width = max(1, round(image.width * scale))
    height = max(1, round(image.height * scale))
    canvas = Image.new("RGB", (size, size), LETTERBOX_FILL)
    with closing(image.resize((width, height), Image.Resampling.BICUBIC)) as resized:
        canvas.paste(resized, ((size - width) // 2, (size - height) // 2))
    return canvas


@dataclass(frozen=True, slots=True)
class ROICrop:
    size: int
    method: ROICropMethod = "roi_resize"
    random_view: bool = False
    scales: tuple[float, ...] = (0.8, 1.0, 1.2)
    shift_ratio: float = 0.15

    def __call__(self, image: Image.Image, box: BBox | None) -> Image.Image:
        if box is None:
            return resize_roi(image, self.size, self.method)
        if self.random_view:
            # Same view ordering and RNG calls as generate_random_27view.
            view = random.randrange(len(self.scales) * len(OFFSET_DIRECTIONS))
            scale = self.scales[view // len(OFFSET_DIRECTIONS)]
            dx, dy = OFFSET_DIRECTIONS[view % len(OFFSET_DIRECTIONS)]
            x1, y1, x2, y2 = box
            width = max(1, round(scale * (x2 - x1)))
            height = max(1, round(scale * (y2 - y1)))
            shift_x = dx * random.uniform(0.0, self.shift_ratio) if dx else 0.0
            shift_y = dy * random.uniform(0.0, self.shift_ratio) if dy else 0.0
            left = round((x1 + x2) / 2 + shift_x * width - width / 2)
            top = round((y1 + y2) / 2 + shift_y * height - height / 2)
            box = (left, top, left + width, top + height)
        # PIL RGB crop pads out-of-image pixels with black, as in multi_roi.
        with closing(image.crop(box)) as crop:
            return resize_roi(crop, self.size, self.method)
