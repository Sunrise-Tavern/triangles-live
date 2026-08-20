"""Turn channel frames into a picture, with no dependencies beyond numpy.

xLights' 3D preview is the high-fidelity oracle, but it needs a GUI and a
sequence -- too slow a loop for "did that change do what I meant".  This draws
the rig from its **real world coordinates** (:mod:`live.geometry`) through a
perspective camera, so the corridor recedes because it actually recedes and
the nets sit where they sit.  It is the same geometry the browser preview
uses, so what you check here is what you will watch live.

PNG is written by hand (zlib + four chunks) rather than pulling in Pillow --
this has to run on a Pi where every install is one more thing to go wrong.

    ./live.sh preview out/capture.fseq --out out/preview.png
    ./live.sh preview out/capture.fseq --sheet 12 --out out/sheet.png
"""

from __future__ import annotations

import struct
import zlib
from pathlib import Path

import numpy as np

from .geometry import PreviewGeometry, build_preview
from .layout import Layout

WIDTH, HEIGHT = 960, 540


# --------------------------------------------------------------------------- #
# PNG
# --------------------------------------------------------------------------- #


def write_png(path: str | Path, image: np.ndarray) -> None:
    """Write an (H, W, 3) uint8 array as a PNG."""
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError("expected an (H, W, 3) uint8 array")
    height, width, _ = image.shape
    # Filter type 0 in front of every scanline; compression does the rest.
    raw = np.concatenate(
        [np.zeros((height, 1), dtype=np.uint8), image.reshape(height, -1)], axis=1
    ).tobytes()

    def chunk(tag: bytes, data: bytes) -> bytes:
        return (struct.pack(">I", len(data)) + tag + data
                + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF))

    Path(path).write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw, 6))
        + chunk(b"IEND", b"")
    )


# --------------------------------------------------------------------------- #
# Drawing
# --------------------------------------------------------------------------- #


class Projection:
    """Rasterises a channel frame onto a canvas using the real 3D geometry.

    Built once; drawing a frame is then one gather and one scatter, so preview
    cost stays off the render budget.
    """

    def __init__(self, layout: Layout | None = None, width: int = WIDTH,
                 height: int = HEIGHT, geometry: PreviewGeometry | None = None,
                 dot: float = 2.2) -> None:
        self.width, self.height = width, height
        self.geometry = geometry or build_preview(layout, aspect=width / height)

        # Expand each pixel into a little disc, so a 3300-dot scatter reads as
        # lit fixtures rather than as noise.  Nearer pixels get bigger discs.
        radius = np.maximum(0, np.rint(self.geometry.size * dot)).astype(int)
        xy = self.geometry.xy * np.array([width - 1, height - 1], dtype=np.float32)
        centres = np.rint(xy).astype(np.int64)

        points, channels = [], []
        for r in np.unique(radius):
            pick = radius == r
            offsets = np.array([(dx, dy) for dy in range(-r, r + 1)
                                for dx in range(-r, r + 1)
                                if dx * dx + dy * dy <= r * r + 1] or [(0, 0)])
            blown = (centres[pick][:, None, :] + offsets[None, :, :]).reshape(-1, 2)
            points.append(blown)
            channels.append(np.repeat(self.geometry.channels[pick],
                                      len(offsets), axis=0))
        self.pixel_index = np.concatenate(points)
        np.clip(self.pixel_index[:, 0], 0, width - 1, out=self.pixel_index[:, 0])
        np.clip(self.pixel_index[:, 1], 0, height - 1, out=self.pixel_index[:, 1])
        self.channel_index = np.concatenate(channels)

    def draw(self, channels: np.ndarray) -> np.ndarray:
        """Render one channel frame to an (H, W, 3) uint8 image."""
        image = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        colours = channels[self.channel_index]
        # Maximum, not assignment: overlapping dots keep the brighter pixel
        # instead of whichever model happened to be drawn last.
        np.maximum.at(image, (self.pixel_index[:, 1], self.pixel_index[:, 0]),
                      colours)
        return image


def contact_sheet(layout: Layout, frames: list[np.ndarray], columns: int = 4,
                  cell: tuple[int, int] = (480, 270)) -> np.ndarray:
    """Tile N frames into one image -- a whole run at a glance."""
    proj = Projection(layout, width=cell[0], height=cell[1], dot=1.4)
    rows = -(-len(frames) // columns)
    sheet = np.zeros((rows * cell[1], columns * cell[0], 3), dtype=np.uint8)
    for i, frame in enumerate(frames):
        r, c = divmod(i, columns)
        sheet[r * cell[1]:(r + 1) * cell[1], c * cell[0]:(c + 1) * cell[0]] = \
            proj.draw(frame)
        # A one-pixel border so the cells read as separate moments.
        sheet[r * cell[1], c * cell[0]:(c + 1) * cell[0]] = 40
        sheet[r * cell[1]:(r + 1) * cell[1], c * cell[0]] = 40
    return sheet
