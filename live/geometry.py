"""Real 3D positions for every pixel, and a camera to flatten them.

The preview used to draw a schematic -- a corridor of nested triangles, nets in
a grid -- which is legible but is *my* idea of the rig, not the rig.  A preview
you watch during a set has to be the real thing, so this reads the actual world
coordinates out of ``xlights_rgbeffects.xml`` and projects them.

Worth knowing, because the offline generator's readme says otherwise: the 24
arches share one ``WorldPos``, but their ``PointData`` does not.  They sit 100
units apart along Z, from -995 to +1305, in exactly the order the ``Tunnel``
group lists them.  The corridor has real depth in the layout.

Coordinate conventions, matching xLights: +X right, +Y up, +Z away from the
viewer.  Model-space points are centred, scaled by ``ScaleX/Y/Z``, rotated Z
then Y then X, then translated to ``WorldPos``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .layout import Layout, load_layout


def _rotation(rx: float, ry: float, rz: float) -> np.ndarray:
    """Rotation matrix for xLights' Z-then-Y-then-X convention, in degrees."""
    x, y, z = np.radians([rx, ry, rz])
    cx, sx, cy, sy, cz, sz = (np.cos(x), np.sin(x), np.cos(y),
                              np.sin(y), np.cos(z), np.sin(z))
    mx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    my = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    mz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return (mx @ my @ mz).astype(np.float32)


def segment_counts(attrs: dict, nodes: int, points: np.ndarray) -> list[int]:
    """Nodes per poly-line segment.

    xLights writes ``PolyNode1..n`` only when the split is uneven; the
    2026-08-30 save omits them for the arches (two equal legs), so with
    none declared the nodes are shared out in proportion to segment length,
    which for a symmetric arch is the 180/180 the old file spelled out.
    """
    declared = [int(attrs.get(f"PolyNode{i + 1}", 0)) for i in range(len(points) - 1)]
    if sum(declared) == nodes:
        return declared
    lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    total = float(lengths.sum())
    if total <= 0.0 or len(lengths) == 0:
        return [nodes]
    counts = [int(round(nodes * float(l) / total)) for l in lengths]
    counts[-1] += nodes - sum(counts)           # rounding remainder
    return counts


def world_positions(layout: Layout) -> dict[str, np.ndarray]:
    """Model name -> ``(nodes, 3)`` world coordinates, one row per pixel."""
    out: dict[str, np.ndarray] = {}
    for name, model in layout.models.items():
        attrs = model.source
        world = np.array([float(attrs.get(f"WorldPos{c}", 0.0)) for c in "XYZ"],
                         dtype=np.float32)

        if model.kind == "arch":
            # PointData is a polyline in world-ish space, offset by WorldPos.
            raw = np.array([float(v) for v in attrs["PointData"].split(",")],
                           dtype=np.float32).reshape(-1, 3)
            counts = segment_counts(attrs, model.nodes, raw)
            points = []
            for i, count in enumerate(counts):
                if count <= 0:
                    continue
                step = np.linspace(0.0, 1.0, count, endpoint=False,
                                   dtype=np.float32)[:, None]
                points.append(raw[i] * (1 - step) + raw[i + 1] * step)
            local = np.vstack(points) if points else raw[:1]
            if len(local) != model.nodes:      # trailing node on the last leg
                local = np.vstack([local, raw[-1]])[: model.nodes]
            out[name] = local + world

        elif model.kind == "net":
            height, width = model.grid
            # Grid row 0 is the top of the model, so y counts down from it.
            local = np.zeros((model.nodes, 3), dtype=np.float32)
            local[:, 0] = model.coords[:, 1] - (width - 1) / 2.0
            local[:, 1] = (height - 1 - model.coords[:, 0]) - (height - 1) / 2.0
            scale = np.array([float(attrs.get(f"Scale{c}", 1.0)) for c in "XYZ"],
                             dtype=np.float32)
            rot = _rotation(*(float(attrs.get(f"Rotate{c}", 0.0)) for c in "XYZ"))
            out[name] = (local * scale) @ rot.T + world

        else:                                   # the par: one point
            out[name] = world.reshape(1, 3).copy()
    return out


@dataclass
class Camera:
    """A pinhole camera, positioned to frame the whole rig."""

    eye: np.ndarray
    target: np.ndarray
    up: np.ndarray = field(default_factory=lambda: np.array([0.0, 1.0, 0.0],
                                                            dtype=np.float32))
    fov: float = 45.0
    aspect: float = 16 / 9

    @classmethod
    def framing(cls, points: np.ndarray, *, yaw: float = 28.0, pitch: float = 14.0,
                distance: float = 1.6, aspect: float = 16 / 9,
                fov: float = 45.0) -> "Camera":
        """Place a camera that sees everything, orbiting the rig's centre.

        ``yaw`` swings around Y (0 looks straight down +Z, so the corridor
        recedes); ``pitch`` lifts the eye; ``distance`` is a multiple of the
        bounding sphere's radius.
        """
        centre = (points.min(axis=0) + points.max(axis=0)) / 2.0
        radius = float(np.linalg.norm(points - centre, axis=1).max())
        a, b = np.radians([yaw, pitch])
        offset = np.array([
            np.sin(a) * np.cos(b), np.sin(b), -np.cos(a) * np.cos(b)
        ], dtype=np.float32) * (radius * distance)
        return cls(eye=(centre + offset).astype(np.float32),
                   target=centre.astype(np.float32), aspect=aspect, fov=fov)

    def project(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """World points -> (``(n, 2)`` in 0..1 screen space, ``(n,)`` depth)."""
        forward = self.target - self.eye
        forward = forward / np.linalg.norm(forward)
        right = np.cross(forward, self.up)
        right = right / np.linalg.norm(right)
        up = np.cross(right, forward)

        rel = points - self.eye
        cam = np.stack([rel @ right, rel @ up, rel @ forward], axis=-1)
        depth = np.maximum(cam[:, 2], 1e-3)
        scale = 1.0 / np.tan(np.radians(self.fov) / 2.0)
        x = (cam[:, 0] / depth) * scale / self.aspect
        y = (cam[:, 1] / depth) * scale
        return np.stack([(x + 1) / 2, (1 - y) / 2], axis=-1).astype(np.float32), depth


@dataclass
class PreviewGeometry:
    """Everything a preview needs: where each dot goes and which channels feed it.

    Sampled down from the full 12 361 pixels -- the eye cannot resolve every
    node of a 465-pixel net on a laptop screen, and a browser does not want
    37 KB per frame.  ``channels`` indexes the *wire* frame, not the canvas, so
    a preview shows what actually went out.
    """

    xy: np.ndarray            # (n, 2) in 0..1
    depth: np.ndarray         # (n,) camera depth, for dot size
    size: np.ndarray          # (n,) dot radius in screen fractions
    channels: np.ndarray      # (n, 3) indices into a channel frame
    model_of: np.ndarray      # (n,) index into `models`
    models: list[str]
    camera: Camera

    def __len__(self) -> int:
        return len(self.xy)

    def sample(self, frame: np.ndarray) -> np.ndarray:
        """Pick this geometry's RGB out of a uint8 channel frame."""
        return frame[self.channels]


def build_preview(layout: Layout | None = None, *, net_stride: int = 2,
                  arch_stride: int = 6, aspect: float = 16 / 9,
                  yaw: float = 28.0, pitch: float = 14.0,
                  distance: float = 1.6) -> PreviewGeometry:
    layout = layout or load_layout()
    positions = world_positions(layout)
    everything = np.vstack(list(positions.values()))
    camera = Camera.framing(everything, yaw=yaw, pitch=pitch,
                            distance=distance, aspect=aspect)

    xy_parts, depth_parts, channel_parts, model_parts, size_parts = [], [], [], [], []
    names = [*layout.nets, *layout.arches, *([layout.par] if layout.par else [])]
    for index, name in enumerate(names):
        model = layout[name]
        stride = {"net": net_stride, "arch": arch_stride}.get(model.kind, 1)
        pick = np.arange(0, model.nodes, stride)
        screen, depth = camera.project(positions[name][pick])
        stride_ch = model.channels_per_node
        # Undo the wire colour order, so a GRB arch does not preview as green.
        slot = np.array([model.order.index(c) for c in (0, 1, 2)])
        channels = (model.start - 1 + pick[:, None] * stride_ch + slot[None, :])

        xy_parts.append(screen)
        depth_parts.append(depth)
        channel_parts.append(channels)
        model_parts.append(np.full(len(pick), index, dtype=np.int32))
        size_parts.append(np.full(len(pick), 6.0 if model.kind == "par" else 1.0,
                                  dtype=np.float32))

    depth = np.concatenate(depth_parts)
    # Nearer pixels draw bigger, which is most of what makes a flat scatter of
    # dots read as a corridor with depth.
    size = np.concatenate(size_parts) * (float(np.median(depth)) / depth) ** 0.9
    return PreviewGeometry(
        xy=np.concatenate(xy_parts), depth=depth, size=size.astype(np.float32),
        channels=np.concatenate(channel_parts).astype(np.int32),
        model_of=np.concatenate(model_parts), models=names, camera=camera,
    )


if __name__ == "__main__":
    layout = load_layout()
    positions = world_positions(layout)
    for name in (layout.nets[0], layout.arches[0], layout.arches[-1],
                 *([layout.par] if layout.par else [])):
        p = positions[name]
        lo, hi = p.min(axis=0).round(0), p.max(axis=0).round(0)
        print(f"{name:<24} {len(p):>4} nodes  x {lo[0]:>8.0f}..{hi[0]:<8.0f} "
              f"y {lo[1]:>6.0f}..{hi[1]:<6.0f} z {lo[2]:>8.0f}..{hi[2]:<8.0f}")
    preview = build_preview(layout)
    print(f"\npreview: {len(preview)} dots from {len(positions)} models")
    print(f"camera eye {preview.camera.eye.round(0)} -> "
          f"target {preview.camera.target.round(0)}")
