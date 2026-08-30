"""The pixel canvas: per-model float buffers, and the geometry effects paint on.

Effects work in **float RGB, 0..1, in source order** and never touch channel
numbers or wire colour order -- :meth:`Canvas.to_channels` does that once per
frame through a precomputed gather.  Keeping the two apart is what let M1's
byte-exact test stay meaningful: an effect cannot accidentally address the
wrong fixture, because it cannot address fixtures at all.

Two facts about this rig make the whole thing vectorise:

* the nets are all triangles on a lattice -- 465 nodes over 59x51 or 435
  over 57x49 -- so they are one ``(n, 465, 3)`` array padded to the largest,
  with per-net geometry vectors of the same shape, and an effect over "the
  nets" is one numpy expression.  Padding slots are painted like any other
  and simply never emitted;
* all 24 arches share one 360-node base->apex->base run, so the corridor is
  one ``(24, 360, 3)`` array and a wave down the tunnel is an outer product.

The geometry vectors below are the effects' whole vocabulary of *where*:
normalised, unit-free, and computed once.
"""

from __future__ import annotations

import numpy as np

from .layout import Layout, load_layout


class NetGeometry:
    """Where some nets' pixels are: ``(k, nodes)`` vectors, one row per net."""

    __slots__ = ("x", "y", "r", "angle")

    def __init__(self, x, y, r, angle) -> None:
        self.x, self.y, self.r, self.angle = x, y, r, angle


class Canvas:
    """One frame's worth of pixels, plus the geometry to paint them by."""

    def __init__(self, layout: Layout | None = None) -> None:
        self.layout = layout or load_layout()
        self.net_names = list(self.layout.nets)
        self.arch_names = list(self.layout.arches)

        nets = [self.layout[n] for n in self.net_names]
        arch = self.layout[self.arch_names[0]]
        self._require_uniform(arch)

        # One flat buffer with the three families as views into it, so a frame
        # can be emitted without concatenating anything.  Nets are padded to
        # the largest node count; ``net_nodes`` says how many are real.
        self.net_nodes = np.array([m.nodes for m in nets], dtype=np.int32)
        width = int(self.net_nodes.max())
        par_slots = (self.layout[self.layout.par].channels_per_node
                     if self.layout.par else 0)
        n_net = len(self.net_names) * width * 3
        n_arch = len(self.arch_names) * arch.nodes * 3
        self._source = np.zeros(n_net + n_arch + par_slots, dtype=np.float32)
        self.nets = self._source[:n_net].reshape(len(self.net_names), width, 3)
        self.arches = self._source[n_net:n_net + n_arch].reshape(
            len(self.arch_names), arch.nodes, 3)
        self.par = self._source[n_net + n_arch:]

        self._geometry(nets, width, arch)
        self._big_geometry()
        self._gather()

    # -- geometry ---------------------------------------------------------- #

    def _require_uniform(self, arch) -> None:
        for name in self.arch_names:
            if self.layout[name].nodes != arch.nodes:
                raise ValueError(f"{name} has {self.layout[name].nodes} nodes, "
                                 f"not {arch.nodes}")

    def _geometry(self, nets, width: int, arch) -> None:
        count = len(nets)
        #: 0 at the left edge, 1 at the right.  ``(nets, nodes)``, per net.
        self.net_x = np.zeros((count, width), dtype=np.float32)
        #: 0 at the **apex** (grid row 0), 1 along the base.  The nets are
        #: triangles: row 0 holds a single node, the bottom row holds thirty.
        self.net_y = np.zeros((count, width), dtype=np.float32)
        #: Distance from the net's centre, normalised so the far corner is ~1.
        self.net_r = np.zeros((count, width), dtype=np.float32)
        #: Angle around the centre, 0..1 -- for pinwheels and spirals.
        self.net_angle = np.zeros((count, width), dtype=np.float32)
        #: True where a slot holds a real pixel.
        self.net_mask = np.zeros((count, width), dtype=bool)
        for i, net in enumerate(nets):
            height, grid_w = net.grid
            n = net.nodes
            rows = net.coords[:, 0].astype(np.float32)
            cols = net.coords[:, 1].astype(np.float32)
            x = cols / max(1.0, grid_w - 1)
            y = rows / max(1.0, height - 1)
            cx = float(x.mean())
            dx, dy = x - cx, y - 0.5
            r = np.hypot(dx, dy).astype(np.float32)
            r /= max(1e-6, float(r.max()))
            self.net_x[i, :n] = x
            self.net_y[i, :n] = y
            self.net_r[i, :n] = r
            self.net_angle[i, :n] = ((np.arctan2(dy, dx) / (2 * np.pi)) % 1.0)
            self.net_mask[i, :n] = True

        t = arch.strip_t.astype(np.float32)
        #: 0..1 along the strip, base -> apex -> base.
        self.arch_t = t
        #: 0 at either base, 1 at the apex -- the arch's height.
        self.arch_h = (1.0 - np.abs(2.0 * t - 1.0)).astype(np.float32)
        #: -1 on the first leg, +1 on the second.
        self.arch_side = np.where(t < 0.5, -1.0, 1.0).astype(np.float32)
        #: 0 at the front of the corridor, 1 at the back.
        n = len(self.arch_names)
        self.depth = (np.arange(n, dtype=np.float32) / max(1, n - 1))

    def _big_geometry(self) -> None:
        """Geometry of the "Big Triangle" nets as *one* surface.

        Four nets -- top, bottom-left, bottom-right and an inverted one in the
        middle -- are mounted as one large triangle.  Painted per net they
        are four small triangles doing the same thing; painted through this
        they are one: ``big.x``/``big.y`` run 0..1 across the whole big
        triangle (y 0 at its apex), ``big.r``/``big.angle`` are about its
        centre.  Same ``(k, nodes)`` shape as the per-net vectors sliced by
        :attr:`big`, so any net effect takes it via its ``geo`` argument.

        Computed from the xLights world positions: the nets are projected
        onto their common plane (they are not quite coplanar -- the mounting
        angles differ by a few degrees -- which is why this is a projection
        and not a lookup).
        """
        from .geometry import world_positions

        self.big = self.net_slice("Big Triangle")
        width = self.nets.shape[1]
        names = self.net_names[self.big]
        positions = world_positions(self.layout)
        points = np.vstack([positions[n] for n in names]).astype(np.float64)
        centre = points.mean(axis=0)
        _, _, vt = np.linalg.svd(points - centre, full_matrices=False)
        normal = vt[2]
        # In-plane "up" is world Y projected into the plane; "right" is
        # across, signed so that x grows from the bottom-left net to the
        # bottom-right one.
        up = np.array([0.0, 1.0, 0.0]) - normal * normal[1]
        up /= max(np.linalg.norm(up), 1e-9)
        right = np.cross(up, normal)
        right /= max(np.linalg.norm(right), 1e-9)
        u = (points - centre) @ right
        v = (points - centre) @ up
        base = [i for i, n in enumerate(names)
                if positions[n][:, 1].mean() < points[:, 1].mean()]
        if len(base) >= 2:
            lo, hi = sorted(base, key=lambda i: positions[names[i]].mean(axis=0) @ right)[::len(base) - 1]
            if positions[names[hi]].mean(axis=0) @ right < positions[names[lo]].mean(axis=0) @ right:
                right, u = -right, -u

        count = len(names)
        x = np.zeros((count, width), dtype=np.float32)
        y = np.zeros((count, width), dtype=np.float32)
        r = np.zeros((count, width), dtype=np.float32)
        angle = np.zeros((count, width), dtype=np.float32)
        u_lo, u_hi = float(u.min()), float(u.max())
        v_lo, v_hi = float(v.min()), float(v.max())
        cursor = 0
        for i, name in enumerate(names):
            n = self.layout[name].nodes
            uu, vv = u[cursor:cursor + n], v[cursor:cursor + n]
            cursor += n
            x[i, :n] = (uu - u_lo) / max(u_hi - u_lo, 1e-9)
            y[i, :n] = 1.0 - (vv - v_lo) / max(v_hi - v_lo, 1e-9)
        # Centre of the big triangle: its centroid, one third up from the base.
        cx, cy = 0.5, 2.0 / 3.0
        dx, dy = x - cx, y - cy
        r[:] = np.hypot(dx, dy)
        r /= max(1e-6, float((r * self.net_mask[self.big]).max()))
        angle[:] = (np.arctan2(dy, dx) / (2 * np.pi)) % 1.0
        self.big_geo = NetGeometry(x, y, r, angle)

    # -- output ------------------------------------------------------------ #

    def _gather(self) -> None:
        """Precompute source -> channel indices for the whole frame.

        Done once, so emitting a frame is two fancy-index operations rather
        than a loop over 33 models.  ``source`` is the concatenation of the
        three buffers in a fixed order; ``dst`` is where each value lands.
        """
        src: list[np.ndarray] = []
        dst: list[np.ndarray] = []
        offset = 0
        width = self.nets.shape[1]
        order = [*self.net_names, *self.arch_names,
                 *([self.layout.par] if self.layout.par else [])]
        for name in order:
            model = self.layout[name]
            stride = model.channels_per_node
            index = np.arange(model.nodes, dtype=np.int64)
            for slot, channel in enumerate(model.order):
                src.append(offset + index * stride + channel)
                dst.append(model.start - 1 + index * stride + slot)
            # A net's buffer row is padded to the widest net; skip the pad.
            offset += (width if model.kind == "net" else model.nodes) * stride
        if offset != self._source.size:
            raise ValueError(
                f"gather covers {offset} values, buffer holds {self._source.size}"
            )
        self._src = np.concatenate(src)
        self._dst = np.concatenate(dst)
        self._scratch = np.empty(self._src.size, dtype=np.float32)

    def net_slice(self, group: str) -> slice:
        """The nets of an xLights model group, as a slice into ``self.nets``.

        A slice rather than an index array on purpose: slices give a writable
        view, so an effect painted through one lands in the canvas.  The show's
        groups happen to be contiguous ("Big Triangle" is nets 1-4, "Small
        Triangle Nets" is 5-8); if that ever stops being true this raises
        rather than quietly painting the wrong nets.
        """
        members = self.layout.groups.get(group)
        if not members:
            raise KeyError(f"no model group {group!r}")
        index = sorted(self.net_names.index(n) for n in members
                       if n in self.net_names)
        if not index:
            raise KeyError(f"group {group!r} holds no nets")
        if index != list(range(index[0], index[-1] + 1)):
            raise ValueError(f"group {group!r} is not a contiguous run of nets")
        return slice(index[0], index[-1] + 1)

    def net_pair(self) -> tuple[slice, slice]:
        """(big nets, small nets): the two groups gestures lead and rest.

        "Big Triangle" is the group xLights declares.  The small nets used to
        be their own group ("Small Triangle Nets"); the 2026-08-30 layout
        dropped it, so they are whatever is left -- provided that is also a
        contiguous run, for the same reason :meth:`net_slice` insists on one.
        """
        big = self.net_slice("Big Triangle")
        if "Small Triangle Nets" in self.layout.groups:
            return big, self.net_slice("Small Triangle Nets")
        rest = [i for i in range(len(self.net_names))
                if not (big.start <= i < big.stop)]
        if not rest:
            raise ValueError("every net is in 'Big Triangle'; nothing is left to rest")
        if rest != list(range(rest[0], rest[-1] + 1)):
            raise ValueError("the nets outside 'Big Triangle' are not contiguous")
        return big, slice(rest[0], rest[-1] + 1)

    def clear(self) -> None:
        self._source[:] = 0.0

    def to_channels(self, out: np.ndarray | None = None, brightness: float = 1.0,
                    gamma: float = 1.0) -> np.ndarray:
        """Flatten the canvas into a uint8 channel array for the wire.

        ``gamma`` defaults to 1.0 -- linear, matching the Falcon's configured
        ``DefaultGammaUnderFullControl=1``.  Change it here and an ``.fseq``
        capture stops matching what the controller would actually show.
        """
        if out is None:
            out = np.zeros(self.layout.channel_count, dtype=np.uint8)
        np.take(self._source, self._src, out=self._scratch)
        if brightness != 1.0:
            self._scratch *= brightness
        np.clip(self._scratch, 0.0, 1.0, out=self._scratch)
        if gamma != 1.0:
            np.power(self._scratch, gamma, out=self._scratch)
        self._scratch *= 255.0
        self._scratch += 0.5
        out[self._dst] = self._scratch.astype(np.uint8)
        return out


if __name__ == "__main__":
    canvas = Canvas()
    print(f"nets   {canvas.nets.shape}  arches {canvas.arches.shape}  "
          f"par {canvas.par.shape}")
    print(f"gather {canvas._src.size} values -> "
          f"{canvas.layout.channel_count} channels")
    canvas.nets[:] = (1.0, 0.0, 0.0)
    canvas.arches[:] = (1.0, 0.0, 0.0)
    if canvas.par.size:
        canvas.par[:] = (1.0, 0.0, 0.0, 0.0)
    frame = canvas.to_channels()
    net0 = canvas.layout[canvas.net_names[0]]
    arch0 = canvas.layout[canvas.arch_names[0]]
    print(f"red on an RGB net  -> {frame[net0.slice][:3]}  (expect [255 0 0])")
    print(f"red on a GRB arch  -> {frame[arch0.slice][:3]}  (expect [0 255 0])")
