"""The pixel canvas: per-model float buffers, and the geometry effects paint on.

Effects work in **float RGB, 0..1, in source order** and never touch channel
numbers or wire colour order -- :meth:`Canvas.to_channels` does that once per
frame through a precomputed gather.  Keeping the two apart is what let M1's
byte-exact test stay meaningful: an effect cannot accidentally address the
wrong fixture, because it cannot address fixtures at all.

Two facts about this rig make the whole thing vectorise:

* all eight nets share one 465-node map over a 59x51 triangular lattice, so
  they are one ``(8, 465, 3)`` array and an effect over "the nets" is one
  numpy expression;
* all 24 arches share one 360-node base->apex->base run, so the corridor is
  one ``(24, 360, 3)`` array and a wave down the tunnel is an outer product.

The geometry vectors below are the effects' whole vocabulary of *where*:
normalised, unit-free, and computed once.
"""

from __future__ import annotations

import numpy as np

from .layout import Layout, load_layout


class Canvas:
    """One frame's worth of pixels, plus the geometry to paint them by."""

    def __init__(self, layout: Layout | None = None) -> None:
        self.layout = layout or load_layout()
        self.net_names = list(self.layout.nets)
        self.arch_names = list(self.layout.arches)

        net = self.layout[self.net_names[0]]
        arch = self.layout[self.arch_names[0]]
        self._require_uniform(net, arch)

        # One flat buffer with the three families as views into it, so a frame
        # can be emitted without concatenating anything.
        par_slots = self.layout[self.layout.par].channels_per_node
        n_net = len(self.net_names) * net.nodes * 3
        n_arch = len(self.arch_names) * arch.nodes * 3
        self._source = np.zeros(n_net + n_arch + par_slots, dtype=np.float32)
        self.nets = self._source[:n_net].reshape(len(self.net_names), net.nodes, 3)
        self.arches = self._source[n_net:n_net + n_arch].reshape(
            len(self.arch_names), arch.nodes, 3)
        self.par = self._source[n_net + n_arch:]

        self._geometry(net, arch)
        self._gather()

    # -- geometry ---------------------------------------------------------- #

    def _require_uniform(self, net, arch) -> None:
        for name in self.net_names:
            other = self.layout[name]
            if other.nodes != net.nodes or not np.array_equal(other.coords, net.coords):
                raise ValueError(
                    f"{name} has a different node map from {net.name}.  The "
                    "renderer assumes all nets are the same model; give each "
                    "net its own buffer here if that ever stops being true."
                )
        for name in self.arch_names:
            if self.layout[name].nodes != arch.nodes:
                raise ValueError(f"{name} has {self.layout[name].nodes} nodes, "
                                 f"not {arch.nodes}")

    def _geometry(self, net, arch) -> None:
        height, width = net.grid
        rows = net.coords[:, 0].astype(np.float32)
        cols = net.coords[:, 1].astype(np.float32)
        #: 0 at the left edge, 1 at the right.
        self.net_x = cols / max(1.0, width - 1)
        #: 0 at the **apex** (grid row 0), 1 along the base.  The nets are
        #: triangles: row 0 holds a single node, the bottom row holds thirty.
        self.net_y = rows / max(1.0, height - 1)
        cx = float(self.net_x.mean())
        dx, dy = self.net_x - cx, self.net_y - 0.5
        #: Distance from the net's centre, normalised so the far corner is ~1.
        self.net_r = np.hypot(dx, dy).astype(np.float32)
        self.net_r /= max(1e-6, float(self.net_r.max()))
        #: Angle around the centre, 0..1 -- for pinwheels and spirals.
        self.net_angle = ((np.arctan2(dy, dx) / (2 * np.pi)) % 1.0).astype(np.float32)

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
        order = [*self.net_names, *self.arch_names, self.layout.par]
        for name in order:
            model = self.layout[name]
            stride = model.channels_per_node
            index = np.arange(model.nodes, dtype=np.int64)
            for slot, channel in enumerate(model.order):
                src.append(offset + index * stride + channel)
                dst.append(model.start - 1 + index * stride + slot)
            offset += model.nodes * stride
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
    canvas.par[:] = (1.0, 0.0, 0.0, 0.0)
    frame = canvas.to_channels()
    net0 = canvas.layout[canvas.net_names[0]]
    arch0 = canvas.layout[canvas.arch_names[0]]
    print(f"red on an RGB net  -> {frame[net0.slice][:3]}  (expect [255 0 0])")
    print(f"red on a GRB arch  -> {frame[arch0.slice][:3]}  (expect [0 255 0])")
