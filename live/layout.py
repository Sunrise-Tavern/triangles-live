"""Channel map: xLights models -> absolute DMX channel ranges, plus geometry.

This is the one place that knows how a pixel buffer becomes bytes on the wire.
Everything is read from ``xlights_rgbeffects.xml`` / ``xlights_networks.xml``
at import time -- rename or re-address something in xLights and this changes
with it, rather than drifting silently.

Three things it resolves that are easy to get wrong by hand:

* **Controller-relative start channels.**  The nets are addressed
  ``!Falcon_F16V5_0E1C:1``, not ``1``.  Controller start channels come from
  walking ``xlights_networks.xml`` in document order.
* **String colour order.**  The nets are ``RGB Nodes``, the arches are
  ``GRB Nodes``.  A renderer produces RGB; the byte order on the wire is not
  the same for both, and getting it wrong looks *almost* right (red/green swap).
* **Custom-model node maps.**  A net is 465 nodes scattered over a 59x51
  grid, or 435 over 57x49 -- the rig has both.  The mapping node -> (row,
  col) lives in ``CustomModelCompressed``.

The DJ par is optional: the 2026-08-30 layout dropped it, and a show with no
DMX fixture is a show, not an error.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# The show folder is the parent of generated/ (same convention as triseq.show).
SHOW_DIR = Path(__file__).resolve().parent.parent.parent
RGB_EFFECTS = SHOW_DIR / "xlights_rgbeffects.xml"
NETWORKS = SHOW_DIR / "xlights_networks.xml"

TUNNEL_GROUP = "Tunnel"

#: Colour order per xLights ``StringType``, as indices into an (R, G, B, W)
#: source tuple.  ``GRB Nodes`` means channel 0 carries green.
STRING_ORDER = {
    "RGB Nodes": (0, 1, 2),
    "GRB Nodes": (1, 0, 2),
    "BRG Nodes": (2, 0, 1),
    "BGR Nodes": (2, 1, 0),
    "RBG Nodes": (0, 2, 1),
    "GBR Nodes": (1, 2, 0),
}


class LayoutError(RuntimeError):
    pass


@dataclass
class Controller:
    """A physical controller and the slice of channel space it owns."""

    name: str
    ip: str
    protocol: str
    start: int          # 1-based
    channels: int

    @property
    def end(self) -> int:
        return self.start + self.channels - 1

    @property
    def slice(self) -> slice:
        return slice(self.start - 1, self.start - 1 + self.channels)


@dataclass
class Model:
    """One addressable fixture and where its pixels land in the channel array."""

    name: str
    kind: str                 # "net" | "arch" | "par"
    start: int                # 1-based absolute start channel
    nodes: int                # pixel count
    channels_per_node: int    # 3 for pixels, 4 for the RGBW par
    order: tuple[int, ...]    # output slot -> index into (R, G, B, W)
    #: Custom models only: node index (0-based) -> (row, col) in the grid.
    grid: tuple[int, int] | None = None
    coords: np.ndarray | None = None   # (nodes, 2) int, [row, col]
    #: Arches only: node index -> position along the strip, 0 (base) .. 1 (base),
    #: peaking at 0.5 at the apex.
    strip_t: np.ndarray | None = None
    world: tuple[float, float, float] = (0.0, 0.0, 0.0)
    #: The model's raw xLights attributes, for anything that needs geometry
    #: beyond the channel map (see :mod:`live.geometry`).
    source: dict[str, str] = field(default_factory=dict)

    @property
    def channels(self) -> int:
        return self.nodes * self.channels_per_node

    @property
    def end(self) -> int:
        """Last channel, 1-based inclusive."""
        return self.start + self.channels - 1

    @property
    def slice(self) -> slice:
        """Slice into a 0-based channel array."""
        return slice(self.start - 1, self.start - 1 + self.channels)

    def pack(self, rgb: np.ndarray, out: np.ndarray) -> None:
        """Write this model's pixels into a full channel array.

        ``rgb`` is (nodes, 3) or (nodes, 4) uint8 in **source** order --
        R, G, B, then W if the fixture has one.  The permutation in
        :attr:`order` is applied here, which is the only place the difference
        between the RGB nets and the GRB arches exists.
        """
        if rgb.shape[0] != self.nodes:
            raise ValueError(
                f"{self.name}: expected {self.nodes} nodes, got {rgb.shape[0]}"
            )
        block = out[self.slice].reshape(self.nodes, self.channels_per_node)
        for slot, source in enumerate(self.order):
            block[:, slot] = rgb[:, source] if source < rgb.shape[1] else 0

    def blank(self) -> np.ndarray:
        """An all-off (nodes, 3) buffer for this model."""
        return np.zeros((self.nodes, 3), dtype=np.uint8)


@dataclass
class Layout:
    models: dict[str, Model]
    #: Arch model names in physical front-to-back order (from the Tunnel group).
    arches: list[str]
    #: Net model names, natural order.
    nets: list[str]
    #: The DMX par's model name, or None when the layout has no DMX fixture.
    par: str | None
    channel_count: int
    groups: dict[str, list[str]] = field(default_factory=dict)
    controllers: dict[str, Controller] = field(default_factory=dict)

    def __getitem__(self, name: str) -> Model:
        return self.models[name]

    def of_kind(self, kind: str) -> list[Model]:
        return [m for m in self.models.values() if m.kind == kind]

    def blank_channels(self) -> np.ndarray:
        """A full, all-off channel array for one frame."""
        return np.zeros(self.channel_count, dtype=np.uint8)

    def _family_slice(self, kind: str) -> slice | None:
        members = [m for m in self.models.values() if m.kind == kind]
        if not members:
            return None
        lo = min(m.start for m in members) - 1
        hi = max(m.end for m in members)
        return slice(lo, hi)

    @property
    def nets_slice(self) -> slice | None:
        """Channel span covering every net.  Diagnostics only."""
        return self._family_slice("net")

    @property
    def arches_slice(self) -> slice | None:
        return self._family_slice("arch")

    def output(self, name: str) -> Controller:
        """A controller by name; raises with the available names if unknown."""
        if name not in self.controllers:
            raise LayoutError(
                f"No controller {name!r}.  Known: {sorted(self.controllers)}"
            )
        return self.controllers[name]

    def ddp_targets(self) -> list[Controller]:
        """Every controller the live engine can actually send to, in channel
        order.

        The show is not one receiver: the nets are on one Falcon and the
        corridor on another, and ``xlights_networks.xml`` is the only place
        that records which is which.  Deriving the list from the layout keeps
        the addresses in one file instead of two -- a second copy in
        ``live.toml`` drifts, and a stale address is silent, because DDP is
        UDP and nothing comes back.

        Entries with no ``<network>`` child (an FPP player) own no channels and
        are skipped; so is anything that is not DDP, since that is the only
        protocol :mod:`live.ddp` speaks.
        """
        return sorted(
            (c for c in self.controllers.values()
             if c.channels and c.protocol.upper() == "DDP"),
            key=lambda c: c.start,
        )

    def unaddressed(self) -> list[Model]:
        """Models that sit outside every controller's channel space.

        Controllers are laid out back to back in document order, so a
        ``MaxChannels`` that is too small does not fail loudly -- it silently
        pushes everything after it off the end of the last controller.  Those
        channels are rendered and then dropped, because there is no receiver
        that claims them.
        """
        spans = [(c.start, c.end) for c in self.controllers.values() if c.channels]
        return [
            m for m in self.models.values()
            if not any(lo <= m.start and m.end <= hi for lo, hi in spans)
        ]

    def describe(self) -> str:
        lines = [f"{self.channel_count} channels total"]
        for m in sorted(self.models.values(), key=lambda m: m.start):
            lines.append(
                f"  {m.name:<24} {m.kind:<5} ch {m.start:>6}-{m.end:<6} "
                f"{m.nodes:>4} nodes  order={m.order}"
            )
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Parsing
# --------------------------------------------------------------------------- #


def _controllers(networks: Path) -> dict[str, Controller]:
    """Controller name -> where it sits in channel space.

    xLights lays controllers out back to back in document order; a controller
    with no ``<network>`` child (an FPP player) consumes no channels.
    """
    if not networks.exists():
        raise LayoutError(f"Could not find {networks}")
    root = ET.parse(networks).getroot()
    found: dict[str, Controller] = {}
    cursor = 1
    for ctrl in root.findall("./Controller"):
        name = ctrl.get("Name") or ""
        size = sum(int(n.get("MaxChannels", "0")) for n in ctrl.findall("./network"))
        found[name] = Controller(
            name=name, ip=ctrl.get("IP", ""), protocol=ctrl.get("Protocol", ""),
            start=cursor, channels=size,
        )
        cursor += size
    return found


def _resolve_start(raw: str, controllers: dict[str, Controller]) -> int:
    raw = (raw or "").strip()
    if raw.startswith("!"):
        name, _, offset = raw[1:].partition(":")
        if name not in controllers:
            raise LayoutError(f"Model references unknown controller {name!r}")
        return controllers[name].start + int(offset) - 1
    if raw.startswith(("@", "#", ">")):
        raise LayoutError(
            f"Start channel form {raw!r} (model- or universe-relative) is not "
            "supported yet -- add it here if the layout starts using it."
        )
    return int(raw)


def _parse_custom(compressed: str, width: int, height: int) -> tuple[int, np.ndarray]:
    """``CustomModelCompressed`` -> (node count, (n, 2) array of [row, col]).

    Entries are ``node,row,col`` with a 1-based node number; the array is
    indexed by 0-based node so ``coords[i]`` is where pixel ``i`` sits.
    """
    entries = [e for e in compressed.split(";") if e]
    coords = np.full((len(entries), 2), -1, dtype=np.int32)
    for entry in entries:
        node, row, col = (int(v) for v in entry.split(","))
        if not 1 <= node <= len(entries):
            raise LayoutError(f"Custom model node {node} out of range 1..{len(entries)}")
        coords[node - 1] = (row, col)
    if (coords < 0).any():
        missing = int((coords[:, 0] < 0).sum())
        raise LayoutError(f"Custom model has {missing} unmapped node(s)")
    if coords[:, 0].max() >= height or coords[:, 1].max() >= width:
        raise LayoutError("Custom model coords exceed the declared grid size")
    return len(entries), coords


def _polyline_t(nodes: int, seg_counts: list[int]) -> np.ndarray:
    """Position of each arch node along the strip, 0..1, base -> apex -> base.

    Only the node *counts* per segment matter for a corridor renderer -- the
    arches are all the same triangle, so a normalised run along the strip is
    the useful coordinate, not world-space metres.
    """
    if sum(seg_counts) != nodes:
        # Fall back to an even run rather than guessing at segment splits.
        return np.linspace(0.0, 1.0, nodes, dtype=np.float32)
    t = np.empty(nodes, dtype=np.float32)
    cursor = 0
    edges = np.linspace(0.0, 1.0, len(seg_counts) + 1)
    for i, count in enumerate(seg_counts):
        t[cursor:cursor + count] = np.linspace(
            edges[i], edges[i + 1], count, endpoint=False
        )
        cursor += count
    return t


def load_layout(rgb_effects: Path | None = None, networks: Path | None = None) -> Layout:
    rgb_path = Path(rgb_effects) if rgb_effects else RGB_EFFECTS
    net_path = Path(networks) if networks else NETWORKS
    if not rgb_path.exists():
        raise LayoutError(
            f"Could not find {rgb_path}.  live/ expects to live in "
            "<show folder>/generated/."
        )

    controllers = _controllers(net_path)
    root = ET.parse(rgb_path).getroot()

    models: dict[str, Model] = {}
    for el in root.findall("./models/model"):
        attrs = el.attrib
        name = attrs["name"]
        display = attrs.get("DisplayAs", "")
        start = _resolve_start(attrs.get("StartChannel", ""), controllers)
        world = (
            float(attrs.get("WorldPosX", 0.0)),
            float(attrs.get("WorldPosY", 0.0)),
            float(attrs.get("WorldPosZ", 0.0)),
        )
        string_type = attrs.get("StringType", "RGB Nodes")

        if display == "Custom":
            width = int(attrs["CustomWidth"])
            height = int(attrs["CustomHeight"])
            compressed = attrs.get("CustomModelCompressed")
            if not compressed:
                raise LayoutError(
                    f"{name}: only CustomModelCompressed layouts are supported; "
                    "re-save the model in a recent xLights."
                )
            count, coords = _parse_custom(compressed, width, height)
            models[name] = Model(
                name=name, kind="net", start=start, nodes=count,
                channels_per_node=3, order=_order(string_type, name),
                grid=(height, width), coords=coords, world=world,
                source=dict(attrs),
            )

        elif display == "Poly Line":
            per_string = int(attrs.get("NodesPerString", 0))
            strings = int(attrs.get("PolyStrings", 1))
            count = per_string * strings
            points = int(attrs.get("NumPoints", 2))
            segs = [int(attrs.get(f"PolyNode{i + 1}", 0)) for i in range(points - 1)]
            if sum(segs) != count and attrs.get("PointData"):
                # Not declared (xLights omits them for an even split): share
                # the nodes out by segment length.  See geometry.segment_counts.
                raw = np.array([float(v) for v in attrs["PointData"].split(",")],
                               dtype=np.float32).reshape(-1, 3)
                lengths = np.linalg.norm(np.diff(raw, axis=0), axis=1)
                if lengths.sum() > 0:
                    segs = [int(round(count * float(l) / float(lengths.sum())))
                            for l in lengths]
                    segs[-1] += count - sum(segs)
            models[name] = Model(
                name=name, kind="arch", start=start, nodes=count,
                channels_per_node=3, order=_order(string_type, name),
                strip_t=_polyline_t(count, segs), world=world,
                source=dict(attrs),
            )

        elif display.startswith("Dmx"):
            width = int(attrs.get("DmxChannelCount", 4))
            slots = [
                int(attrs.get("DmxRedChannel", 1)),
                int(attrs.get("DmxGreenChannel", 2)),
                int(attrs.get("DmxBlueChannel", 3)),
                int(attrs.get("DmxWhiteChannel", 0)),
            ]
            # order[i] = which of (R,G,B,W) drives DMX slot i+1; 0 = unused.
            order = [0] * width
            for src, slot in enumerate(slots):
                if 1 <= slot <= width:
                    order[slot - 1] = src
            models[name] = Model(
                name=name, kind="par", start=start, nodes=1,
                channels_per_node=width, order=tuple(order), world=world,
                source=dict(attrs),
            )

        else:  # pragma: no cover - nothing else exists in this show yet
            raise LayoutError(f"{name}: unsupported DisplayAs {display!r}")

    groups = {
        g.get("name"): [n.strip() for n in (g.get("models") or "").split(",") if n.strip()]
        for g in root.findall("./modelGroups/modelGroup")
    }
    if TUNNEL_GROUP not in groups:
        raise LayoutError(f"No {TUNNEL_GROUP!r} model group -- corridor order is unknown.")
    # The order of that attribute IS the physical front-to-back order.  Do not sort.
    arches = groups[TUNNEL_GROUP]
    unknown = [n for n in arches if n not in models]
    if unknown:
        raise LayoutError(f"{TUNNEL_GROUP!r} names models that do not exist: {unknown}")

    nets = sorted((m.name for m in models.values() if m.kind == "net"), key=_natural_key)
    pars = [m.name for m in models.values() if m.kind == "par"]

    channel_count = max(m.end for m in models.values())

    return Layout(
        models=models, arches=arches, nets=nets,
        par=sorted(pars)[0] if pars else None,
        channel_count=channel_count, groups=groups, controllers=controllers,
    )


def _order(string_type: str, name: str) -> tuple[int, ...]:
    if string_type not in STRING_ORDER:
        raise LayoutError(
            f"{name}: unhandled StringType {string_type!r}.  Add it to "
            "STRING_ORDER -- guessing here swaps colours on the wire."
        )
    return STRING_ORDER[string_type]


def _natural_key(name: str):
    """Sort by the first number in the name, then the name.

    Net names now carry a position after the number ("Net 5 top", "Net 8
    Bottom left from front"), so a trailing-digit key would sort them as
    text; the number is what orders them.
    """
    match = re.search(r"\d+", name)
    return (name[:match.start()] if match else name,
            int(match.group()) if match else 0, name)


if __name__ == "__main__":
    layout = load_layout()
    print(layout.describe())
    print()
    print("controllers:")
    for c in layout.controllers.values():
        span = f"ch {c.start}-{c.end}" if c.channels else "no channels"
        print(f"  {c.name:<20} {c.protocol:<12} {c.ip:<16} {span}")
    orphans = layout.unaddressed()
    if orphans:
        print(f"\n!! {len(orphans)} model(s) sit outside every controller's channel "
              f"space and cannot be driven over DDP:")
        print(f"   {orphans[0].name} (ch {orphans[0].start}) ... "
              f"{orphans[-1].name} (ch {orphans[-1].end})")
    print()
    print(f"corridor order : {layout.arches[0]} ... {layout.arches[-1]} "
          f"({len(layout.arches)} arches)")
    print(f"nets           : {', '.join(layout.nets)}")
    print(f"par            : {layout.par or 'none'}")
