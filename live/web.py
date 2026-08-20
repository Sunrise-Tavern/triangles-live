"""The browser UI: preview, controls, presets -- served from the daemon itself.

One process holds the render thread and this server, so the page is both the
dev preview on the Mac and the control surface at the rig.  No build toolchain:
the front end is one HTML file, one JS file and one stylesheet, served as-is.

Two channels to the browser:

* **REST** for anything with a result worth confirming -- reading the schema,
  patching settings, saving and loading presets, moving the camera.
* **one WebSocket** carrying both status (JSON text, ~5 Hz) and preview frames
  (raw RGB bytes, at the engine's own frame rate).  Binary, because 3 300 dots
  of JSON is about 300 KB/s of quoting and commas for no benefit.

The preview runs at the render rate on purpose.  It was capped at 15 Hz first,
and that reads as stutter even though the engine is keeping perfect time --
you were watching 15 of every 40 frames.  At 40 Hz the feed is ~390 KB/s of
which most is skipped: identical frames are not resent, so a held look or a
blackout costs nothing.

The preview samples the **wire frame**, not the canvas, so what you watch is
what the Falcon is being sent -- including blackout and master brightness.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import numpy as np
from aiohttp import WSMsgType, web

from . import settings as knobs
from .engine import Engine
from .geometry import build_preview

STATIC = Path(__file__).resolve().parent / "static"

STATUS_HZ = 5.0
#: Cap on the preview feed.  ``None`` means "match the engine", which is the
#: default; turn it down only for a slow client over a bad link.
PREVIEW_HZ: float | None = None


class Server:
    def __init__(self, engine: Engine, *, net_stride: int = 2,
                 arch_stride: int = 6, preview_fps: float | None = None,
                 detail: float = 1.0) -> None:
        self.engine = engine
        self.preview_fps = preview_fps or PREVIEW_HZ or engine.fps
        # Detail is spatial, frame rate is temporal -- two separate dials.
        # Raising detail costs bandwidth linearly: 1.0 is ~3 300 dots.
        self.strides = (max(1, round(net_stride / detail)),
                        max(1, round(arch_stride / detail)))
        self.camera = {"yaw": 28.0, "pitch": 14.0, "distance": 1.6,
                       "aspect": 16 / 9}
        self.generation = 0
        self.geometry = self._build()

    def _build(self):
        return build_preview(self.engine.layout, net_stride=self.strides[0],
                             arch_stride=self.strides[1], **self.camera)

    # -- REST -------------------------------------------------------------- #

    async def get_schema(self, request: web.Request) -> web.Response:
        return web.json_response({
            "settings": knobs.describe(),
            "models": self.geometry.models,
            "channels": self.engine.layout.channel_count,
            "fps": self.engine.fps,
            "unaddressed": [m.name for m in self.engine.layout.unaddressed()],
        })

    async def get_geometry(self, request: web.Request) -> web.Response:
        geo = self.geometry
        return web.json_response({
            "generation": self.generation,
            "camera": self.camera,
            "count": len(geo),
            # Rounded to keep the payload small; the canvas is ~1000 px wide,
            # so four decimals is already sub-pixel.
            "xy": [round(float(v), 4) for v in geo.xy.reshape(-1)],
            "size": [round(float(v), 3) for v in geo.size],
            "model_of": geo.model_of.tolist(),
            "models": geo.models,
        })

    async def post_camera(self, request: web.Request) -> web.Response:
        patch = await request.json()
        for key in ("yaw", "pitch", "distance", "aspect"):
            if key in patch:
                self.camera[key] = float(patch[key])
        self.camera["distance"] = min(max(self.camera["distance"], 0.4), 6.0)
        self.geometry = self._build()
        self.generation += 1
        return web.json_response({"generation": self.generation,
                                  "camera": self.camera})

    async def get_settings(self, request: web.Request) -> web.Response:
        return web.json_response(self.engine.settings.to_dict())

    async def post_settings(self, request: web.Request) -> web.Response:
        try:
            changed = self.engine.settings.apply(await request.json())
        except (KeyError, ValueError) as exc:
            raise web.HTTPBadRequest(text=str(exc)) from exc
        return web.json_response({"changed": changed,
                                  "settings": self.engine.settings.to_dict()})

    async def get_presets(self, request: web.Request) -> web.Response:
        return web.json_response({"presets": knobs.list_presets()})

    async def put_preset(self, request: web.Request) -> web.Response:
        name = request.match_info["name"]
        try:
            knobs.save_preset(name, self.engine.settings)
        except ValueError as exc:
            raise web.HTTPBadRequest(text=str(exc)) from exc
        return web.json_response({"saved": name, "presets": knobs.list_presets()})

    async def load_preset(self, request: web.Request) -> web.Response:
        name = request.match_info["name"]
        try:
            self.engine.settings.load(knobs.read_preset(name))
        except (ValueError, FileNotFoundError) as exc:
            raise web.HTTPNotFound(text=str(exc)) from exc
        return web.json_response({"loaded": name,
                                  "settings": self.engine.settings.to_dict()})

    async def delete_preset(self, request: web.Request) -> web.Response:
        name = request.match_info["name"]
        try:
            knobs.delete_preset(name)
        except ValueError as exc:
            raise web.HTTPBadRequest(text=str(exc)) from exc
        return web.json_response({"deleted": name, "presets": knobs.list_presets()})

    # -- WebSocket --------------------------------------------------------- #

    async def websocket(self, request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse(heartbeat=20.0)
        await ws.prepare(request)

        pump = asyncio.create_task(self._pump(ws))
        try:
            async for message in ws:
                # The browser only talks to us over REST; anything arriving
                # here is a stray, and ignoring it keeps the pump alive.
                if message.type == WSMsgType.ERROR:
                    break
        finally:
            pump.cancel()
            await asyncio.gather(pump, return_exceptions=True)
        return ws

    async def _pump(self, ws: web.WebSocketResponse) -> None:
        """Push preview frames and, less often, status."""
        period = 1.0 / self.preview_fps
        status_every = max(1, round(self.preview_fps / STATUS_HZ))
        tick = 0
        last: bytes | None = None
        loop = asyncio.get_running_loop()
        next_at = loop.time()
        try:
            while not ws.closed:
                next_at += period
                delay = next_at - loop.time()
                if delay < -2 * period:
                    # A slow client dragged us behind -- resync rather than
                    # sprinting to catch up on frames nobody will ever see.
                    next_at = loop.time()
                    delay = 0.0
                await asyncio.sleep(max(0.0, delay))

                payload = np.ascontiguousarray(
                    self.geometry.sample(self.engine.frame)).tobytes()
                if payload != last:
                    # A held look, a blackout, or a paused engine costs nothing.
                    await ws.send_bytes(payload)
                    last = payload

                if tick % status_every == 0:
                    await ws.send_str(json.dumps({
                        "status": self.engine.status.to_dict(),
                        "settings": self.engine.settings.to_dict(),
                        "generation": self.generation,
                    }))
                tick += 1
        except (ConnectionResetError, asyncio.CancelledError):
            pass

    # -- wiring ------------------------------------------------------------ #

    def app(self) -> web.Application:
        app = web.Application()
        app.add_routes([
            web.get("/", self.index),
            web.get("/api/schema", self.get_schema),
            web.get("/api/geometry", self.get_geometry),
            web.post("/api/camera", self.post_camera),
            web.get("/api/settings", self.get_settings),
            web.post("/api/settings", self.post_settings),
            web.get("/api/presets", self.get_presets),
            web.put("/api/presets/{name}", self.put_preset),
            web.post("/api/presets/{name}/load", self.load_preset),
            web.delete("/api/presets/{name}", self.delete_preset),
            web.get("/ws", self.websocket),
            web.static("/static", STATIC),
        ])
        return app

    async def index(self, request: web.Request) -> web.FileResponse:
        return web.FileResponse(STATIC / "index.html")


def serve(engine: Engine, host: str = "0.0.0.0", port: int = 8080,
          preview_fps: float | None = None, detail: float = 1.0) -> None:
    """Run the engine and the server until interrupted."""
    server = Server(engine, preview_fps=preview_fps, detail=detail)
    engine.start()
    try:
        web.run_app(server.app(), host=host, port=port, print=None)
    finally:
        engine.stop()
