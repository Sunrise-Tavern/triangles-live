"""Live (real-time) lighting engine for the Triangles rig.

Audio in -> analysis -> beat clock -> arranger -> renderer -> DDP -> Falcon.
See ../LIVE_PLAN.md.  Everything here is import-light and runs on a Pi;
the offline generator in ``triseq/`` stays untouched.
"""

__all__ = ["layout", "ddp", "fseq", "timing", "fake_falcon"]
