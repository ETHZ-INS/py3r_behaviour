"""Asset type registry for Features geometric assets.

Geometric assets follow a static/dynamic pair pattern:
- StaticX  — resolved numeric coordinates, fixed across frames.
- DynamicX — keypoint name references, resolved per frame from tracking data.

To add a new asset type, create its dataclass (with ``to_dict`` / ``from_dict`` /
``with_name``), give it a unique ``"kind"`` string, and add one entry here.
"""

from __future__ import annotations

from py3r.behaviour.features.axis import DynamicAxis, StaticAxis
from py3r.behaviour.features.boundary import DynamicBoundary, StaticBoundary

# Maps serialised "kind" strings to asset classes.
_ASSET_KINDS: dict[str, type] = {
    "static_boundary": StaticBoundary,
    "dynamic_boundary": DynamicBoundary,
    "static_axis": StaticAxis,
    "dynamic_axis": DynamicAxis,
}

# Backward compatibility: files saved before the kind rename used "static" / "dynamic".
_LEGACY_ASSET_KINDS: dict[str, type] = {
    "static": StaticBoundary,
    "dynamic": DynamicBoundary,
}
