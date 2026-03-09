"""Animation API facade.

This package keeps the public animation API stable while the implementation
is incrementally refactored away from the legacy module.
"""

from .animation_stream import (
    AnimationStream,
    build_animation_stream,
    build_animation_stream_from_points,
    collect_dynamic_source_names_from_style,
)

__all__ = [
    "AnimationStream",
    "build_animation_stream",
    "build_animation_stream_from_points",
    "collect_dynamic_source_names_from_style",
]
