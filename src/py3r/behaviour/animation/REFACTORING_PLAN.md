# Animation Simplification Plan

## Current Assessment

The legacy implementation works and already has solid feature coverage:

- 2D and 3D points (including orthographic/perspective projection)
- point/line/boundary rendering
- dynamic styles driven by per-frame sources
- text overlays with formatting, colormaps, and background panels
- interactive playback and video export

The main issue is complexity concentration in one file. `animation_stream.py`
currently mixes:

- input validation and data normalization
- style parsing and dynamic style compilation
- coordinate projection and bounds handling
- OpenCV drawing logic
- stream state (`read`, iterator cursor)
- playback and encode/decode I/O

This coupling makes it harder to reason about changes and test components in isolation.

## Proposed Simpler Architecture

Keep public API signatures unchanged:

- `AnimationStream`
- `build_animation_stream`
- `build_animation_stream_from_points`

Split internals into four focused units:

1. `models.py`
   - small dataclasses for immutable render inputs:
     - `GeometryData`
     - `StyleProgram`
     - `TextOverlaySpec`
2. `compiler.py`
   - pure functions:
     - validate/normalize points, lines, boundaries
     - compile dynamic style arrays from `style_sources`
     - resolve optional 3D projection to 2D
3. `renderer.py`
   - single responsibility: `render_frame(idx, target=None)`
   - no file I/O, no window handling
4. `io.py`
   - playback + save wrappers that consume renderer output
   - all OpenCV capture/writer concerns isolated

`AnimationStream` then becomes a thin coordinator around `renderer + io`,
holding only stream cursor state and convenience methods.

## Migration Strategy

1. Keep this facade package stable (already done).
2. Introduce `compiler.py` first and port style compilation there.
3. Add `renderer.py` and migrate `render_into`/`get_frame`.
4. Move `play`/`save` to `io.py`.
5. Run existing `tests/test_animation_stream.py` unchanged for parity.
6. Remove direct dependencies on `animation_bak` once parity is complete.

## Why This Is Simpler

- Each file answers one question (compile, render, or I/O).
- Rendering can be unit-tested without video capture/writer setup.
- Style bugs no longer require touching playback code.
- New render features become additive instead of cross-cutting edits.
