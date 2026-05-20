# Tracking — AI context page

*This page is written for AI assistants. See [Use with AI](index.md) for the gateway prompt.*

*For all method signatures and parameter details, see the [Tracking API reference](../api/tracking.md).*

---

## What Tracking is

`Tracking` wraps a single recording's keypoint coordinate data as a
`pandas.DataFrame` (columns like `nose.x`, `nose.y`, `nose.likelihood`;
index named `frame`). It is primarily used for exploration — real analysis
runs through `TrackingCollection`. See [Collections](collections.md).

---

## TrackingMV — multi-view recordings

`TrackingMV` is the multi-camera variant of `Tracking`. It stores a dict of
view name → `Tracking` (one per camera) plus a calibration dict, and exposes
a `.stereo_triangulate()` method that returns a standard 3D `Tracking` object.

At collection level, pass `tracking_cls=TrackingMV` to the folder loader and
then call `.stereo_triangulate()` on the collection to convert the whole dataset
to 3D `Tracking` objects before proceeding with the normal pipeline:

```python
tc_mv = TrackingCollection.from_dlc_folder('data/', fps=30, tracking_cls=TrackingMV)
tc = tc_mv.stereo_triangulate()   # returns a normal TrackingCollection
```

From this point, `tc` is a standard `TrackingCollection` and the rest of the
pipeline is identical to the 2D case.

---

## Loading a single recording

| Format | Method |
|---|---|
| DeepLabCut single-animal CSV | `Tracking.from_dlc(filepath, handle, fps)` |
| DeepLabCut multi-animal CSV | `Tracking.from_dlcma(filepath, handle, fps)` |
| YOLO3R CSV | `Tracking.from_yolo3r(filepath, handle, fps)` |
| Saved directory | `Tracking.load(dirpath)` |

`handle` is a string identifier for the recording. `fps` is required for any
time-based summary statistics. See the API reference for full signatures.

---

## Standard preprocessing sequence

This is the expected order. Skipping steps will trigger warnings downstream in Features.

```python
t = Tracking.from_dlc('recording.csv', handle='subj01', fps=30)

# 1. Drop low-confidence detections
t.filter_likelihood(...)

# 2. Interpolate gaps left by step 1
t.interpolate(...)

# 3. Smooth trajectories
t.smooth_all(...)

# 4. Calibrate to real-world units (if needed)
t.rescale_by_known_distance(...)

# 5. Convert to Features
f = t.to_features()
```

See the API reference for the parameters of each method. All preprocessing
methods are **inplace by default** — pass `inplace=False` to get a new object.

---

## Defining synthetic keypoints

Use these before calling `.to_features()` if you need derived spatial references.

```python
# Midpoint between two tracked points
t.define_midpoint('body_centre', points=['nose', 'tail'])

# Fixed offset from a reference point
t.define_offset_point('virtual_ref', ref='nose', offset=...)
```

---

## Slicing

```python
t_trimmed = t.trim(startframe=..., endframe=...)   # inplace by default
t_slice = t.loc[start:end]                         # returns new Tracking
```

---

## AnimationStream

Call `.animation_stream()` on a `Tracking` to get an `AnimationStream` object
for rendering annotated video overlays. The stream is configured via the
`AnimationStream` API; see the [AnimationStream API reference](../api/animation.md).

```python
stream = t.animation_stream(...)
```

Feature-annotated overlays (with computed columns overlaid) are created from
`Features` instead — see the [Features page](features.md).

---

## Key gotchas

- Filter before interpolating: `filter_likelihood` replaces low-confidence frames
  with `NaN`; interpolation fills those gaps; smoothing then runs on clean data.
- `smooth_all` accepts per-point overrides if different keypoints need different
  smoothing windows — see the API reference.
- `rescale_by_known_distance` requires two keypoints whose true physical
  separation is known. Check the API reference for the expected unit.
- If the recording has no calibration, features involving distance will be in
  pixel units — note this for the user.
