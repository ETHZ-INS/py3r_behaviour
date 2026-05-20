# Features — AI context page

*This page is written for AI assistants. See [Use with AI](index.md) for the gateway prompt.*

*For all method signatures and parameter details, see the [Features API reference](../api/features.md).*

---

## What Features is

`Features` wraps a `Tracking` object and accumulates derived time-series as
named columns in `features.data` (a `pandas.DataFrame`, one row per frame).
It is the computational layer of the pipeline.

Create it from a `Tracking` object:

```python
f = t.to_features()
# or equivalently:
f = Features(t)
```

---

## The store pattern — critical

**All feature computation methods return a `FeaturesResult`. Nothing is stored
automatically.** The user must call `.store()` to persist the result.

```python
result = f.distance_between('nose', 'tail')
result.store(name='nose_tail_dist')
# Now: f.data['nose_tail_dist'] is a pd.Series of per-frame distances
```

If the user asks "why is my feature not in features.data", they forgot `.store()`.

---

## Motion features

| What | Method |
|---|---|
| Frame-to-frame distance between two points | `f.distance_between(p1, p2)` |
| Boolean: is p1 within distance of p2? | `f.within_distance(p1, p2, ...)` |
| Speed of a single point | `f.speed(point)` |
| Acceleration of a single point | `f.acceleration(point)` |
| Heading angle from p1 to p2 | `f.azimuth(p1, p2)` |

See the API reference for parameter details and units.

---

## Boundaries

A boundary is a polygon. It can be **static** (fixed vertices computed once from
keypoint medians) or **dynamic** (vertices recomputed every frame from live
keypoint positions).

**Use static boundaries** for fixed arena regions (walls, objects, regions defined
at the start of the recording).

**Use dynamic boundaries** for body-relative regions (a polygon defined by the
animal's own keypoints, e.g. the area enclosed by the trunk).

```python
# Static: define once from median positions
arena_wall = f.define_static_boundary(
    points=['corner1', 'corner2', 'corner3', 'corner4']
)

# Dynamic: recomputed per frame
body_boundary = f.define_dynamic_boundary(
    points=['shoulder', 'hip', 'tail']
)

# Query: is nose inside the boundary?
result = f.within_boundary('nose', arena_wall)
result.store(name='in_arena')

# Distance to boundary edge
result = f.distance_to_boundary('nose', arena_wall, signed=True)
result.store(name='dist_to_wall')
```

Boundaries are stored in `f._assets` and persist through `.save()`. Pass a
`name` when defining to register it by string for later lookup:

```python
arena_wall = f.define_static_boundary(
    ['corner1', 'corner2', 'corner3', 'corner4'], name='arena'
)
# arena_wall is also stored in f._assets['arena']

# Later — pass the string name instead of the object:
result = f.within_boundary('nose', 'arena')
```

---

## Axes

An axis is an infinite line defined by two points. It can be **static** or
**dynamic** for the same reasons as boundaries.

```python
# Static axis from two median positions
midline = f.define_static_axis('left_wall', 'right_wall')

# Dynamic axis: direction shifts per frame
body_axis = f.define_dynamic_axis('nose', 'tail', offset=0.0)

# Perpendicular distance from nose to the axis
result = f.distance_to_axis('nose', midline, signed=True)
result.store(name='lateral_position')

# Does the axis intersect a boundary?
result = f.axis_intersects_boundary(body_axis, arena_wall)
result.store(name='facing_wall')
```

`signed=True` on `distance_to_axis` gives positive/negative values depending
on which side of the line the point is on. Use this to distinguish left vs. right.

---

## Combining features with boolean logic

Features stored in `features.data` can be combined using standard pandas boolean
operators. The result can then be stored as a new feature:

```python
moving = f.data['nose_speed'] > threshold   # threshold is a value the user supplies
in_arena = f.data['in_arena']
active_in_arena = moving & in_arena
f.store(active_in_arena.rename('active_in_arena'))
```

---

## Scaling and anchoring boundaries

Both boundary types accept scaling parameters to expand or shrink the polygon
about an anchor centre. See the API reference for the parameter names and
behaviour.

---

## Inspecting stored features and assets

```python
f.data.columns          # all stored feature names
f.list_assets()         # DataFrame of all stored boundaries and axes
f.list_boundaries()     # boundaries only
```

---

## Saving and loading

```python
f.save('path/to/features_dir/')
f2 = Features.load('path/to/features_dir/')
```

Saved features include the nested `Tracking` and all stored assets.

---

## AnimationStream

Call `.animation_stream()` on a `Features` object to get an `AnimationStream`
for rendering video overlays that include computed feature columns (e.g. speed
colour maps, boundary overlays). This is the preferred entry point for
feature-annotated video.

```python
stream = f.animation_stream(...)
```

For overlays on raw tracking data without feature annotations, use
`tracking.animation_stream(...)` instead. See the
[AnimationStream API reference](../api/animation.md) for configuration.

---

## Moving to Summary

```python
s = f.to_summary()
# or equivalently:
s = Summary(f)
```

Only features that have been stored with `.store()` will be available in Summary.
