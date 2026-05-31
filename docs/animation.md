# Animation and video overlay

`animation_stream` renders keypoint coordinates, lines, boundaries, and
per-frame feature values as an OpenCV-backed frame stream supporting
random-access rendering (`get_frame`), sequential playback (`play`), and
saving to a video file (`save`).

---

## Rendering modes

The stream can be used in two ways:

| Mode | When to use | Key arguments |
|---|---|---|
| **Overlay on source video** | Composite geometry on top of real frames | `pixel_coords=True`, `undo_meta_scaling=True` |
| **Standalone canvas** | Render geometry on a blank background, scaling to fit | `pixel_coords=False` (default) |

In overlay mode, coordinates must be in pixels — matching the original video.
If tracking data has been rescaled to real-world units (e.g. metres),
`undo_meta_scaling=True` reverses that scaling back to pixels before rendering.
In standalone mode, coordinates are scaled automatically to fill the canvas,
so units do not matter.

---

## Entry points

| Entry point | When to use |
|---|---|
| [`Tracking.animation_stream`][py3r.behaviour.tracking.tracking.Tracking.animation_stream] | Keypoints and static styling only |
| [`Features.animation_stream`][py3r.behaviour.features.features.Features.animation_stream] | Keypoints + feature columns, boundaries, and dynamic (data-driven) styling |

Both return an [`AnimationStream`][py3r.behaviour.animation.animation_stream.AnimationStream]
object. `AnimationStream` is not intended to be constructed directly.

---

## Basic usage

```python
# Overlay on a source video (coordinates must be in pixels)
stream = f.animation_stream(
    points=["nose", "bodycentre", "tailbase"],
    lines=[("nose", "bodycentre"), ("bodycentre", "tailbase")],
    boundaries=["centre"],
    features={"Speed": "speed_of_bodycentre_in_xy", "In centre": "in_centre"},
    pixel_coords=True,
    undo_meta_scaling=True,
    canvas_size=(1280, 1024),
    style=style,
)
stream.play(video_path="recording.avi")
stream.save(video_path="recording.avi", out_path="output.mp4")

# Standalone canvas (auto-scaling, no video required)
stream = t.animation_stream(
    points=["nose", "bodycentre"],
    canvas_size=(800, 600),
    style=style,
)
```

---

## The style dict

The style dict controls the visual appearance of every element in the animation —
colours, sizes, and transparency — and can drive any of these properties
dynamically from per-frame feature data, making it possible to build rich,
data-driven visualisations. This flexibility means it can become quite involved,
but most use cases only need a small subset of its options.

The style dict is entirely optional — calling `animation_stream` with no
`style` argument renders everything in default colours. Style properties
that are present in the dict but not applicable to the current animation
(e.g. styling for a boundary that is not being drawn) are silently ignored,
so a lab-wide style file can be loaded from JSON and reused across recordings
without modification.

```python
import json
style = json.load(open("lab_style.json"))
```

### Structure

The dict has one section per geometry type. Within each section, `"default"`
applies to every element of that type; a named key overrides only that specific
element. Named overrides are merged on top of `"default"` rather than replacing
it, so you only need to specify what differs. This pattern is the same across
all sections, including `"text"` (where the named keys are the display label
strings passed via `features=` in `Features.animation_stream`).

The `"text"` section has one addition, however: `"origin"`
sits at the top level of the section (outside `"default"`) because it
positions the whole text panel and is not a per-label property.

```python
style = {
    "points":     { "default": {...}, "point_name": {...}, ... },
    "lines":      { "default": {...}, ("p1", "p2"): {...}, ... },
    "boundaries": { "default": {...}, "boundary_name": {...}, ... },
    "axes":       { "default": {...}, "axis_name": {...}, ... },
    "text": {           # styles the feature-value text overlay panel
        "origin": (x, y),   # pixel position of the text block — top-level only
        "default": {...},
        "label_name": {...},
        ...
    },
}
```

---

### Draw order

Elements are composited in this order, so later elements appear on top of earlier ones:

1. Boundary fills
2. Boundary edges
3. Axes
4. Lines
5. Points
6. Text overlay panel

This matters when using `fill_alpha` — a semi-transparent boundary fill will
show the video frame underneath, but lines and points drawn afterwards will
appear on top of the fill.


---

### Points

| Property | Type | Default |
|---|---|---|
| `color` | `(B, G, R)` | `(0, 255, 255)` |
| `radius` | `int` | `3` |

```python
"points": {
    "default": {"color": (170, 170, 170), "radius": 2},
    "nose":    {"color": (0, 255, 0), "radius": 5},
}
```

---

### Lines

Lines are identified by the same `(point1, point2)` tuples passed to
`lines=` in `animation_stream`.

| Property | Type | Default |
|---|---|---|
| `color` | `(B, G, R)` | `(255, 255, 255)` |
| `width` | `int` | `1` |

```python
"lines": {
    "default": {"color": (200, 200, 200), "width": 1},
    ("nose", "bodycentre"): {"color": (0, 255, 0), "width": 2},
}
```

---

### Boundaries

| Property | Type | Default |
|---|---|---|
| `edge_color` | `(B, G, R)` or `None` | `(0, 255, 0)` |
| `edge_width` | `int` | `1` |
| `fill_color` | `(B, G, R)` or `None` | `None` |
| `fill_alpha` | `float` 0–1 | `0.0` |
| `fill_mode` | `"normal"` or `"erase"` | `"normal"` |

```python
"boundaries": {
    "centre": {"fill_color": (50, 50, 200), "fill_alpha": 0.3, "edge_width": 0},
}
```

---

### Text overlays (`"text"` section)

The `"text"` section styles the on-screen panel that displays per-frame
feature values — the same columns passed via `features=` to `animation_stream`.
The section name is `"text"` in the style dict, but the content it controls
comes from `Features.data`; there is no separate `"features"` section. The `features=` argument to `animation_stream` controls which
columns to show. Pass a **dict** to control the display label independently
of the column name:

```python
# List form — label is the column name
features=["speed_of_bodycentre_in_xy", "in_centre"]

# Dict form — keys are display labels, values are column names
features={"Speed (m/s)": "speed_of_bodycentre_in_xy", "In centre": "in_centre"}
```

Pass `None` or `""` in a list, or `None` as a dict value, to insert a blank
spacer line between groups of labels.

`"origin"` is a **top-level key inside `"text"`**; all other properties go
inside `"default"` or per-label:

| Property | Where | Default |
|---|---|---|
| `origin` | top-level `"text"` | `(10, 20)` |
| `color` | `"default"` or per-label | `(255, 255, 255)` |
| `format` | `"default"` or per-label | `".3f"` |
| `font_scale` | `"default"` or per-label | `0.5` |
| `thickness` | `"default"` or per-label | `1` |
| `outline_color` | `"default"` or per-label | `(0, 0, 0)` |
| `outline_thickness` | `"default"` or per-label | `2` |
| `line_height` | `"default"` or per-label | `18` |
| `as_bool` | per-label | `False` |

```python
"text": {
    "origin": (20, 30),
    "default": {
        "color": (255, 255, 255),
        "format": ".2f",
        "font_scale": 0.45,
        "outline_color": (0, 0, 0),
        "outline_thickness": 2,
        "line_height": 17,
    },
    "In centre": {"as_bool": True, "font_scale": 0.6},
}
```

---

## Dynamic (data-driven) styling

Any color or numeric style property can be driven by a column in
`Features.data` instead of a fixed value. This applies to colors, radii,
line widths, `fill_alpha`, font sizes — anything numeric. Dynamic specs
require `Features.animation_stream`; they are resolved once at construction
time, not during rendering.

There are two dynamic spec forms:

### Colormap (`cmap`)

Maps a numeric column through a matplotlib colormap.

```python
{"from": "column_name", "cmap": "viridis", "vmin": 0.0, "vmax": 1.0}
```

`vmin`/`vmax` are optional — if omitted they are inferred from the full
recording. Add `"nan_color": (B, G, R)` to control the colour for NaN frames.

### Value map (`map`)

Maps discrete values to specific property values via an explicit lookup dict.
**Always include a `"default"` key** to handle unexpected values and NaN
frames, especially when the source column is a nullable type such as
`pd.BooleanDtype`:

```python
{"from": "cluster_label", "map": {0: (0, 255, 0), 1: (0, 0, 255), "default": (128, 128, 128)}}
```

### Example — cmap on points and lines

```python
style = {
    "points": {
        "default": {
            "color": {"from": "speed_of_bodycentre_in_xy", "cmap": "turbo", "vmin": 0.0, "vmax": 0.5},
        },
    },
    "lines": {
        "default": {
            "color": {"from": "speed_of_bodycentre_in_xy", "cmap": "turbo", "vmin": 0.0, "vmax": 0.5},
        },
    },
}
```

### Example — static and dynamic overrides

```python
style = {
    "points": {
        "default": {"color": (170, 170, 170), "radius": 2},
        # static override: bodycentre always white and larger
        "bodycentre": {"color": (255, 255, 255), "radius": 6},
        # dynamic override: nose coloured by its own speed
        "nose": {
            "color": {"from": "speed_of_nose_in_xy", "cmap": "turbo", "vmin": 0.0, "vmax": 0.5},
        },
    },
    "lines": {
        # default: colour all lines by speed via cmap
        "default": {
            "color": {"from": "speed_of_bodycentre_in_xy", "cmap": "turbo", "vmin": 0.0, "vmax": 0.5},
        },
        # static override: spine line always drawn in white, thicker
        ("neck", "tailbase"): {"color": (255, 255, 255), "width": 3},
        # dynamic override: head line coloured by whether animal is in centre
        ("nose", "headcentre"): {
            "color": {
                "from": "in_centre",
                "map": {True: (0, 255, 0), False: (0, 100, 255), "default": (128, 128, 128)},
            },
        },
    },
    "text": {
        "origin": (20, 30),
        "default": {
            "color": (200, 200, 200),
            "format": ".2f",
            "font_scale": 0.45,
            "outline_color": (0, 0, 0),
            "outline_thickness": 2,
        },
        # static override: speed label rendered larger
        "Speed": {"font_scale": 0.6, "line_height": 22},
        # dynamic override: zone label changes colour with the data
        "In centre": {
            "as_bool": True,
            "color": {
                "from": "in_centre",
                "map": {True: (0, 255, 0), False: (0, 100, 255), "default": (200, 200, 200)},
            },
        },
    },
}
```
