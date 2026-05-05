This guide explains the end‑to‑end workflow with collections as the primary entry point.
It focuses on "what" and "where" — not code.

*For detailed methods and syntax examples, see API*

*For a runnable code pipeline, see Examples*

---

## Core objects

The library is organised around three paired layers — a single-recording object and a
collection that batches over many recordings. In normal use you operate on collections
and let `.each` dispatch to individual recordings.

| Layer | Purpose | Single | Collection | Created via |
|---|---|---|---|---|
| **Tracking** | Raw keypoint coordinates — load, preprocess, QA | [`Tracking`][py3r.behaviour.tracking.tracking.Tracking] | [`TrackingCollection`][py3r.behaviour.tracking.tracking_collection.TrackingCollection] | loaders (`from_dlc_folder`, …) |
| **Features** | Per-frame derived signals — speeds, distances, boundaries, cluster labels | [`Features`][py3r.behaviour.features.features.Features] | [`FeaturesCollection`][py3r.behaviour.features.features_collection.FeaturesCollection] | `tc.to_features()` |
| **Summary** | Per-recording scalars and statistics — time in zone, total distance, transition matrices | [`Summary`][py3r.behaviour.summary.summary.Summary] | [`SummaryCollection`][py3r.behaviour.summary.summary_collection.SummaryCollection] | `fc.to_summary()` |

A fourth object, [`TrackingMV`][py3r.behaviour.tracking.tracking_mv.TrackingMV], wraps
multiple views plus calibration for 3D / stereo setups.

---

## Collection helpers

These behaviours are shared by all three collection types (`TrackingCollection`,
`FeaturesCollection`, `SummaryCollection`).

### `.each` batch dispatch

`.each` is the primary way to call a method on every element of a collection at once.
`coll.each.method(arg)` runs `element.method(arg)` for each element and collects the
results.

- **Inplace methods** (most `Tracking` preprocessing) return a
  `BatchResult` — a handle-keyed dict of the return values.
- **Non-inplace methods** (when `inplace=False`) return a new collection of the same
  type.
- A `BatchResult` passed as an argument to a subsequent `.each` call is automatically
  mapped per-handle rather than broadcast.

### Grouping

Collections can be grouped by any subset of tags. Groupings persist when you convert
between layers (e.g. a grouped `TrackingCollection` produces a grouped
`FeaturesCollection`).

- [`.groupby(tags)`][py3r.behaviour.tracking.tracking_collection.TrackingCollection.groupby]
  — returns a grouped view keyed by `(tag_value, …)` tuples.
- [`.flatten()`][py3r.behaviour.tracking.tracking_collection.TrackingCollection.groupby]
  — collapse back to a flat collection.
- [`.regroup()`][py3r.behaviour.tracking.tracking_collection.TrackingCollection.groupby]
  — recompute grouping after tags have changed.
- [`.get_group(key)`][py3r.behaviour.tracking.tracking_collection.TrackingCollection.groupby]
  — retrieve a sub-collection for one group.

### Indexing

- `coll["recording1"]` — by handle string
- `coll[0]` — by integer position
- `coll[0:3]` — by slice
- `.loc[]` / `.iloc[]` — label- or integer-based slicing applied in batch to each leaf

### Metadata

- `.tags_info()` — coverage and unique-value counts per tag across the collection.
- `.stored_info()` — overview of what is stored in each element
  (tracked points / feature columns / summary metrics depending on layer).

### Save and load

All objects and collections implement `.save(path)` / `.load(path)` that serialise
the full object tree — data, metadata, boundaries, groupings, and tags.
Parquet is the default format; CSV is available as a fallback.

### `copy()`

All objects and collections implement `.copy()` for deep copies — useful when
exploring parameters without altering the main pipeline object.

---

## Typical workflow

The typical flow is: **load → preprocess → features and clustering → summary stats → group comparisons and plots**.

### 1. Load a `TrackingCollection`

`Tracking` holds the raw frame-by-frame keypoint coordinates for one recording, together
with fps, rescaling metadata, and tags. A `TrackingCollection` is a keyed mapping of
these, loaded from a folder in a single call — each file becomes a `Tracking` keyed by
its filename stem.

- [`TrackingCollection.from_dlc_folder`][py3r.behaviour.tracking.tracking_collection.TrackingCollection.from_dlc_folder]
- [`TrackingCollection.from_yolo3r_folder`][py3r.behaviour.tracking.tracking_collection.TrackingCollection.from_yolo3r_folder]
- [`TrackingCollection.from_dlcma_folder`][py3r.behaviour.tracking.tracking_collection.TrackingCollection.from_dlcma_folder]

Or from an explicit `{handle: filepath}` mapping:
[`TrackingCollection.from_dlc`][py3r.behaviour.tracking.tracking_collection.TrackingCollection.from_dlc].

If files are grouped into a few separate directories, each directory can be loaded into a separate `TrackingCollection` using the relevant from_folder method and then combined using [`TrackingCollection.merge`][py3r.behaviour.tracking.tracking_collection.TrackingCollection.merge].

---

### 2. Add experimental tags

A tags CSV maps handles to experimental metadata (treatment, genotype, timepoint, …).
The `handle` column must match the filename stems. Tags are preserved through all
downstream conversions and drive grouping and CSV export.

- [`TrackingCollection.add_tags_from_csv`][py3r.behaviour.tracking.tracking_collection.TrackingCollection.add_tags_from_csv]
  — mutates all matching `Tracking` objects in‑place.

---

### 3. Batch preprocessing

Standard preprocessing order — all run via `.each`:

| Step | Method |
|---|---|
| Remove low-confidence detections | [`Tracking.filter_likelihood`][py3r.behaviour.tracking.tracking.Tracking.filter_likelihood] |
| Interpolate short gaps | [`Tracking.interpolate`][py3r.behaviour.tracking.tracking.Tracking.interpolate] |
| Smooth trajectories | [`Tracking.smooth_all`][py3r.behaviour.tracking.tracking.Tracking.smooth_all] |
| Rescale pixels → metres | [`Tracking.rescale_by_known_distance`][py3r.behaviour.tracking.tracking.Tracking.rescale_by_known_distance] |
| Trim start/end | [`Tracking.trim`][py3r.behaviour.tracking.tracking.Tracking.trim] |

Most preprocessing methods guard against re-application; use `inplace=False` on a
copy when iterating on parameters.

**QA checks:**

- [`Tracking.time_as_expected`][py3r.behaviour.tracking.tracking.Tracking.time_as_expected]
  — assert recording duration is within bounds.
- [`TrackingCollection.plot`][py3r.behaviour.tracking.tracking_collection.TrackingCollection.plot]
  — batch-save trajectory plots; call on a single `Tracking` for inline QC.

---

### 4. Create a `FeaturesCollection`

`Features` wraps a `Tracking` and provides methods that compute per-frame derived signals
— speeds, distances, boundary membership, cluster labels — stored as named columns in
`Features.data`. Convert the whole preprocessed collection:

```
fc = tc.to_features()
```

[`TrackingCollection.to_features`][py3r.behaviour.tracking.tracking_collection.TrackingCollection.to_features]
preserves grouped structure. The underlying classmethod is
[`FeaturesCollection.from_tracking_collection`][py3r.behaviour.features.features_collection.FeaturesCollection.from_tracking_collection].

---

### 5. Compute features

All feature methods return a `FeaturesResult`
(a `pd.Series` subclass). Call `.store()` on the result to persist it with an
auto-generated name in `Features.data`.  Batch via `.each`:
`fc.each.speed("bodycentre").store()`.

#### Distance and movement

- [`Features.speed`][py3r.behaviour.features.features.Features.speed]
- [`Features.distance_between`][py3r.behaviour.features.features.Features.distance_between]
- [`Features.distance_change`][py3r.behaviour.features.features.Features.distance_change]
- [`Features.acceleration`][py3r.behaviour.features.features.Features.acceleration]

#### Orientation

- [`Features.azimuth`][py3r.behaviour.features.features.Features.azimuth]
- [`Features.azimuth_deviation`][py3r.behaviour.features.features.Features.azimuth_deviation]
- [`Features.within_azimuth_deviation`][py3r.behaviour.features.features.Features.within_azimuth_deviation]

#### Boundaries and locations

Define named boundary regions first, then query against them:

- [`Features.define_static_boundary`][py3r.behaviour.features.features.Features.define_static_boundary]
  — convex hull from named keypoints; supports `anchor` and `scale_dim1`/`scale_dim2` for sizing.
- [`Features.define_dynamic_boundary`][py3r.behaviour.features.features.Features.define_dynamic_boundary]
  — convex hull that moves with the animal (uses per-frame keypoint positions).
- [`Features.import_static_boundary`][py3r.behaviour.features.features.Features.import_static_boundary]
  — define a boundary from explicit vertex coordinates.
- [`Features.within_boundary`][py3r.behaviour.features.features.Features.within_boundary]
  — Boolean per-frame membership; accepts a boundary object or stored name.
- [`Features.distance_to_boundary`][py3r.behaviour.features.features.Features.distance_to_boundary]
- [`Features.area_of_boundary`][py3r.behaviour.features.features.Features.area_of_boundary]
- [`Features.list_boundaries`][py3r.behaviour.features.features.Features.list_boundaries]
  — tabular summary of all named boundaries.

#### Speed thresholds

- [`Features.above_speed`][py3r.behaviour.features.features.Features.above_speed],
  [`Features.all_above_speed`][py3r.behaviour.features.features.Features.all_above_speed]
- [`Features.below_speed`][py3r.behaviour.features.features.Features.below_speed],
  [`Features.all_below_speed`][py3r.behaviour.features.features.Features.all_below_speed]

#### BatchResult composition

`BatchResult` objects returned by `.each` support element-wise operations, letting you
build compound features without loops:

- Logical: `in_centre & above_speed`, `~in_centre`, `a | b`
- Arithmetic: `in_centre.astype("Int64") * dist_change`
- Comparison: `(fc.each.speed("bodycentre") * 100) > 10.0`

The composed `BatchResult` can be stored directly with `.store("name")`.

#### State composition from booleans

[`Features.compose_state_from_booleans`][py3r.behaviour.features.features.Features.compose_state_from_booleans]
collapses multiple Boolean columns into a single categorical state column via a
`{label: series_or_column_name}` dict. Frames that are unassigned or overlap become
`NaN`.

---

### 6. Embeddings and clustering

Build time-shifted embeddings and run k-means to obtain per-frame behavioural cluster
labels, then store them as a feature column for downstream summary.

- [`FeaturesCollection.cluster_embedding_stream`][py3r.behaviour.features.features_collection.FeaturesCollection.cluster_embedding_stream]
  — **preferred** for multi-recording datasets: streaming MiniBatchKMeans that
  processes one `Features` at a time in multiple epochs, keeping memory low.
  Returns `(cluster_labels BatchResult, centroids DataFrame, scaling_factors dict)`.
- [`FeaturesCollection.cluster_embedding`][py3r.behaviour.features.features_collection.FeaturesCollection.cluster_embedding]
  — standard in-memory k-means; same return signature.
- [`Features.assign_clusters_by_centroids`][py3r.behaviour.features.features.Features.assign_clusters_by_centroids]
  — assign labels from precomputed centroids (e.g. from a reference dataset).
- [`FeaturesCollection.cluster_diagnostics`][py3r.behaviour.features.features_collection.FeaturesCollection.cluster_diagnostics]
  — per-cluster size statistics to assess quality.
- [`Features.embedding_df`][py3r.behaviour.features.features.Features.embedding_df]
  — build an embedding DataFrame directly (for inspection or custom clustering).

Use [`fc.stored_info()`][py3r.behaviour.features.features_collection.FeaturesCollection.stored_info]
to inspect all stored feature columns.

---

### 7. Create a `SummaryCollection`

`Summary` collapses the per-frame feature time-series for one recording into scalars and
Series — total distance, time in zone, state-transition matrices. `SummaryCollection`
batches these and provides group-level analyses and export.

```
sc = fc.to_summary()
```

[`FeaturesCollection.to_summary`][py3r.behaviour.features.features_collection.FeaturesCollection.to_summary]
preserves grouped structure. The underlying classmethod is
[`SummaryCollection.from_features_collection`][py3r.behaviour.summary.summary_collection.SummaryCollection.from_features_collection].

---

### 8. Generate summary statistics

All summary methods return a `SummaryResult`. Call `.store()` to persist with a name.
Batch via `sc.each.method(...).store(...)`.

#### Scalar metrics

- [`Summary.total_distance`][py3r.behaviour.summary.summary.Summary.total_distance]
- [`Summary.time_true`][py3r.behaviour.summary.summary.Summary.time_true],
  [`Summary.time_false`][py3r.behaviour.summary.summary.Summary.time_false]
- [`Summary.sum_column`][py3r.behaviour.summary.summary.Summary.sum_column]
- [`Summary.mean_column`][py3r.behaviour.summary.summary.Summary.mean_column],
  [`Summary.median_column`][py3r.behaviour.summary.summary.Summary.median_column],
  [`Summary.max_column`][py3r.behaviour.summary.summary.Summary.max_column],
  [`Summary.min_column`][py3r.behaviour.summary.summary.Summary.min_column]

#### State / sequence metrics (return Series)

- [`Summary.time_in_state`][py3r.behaviour.summary.summary.Summary.time_in_state]
  — seconds spent in each state.
- [`Summary.count_state_onsets`][py3r.behaviour.summary.summary.Summary.count_state_onsets]
  — number of entries into each state.
- [`Summary.transition_matrix`][py3r.behaviour.summary.summary.Summary.transition_matrix]
  — state-transition matrix.

Pass `all_states=` to any of these to guarantee a fixed state domain (including states
absent in a recording) — important for comparisons across recordings.

#### Per-state dispatch with `.by_state`

[`Summary.by_state(column)`][py3r.behaviour.summary.summary.Summary.by_state] returns a
proxy that re-runs an aggregate method separately for each value of a state column,
returning a Series indexed by state. Supports `mean_column`, `median_column`,
`max_column`, `min_column`.

#### Temporal binning

- [`Summary.make_bin`][py3r.behaviour.summary.summary.Summary.make_bin]
  — create a `Summary` from a frame-range slice.
- [`Summary.make_bins`][py3r.behaviour.summary.summary.Summary.make_bins]
  — divide a recording into `n` equal-length `Summary` objects.

---

### 9. Export per-recording results

- [`SummaryCollection.to_df`][py3r.behaviour.summary.summary_collection.SummaryCollection.to_df]
  — flatten stored scalar metrics + tag columns into one tidy DataFrame indexed by
  handle. Use `series="separate"` to also export Series metrics (time_in_state, …) as
  separate DataFrames.
- [`SummaryCollection.stored_info`][py3r.behaviour.summary.summary_collection.SummaryCollection.stored_info]
  — overview of stored metrics across the collection.

---

### 10. Group analyses and plotting

Group-level analyses require a grouped `SummaryCollection` — i.e. one produced from a
grouped `FeaturesCollection`, or grouped after the fact with `.groupby(tags)`.
Groupings control how data is split for comparisons, how plot colours are assigned, and
which groups BFA contrasts.

#### seaborn wrapper plots

`SummaryCollection` provides a family of `sns*` convenience methods that turn stored
metrics into publication-ready group comparison plots. Each accepts a metric name (or a
`BatchResult`) and handles tidy-data preparation, group colouring, and axis layout
automatically. All support `annotate=` for statistical annotations (see below).

- [`SummaryCollection.snsstrip`][py3r.behaviour.summary.summary_collection.SummaryCollection.snsstrip]
  — strip plot (jittered scatter, one point per recording).
- [`SummaryCollection.snsswarm`][py3r.behaviour.summary.summary_collection.SummaryCollection.snsswarm]
  — swarm plot (non-overlapping scatter).
- [`SummaryCollection.snsbox`][py3r.behaviour.summary.summary_collection.SummaryCollection.snsbox]
  — box plot.
- [`SummaryCollection.snsviolin`][py3r.behaviour.summary.summary_collection.SummaryCollection.snsviolin]
  — violin plot.
- [`SummaryCollection.snsbar`][py3r.behaviour.summary.summary_collection.SummaryCollection.snsbar]
  — bar plot (mean ± error).
- [`SummaryCollection.snspoint`][py3r.behaviour.summary.summary_collection.SummaryCollection.snspoint]
  — point plot.
- [`SummaryCollection.snssuperplot`][py3r.behaviour.summary.summary_collection.SummaryCollection.snssuperplot]
  — superplot (strip plot overlaid on bar plot).
- [`SummaryCollection.prepare_plot`][py3r.behaviour.summary.summary_collection.SummaryCollection.prepare_plot]
  — prepare the tidy DataFrame and seaborn kwargs without drawing, for full manual
  control.

#### Statistical annotations

All `sns*` methods accept an `annotate=` argument that passes through to
[`statannotations`](https://github.com/trevismd/statannotations). Pass `"help"` to
print available tests and options, or a dict with `pairs`, `test`, `correction`, and
other `statannotations.Annotator.configure` kwargs.

#### Chord diagrams (state transitions)

- [`SummaryCollection.plot_chord`][py3r.behaviour.summary.summary_collection.SummaryCollection.plot_chord]
  — batch chord diagrams across the collection (one per recording or per group).

#### Transition UMAP

- [`SummaryCollection.plot_transition_umap`][py3r.behaviour.summary.summary_collection.SummaryCollection.plot_transition_umap]
  — UMAP embedding of per-recording transition matrices, coloured by group; useful for
  visualising behavioural similarity structure across the dataset.

#### Behaviour Flow Analysis (BFA)

BFA tests whether state-transition patterns differ between groups using a
shuffle-based null distribution over Manhattan distances between group-level transition
matrices.

- [`SummaryCollection.bfa`][py3r.behaviour.summary.summary_collection.SummaryCollection.bfa]
  — compute observed and shuffled Manhattan distances for each group pair.
- [`SummaryCollection.bfa_stats`][py3r.behaviour.summary.summary_collection.SummaryCollection.bfa_stats]
  — derive percentile, z-score, and right-tail p from BFA results.
- [`SummaryCollection.plot_bfa_results`][py3r.behaviour.summary.summary_collection.SummaryCollection.plot_bfa_results]
  — histograms of BFA distance distributions with significance annotations per compared group pair.

---

## Working with individual recordings

All methods are available directly on the leaf objects when you don't need the
collection layer.

- `Tracking` → `Features`: [`Tracking.to_features()`][py3r.behaviour.tracking.tracking.Tracking.to_features]
- `Features` → `Summary`: [`Features.to_summary()`][py3r.behaviour.features.features.Features.to_summary]
