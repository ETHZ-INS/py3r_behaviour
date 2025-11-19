This guide explains the end‑to‑end workflow and links to the relevant APIs. It is intentionally minimal and focuses on “what” and “where,” not code. 

*For detailed methods and syntax examples, see API*

*For code pipeline examples, see Examples*

### Core objects
- [`Tracking`][py3r.behaviour.tracking.tracking.Tracking]: store and preprocess single‑view tracked keypoints with auto-generated metadata.
- [`TrackingMV`][py3r.behaviour.tracking.tracking_mv.TrackingMV]: like `Tracking` but with multi‑view plus calibration (for 3D).
- [`Features`][py3r.behaviour.features.features.Features]: (generated from a `Tracking` object) derive per‑frame signals from a single `Tracking` (e.g. speeds, distances, location booleans, behavioural clusters).
- [`Summary`][py3r.behaviour.summary.summary.Summary]: (generated from a `Features` object) derive scalar/statistical results from a single `Features` (e.g. av speeds, total distances, time in location, behavioural flow analysis stats).

### Collections
- [`TrackingCollection`][py3r.behaviour.tracking.tracking_collection.TrackingCollection]: batch load/process and group `Tracking` (or `TrackingMV`) objects.
- [`FeaturesCollection`][py3r.behaviour.features.features_collection.FeaturesCollection]: (generated from `TrackingCollection` object) batch process and group `Features` objects, and perform whole-dataset operations (e.g. behavioural clustering from time-series embeddings).
- [`SummaryCollection`][py3r.behaviour.summary.summary_collection.SummaryCollection]: (generated from `FeaturesCollection` object) mapping of handle → `Summary` with grouping/batch helpers.

### Collection helpers
- [`.groupby`][py3r.behaviour.tracking.tracking_collection.TrackingCollection.groupby] and [`.flatten`][py3r.behaviour.tracking.tracking_collection.TrackingCollection.groupby] allow dynamic reorganisation of collections based on arbitrary subsets of `tags` assigned to individual elements. Groupings persist when generating e.g. a `FeaturesCollection` from a `TrackingCollection`
- flexible collection indexing by handle (e.g. `coll['recording1']`), integer (e.g. `coll[0]`) and slice (e.g. `coll[0:2]`)

### General helpers
- All core objects and collections have [`.save`][py3r.behaviour.tracking.tracking.Tracking.save] and [`.load`][py3r.behaviour.tracking.tracking.Tracking.load] methods that preserve all data/metadata/groupings
- Objects and collections allow `.loc[]` and `.iloc[]` (batch) slicing and indexing of the DataFrames in all `Tracking`, `TrackingCollection`, `Features` and `FeaturesCollection` objects

### Typical workflow
1.  Load a dataset of single-view tracking files from DeepLabCut: [`TrackingCollection.from_dlc_folder`][py3r.behaviour.tracking.tracking_collection.TrackingCollection.from_dlc_folder]

1.  Add tags to each recording (e.g. 'treatment', 'genotype'): [`TrackingCollection.add_tags_from_csv`][py3r.behaviour.tracking.tracking_collection.TrackingCollection.add_tags_from_csv]

1. Group the collection by any subset of tags: [`TrackingCollection.groupby`][py3r.behaviour.tracking.tracking_collection.TrackingCollection.groupby] (grouping persists upon `FeaturesCollection` and `SummaryCollection` generation)

1. Perform various QA checks:
    1. ensure recording length is as expected: [`Tracking.time_as_expected`][py3r.behaviour.tracking.tracking.Tracking.time_as_expected]
    1. plot tracked point trajectories: [`TrackingCollection.plot`][py3r.behaviour.tracking.tracking_collection.TrackingCollection.plot]

1. Perform various batch pre-processing steps: 
    1. remove low-likelihood tracked points: [`Tracking.filter_likelihood`][py3r.behaviour.tracking.tracking.Tracking.filter_likelihood]
    1. smooth data: [`Tracking.smooth_all`][py3r.behaviour.tracking.tracking.Tracking.smooth_all]
    1. interpolate gaps: [`Tracking.interpolate`][py3r.behaviour.tracking.tracking.Tracking.interpolate]
    1. rescale pixels to metres: [`Tracking.rescale_by_known_distance`][py3r.behaviour.tracking.tracking.Tracking.rescale_by_known_distance]
    1. trim the start/end of the recording: [`trim`][py3r.behaviour.tracking.tracking.Tracking.trim]

1. Generate a `FeaturesCollection` object from the `TrackingCollection`: [`FeaturesCollection.from_tracking_collection`][py3r.behaviour.features.features_collection.FeaturesCollection.from_tracking_collection]

1. Calculate various features and [`store`][py3r.behaviour.features.features_collection.FeaturesCollection.store] them with auto-generated names and metadata:
    - Distance and movement:
        - [`Features.distance_between`][py3r.behaviour.features.features.Features.distance_between]
        - [`Features.distance_change`][py3r.behaviour.features.features.Features.distance_change]
        - [`Features.speed`][py3r.behaviour.features.features.Features.speed]
        - [`Features.acceleration`][py3r.behaviour.features.features.Features.acceleration]
    - Boundaries and locations:
        - Define static boundary from tracked points: [`Features.define_boundary`][py3r.behaviour.features.features.Features.define_boundary]
        - Static membership: [`Features.within_boundary_static`][py3r.behaviour.features.features.Features.within_boundary_static]
        - Dynamic membership: [`Features.within_boundary_dynamic`][py3r.behaviour.features.features.Features.within_boundary_dynamic]
        - Boundary area: [`Features.area_of_boundary`][py3r.behaviour.features.features.Features.area_of_boundary]
        - Distance to boundary (static/dynamic): [`Features.distance_to_boundary_static`][py3r.behaviour.features.features.Features.distance_to_boundary_static], [`Features.distance_to_boundary_dynamic`][py3r.behaviour.features.features.Features.distance_to_boundary_dynamic]
    - Orientation:
        - [`Features.azimuth`][py3r.behaviour.features.features.Features.azimuth]
        - [`Features.azimuth_deviation`][py3r.behaviour.features.features.Features.azimuth_deviation]
        - [`Features.within_azimuth_deviation`][py3r.behaviour.features.features.Features.within_azimuth_deviation]
    - Thresholds:
        - [`Features.above_speed`][py3r.behaviour.features.features.Features.above_speed], [`Features.all_above_speed`][py3r.behaviour.features.features.Features.all_above_speed]
        - [`Features.below_speed`][py3r.behaviour.features.features.Features.below_speed], [`Features.all_below_speed`][py3r.behaviour.features.features.Features.all_below_speed]
    - Embeddings and clustering:
        - Build time‑shifted embeddings: [`Features.embedding_df`][py3r.behaviour.features.features.Features.embedding_df]
        - Batch k‑means on embeddings: [`FeaturesCollection.cluster_embedding`][py3r.behaviour.features.features_collection.FeaturesCollection.cluster_embedding]
        - Assign to precomputed centroids: [`Features.assign_clusters_by_centroids`][py3r.behaviour.features.features.Features.assign_clusters_by_centroids]

1. Generate a `SummaryCollection` object from the `FeaturesCollection`: [`SummaryCollection.from_features_collection`][py3r.behaviour.summary.summary_collection.SummaryCollection.from_features_collection]

1. Generate summary statistics and export:
    - Per‑recording metrics (batched over the collection): [`Summary.time_true`][py3r.behaviour.summary.summary.Summary.time_true], [`Summary.time_false`][py3r.behaviour.summary.summary.Summary.time_false], [`Summary.total_distance`][py3r.behaviour.summary.summary.Summary.total_distance], [`Summary.transition_matrix`][py3r.behaviour.summary.summary.Summary.transition_matrix], [`Summary.count_state_onsets`][py3r.behaviour.summary.summary.Summary.count_state_onsets], [`Summary.time_in_state`][py3r.behaviour.summary.summary.Summary.time_in_state]
    - Collate scalar outputs into a tidy table: [`SummaryCollection.to_df`][py3r.behaviour.summary.summary_collection.SummaryCollection.to_df]
    - Behaviour Flow Analysis on grouped collections: [`SummaryCollection.bfa`][py3r.behaviour.summary.summary_collection.SummaryCollection.bfa] and post‑processing stats via [`SummaryCollection.bfa_stats`][py3r.behaviour.summary.summary_collection.SummaryCollection.bfa_stats]
