This guide explains the end‑to‑end workflow and links to the relevant APIs. It is intentionally minimal and focuses on “what” and “where,” not code.

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
- [`.save`][py3r.behaviour.tracking.tracking_collection.TrackingCollection.save] and [`.load`][py3r.behaviour.tracking.tracking_collection.TrackingCollection.load] method exist for all objects and preserve all data/metadata/groupings
- `.loc`, `.iloc` allow (batch) slicing and indexing of the DataFrames in all `Tracking`, `TrackingCollection`, `Features` and `FeaturesCollection` objects

### Typical workflow
1.  Load a dataset of single-view tracking files from DeepLabCut: [`TrackingCollection.from_dlc_folder`][py3r.behaviour.tracking.tracking_collection.TrackingCollection.from_dlc_folder]

1.  Add tags to each recording (e.g. 'treatment', 'genotype'): [`TrackingCollection.add_tags_from_csv`][py3r.behaviour.tracking.tracking_collection.TrackingCollection.add_tags_from_csv]

1. Group the collection by any subset of tags: [`TrackingCollection.groupby`][py3r.behaviour.tracking.tracking_collection.TrackingCollection.groupby] (grouping persists upon `FeaturesCollection` and `SummaryCollection` generation)

1. Perform various QA checks:
    1. ensure recording length is as expected: [`TrackingCollection.time_as_expected`][py3r.behaviour.tracking.tracking_collection.TrackingCollection.time_as_expected]
    1. plot tracked point trajectories: [`TrackingCollection.plot`][py3r.behaviour.tracking.tracking_collection.TrackingCollection.plot]

1. Perform various batch pre-processing steps: 
    1. remove low-likelihood tracked points: [`TrackingCollection.filter_likelihood`][py3r.behaviour.tracking.tracking_collection.TrackingCollection.filter_likelihood]
    1. smooth data: [`TrackingCollection.smooth_all`][py3r.behaviour.tracking.tracking_collection.TrackingCollection.smooth_all]
    1. interpolate gaps: [`TrackingCollection.interpolate`][py3r.behaviour.tracking.tracking_collection.TrackingCollection.interpolate]
    1. rescale pixels to metres: [`TrackingCollection.rescale_by_known_distance`][py3r.behaviour.tracking.tracking_collection.TrackingCollection.rescale_by_known_distance]
    1. trim the start/end of the recording: [`TrackingCollection.trim`][py3r.behaviour.tracking.tracking_collection.TrackingCollection.trim]

1. Generate a `FeaturesCollection` object from the `TrackingCollection`: [`FeaturesCollection.from_tracking_collection`][py3r.behaviour.features.features_collection.FeaturesCollection.from_tracking_collection]

1. Calculate various features and [`store`][py3r.behaviour.features.features_collection.FeaturesCollection.store] them with auto-generated names and metadata:
    1. 




`
   - Single‑view from folder of DLC CSVs: 
   - Single file: [`Tracking.from_dlc`][py3r.behaviour.tracking.tracking.Tracking.from_dlc]
   - Multi‑view from folder of recordings (each with views + calibration.json): [`TrackingCollection.from_dlc_folder`][py3r.behaviour.tracking.tracking_collection.TrackingCollection.from_dlc_folder] with `tracking_cls=`[`TrackingMV`][py3r.behaviour.tracking.tracking_mv.TrackingMV], or per‑recording [`TrackingMV.from_dlc`][py3r.behaviour.tracking.tracking_mv.TrackingMV.from_dlc]

2) Inspect and slice
   - Access a frame‑indexed DataFrame via `.data`. Use [`Tracking.loc`][py3r.behaviour.tracking.tracking.Tracking.loc] / [`Tracking.iloc`][py3r.behaviour.tracking.tracking.Tracking.iloc] on the object.

3) Tag and group
   - Attach arbitrary tags for later grouping: [`Tracking.add_tag`][py3r.behaviour.tracking.tracking.Tracking.add_tag]
   - Group any collection by tags: [`groupby`][py3r.behaviour.util.base_collection.BaseCollection.groupby] (inherited)
   - Flatten a grouped view: [`flatten`][py3r.behaviour.util.base_collection.BaseCollection.flatten]

4) Generate features
   - Per recording: build [`Features`][py3r.behaviour.features.features.Features] and compute methods like [`distance_between`][py3r.behaviour.features.features.Features.distance_between], [`speed`][py3r.behaviour.features.features.Features.speed], etc.; persist with [`store`][py3r.behaviour.features.features.Features.store].
   - Over a collection: [`FeaturesCollection.from_tracking_collection`][py3r.behaviour.features.features_collection.FeaturesCollection.from_tracking_collection] and batch [`store`][py3r.behaviour.features.features_collection.FeaturesCollection.store].

5) Summarise
   - Per recording: [`Summary`][py3r.behaviour.summary.summary.Summary] with metrics like [`time_true`][py3r.behaviour.summary.summary.Summary.time_true], [`total_distance`][py3r.behaviour.summary.summary.Summary.total_distance].
   - Over a collection: [`SummaryCollection.from_features_collection`][py3r.behaviour.summary.summary_collection.SummaryCollection.from_features_collection] and collate with [`to_df`][py3r.behaviour.summary.summary_collection.SummaryCollection.to_df].

6) Save and load (round‑trip)
   - Each core object supports `save(dirpath, data_format=...)` and `load(dirpath)`:
     - [`Tracking.save`][py3r.behaviour.tracking.tracking.Tracking.save] / [`Tracking.load`][py3r.behaviour.tracking.tracking.Tracking.load]
     - [`Features.save`][py3r.behaviour.features.features.Features.save] / [`Features.load`][py3r.behaviour.features.features.Features.load]
     - [`Summary.save`][py3r.behaviour.summary.summary.Summary.save] / [`Summary.load`][py3r.behaviour.summary.summary.Summary.load]
   - Collections can be persisted too: see [`save`][py3r.behaviour.util.base_collection.BaseCollection.save] and [`load`][py3r.behaviour.util.base_collection.BaseCollection.load].

7) Multi‑view notes
   - Prefer folder‑based loaders:
     - Per recording: [`TrackingMV.from_dlc`][py3r.behaviour.tracking.tracking_mv.TrackingMV.from_dlc]
     - Batch: [`TrackingCollection.from_dlc_folder`][py3r.behaviour.tracking.tracking_collection.TrackingCollection.from_dlc_folder] with `tracking_cls=TrackingMV`

### Next steps
- Install and environment: see Install
- Browse API pages for doctestable examples:
  - Tracking, TrackingMV, TrackingCollection
  - Features, FeaturesCollection
  - Summary, SummaryCollection