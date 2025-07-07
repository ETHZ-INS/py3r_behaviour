# API documentation

## Tracking

represents a single animal's tracking data

**constructor:**
```python
Tracking(data: pd.DataFrame, meta: dict, handle: str)
```

**factory methods**
- `from_dlc(filepath, handle, options)` — load from DLC csv (class method)
- `from_dlcma(filepath, handle, options)` — load from DLC multi-animal csv (class method)
- `from_yolo3r(filepath, handle, options)` — load from YOLO3R csv (class method)

**main methods:**
- `get_point_names()` — list of tracked point names
- `distance_between_points(point1, point2, dims=('x','y'))` — framewise distance between two points
- `rescale_by_known_distance(point1, point2, distance_in_metres, dims=('x','y'))` — rescale all dims by known distance
- `smooth(smoothing_dict)` — apply smoothing to tracking data
- `trim(startframe=None, endframe=None)` — trim the tracking data between frames
- `filter_likelihood(threshold)` — set positions with likelihood below threshold to NaN
- `time_as_expected(mintime, maxtime)` — check if total length is within expected time
- `generate_smoothdict(pointslists, windows, smoothtypes)` — create smoothing parameter dict
- `distance_between_points(point1, point2, dims=('x','y'))` — framewise distance between two points


**example:**
```python
from rrr_behaviour.util.tracking import Tracking
tracking = Tracking.from_dlc('file.csv', handle='animal1', options=opts)
print(tracking.get_point_names())
```

---

## TrackingMV

multi-view tracking object for stereo or multi-camera setups

**constructor:**
```python
TrackingMV(views: dict[str, Tracking], calibration: dict, handle: str)
```

**factory methods**
- `from_dlc(filepaths, handle, options, calibration)` — load from dict of view name -> DLC csv (class method)
- `from_dlcma(filepaths, handle, options, calibration)` — load from dict of view name -> DLC multi-animal csv (class method)
- `from_yolo3r(filepaths, handle, options, calibration)` — load from dict of view name -> YOLO3R csv (class method)

**main methods:**
- `stereo_triangulate()` — triangulate two views to produce 3d tracking
- batch access to all Tracking methods via `__getattr__`


**example:**
```python
mv = TrackingMV.from_dlc({'left': 'left.csv', 'right': 'right.csv'}, handle='rec1', options=opts, calibration=calib)
tri = mv.stereo_triangulate()
```

---

## TrackingCollection

a collection of Tracking or TrackingMV objects, keyed by handle

**constructor:**
```python
TrackingCollection(tracking_dict: dict[str, Tracking])
```

**factory methods:**
- `from_dlc(handles_and_filepaths, options, tracking_cls=Tracking)` — create from dict of DLC files (class method)
- `from_yolo3r(handles_and_filepaths, options, tracking_cls=Tracking)` — create from dict of YOLO3R files (class method)
- `from_dlcma(handles_and_filepaths, options, tracking_cls=Tracking)` — create from dict of DLC multi-animal files (class method)
- `from_list(tracking_list)` — create from list of Tracking objects (class method)
- `from_dogfeather(handles_and_filepaths, options, tracking_cls=Tracking)` — create from dict of dogfeather files (class method)
- `from_dlc_folder(folder_path, options, tracking_cls=Tracking)` — create from folder of DLC files (class method)
- `from_yolo3r_folder(folder_path, options, tracking_cls=Tracking)` — create from folder of YOLO3R files (class method)
- `from_dlcma_folder(folder_path, options, tracking_cls=Tracking)` — create from folder of DLC multi-animal files (class method)

**batch processing:**
any method of Tracking can be called on the collection and will be applied to all objects:
```python
collection = TrackingCollection({...})
result = collection.get_point_names()  # dict: handle -> point names
```

---

## MultipleTrackingCollection

a collection of TrackingCollection objects, keyed by group/condition

**constructor:**
```python
MultipleTrackingCollection(tracking_collections: dict[str, TrackingCollection])
```

**factory methods:**
- `from_dict(trackingcollections)` — create from dict of TrackingCollection objects (class method)
- `from_dlc_folder(parent_folder, options, tracking_cls=Tracking)` — create from folder of subfolders of DLC files (class method)
- `from_yolo3r_folder(parent_folder, options, tracking_cls=Tracking)` — create from folder of subfolders of YOLO3R files (class method)
- `from_dlcma_folder(parent_folder, options, tracking_cls=Tracking)` — create from folder of subfolders of DLC multi-animal files (class method)

**batch processing:**
any method of TrackingCollection can be called on the multiple collection and will be applied to all collections:
```python
multi = MultipleTrackingCollection({...})
result = multi.get_point_names()  # dict: group -> (dict: handle -> point names)
```

---

## Features

generates features from a pre-processed Tracking object. Features are calculated quantities that have one value for each frame in the Tracking object.

**constructor:**
```python
Features(tracking: Tracking)
```

**main methods:**
- `distance_from(point1, point2, dims=('x','y'))` — returns a FeaturesResult: distance from point1 to point2
- `within_distance(point1, point2, distance, dims=('x','y'))` — returns a FeaturesResult: True if point1 is within distance of point2
- `get_point_median(point, dims=('x','y'))` — median coordinates of point (tuple)
- `speed(point, dims=('x','y'))` — returns a FeaturesResult: speed of point
- `above_speed(point, speed, dims=('x','y'))` — returns a FeaturesResult: True if point is above speed
- `all_above_speed(points, speed, dims=('x','y'))` — returns a FeaturesResult: True if all points are above speed
- `below_speed(point, speed, dims=('x','y'))` — returns a FeaturesResult: True if point is below speed
- `all_below_speed(points, speed, dims=('x','y'))` — returns a FeaturesResult: True if all points are below speed
- `distance_change(point, dims=('x','y'))` — returns a FeaturesResult: unsigned distance moved by point per frame
- `azimuth(point1, point2)` — returns a FeaturesResult: azimuthal angle from point1 to point2
- `azimuth_deviation(basepoint, pointdirection1, pointdirection2)` — returns a FeaturesResult: deviation between two azimuths
- `within_azimuth_deviation(basepoint, pointdirection1, pointdirection2, deviation)` — returns a FeaturesResult: True if azimuth deviation is within threshold
- `acceleration(point, dims=('x','y'))` — returns a FeaturesResult: acceleration of point in specified dims
- `store(series, name, overwrite=False, meta=None)` — store a calculated feature manually
- `embedding_df(embedding: dict[str, list[int]])` — generate a time series embedding DataFrame
- `cluster_embedding(embedding: dict[str, list[int]], n_clusters: int)` — cluster the embedding using k-means

**Note:** Most feature calculation methods return a `FeaturesResult` object, which behaves like a pandas Series and supports `.store()` to save the result in the parent Features object.

**example:**
```python
features = Features(tracking)
speed = features.speed('nose')  # FeaturesResult
speed.mean()
speed.store()
```

---

## FeaturesResult

A wrapper class for the result of a feature calculation from a Features object. Subclasses `pd.Series` and adds metadata and a convenient `.store()` method.

**Main attributes and methods:**
- `.value` — the underlying Series (in practice, you can use all Series methods directly on the FeaturesResult)
- `.store(name=None, meta=None, overwrite=False)` — store the result in the parent Features object, with optional custom name and metadata
- Metadata fields: `_features_obj`, `_column_name`, `_params` (internal)

**Usage:**
```python
result = features.speed('nose')  # FeaturesResult
result.mean()  # can use as a Series
result.store()  # stores in features.data with an auto-generated name
result.store(name='custom_speed')  # stores with a custom name
```

When using batch methods on a FeaturesCollection or MultipleFeaturesCollection, you get dicts of FeaturesResult objects, which can be batch-stored using the collection's `.store()` method.

---

## FeaturesCollection

a collection of Features objects, keyed by handle

**constructor:**
```python
FeaturesCollection(features_dict: dict[str, Features])
```
**factory methods**
- `from_tracking_collection(tracking_collection, feature_cls=Features)` — create from TrackingCollection (class method)
- `from_list(features_list)` — create from list of Features objects (class method)

**main methods:**
- `cluster_embedding(embedding_dict, n_clusters, random_state=0)` — k-means clustering on embeddings
- `train_knn_regressor(embedding, target_embedding, n_neighbors=5, **kwargs)` — train kNN regressor
- `store(results_dict: dict[str, FeaturesResult], name=None, meta=None, overwrite=False)` — store all batch FeaturesResult objects in the collection (see below)

**batch processing:**
any method of Features can be called on the collection and will be applied to all objects:
```python
fc = FeaturesCollection({...})
speeds = fc.speed('nose')  # dict: handle -> FeaturesResult
fc.store(speeds)  # stores all results in the underlying Features objects
```

---

## MultipleFeaturesCollection

a collection of FeaturesCollection objects, keyed by group/condition

**constructor:**
```python
MultipleFeaturesCollection(features_collections: dict[str, FeaturesCollection])
```

**factory methods**
- `from_multiple_tracking_collection(multiple_tracking_collection)` — create from MultipleTrackingCollection (class method)

**main methods:**
- `store(results_dict: dict[str, dict[str, FeaturesResult]], name=None, meta=None, overwrite=False)` — store all batch FeaturesResult objects in all collections (see below)
- `cluster_embedding(embedding_dict, n_clusters, random_state=0)` — k-means clustering on all collections
- `knn_cross_predict_rms_matrix(source_embedding, target_embedding, n_neighbors=5, ...)` — cross-predict RMS error matrix
- `cross_predict_rms(source_embedding, target_embedding, n_neighbors=5, ...)` — cross-prediction RMS error (within/between collections)
- `plot_cross_predict_results(results, ...)` — plot cross-prediction results (static method)
- `dumbbell_plot_cross_predict(results, within_key, between_key, ...)` — plot dumbbell plot (static method)

**batch processing:**
any method of Features or FeaturesCollection can be called on the MultipleCollection and will be applied to all objects:
```python
mfc = MultipleFeaturesCollection({...})
speeds = mfc.speed('nose')  # dict: collection -> (dict: handle -> FeaturesResult)
mfc.store(speeds)  # stores all results in all underlying Features objects
```

---

## Summary

stores and computes summary statistics from features objects

**constructor:**
```python
Summary(trackingfeatures: Features)
```

**main methods:**
- `total_distance(point, startframe=None, endframe=None)` — returns a SummaryResult: total distance traveled by a point
- `time_true(column)` — returns a SummaryResult: time in seconds that a boolean condition is true in the given column
- `transition_matrix(column, all_states=None)` — returns a SummaryResult: transition matrix for a column
- `count_onset(column)` — returns a SummaryResult: number of times a boolean column changes from False to True
- `count_state_onsets(column)` — returns a SummaryResult: number of times a state is entered in a given column
- `time_in_state(column)` — returns a SummaryResult: time spent in each state in a given column
- `store(summarystat, name, overwrite=False, meta=None)` — store a summary statistic manually
- `make_bin(startframe, endframe)` — create a binned Summary object
- `make_bins(numbins)` — create a list of binned Summary objects

**Note:** Most summary calculation methods return a `SummaryResult` object, which behaves like the underlying value (scalar, Series, or DataFrame) and supports `.store()` to save the result in the parent Summary object.

**example:**
```python
summary = Summary(features)
total = summary.total_distance('nose')  # SummaryResult
float(total)  # or total.value
summary.time_true('is_running').store()
```

---

## SummaryResult

A wrapper class for the result of a summary calculation from a Summary object. Forwards attribute access to the underlying value (scalar, Series, or DataFrame) and adds a convenient `.store()` method.

**Main attributes and methods:**
- `.value` — the underlying value (scalar, Series, or DataFrame)
- `.store(name=None, meta=None, overwrite=False)` — store the result in the parent Summary object, with optional custom name and metadata
- Metadata fields: `_summary_obj`, `_func_name`, `_params` (internal)

**Usage:**
```python
result = summary.time_true('is_running')  # SummaryResult
float(result)  # or result.value, or use as a scalar/Series/DataFrame
result.store()  # stores in summary.data with an auto-generated name
result.store(name='custom_time_true')  # stores with a custom name
```

When using batch methods on a SummaryCollection or MultipleSummaryCollection, you get dicts of SummaryResult objects, which can be batch-stored using the collection's `.store()` method.

---

## SummaryCollection

a collection of Summary objects, keyed by handle

**constructor:**
```python
SummaryCollection(summary_dict: dict[str, Summary])
```

**factory methods**
- `from_features_collection(features_collection, summary_cls=Summary)` — create from FeaturesCollection (class method)
- `from_list(summary_list)` — create from list of Summary objects (class method)

**main methods:**
- `store(results_dict: dict[str, SummaryResult], name=None, meta=None, overwrite=False)` — store all batch SummaryResult objects in the collection (see below)

**batch processing:**
any method of Summary can be called on the collection and will be applied to all objects:
```python
sc = SummaryCollection({...})
times = sc.time_true('is_running')  # dict: handle -> SummaryResult
sc.store(times)  # stores all results in the underlying Summary objects
```

---

## MultipleSummaryCollection

a collection of SummaryCollection objects, keyed by group/condition

**constructor:**
```python
MultipleSummaryCollection(dict_of_summary_collections: dict[str, SummaryCollection])
```

**factory methods**
- `from_multiple_features_collection(multiple_features_collection)` — create from MultipleFeaturesCollection (class method)

**main methods:**
- `bfa(column, all_states=None, numshuffles=1000)` — behaviour flow analysis
- `bfa_stats(bfa_results)` — compute stats for behaviour flow analysis (static method)
- `store(results_dict: dict[str, dict[str, SummaryResult]], name=None, meta=None, overwrite=False)` — store all batch SummaryResult objects in all collections (see below)

**batch processing:**
any method of Summary or SummaryCollection can be called on the MultipleCollection and will be applied to all objects:
```python
msc = MultipleSummaryCollection({...})
times = msc.time_true('is_running')  # dict: group -> (dict: handle -> SummaryResult)
msc.store(times)  # stores all results in all underlying Summary objects
```

---

for more details, see the code and docstrings in each module. 