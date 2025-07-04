# API documentation

## Tracking

represents a single animal's tracking data

**constructor:**
```python
Tracking(data: pd.DataFrame, meta: dict, handle: str)
```

**main methods:**
- `get_point_names()` — list of tracked point names
- `distance_between_points(point1, point2, dims=('x','y'))` — framewise distance between two points
- `rescale_by_known_distance(point1, point2, distance_in_metres, dims=('x','y'))` — rescale all dims by known distance
- `smooth_tracking(smoothing_dict)` — apply smoothing to tracking data
- `trim(startframe=None, endframe=None)` — trim the tracking data between frames
- `filter_likelihood(threshold)` — set positions with likelihood below threshold to NaN
- `time_as_expected(mintime, maxtime)` — check if total length is within expected time
- `generate_smoothdict(pointslists, windows, smoothtypes)` — create smoothing parameter dict
- `distance_between_points(point1, point2, dims=('x','y'))` — framewise distance between two points
- `get_point_names()` — list of tracked point names
- `rescale_by_known_distance(point1, point2, distance_in_metres, dims=('x','y'))` — rescale all dims by known distance
- `smooth_tracking(smoothing_params)` — apply smoothing to tracking data
- `from_dlc(filepath, handle, options)` — load from DLC csv (class method)
- `from_dlcma(filepath, handle, options)` — load from DLC multi-animal csv (class method)
- `from_yolo3r(filepath, handle, options)` — load from YOLO3R csv (class method)

**example:**
```python
from rrr_behaviour.util.tracking import Tracking
tracking = Tracking.from_dlc('file.csv', handle='animal1', options=opts)
print(tracking.get_point_names())
```

---

## TrackingCollection

a collection of Tracking objects, keyed by handle

**constructor:**
```python
TrackingCollection(tracking_dict: dict[str, Tracking])
```

**main methods:**
- batch access to all Tracking methods via `__getattr__`
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

**main methods:**
- batch access to all TrackingCollection methods via `__getattr__`
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

## TrackingMV

multi-view tracking object for stereo or multi-camera setups

**constructor:**
```python
TrackingMV(views: dict[str, Tracking], calibration: dict, handle: str)
```

**main methods:**
- `stereo_triangulate()` — triangulate two views to produce 3d tracking
- batch access to all Tracking methods via `__getattr__`
- `from_dlc(filepaths, handle, options, calibration)` — load from dict of view name -> DLC csv (class method)
- `from_dlcma(filepaths, handle, options, calibration)` — load from dict of view name -> DLC multi-animal csv (class method)
- `from_yolo3r(filepaths, handle, options, calibration)` — load from dict of view name -> YOLO3R csv (class method)

**example:**
```python
mv = TrackingMV.from_dlc({'left': 'left.csv', 'right': 'right.csv'}, handle='rec1', options=opts, calibration=calib)
tri = mv.stereo_triangulate()
```

---

## Features

generates features from a pre-processed Tracking object. Features are calculated quantities that have one value for each frame in the Tracking object.

**constructor:**
```python
Features(tracking: Tracking)
```

**main methods:**
- `distance_from(point1, point2, dims=('x','y'))` — distance from point1 to point2
- `within_distance(point1, point2, distance, dims=('x','y'))` — True if point1 is within distance of point2
- `get_point_median(point, dims=('x','y'))` — median coordinates of point
- `find_speed(point, dims=('x','y'))` — speed of point
- `above_speed(point, speed, dims=('x','y'))` — True if point is above speed
- `all_above_speed(points, speed, dims=('x','y'))` — True if all points are above speed
- `below_speed(point, speed, dims=('x','y'))` — True if point is below speed
- `all_below_speed(points, speed, dims=('x','y'))` — True if all points are below speed
- `distance_change(point, dims=('x','y'))` — unsigned distance moved by point per frame
- `find_angle(point1, point2)` — angle from point1 to point2
- `find_angle_deviation(basepoint, pointdirection1, pointdirection2)` — angle deviation between two directions
- `within_angle_deviation(basepoint, pointdirection1, pointdirection2, deviation)` — True if angle deviation is within threshold
- `acceleration(point, dims=('x','y'))` — acceleration of point in specified dims
- `store()

**example:**
```python
features = Features(tracking)
speed = features.find_speed('nose')
```

---

## FeaturesCollection

a collection of Features objects, keyed by handle

**constructor:**
```python
FeaturesCollection(features_dict: dict[str, Features])
```

**main methods:**
- batch access to all Features methods via `__getattr__`
- `from_tracking_collection(tracking_collection, feature_cls=Features)` — create from TrackingCollection (class method)
- `from_list(features_list)` — create from list of Features objects (class method)
- `cluster_embedding(embedding_dict, n_clusters, random_state=0)` — k-means clustering on embeddings
- `train_knn_regressor(embedding, target_embedding, n_neighbors=5, **kwargs)` — train kNN regressor
- `store(results, name, overwrite=False, meta=None)` — store results in all Features objects

**batch processing:**
any method of Features can be called on the collection and will be applied to all objects:
```python
fc = FeaturesCollection({...})
speeds = fc.find_speed('nose')  # dict: handle -> speed series
```
methods that need to be adjusted (e.g. store) are overloaded in the Collection class

---

## MultipleFeaturesCollection

a collection of FeaturesCollection objects, keyed by group/condition

**constructor:**
```python
MultipleFeaturesCollection(features_collections: dict[str, FeaturesCollection])
```

**main methods:**
- batch access to all FeaturesCollection methods via `__getattr__`
- `from_multiple_tracking_collection(multiple_tracking_collection)` — create from MultipleTrackingCollection (class method)
- `store(results, name, overwrite=False, meta=None)` — store results in all FeaturesCollection objects
- `cluster_embedding(embedding_dict, n_clusters, random_state=0)` — k-means clustering on all collections
- `knn_cross_predict_rms_matrix(source_embedding, target_embedding, n_neighbors=5, ...)` — cross-predict RMS error matrix
- `cross_predict_rms(source_embedding, target_embedding, n_neighbors=5, ...)` — cross-prediction RMS error (within/between collections)
- `plot_cross_predict_results(results, ...)` — plot cross-prediction results (static method)
- `dumbbell_plot_cross_predict(results, within_key, between_key, ...)` — plot dumbbell plot (static method)

**batch processing:**
any method of Features or FeaturesCollection can be called on the MultipleCollection and will be applied to all objects:
```python
mfc = MultipleFeaturesCollection({...})
speeds = mfc.find_speed('nose')  # dict: collection -> (dict: handle -> speed series)
```

---

## Summary

stores and computes summary statistics from features objects

**constructor:**
```python
Summary(trackingfeatures: Features)
```

**main methods:**
- `total_distance(point, startframe=None, endframe=None)` — total distance traveled by a point
- `time_true(series)` — time a condition is true
- `transition_matrix(column, all_states=None)` — transition matrix for a column
- `count_onset(series)` — number of times a boolean series changes from False to True (static method)
- `store(summarystat, name, overwrite=False, meta=None)` — store a summary statistic
- `make_bin(startframe, endframe)` — create a binned Summary object
- `make_bins(numbins)` — create a list of binned Summary objects
- `count_state_onsets(column)` — count number of times a state is entered
- `time_in_state(column)` — time spent in each state

**example:**
```python
summary = Summary(features)
total = summary.total_distance('nose')
```

---

## SummaryCollection

a collection of Summary objects, keyed by handle

**constructor:**
```python
SummaryCollection(summary_dict: dict[str, Summary])
```

**main methods:**
- batch access to all Summary methods via `__getattr__`
- `from_features_collection(features_collection, summary_cls=Summary)` — create from FeaturesCollection (class method)
- `from_list(summary_list)` — create from list of Summary objects (class method)

---

## MultipleSummaryCollection

a collection of SummaryCollection objects, keyed by group/condition

**constructor:**
```python
MultipleSummaryCollection(dict_of_summary_collections: dict[str, SummaryCollection])
```

**main methods:**
- batch access to all SummaryCollection methods via `__getattr__`
- `from_multiple_features_collection(multiple_features_collection)` — create from MultipleFeaturesCollection (class method)
- `bfa(column, all_states=None, numshuffles=1000)` — behaviour flow analysis
- `bfa_stats(bfa_results)` — compute stats for behaviour flow analysis (static method)

---

for more details, see the code and docstrings in each module. 