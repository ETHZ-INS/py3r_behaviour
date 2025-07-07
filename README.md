# py3r.behaviour

This package is part of the `py3r` namespace and provides tools for behaviour analysis, including tracking, feature extraction, and summary statistics.

## Structure

- `src/py3r/behaviour/tracking.py` — tracking data structures and utilities
- `src/py3r/behaviour/features.py` — feature extraction from tracking data
- `src/py3r/behaviour/summary.py` — summary statistics and aggregation
- `src/py3r/behaviour/util/` — utility functions (misc, bmicro, three_d, exceptions)

## Installation (Development)

From the root of this repository:

```bash
pip install -e .
```

This will install the package in editable mode, making it available as `py3r.behaviour` in your Python environment.

## Packaging

This project uses a single `pyproject.toml` file for modern Python packaging with [setuptools](https://setuptools.pypa.io/en/latest/userguide/pyproject_config.html). No `setup.py` is required.

## Usage

```python
from py3r.behaviour.tracking import Tracking
from py3r.behaviour.features import Features
from py3r.behaviour.summary import Summary
```

## Notes
- This package uses [PEP 420](https://www.python.org/dev/peps/pep-0420/) namespace packages (empty `__init__.py` files).
- All internal imports use the `py3r.behaviour` namespace.

## Overview

- Load, process, and analyze 2D and 3D tracking data from DeepLabCut, YOLO3R, and other sources.
- Batch processing via collection classes for individuals, groups, and conditions.
- Extensible to multi-view (stereo) and multi-animal tracking.

## Quickstart

```python
from py3r.behaviour.tracking import Tracking, TrackingCollection, MultipleTrackingCollection, TrackingMV
from py3r.behaviour.features import Features, FeaturesCollection, MultipleFeaturesCollection
from py3r.behaviour.summary import Summary, SummaryCollection, MultipleSummaryCollection

# load a single tracking file
t = Tracking.from_dlc('path/to/file.csv', handle='animal1', options=opts)

# load a folder of tracking files as a collection
tc = TrackingCollection.from_dlc_folder('path/to/folder', options=opts)

# load multiple collections of tracking files
mtc = MultipleTrackingCollection.from_dlc_folder('path/to/parent/folder', options=opts)

# batch process: get point names for all animals
point_names = collection.get_point_names()  # returns dict: handle -> point names

# multi batch process: get point names for all animals in all collections
point_names = multicollection.get_point_names() # returns dict: collection -> (handle -> point names)

# generate features for all animals
fc = FeaturesCollection.from_tracking_collection(tc)

# generate features for all collections
mfc = MultipleFeaturesCollection.from_multiple_tracking_collection(mtc)
```

## Documentation

See [API.md](./API.md) for full class and method documentation, including batch processing and advanced usage.

## Batch Processing

All `XCollection` and `MultipleXCollection` classes support batch processing: any method of the base class can be called on the collection and will be applied to all contained objects, returning a dictionary of results.

---

For more details, see the API documentation. 