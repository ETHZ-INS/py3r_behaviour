# py3r.behaviour

This package is part of the `py3r` namespace and provides tools for behaviour analysis, including tracking, feature extraction, and summary statistics.

## Structure

- `src/py3r/behaviour/tracking/` — tracking data structures and utilities
- `src/py3r/behaviour/features/` — feature extraction from tracking data
- `src/py3r/behaviour/summary/` — summary statistics and aggregation
- `src/py3r/behaviour/util/` — utility functions (misc, bmicro, three_d, exceptions)

## Installation (User)

*note: requires pip and git; checks and upgrades version of pip if neccessary*

### Windows

```
powershell -Command "$repo='ETH-INS/py3r_behaviour'; $min_ver='21.3'; \
$ver = (python -c 'import pip; print(pip.__version__)'); \
if (([Version]$ver) -lt ([Version]$min_ver)) { python -m pip install --upgrade pip }; \
$latest = Invoke-RestMethod -Uri \"https://api.github.com/repos/$repo/releases/latest\"; \
$tag = $latest.tag_name; \
pip install --upgrade git+https://github.com/$repo.git@$tag"


```

### Linux/Mac OS
```bash
repo="ETH-INS/py3r_behaviour"
min_ver="21.3"
current_ver=$(python -c 'import pip; print(pip.__version__)')
if python -c "from packaging import version; import sys; sys.exit(0 if version.parse('$current_ver') >= version.parse('$min_ver') else 1)"; then
    echo "pip $current_ver OK"
else
    python -m pip install --upgrade pip
fi
latest_tag=$(curl -s https://api.github.com/repos/$repo/releases/latest | grep -Po '"tag_name": "\K.*?(?=")')
pip install --upgrade git+https://github.com/$repo.git@$latest_tag


```

## Installation (Developer only)

From the root of this repository:

```bash
pip install -e .
```

This will install the package in editable mode, making it available as `py3r.behaviour` in your Python environment.

## Packaging

This project uses a single `pyproject.toml` file for modern Python packaging with [setuptools](https://setuptools.pypa.io/en/latest/userguide/pyproject_config.html). No `setup.py` is required.

## Notes
- This package uses [PEP 420](https://www.python.org/dev/peps/pep-0420/) namespace packages (empty `__init__.py` files).
- All internal imports use the `py3r.behaviour` namespace.

## Overview

- Load, process, and analyze 2D and 3D tracking data from DeepLabCut, YOLO3R, and other sources.
- Batch processing via collection classes for individuals, groups, and conditions.
- Extensible to multi-view (stereo) and multi-animal tracking.

## Quickstart

```python
import py3r.behaviour as bv

# load a single tracking file
# (now also supports .interpolate, .add_usermeta, .save, etc.)
t = bv.Tracking.from_dlc('path/to/file.csv', handle='animal1', options=opts)
t.interpolate(method="linear")
t.add_usermeta({"experiment": "test"})
t.save('path/to/file.csv')

# load a folder of tracking files as a collection
tc = bv.TrackingCollection.from_dlc_folder('path/to/folder', options=opts)

# load multiple collections of tracking files
mtc = bv.MultipleTrackingCollection.from_dlc_folder('path/to/parent/folder', options=opts)


# multi batch process: get point names for all animals in all collections
point_names = mtc.get_point_names() # returns dict: collection -> (handle -> point names)

# generate features for all animals
fc = bv.FeaturesCollection.from_tracking_collection(tc)

# generate features for all collections
mfc = bv.MultipleFeaturesCollection.from_multiple_tracking_collection(mtc)

### Working with Collections and Batch Methods

You can call any feature/statistic method on a collection, and it will be applied to all contained objects, returning a batch result. Batch results support `.plot()` and `.store()` directly:

```python
results = mfc.speed('nose')
results.plot()
results.store()
```

### Slicing and Indexing

All X, XCollection, and MultipleXCollection classes support pandas-like slicing and flexible indexing:

```python
# Slice by frame range
subset = mfc.loc[100:200]
# Slice by integer position
subset = mfc.iloc[0:10]
# Dict-style access
animal = mfc['group1']['video1']
# Integer index
animal = mfc[0][2]
# Slice collections
subset = mfc[0:2]  # Returns a MultipleFeaturesCollection with the first two groups
```

### Iterating and Inspecting Collections

All collections support `.keys()`, `.values()`, and `.items()`:

```python
for group, collection in mfc.items():
    print(group, collection)
```

## Documentation

See [API.md](./API.md) for full class and method documentation, including batch processing and advanced usage.

## Batch Processing

All `XCollection` and `MultipleXCollection` classes support batch processing: any method of the base class can be called on the collection and will be applied to all contained objects, returning a dictionary of results.

## New API Features

### Batch Results: `.store()` and `.plot()`
You can now call `.store()` and `.plot()` directly on the result of a batch method:

```python
results = mfc.speed('nose')
results.plot()
results.store()
```

### Slicing with `.loc` and `.iloc`
All X, XCollection, and MultipleXCollection classes support pandas-like slicing, where X is Features or Tracking:

```python
# Slice by frame range
subset = mfc.loc[100:200]
# Slice by integer position
subset = mfc.iloc[0:10]
```

### Indexing and Slicing Collections
You can access objects in collections by key, integer, or slice:

```python
# Dict-style access
animal = mfc['group1']['video1']
# Integer index
animal = mfc[0][2]
# Slice
subset = mfc[0:2]  # Returns a MultipleFeaturesCollection with the first two groups
```

### Iterating and Inspecting
All collections support `.keys()`, `.values()`, and `.items()`:

```python
for group, collection in mfc.items():
    print(group, collection)
```

### FeaturesResult and SummaryResult usage
FeaturesResult acts as a pandas Series (no .value property, use Series methods directly)
SummaryResult has a .value property for the underlying value

---

For more details, see the API documentation. 