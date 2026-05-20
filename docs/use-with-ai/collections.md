# Collections — AI context page

*This page is written for AI assistants. See [Use with AI](index.md) for the gateway prompt.*

---

## What collections are

Collections are the **primary way to use py3r.behaviour**. The entire pipeline is
designed to be run through collections, with individual-recording operations
dispatched via `.each`. Working with a single `Tracking` or `Features` object
directly is for exploration only — a real dataset always goes through a collection.

Every pipeline layer has a collection variant:

| Single (exploration) | Collection (normal use) |
|---|---|
| `Tracking` | `TrackingCollection` |
| `Features` | `FeaturesCollection` |
| `Summary` | `SummaryCollection` |

A collection is a dict-like container (handles are keys, objects are values).
The `.each` property dispatches any method call across all recordings at once.

---

## Loading a collection

Choose the loader that matches the tracking format. DLC is DeepLabCut;
YOLO3R is the 3R Hub internal tracking format.

```python
# All CSVs in a folder — use the format-specific convenience method
tc = TrackingCollection.from_dlc_folder('data/', fps=30)
tc = TrackingCollection.from_dlcma_folder('data/', fps=30)   # DLC multi-animal
tc = TrackingCollection.from_yolo3r_folder('data/', fps=30)

# Explicit handle → filepath mapping (when filenames aren't your handles)
tc = TrackingCollection.from_dlc(
    {'subj01': 'data/subj01.csv', 'subj02': 'data/subj02.csv'},
    fps=30
)

# From a saved directory
tc = TrackingCollection.load('saved/tracking/')
```

Do not use `from_folder()` directly — it is an internal method. Always use
the format-specific loaders above.

---

## Building a collection from disparate sources — `merge()`

When data is spread across multiple folders, cohorts, or time-points, load each
batch separately and merge them into one collection:

```python
batch1 = TrackingCollection.from_dlc_folder('data/cohort1/', fps=30)
batch2 = TrackingCollection.from_dlc_folder('data/cohort2/', fps=30)
batch3 = TrackingCollection.load('saved/cohort3/')

tc = TrackingCollection.merge([batch1, batch2, batch3])
```

`merge()` flattens grouped inputs automatically, so it accepts collections in
any state. All handles must be unique across the inputs — a `ValueError` is
raised if there are collisions. A warning is issued if the tag schemas differ
across inputs (e.g. one batch has been tagged but another has not).

To get an independent copy of all leaves rather than shared references:

```python
tc = TrackingCollection.merge([batch1, batch2], copy=True)
```

To build a collection from individually-loaded objects:

```python
t1 = Tracking.from_dlc('subj01.csv', handle='subj01', fps=30)
t2 = Tracking.from_dlc('subj02.csv', handle='subj02', fps=30)
tc = TrackingCollection.from_list([t1, t2])
```

**Do not use direct assignment** (`coll['new'] = obj`) — it is deprecated.
Use `merge()` or `from_list()` instead.

---

## The `.each` pattern

`.each.<method>(...)` broadcasts any method call across all objects in the
collection. This is the correct way to batch — do NOT write a manual for-loop.

```python
tc.each.filter_likelihood(0.6)
tc.each.smooth_all(window=11, method='savgol')

# Convert the whole collection to Features
fc = tc.to_features()
```

**Return value:** if all leaf results are the collection's element type (e.g.
every result is a `Tracking`), `.each` upcasts to a collection. Otherwise it
returns a `BatchResult` — a dict subclass of `handle → result`.

**Chaining:** `BatchResult` forwards any method call to its leaves, so results
can be chained directly:

```python
# Compute a feature on all recordings and store it in one chain
fc.each.distance_between('nose', 'tail').store(name='nose_tail_dist')
fc.each.speed('nose').store(name='nose_speed')

# Summary statistics work the same way
sc.each.time_true('in_arena').store(name='time_in_arena')
```

**Per-handle arguments:** pass a `BatchResult` as an argument and `.each` will
map each handle's value to the matching recording. Only `BatchResult` objects
get this mapped treatment — plain dicts are always scalar-broadcast to all
leaves.

```python
# Different calibration value per animal — stored in a BatchResult
distances = BatchResult({'subj01': ..., 'subj02': ...}, tc)
tc.each.rescale_by_known_distance('corner1', 'corner2', distances)
```

Use `.each.forcebatch.<method>(...)` when you need to guarantee a `BatchResult`
even if the result type would normally be upcast to a collection.

---

## Tagging recordings

Tags are key-value string pairs attached to each `Tracking` object. They are
the basis for grouping and are the only way to label recordings.

**The normal way** is a CSV file with a `handle` column and one column per tag.
The handle values must match the recording handles in the collection exactly.

```
handle,genotype,sex
subj01,WT,M
subj02,KO,F
```

```python
tc.add_tags_from_csv('metadata.csv')
```

This is the right approach for any real dataset. The CSV can be prepared in a
spreadsheet and lives alongside the data files.

For one-off or programmatic tagging, use `add_tag()` on individual recordings:

```python
tc['subj01'].add_tag('genotype', 'WT')
```

Inspect tag coverage after tagging:

```python
tc.tags_info()                           # coverage summary
tc.tags_info(include_value_counts=True)  # also shows value → count per tag
```

---

## Grouping

```python
grouped = tc.groupby('genotype')

grouped.is_grouped      # True
grouped.group_keys      # [('WT',), ('KO',)]   — always tuples
grouped.groupby_tags    # ['genotype']
```

**Group keys are always tuples**, even for a single tag. Access a group with
its full tuple key:

```python
grouped[('WT',)]          # TrackingCollection of WT animals
grouped.get_group(('WT',))  # equivalent, more explicit
```

Group by multiple tags — keys become tuples of all tag values in order:

```python
grouped2 = tc.groupby(['genotype', 'sex'])
grouped2.group_keys    # [('WT', 'M'), ('WT', 'F'), ('KO', 'M'), ...]
grouped2[('WT', 'M')]  # sub-collection of WT males
```

`.each` works on grouped collections and returns a nested `BatchResult`:

```python
grouped.each.filter_likelihood(0.6)
```

If you change tags after grouping, use `.regroup()` to recompute the groups
without flattening and re-calling `.groupby()`:

```python
tc['subj02'].add_tag('genotype', 'KO', overwrite=True)
grouped = grouped.regroup()
```

Flatten back to a flat collection at any time:

```python
flat = grouped.flatten()
flat.is_grouped   # False
```

Grouping does not copy data — it is a view over the same leaves.

---

## Building a full batch pipeline

```python
# Load from multiple sources and merge
batch1 = TrackingCollection.from_dlc_folder('data/cohort1/', fps=30)
batch2 = TrackingCollection.from_dlc_folder('data/cohort2/', fps=30)
tc = TrackingCollection.merge([batch1, batch2])

# Tag
for handle, t in tc.items():
    t.add_tag('group', handle.split('_')[0])   # e.g. 'WT_01' → 'WT'

# Preprocess (see Tracking API reference for parameter details)
tc.each.filter_likelihood(...)
tc.each.interpolate(...)
tc.each.smooth_all(...)
tc.each.rescale_by_known_distance(...)

# Convert to Features
fc = tc.to_features()

# Compute features — chain .store() directly on the BatchResult
fc.each.distance_between('nose', 'tail').store(name='nose_tail_dist')
fc.each.speed('nose').store(name='nose_speed')

# Define a static boundary on all recordings and query membership
fc.each.define_static_boundary(
    ['corner1', 'corner2', 'corner3', 'corner4'], name='arena'
)
fc.each.within_boundary('nose', 'arena').store(name='in_arena')

# Convert to Summary and compute statistics
sc = fc.to_summary()
sc.each.time_true('in_arena').store(name='time_in_arena')
```

---

## Saving and loading collections

```python
tc.save('saved/tracking/')
tc2 = TrackingCollection.load('saved/tracking/')
```

The saved directory contains a `manifest.json` plus per-recording parquet files.
The same pattern applies to `FeaturesCollection` and `SummaryCollection`.

---

## Accessing individual objects

```python
tc['subj01']          # by handle (string key)
tc[0]                 # by integer index
tc[0:5]               # by slice — returns a new collection
list(tc.keys())       # all handles
list(tc.values())     # all objects
list(tc.items())      # (handle, object) pairs
len(tc)               # number of recordings
```

---

## Key gotchas

- **Do not loop manually.** `for t in tc.values(): t.filter_likelihood(...)` works
  but is an antipattern. Use `.each`.
- **Group keys are tuples.** `grouped[('WT',)]` not `grouped['WT']`.
- **Only `BatchResult` gets per-handle dispatch.** Plain dicts passed to `.each`
  are always scalar-broadcast to all leaves.
- **Tags must exist before grouping.** Every leaf must have the tag you group by,
  or `groupby()` raises `ValueError` listing what is missing.
- **`merge()` requires unique handles.** If two recordings share a handle, raise
  before merging and rename one.
