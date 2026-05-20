# Summary — AI context page

*This page is written for AI assistants. See [Use with AI](index.md) for the gateway prompt.*

---

## What Summary is

`Summary` consumes a `Features` object and computes aggregate statistics — scalar
values, per-state breakdowns, or transition matrices — across the recording.

```python
s = f.to_summary()
# or equivalently:
s = Summary(f)
```

Statistics are stored in `s.data` (a dict of name → value) once explicitly
stored with `.store()`.

---

## The store pattern — same as Features

**Summary methods return a `SummaryResult`. Nothing is stored automatically.**

```python
result = s.time_true('in_arena')
result.store(name='time_in_arena')
# Now: s.data['time_in_arena'] is a scalar (seconds)
```

---

## Standard statistics

All methods below take a column name from `features.data`.

| What | Method | Returns |
|---|---|---|
| Total seconds a boolean column is True | `s.time_true(column)` | scalar |
| Total seconds a boolean column is False | `s.time_false(column)` | scalar |
| Number of False→True transitions | `s.count_onset(column)` | scalar |
| Total distance traveled by a point | `s.total_distance(point)` | scalar |
| Sum / mean / median / max / min of a column | `s.sum_column(column)` etc. | scalar |
| Frame index of the N-th event onset | `s.calculate_latency_nth_onset(column, target, op, nth)` | scalar |

For full parameter signatures, see the [Summary API reference](../api/summary.md).

---

## State-based analysis

When `features.data` contains a categorical column (a state label per frame),
use `.by_state()` to apply a method per state and collect the results into a
`pd.Series` indexed by state name.

```python
# features.data['behaviour'] contains labels like 'moving', 'grooming', 'still'

result = s.by_state('behaviour', all_states=['moving','grooming','still']).time_true('in_arena')
result.store(name='time_in_arena_by_state')
# result.value is a pd.Series: {'moving': 12.4, 'grooming': 5.1, 'still': 8.2}
```

**Important constraints:**

- `.by_state()` only works with methods explicitly marked as compatible. The
  supported set is: `time_true`, `time_false`, `total_distance`, `sum_column`,
  `mean_column`, `median_column`, `max_column`, `min_column`. Calling an
  unsupported method raises `NotImplementedError`.
- Each per-state call must return a scalar. Methods that return a Series or
  DataFrame per call (e.g. `transition_matrix`) cannot be used with `.by_state()`.
- `all_states` is the complete expected state list. Missing states are filled
  with 0 or NaN rather than being silently absent — always supply it.

For counting onsets or time per state across the whole recording without
subsetting, use the dedicated shorthand methods:

```python
s.time_in_state('behaviour', all_states=[...])      # seconds per state
s.count_state_onsets('behaviour', all_states=[...]) # onset count per state
s.transition_matrix('behaviour', all_states=[...])  # DataFrame of transitions
```

---

## Temporal bins

Divide the recording into time bins before summarising:

```python
# Split into 3 equal-duration bins
bins = s.make_bins(numbins=3)
for i, bin_summary in enumerate(bins):
    result = bin_summary.time_true('in_arena')
    result.store(name=f'time_in_arena_bin{i}')
```

Or define an explicit bin by frame range:

```python
first_minute = s.make_bin(startframe=0, endframe=1800)  # at 30 fps
```

---

## Accessing stored results

```python
s.data                  # dict of name → value
s.data['time_in_arena'] # scalar, pd.Series, or pd.DataFrame depending on method
s.meta['time_in_arena'] # metadata dict for that statistic
```

---

## Saving and loading

```python
s.save('path/to/summary_dir/')
s2 = Summary.load('path/to/summary_dir/')
```

---

## What Summary does NOT do

- Compute per-frame features — that is `Features`'s job.
- Modify the underlying `Tracking` or `Features` data.
- Produce publication figures directly (though `.plot_chord()` is available for
  transition matrices).

If the user wants a per-frame derived quantity, they need to go back to `Features`.
