# Plotting — AI context page

*This page is written for AI assistants. See [Use with AI](index.md) for the gateway prompt.*

*For full signatures and parameter details, see the [Summary API reference](../api/summary.md).*

---

## Where plotting lives

Plotting is on `SummaryCollection`, not on individual `Summary` objects. The
collection must be **grouped** before plotting — grouping determines the x-axis
structure and colour assignment.

```python
sc_grouped = sc.groupby('genotype')
sc_grouped.snsstrip('time_in_arena')
```

There is no standalone plotting module to import. All `sns*` methods are called
directly on a grouped `SummaryCollection`.

---

## The `sns*` methods

Each method wraps a seaborn categorical plot type. They all share the same
interface and accept the same core parameters.

| Method | Plot type |
|---|---|
| `sc.snsstrip(metric)` | Jittered scatter (stripplot) |
| `sc.snsswarm(metric)` | Non-overlapping scatter (swarmplot) |
| `sc.snsbar(metric)` | Bar plot |
| `sc.snsbox(metric)` | Box plot |
| `sc.snsviolin(metric)` | Violin plot |
| `sc.snspoint(metric)` | Point plot (mean ± CI) |
| `sc.snssuperplot(metric)` | Bar (mean) + strip (individual points) overlay |

`snssuperplot` is the recommended default for publication figures — it shows
both the group summary and individual data points.

The `metric` argument is either a stored metric name (string) or a
`BatchResult` from a chained `.each` call:

```python
# From a stored metric name
sc_grouped.snsstrip('time_in_arena')

# Directly from a batch result — no need to store first
sc_grouped.snsstrip(sc_grouped.each.time_true('in_arena'))
```

All methods return `(fig, ax, df)`.

---

## Group order and axis layout

Use `group_order` to control the order groups appear on the x-axis:

```python
sc_grouped.snsstrip(
    'time_in_arena',
    group_order={'genotype': ['WT', 'KO']}
)
```

Use `sort_by` to change the primary sort axis without affecting colour
assignment. This is useful when the collection is grouped by multiple tags
and you want a different tag to drive the x-axis layout:

```python
sc_grouped2 = sc.groupby(['genotype', 'timepoint'])
sc_grouped2.snsstrip('time_in_arena', sort_by='timepoint')
```

---

## Statistical annotations via `statannotations`

All `sns*` methods accept an `annotate` parameter that drives the
`statannotations` library. `statannotations` is an optional dependency — install
it separately if needed (`pip install statannotations`).

Pass `annotate="help"` to print the available tests, corrections, and the
exact group label strings present in the current plot (useful for constructing
the `pairs` list):

```python
sc_grouped.snsstrip('time_in_arena', annotate="help")
```

To add annotations, pass a dict with at least `"pairs"`:

```python
sc_grouped.snsstrip(
    'time_in_arena',
    annotate={
        'pairs': [('WT', 'KO')],
        'test': 'Mann-Whitney',
        'text_format': 'star',
    }
)
```

Additional keys in the `annotate` dict are passed through to
`statannotations.Annotator`. See the API reference for all supported keys,
including `correction` (e.g. `'bonferroni'`, `'fdr_bh'`) and `headroom`
(extra vertical space for annotation brackets).

---

## Power-user path: `prepare_plot`

For full control over the seaborn call, use `prepare_plot()` directly. It
returns a `PlotSpec` with a ready-to-unpack `sns_kwargs` dict, a tidy
long-form DataFrame, and the figure and axes:

```python
import seaborn as sns

spec = sc_grouped.prepare_plot('time_in_arena', group_order=...)
sns.boxplot(**spec.sns_kwargs, width=0.6)
spec.ax.set_title('My title')
spec.fig.savefig('output.png', dpi=300)
```

This is also the right path for composing multiple seaborn layers on one axes
(e.g. bar + strip overlay with custom styling).

---

## Plotting from a single Summary (ungrouped)

`Summary` (not `SummaryCollection`) also has `sns*` methods for quick
single-recording plots. The interface is the same but there is no grouping
structure — one point per metric. Use this only for exploratory work; for
publication figures always use the grouped collection.
