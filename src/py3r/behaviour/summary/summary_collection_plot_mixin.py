"""Seaborn plotting wrappers for SummaryCollection.

This mixin is mixed into :class:`SummaryCollection` and provides all
``sns*`` categorical-plot helpers plus the shared infrastructure they rely on
(data tidying, figure sizing, palette construction, pre/post plot hooks).
"""

from __future__ import annotations

import pandas as pd


class SummaryCollectionPlotMixin:
    """Mixin supplying seaborn plotting methods to SummaryCollection."""

    # -------------------------------------------------------------------------
    # Static helpers — group labelling, sorting, palettes
    # -------------------------------------------------------------------------

    @staticmethod
    def _gkey_to_label(gkey):
        """Format a group key tuple as a clean string label."""
        if isinstance(gkey, tuple):
            if len(gkey) == 1:
                return str(gkey[0])
            return ", ".join(str(x) for x in gkey)
        return str(gkey)

    @staticmethod
    def _rank_tuple(parts, tags, group_order):
        """Build a sort-key tuple from *parts* using *group_order* for *tags*.

        For each position, if the tag appears in *group_order*, the rank is the
        index within that list (unlisted values sort last).  Otherwise falls
        back to numeric-aware ordering.
        """
        ranks = []
        for i, tag in enumerate(tags):
            val = parts[i] if i < len(parts) else ""
            if tag in group_order:
                try:
                    ranks.append(group_order[tag].index(val))
                except ValueError:
                    ranks.append(len(group_order[tag]))
            else:
                try:
                    ranks.append((0, float(val)))
                except (ValueError, TypeError):
                    ranks.append((1, str(val)))
        return ranks

    @staticmethod
    def _sort_group_labels(label_to_tuple, group_order=None, groupby_tags=None, sort_by=None):
        """
        Return group labels sorted according to *group_order*.

        Parameters
        ----------
        label_to_tuple : dict[str, tuple]
            Mapping of display label → raw group-key tuple.
        group_order : dict[str, list] | None
            ``{tag_name: [val, ...]}`` giving the desired order per tag.
            Tags not present fall back to :meth:`_smart_sort_labels`.
        groupby_tags : list[str] | None
            Tag names corresponding to tuple positions (from ``groupby(tags=...)``).
        sort_by : list[str] | str | None
            Tag names in the desired *spatial* priority order.  When provided,
            the sort key is built using this tag order instead of *groupby_tags*,
            allowing spatial arrangement to differ from palette colour assignment
            (which always follows *groupby_tags*).

            Accepts a single tag name (string) to promote that tag to primary
            sort position while keeping the remaining tags in their original
            ``groupby`` order.

        Returns
        -------
        list[str]
            Group labels in the desired order.
        """
        if not group_order or not groupby_tags:
            return SummaryCollectionPlotMixin._smart_sort_labels(label_to_tuple.keys())

        # Resolve sort_by into a full tag ordering
        if sort_by is not None:
            if isinstance(sort_by, str):
                sort_by = [sort_by]
            # Validate all sort_by tags exist in groupby_tags
            unknown = set(sort_by) - set(groupby_tags)
            if unknown:
                raise ValueError(f"sort_by tag(s) {unknown} not in groupby tags {groupby_tags}")
            # Append any groupby_tags not mentioned in sort_by (preserve their
            # relative order as a tiebreaker)
            sort_tags = list(sort_by) + [t for t in groupby_tags if t not in sort_by]
        else:
            sort_tags = groupby_tags

        def _key(label):
            raw = label_to_tuple[label]
            # Rearrange tuple elements to match sort_tags order
            reordered = tuple(raw[groupby_tags.index(t)] for t in sort_tags)
            return SummaryCollectionPlotMixin._rank_tuple(reordered, sort_tags, group_order)

        return sorted(label_to_tuple.keys(), key=_key)

    @staticmethod
    def _build_group_palette(label_to_tuple, group_order=None, groupby_tags=None):
        """
        Build a colour palette from group labels mapped to their raw tuples.

        For multi-tag groups the first n-1 tuple elements choose a base colour
        (from ``tab10``) and the last element creates a light-to-dark gradient
        within that base.  Single-tag groups return ``None`` (seaborn default).

        When *group_order* and *groupby_tags* are provided, the ordering of
        base-colour groups (first n-1 tags) and shade values (last tag) respects
        the user-specified order.
        """
        import colorsys

        import matplotlib.pyplot as plt

        max_depth = max(len(t) for t in label_to_tuple.values())
        if max_depth <= 1:
            return None

        # Resolve base-group ordering (first n-1 tags)
        raw_bases = list(set(t[:-1] for t in label_to_tuple.values()))
        if group_order and groupby_tags and len(groupby_tags) > 1:
            base_tags = groupby_tags[:-1]
            base_groups = sorted(
                raw_bases,
                key=lambda bg: SummaryCollectionPlotMixin._rank_tuple(bg, base_tags, group_order),
            )
        else:
            base_groups = sorted(raw_bases)

        base_colors = plt.cm.tab10.colors
        base_map = {bg: base_colors[i % len(base_colors)] for i, bg in enumerate(base_groups)}

        # Resolve shade ordering (last tag)
        raw_shades = set(t[-1] for t in label_to_tuple.values())
        last_tag = groupby_tags[-1] if groupby_tags else None
        if group_order and last_tag and last_tag in group_order:
            specified = group_order[last_tag]
            shade_values = [v for v in specified if v in raw_shades]
            # Append any values not in the specified order
            shade_values += SummaryCollectionPlotMixin._smart_sort_labels(
                raw_shades - set(shade_values)
            )
        else:
            shade_values = SummaryCollectionPlotMixin._smart_sort_labels(raw_shades)
        n_shades = len(shade_values)

        palette = {}
        for label, parts in label_to_tuple.items():
            base_rgb = base_map[parts[:-1]]
            shade_idx = shade_values.index(parts[-1])

            # Blend from light (towards white) to full colour
            if n_shades > 1:
                t = 0.35 + (shade_idx / (n_shades - 1)) * 0.65
            else:
                t = 0.8

            # Adjust saturation via HLS so hue stays stable
            h, _l, s = colorsys.rgb_to_hls(*base_rgb[:3])
            lightness = 0.85 - t * 0.45  # range ~0.85 (light) → ~0.40 (dark)
            r, g, b = colorsys.hls_to_rgb(h, lightness, s)
            palette[label] = (r, g, b)

        return palette

    # -------------------------------------------------------------------------
    # Data tidying
    # -------------------------------------------------------------------------

    def _metric_to_tidy(
        self, metric, group_order=None, sort_by=None
    ) -> tuple[pd.DataFrame, str, dict | None, list | None]:
        """
        Convert a metric (string key or BatchResult) to a tidy (long-form) DataFrame.

        Parameters
        ----------
        metric : str or BatchResult
            Metric to convert.
        group_order : dict[str, list] | None
            ``{tag_name: [value, ...]}`` controlling group display order.
        sort_by : list[str] | str | None
            Override spatial sort priority.  See :meth:`_sort_group_labels`.

        Returns
        -------
        tuple[pd.DataFrame, str, dict | None, list | None, str | None]
            - DataFrame with columns: _handle, _group (if grouped), component, value
            - metric_name string for labeling
            - palette dict (label → RGB) or None
            - sorted group labels list or None
            - ylabel string (from SummaryResult) or None
        """
        from py3r.behaviour.util.collection_utils import BatchResult

        flat_self = self.flatten()
        is_grouped = getattr(self, "is_grouped", False)

        # Extract data based on metric type
        metric_name = None
        auto_ylabel = None
        if isinstance(metric, str):
            metric_name = metric
            if is_grouped:
                data_map = {}
                for gkey, subcoll in self.items():
                    data_map[gkey] = {}
                    for handle, summary in subcoll.items():
                        if metric not in summary.data:
                            raise KeyError(f"Metric '{metric}' not found in Summary '{handle}'")
                        data_map[gkey][handle] = summary.data[metric]
                        if auto_ylabel is None:
                            stored_meta = summary.meta.get(metric)
                            if isinstance(stored_meta, dict):
                                auto_ylabel = stored_meta.get("_ylabel")
            else:
                data_map = {}
                for handle, summary in flat_self.items():
                    if metric not in summary.data:
                        raise KeyError(f"Metric '{metric}' not found in Summary '{handle}'")
                    data_map[handle] = summary.data[metric]
                    if auto_ylabel is None:
                        stored_meta = summary.meta.get(metric)
                        if isinstance(stored_meta, dict):
                            auto_ylabel = stored_meta.get("_ylabel")

        elif isinstance(metric, (dict, BatchResult)):
            raw = dict(metric) if isinstance(metric, BatchResult) else metric
            first_val = next(iter(raw.values()))
            if isinstance(first_val, dict):
                # Grouped structure: {group: {handle: SummaryResult}}
                is_grouped = True
                data_map = {}
                for gkey, subdict in raw.items():
                    data_map[gkey] = {}
                    for handle, sr in subdict.items():
                        val = sr.value if hasattr(sr, "value") else sr
                        data_map[gkey][handle] = val
                        if metric_name is None and hasattr(sr, "_func_name"):
                            metric_name = sr._func_name
                        if auto_ylabel is None and hasattr(sr, "_ylabel"):
                            auto_ylabel = sr._ylabel
            else:
                # Flat structure: {handle: SummaryResult}
                data_map = {}
                for handle, sr in raw.items():
                    val = sr.value if hasattr(sr, "value") else sr
                    data_map[handle] = val
                    if metric_name is None and hasattr(sr, "_func_name"):
                        metric_name = sr._func_name
                    if auto_ylabel is None and hasattr(sr, "_ylabel"):
                        auto_ylabel = sr._ylabel
        else:
            raise TypeError(f"metric must be str or BatchResult/dict, got {type(metric).__name__}")

        if metric_name is None:
            metric_name = "value"

        # Build tidy DataFrame: one row per (handle, component) pair
        # For grouped data, also track raw tuples for palette construction.
        rows = []
        label_to_tuple = {}  # {label_str: raw_tuple}
        if is_grouped:
            for gkey, subdict in data_map.items():
                gkey_tuple = gkey if isinstance(gkey, tuple) else (gkey,)
                gkey_str = self._gkey_to_label(gkey)
                label_to_tuple[gkey_str] = gkey_tuple
                for handle, val in subdict.items():
                    if isinstance(val, pd.Series):
                        for comp, v in val.items():
                            rows.append(
                                {
                                    "_handle": handle,
                                    "_group": gkey_str,
                                    "component": str(comp),
                                    "value": v,
                                }
                            )
                    else:
                        rows.append(
                            {
                                "_handle": handle,
                                "_group": gkey_str,
                                "component": metric_name,
                                "value": float(val),
                            }
                        )
        else:
            for handle, val in data_map.items():
                if isinstance(val, pd.Series):
                    for comp, v in val.items():
                        rows.append({"_handle": handle, "component": str(comp), "value": v})
                else:
                    rows.append(
                        {
                            "_handle": handle,
                            "component": metric_name,
                            "value": float(val),
                        }
                    )

        df = pd.DataFrame(rows)
        if len(df) == 0:
            raise ValueError("No data to plot.")

        # Build palette and sorted group labels from group structure
        groupby_tags = getattr(self, "_groupby_tags", None)
        if label_to_tuple:
            palette = self._build_group_palette(label_to_tuple, group_order, groupby_tags)
            sorted_groups = self._sort_group_labels(
                label_to_tuple, group_order, groupby_tags, sort_by=sort_by
            )
        else:
            palette = None
            sorted_groups = None

        return df, metric_name, palette, sorted_groups, auto_ylabel

    # -------------------------------------------------------------------------
    # Figure sizing
    # -------------------------------------------------------------------------

    # Default sizing constants for seaborn wrappers
    _SNS_HEIGHT = 4.0  # fixed vertical size (inches)
    _SNS_MIN_WIDTH = 2.0  # minimum figure width (inches)
    _SNS_BASE_PER_TICK = 0.35  # base horizontal inches per x-tick
    _SNS_EXTRA_PER_GROUP = 0.15  # additional inches per dodged sub-bar

    @staticmethod
    def _auto_figsize(n_components: int, n_groups: int = 1, figsize=None) -> tuple[float, float]:
        """Compute default figure size.

        Width per tick position scales with the number of sub-bars that
        need to fit (i.e. *n_groups* when dodge is active).  This keeps
        bars compact when ungrouped and gives enough room when several
        groups are dodged within each component.
        """
        if figsize is not None:
            return figsize
        n_ticks = n_groups if n_components == 1 else n_components
        # How many sub-bars per tick? 1 when ungrouped or single-component-grouped.
        bars_per_tick = n_groups if n_components > 1 else 1
        width_per_tick = (
            SummaryCollectionPlotMixin._SNS_BASE_PER_TICK
            + bars_per_tick * SummaryCollectionPlotMixin._SNS_EXTRA_PER_GROUP
        )
        width = max(SummaryCollectionPlotMixin._SNS_MIN_WIDTH, n_ticks * width_per_tick)
        return (width, SummaryCollectionPlotMixin._SNS_HEIGHT)

    # -------------------------------------------------------------------------
    # Misc static helpers
    # -------------------------------------------------------------------------

    @staticmethod
    def _smart_sort_labels(labels):
        """Sort labels numerically when possible, otherwise alphabetically."""

        def _sort_key(label):
            try:
                return (0, float(label))
            except (ValueError, TypeError):
                return (1, str(label))

        return sorted(labels, key=_sort_key)

    @staticmethod
    def _prettify_metric_name(metric_name):
        """Convert a snake_case metric key into a human-readable title.

        ``"total_distance_bodycentre"`` → ``"Total Distance Bodycentre"``
        """
        return metric_name.replace("_", " ").strip().title()

    @staticmethod
    def _slugify_metric_name(metric_name):
        """Convert a metric name into a safe filename slug.

        Replaces spaces and special characters with underscores and
        strips leading/trailing underscores.
        """
        import re

        slug = re.sub(r"[^a-zA-Z0-9]+", "_", metric_name)
        return slug.strip("_").lower()

    # -------------------------------------------------------------------------
    # Seaborn kwargs builder
    # -------------------------------------------------------------------------

    @staticmethod
    def _build_sns_kwargs(df, ax, palette=None, sorted_groups=None, y_col="value"):
        """
        Build core seaborn plot kwargs based on data shape.

        - **Ungrouped**: ``x=component``, no hue.
        - **Grouped, single component** (scalar metric): ``x=_group``,
          ``hue=_group`` for colouring.  Legend is hidden (info in tick labels).
        - **Grouped, multi-component**: ``x=component``, ``hue=_group``,
          ``dodge=True``.  Seaborn packs groups tightly within each component
          position with natural gaps between components.

        Parameters
        ----------
        y_col : str
            Name of the y-value column in *df*.  Defaults to ``"value"``
            but will be the ylabel string when called from
            :meth:`prepare_plot`.

        Returns
        -------
        tuple[dict, bool]
            (plot_kwargs, hide_legend)
        """
        is_grouped = "_group" in df.columns
        n_components = df["component"].nunique()

        if not is_grouped:
            components = SummaryCollectionPlotMixin._smart_sort_labels(df["component"].unique())
            return {
                "data": df,
                "x": "component",
                "y": y_col,
                "hue": "component",
                "dodge": False,
                "order": components,
                "hue_order": components,
                "ax": ax,
            }, True

        groups = sorted_groups if sorted_groups is not None else sorted(df["_group"].unique())

        if n_components == 1:
            # Scalar grouped: groups as tick labels, hue for colouring only
            kwargs = {
                "data": df,
                "x": "_group",
                "y": y_col,
                "hue": "_group",
                "dodge": False,
                "order": groups,
                "hue_order": groups,
                "ax": ax,
            }
            if palette:
                kwargs["palette"] = palette
            return kwargs, True  # legend redundant (info in tick labels)

        # Multi-component grouped: seaborn dodge handles spacing natively
        components = SummaryCollectionPlotMixin._smart_sort_labels(df["component"].unique())
        kwargs = {
            "data": df,
            "x": "component",
            "y": y_col,
            "hue": "_group",
            "dodge": True,
            "order": components,
            "hue_order": groups,
            "ax": ax,
        }
        if palette:
            kwargs["palette"] = palette
        return kwargs, False

    @staticmethod
    def _sns_post_plot(
        fig,
        ax,
        *,
        metric_name,
        title=None,
        ylabel="Value",
        n_components=1,
        n_groups=1,
        hide_legend=True,
        legend_n_entries=None,
        savedir=None,
        filename=None,
        filename_prefix=None,
        default_suffix="plot",
        show=True,
        created_fig=True,
    ):
        """Apply styling, manage legend, save/show figure.

        Parameters
        ----------
        legend_n_entries : int | None
            When not None, deduplicate a multi-layer legend by keeping only
            the first *legend_n_entries* handles (used by superplot).
        filename_prefix : str | None
            Optional prefix for the auto-generated filename (e.g. a handle).
        created_fig : bool
            Whether this method's caller created the figure (True) or the
            user passed an external *ax* (False).  When False the figure is
            never closed, because the user owns it.
        """
        import os

        import matplotlib.pyplot as plt

        ax.set_xlabel("")
        ax.set_ylabel(ylabel)
        pretty_title = title or SummaryCollectionPlotMixin._prettify_metric_name(metric_name)
        ax.set_title(pretty_title)

        # Rotate x-tick labels if many ticks
        n_ticks = n_groups if n_components == 1 else n_components
        if n_ticks > 3:
            ax.tick_params(axis="x", rotation=45)
            for label in ax.get_xticklabels():
                label.set_ha("right")

        # Legend: hide when redundant, otherwise place outside plot
        legend = ax.get_legend()
        if legend is not None:
            if hide_legend:
                legend.remove()
            elif legend_n_entries is not None:
                handles, labels = ax.get_legend_handles_labels()
                ax.legend(
                    handles[:legend_n_entries],
                    labels[:legend_n_entries],
                    title="Group",
                    bbox_to_anchor=(1.02, 1),
                    loc="upper left",
                )
            else:
                legend.set_bbox_to_anchor((1.02, 1))
                legend.set_loc("upper left")

        plt.tight_layout()

        if savedir:
            os.makedirs(savedir, exist_ok=True)
            if filename is None:
                slug = SummaryCollectionPlotMixin._slugify_metric_name(metric_name)
                parts = [filename_prefix, slug, default_suffix]
                fname = "_".join(p for p in parts if p) + ".png"
            else:
                fname = filename
            fig.savefig(os.path.join(savedir, fname), dpi=150, bbox_inches="tight")

        if show:
            plt.show()
        elif created_fig:
            # Close figure to suppress automatic inline-backend display in
            # Jupyter/IPython.  The returned *fig* object remains usable for
            # saving (fig.savefig(...)) or explicit display(fig).
            # Only close if we created the figure; user-provided axes are
            # left open so the user retains control.
            plt.close(fig)

    # ------------------------------------------------------------------
    # Statistical annotations (statannotations integration)
    # ------------------------------------------------------------------

    @staticmethod
    def _apply_annotations(ax, df, annotate, sns_kw):
        """Apply statistical annotations to a seaborn plot.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            The axes containing the plot.
        df : pandas.DataFrame
            The tidy DataFrame (used to extract group labels for help mode).
        annotate : str or dict or None
            - ``None``: no annotations (returns immediately).
            - ``"help"``: prints a guide with available tests, corrections,
              and the current group labels, then returns without annotating.
            - ``dict``: must contain ``"pairs"``; all other keys are passed
              to ``Annotator.configure()``.  Defaults::

                  test="Mann-Whitney", text_format="star", loc="inside"
        sns_kw : dict
            Seaborn call parameters (``data``, ``x``, ``y``, etc.) stashed
            by the plot function.

        Returns
        -------
        bool
            True if annotations were applied, False otherwise.
        """
        if annotate is None:
            return False

        # ---- Help mode ----
        is_grouped = "_group" in df.columns
        if annotate == "help":
            if is_grouped:
                labels = sorted(df["_group"].unique().tolist())
            else:
                labels = sorted(df["component"].unique().tolist())

            print(
                "=== Statistical Annotation Guide ===\n"
                "\n"
                "annotate={\n"
                '    "pairs": [("groupA", "groupB"), ...],  # REQUIRED\n'
                '    "test": "Mann-Whitney",                # see below\n'
                '    "correction": None,                    # see below\n'
                '    "text_format": "star",                 # "star", "simple", "full"\n'
                '    "headroom": None,                      # float multiplier, see below\n'
                "}\n"
                "\n"
                "Available tests:\n"
                "  Parametric:     t-test_ind, t-test_welch, t-test_paired\n"
                "  Non-parametric: Mann-Whitney, Wilcoxon, Kruskal, Brunner-Munzel\n"
                "  Other:          Levene (variance equality)\n"
                "\n"
                "  Tip: Mann-Whitney is a safe default for most behavioural data.\n"
                "  Use paired tests (t-test_paired, Wilcoxon) for repeated measures.\n"
                "  Use parametric tests only if data is normally distributed.\n"
                "\n"
                "Multiple comparisons correction (recommended for >3 pairs):\n"
                "  FWER (conservative): bonferroni, holm\n"
                "  FDR  (less conservative): fdr_bh (Benjamini-Hochberg), fdr_by\n"
                "\n"
                "Headroom:\n"
                "  Extra vertical space for brackets, as a fraction of the y range.\n"
                "  E.g. headroom=0.3 adds 30%% extra room above the data.\n"
                "\n"
                f"Your labels: {labels}\n"
            )
            return False

        # ---- Annotation mode ----
        if not isinstance(annotate, dict):
            raise TypeError(
                'annotate must be None, "help", or a dict with at least "pairs". '
                'Pass annotate="help" for a guide.'
            )

        try:
            from statannotations.Annotator import Annotator
        except ImportError as e:
            raise ImportError(
                "statannotations is required for plot annotations. "
                "Install with: pip install statannotations"
            ) from e

        pairs = annotate.get("pairs")
        if not pairs:
            raise ValueError(
                'annotate dict must contain "pairs", e.g. '
                'annotate={"pairs": [("groupA", "groupB")]}'
            )

        # Separate Annotator.configure kwargs from our own
        configure_kw = {k: v for k, v in annotate.items() if k != "pairs"}

        # Sensible defaults
        configure_kw.setdefault("test", "Mann-Whitney")
        configure_kw.setdefault("text_format", "star")
        configure_kw.setdefault("loc", "inside")

        # Map our friendly "correction" key to statannotations' name
        correction = configure_kw.pop("correction", None)
        if correction is not None:
            configure_kw["comparisons_correction"] = correction

        # Pop our custom headroom key before it reaches statannotations
        headroom = configure_kw.pop("headroom", None)

        # Optional manual headroom: fraction of extra vertical space, e.g.
        # 0.3 means 30% extra room above the data for annotation brackets.
        if headroom:
            ymin, ymax = ax.get_ylim()
            ax.set_ylim(ymin, ymax * (1 + headroom))

        annotator = Annotator(ax, pairs, **sns_kw)
        annotator.configure(**configure_kw)
        annotator.apply_and_annotate()
        return True

    # ------------------------------------------------------------------
    # Main single-function wrapper
    # ------------------------------------------------------------------

    def _sns_plot_common(
        self,
        plot_func,
        metric,
        *,
        group_order: dict | None = None,
        sort_by: list | str | None = None,
        annotate=None,
        ax=None,
        show: bool = True,
        savedir: str | None = None,
        filename: str | None = None,
        title: str | None = None,
        ylabel: str | None = None,
        **kwargs,
    ):
        """
        Common wrapper logic for seaborn categorical plots.

        Parameters
        ----------
        plot_func : callable
            The seaborn plotting function (e.g., sns.stripplot).
        metric : str or BatchResult
            Metric to plot.
        group_order : dict[str, list] | None
            Control group display order. Keys are tag names (matching those
            passed to ``groupby(tags=...)``), values are lists of tag values
            in the desired order.  Example::

                group_order={"treatment": ["control", "FST"],
                             "timepoint": ["45m", "1d"]}
        sort_by : list[str] | str | None
            Override spatial sort priority on the x-axis without affecting
            colour assignment.  Accepts a list of tag names in the desired
            sort priority, or a single tag name (string) to promote it to
            primary.  Colours always follow the ``groupby(tags=...)`` order.
            Example::

                # groupby(tags=["treatment", "timepoint"])
                # colours: treatment = base colour, timepoint = shade
                sort_by=["timepoint", "treatment"]
                # → control,45m | FST,45m | control,1d | FST,1d

        annotate : str or dict or None
            Statistical annotations via ``statannotations``.

            - ``None`` (default): no annotations.
            - ``"help"``: print a guide with available tests, corrections,
              and the group labels present in this plot.
            - ``dict``: must contain ``"pairs"``; everything else is
              optional with sensible defaults::

                  annotate={
                      "pairs": [("control, 45m", "FST, 45m")],
                      "test": "Mann-Whitney",    # default
                      "correction": None,         # "bonferroni", "holm", "fdr_bh", …
                      "text_format": "star",      # "star", "simple", "full"
                      "headroom": None,           # float; vertical space multiplier
                  }

              ``headroom`` (float, optional): extra vertical space for
              annotation brackets, as a fraction of the y range.  E.g.
              ``0.3`` adds 30% extra room above the data.
        ax : matplotlib.axes.Axes, optional
            Axes to plot on. If None, creates new figure.
        show : bool
            Display the plot.
        savedir : str | None
            Directory to save figure.
        filename : str | None
            Custom filename.
        title : str | None
            Plot title.
        ylabel : str | None
            Y-axis label. When *None*, uses the unit label from the
            ``SummaryResult`` (e.g. ``"Time (s)"``), falling back to
            ``"Value"`` if unavailable.
        **kwargs
            Passed to the seaborn plot function.

        Returns
        -------
        tuple[Figure, Axes, DataFrame]
        """
        spec = self.prepare_plot(
            metric,
            group_order=group_order,
            sort_by=sort_by,
            ax=ax,
            figsize=kwargs.pop("figsize", None),
        )
        spec.sns_kwargs.update(kwargs)
        plot_func(**spec.sns_kwargs)

        # Build seaborn params dict for statannotations passthrough
        sns_kw = {
            k: v
            for k, v in spec.sns_kwargs.items()
            if k in ("data", "x", "y", "hue", "order", "hue_order")
        }

        # Apply statistical annotations (before save/show)
        self._apply_annotations(spec.ax, spec.df, annotate, sns_kw)

        self._sns_post_plot(
            spec.fig,
            spec.ax,
            metric_name=spec.metric_name,
            title=title,
            ylabel=ylabel or spec.ylabel,
            filename_prefix=spec.filename_prefix,
            n_components=spec.n_components,
            n_groups=spec.n_groups,
            hide_legend=spec.hide_legend,
            savedir=savedir,
            filename=filename,
            default_suffix=plot_func.__name__,
            show=show,
            created_fig=spec.created_fig,
        )

        return spec.fig, spec.ax, spec.df

    # ------------------------------------------------------------------
    # Public prepare_plot() — power-user escape hatch
    # ------------------------------------------------------------------

    def prepare_plot(
        self,
        metric,
        *,
        group_order: dict | None = None,
        sort_by: list | str | None = None,
        ax=None,
        figsize=None,
    ):
        """
        Prepare a tidy DataFrame and seaborn kwargs without drawing anything.

        This is the single entry point for all plot data preparation.  The
        convenience ``sns*`` methods call this internally; power users can
        call it directly for full control over the seaborn call.

        Parameters
        ----------
        metric : str or BatchResult
            Metric to prepare.
        group_order : dict[str, list] | None
            ``{tag_name: [value, ...]}`` controlling within-tag value ordering.
        sort_by : list[str] | str | None
            Override spatial sort priority (which tag is the primary x-axis
            sort dimension).  Colours are unaffected — they always follow
            the ``groupby(tags=...)`` order.  Accepts a single tag name or a
            list.  See :meth:`_sns_plot_common` for details.
        ax : matplotlib.axes.Axes, optional
            Axes to plot on.  If *None*, a new figure is created with
            auto-calculated size.
        figsize : tuple[float, float], optional
            Override the automatic figure size.

        Returns
        -------
        PlotSpec
            A namespace with the following attributes:

            - **fig** — the :class:`~matplotlib.figure.Figure`
            - **ax** — the :class:`~matplotlib.axes.Axes`
            - **df** — tidy long-form :class:`~pandas.DataFrame`
            - **sns_kwargs** — ``dict`` ready to unpack into any seaborn
              categorical plot function (contains ``data``, ``x``, ``y``,
              ``hue``, ``order``, ``hue_order``, ``palette``, ``dodge``,
              ``ax``)
            - **metric_name** — raw metric name string
            - **ylabel** — auto-detected y-axis label (or ``"Value"``)
            - **hide_legend** — ``bool`` hint for legend handling
            - **created_fig** — ``bool`` whether the figure was created here
            - **n_components** — ``int`` number of unique components
            - **n_groups** — ``int`` number of unique groups (1 if ungrouped)
            - **filename_prefix** — ``str | None`` handle slug for auto-filenames

        Examples
        --------
        Basic power-user workflow::

            import seaborn as sns

            spec = sc_grouped.prepare_plot(
                "total_distance",
                group_order=GROUP_ORDER,
                sort_by="timepoint",
            )

            # Full seaborn control — override anything you like
            sns.boxplot(**spec.sns_kwargs, width=0.6)
            spec.ax.set_title("My custom title")
            spec.fig.savefig("custom.png", dpi=300)

        Composing multiple layers::

            spec = sc_grouped.prepare_plot(metric, group_order=ORDER)
            sns.barplot(**spec.sns_kwargs, errorbar=None, alpha=0.4)
            sns.stripplot(**spec.sns_kwargs, size=4, jitter=True)
        """
        from types import SimpleNamespace

        import matplotlib.pyplot as plt

        df, metric_name, palette, sorted_groups, auto_ylabel = self._metric_to_tidy(
            metric, group_order, sort_by=sort_by
        )
        is_grouped = "_group" in df.columns
        n_components = df["component"].nunique()
        n_groups = df["_group"].nunique() if is_grouped else 1

        # Rename the y-column from generic "value" to the actual label so
        # seaborn auto-labels the y-axis correctly (e.g. "Time (s)").
        ylabel = auto_ylabel or "Value"
        df = df.rename(columns={"value": ylabel})

        created_fig = ax is None
        if created_fig:
            figsize = self._auto_figsize(n_components, n_groups, figsize)
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.figure

        plot_kwargs, hide_legend = self._build_sns_kwargs(
            df, ax, palette, sorted_groups, y_col=ylabel
        )

        # For single-item collections, prefix auto-filename with the handle
        unique_handles = df["_handle"].unique()
        filename_prefix = (
            self._slugify_metric_name(unique_handles[0]) if len(unique_handles) == 1 else None
        )

        return SimpleNamespace(
            fig=fig,
            ax=ax,
            df=df,
            sns_kwargs=plot_kwargs,
            metric_name=metric_name,
            ylabel=ylabel,
            hide_legend=hide_legend,
            created_fig=created_fig,
            n_components=n_components,
            n_groups=n_groups,
            filename_prefix=filename_prefix,
        )

    # ------------------------------------------------------------------
    # Public seaborn wrappers
    # ------------------------------------------------------------------

    def snsstrip(
        self,
        metric,
        *,
        group_order: dict | None = None,
        sort_by: list | str | None = None,
        annotate=None,
        ax=None,
        show: bool = True,
        savedir: str | None = None,
        filename: str | None = None,
        title: str | None = None,
        **kwargs,
    ):
        """
        Strip plot (jittered scatter) using seaborn.

        Parameters
        ----------
        metric : str or BatchResult
            Either a key from Summary.data, or a BatchResult from a batch method.
        group_order : dict[str, list] | None
            Control group display order.  See :meth:`_sns_plot_common`.
        sort_by : list[str] | str | None
            Override spatial sort priority.  See :meth:`_sns_plot_common`.
        ax : matplotlib.axes.Axes, optional
            Axes to plot on. If None, creates new figure.
        show : bool
            Display the plot. Default True.
        savedir : str | None
            Directory to save figure.
        filename : str | None
            Custom filename.
        title : str | None
            Plot title.
        **kwargs
            Passed to seaborn.stripplot (e.g., jitter, alpha, size, palette).

        Returns
        -------
        tuple[Figure, Axes, DataFrame]

        Examples
        --------
        ```pycon
        >>> import tempfile, shutil
        >>> from pathlib import Path
        >>> import pandas as pd
        >>> from py3r.behaviour.util.docdata import data_path
        >>> from py3r.behaviour.tracking.tracking_collection import TrackingCollection
        >>> from py3r.behaviour.features.features_collection import FeaturesCollection
        >>> from py3r.behaviour.summary.summary_collection import SummaryCollection
        >>> with tempfile.TemporaryDirectory() as d:
        ...     d = Path(d)
        ...     with data_path('py3r.behaviour.tracking._data', 'dlc_single.csv') as p:
        ...         _ = shutil.copy(p, d / 'A.csv'); _ = shutil.copy(p, d / 'B.csv')
        ...     tc = TrackingCollection.from_dlc({'A': str(d/'A.csv'), 'B': str(d/'B.csv')}, fps=30)
        >>> fc = FeaturesCollection.from_tracking_collection(tc)
        >>> for f in fc.values():
        ...     idx = f.tracking.data.index[:30]
        ...     f.store(pd.Series(([True, False] * 15)[:len(idx)], index=idx),
        ...             'active', meta={})
        >>> sc = SummaryCollection.from_features_collection(fc)
        >>> fig, ax, df = sc.snsstrip(sc.each.time_in_state('active'), show=False)
        >>> isinstance(df, pd.DataFrame)
        True

        ```
        """
        import seaborn as sns

        defaults = {"alpha": 0.7, "jitter": True, "size": 5}
        for k, v in defaults.items():
            kwargs.setdefault(k, v)

        return self._sns_plot_common(
            sns.stripplot,
            metric,
            group_order=group_order,
            sort_by=sort_by,
            annotate=annotate,
            ax=ax,
            show=show,
            savedir=savedir,
            filename=filename,
            title=title,
            **kwargs,
        )

    def snsswarm(
        self,
        metric,
        *,
        group_order: dict | None = None,
        sort_by: list | str | None = None,
        annotate=None,
        ax=None,
        show: bool = True,
        savedir: str | None = None,
        filename: str | None = None,
        title: str | None = None,
        **kwargs,
    ):
        """
        Swarm plot (non-overlapping scatter) using seaborn.

        Parameters
        ----------
        metric : str or BatchResult
            Either a key from Summary.data, or a BatchResult from a batch method.
        group_order : dict[str, list] | None
            Control group display order.  See :meth:`_sns_plot_common`.
        sort_by : list[str] | str | None
            Override spatial sort priority.  See :meth:`_sns_plot_common`.
        ax, show, savedir, filename, title
            Save/display options.
        **kwargs
            Passed to seaborn.swarmplot (e.g., size, palette).

        Returns
        -------
        tuple[Figure, Axes, DataFrame]
        """
        import seaborn as sns

        defaults = {"size": 5}
        for k, v in defaults.items():
            kwargs.setdefault(k, v)

        return self._sns_plot_common(
            sns.swarmplot,
            metric,
            group_order=group_order,
            sort_by=sort_by,
            annotate=annotate,
            ax=ax,
            show=show,
            savedir=savedir,
            filename=filename,
            title=title,
            **kwargs,
        )

    def snsbar(
        self,
        metric,
        *,
        group_order: dict | None = None,
        sort_by: list | str | None = None,
        annotate=None,
        ax=None,
        show: bool = True,
        savedir: str | None = None,
        filename: str | None = None,
        title: str | None = None,
        **kwargs,
    ):
        """
        Bar plot with error bars using seaborn.

        Parameters
        ----------
        metric : str or BatchResult
            Either a key from Summary.data, or a BatchResult from a batch method.
        group_order : dict[str, list] | None
            Control group display order.  See :meth:`_sns_plot_common`.
        sort_by : list[str] | str | None
            Override spatial sort priority.  See :meth:`_sns_plot_common`.
        ax, show, savedir, filename, title
            Save/display options.
        **kwargs
            Passed to seaborn.barplot (e.g., errorbar, palette, saturation).

        Returns
        -------
        tuple[Figure, Axes, DataFrame]
        """
        import seaborn as sns

        defaults = {"errorbar": "se", "capsize": 0.1}
        for k, v in defaults.items():
            kwargs.setdefault(k, v)

        return self._sns_plot_common(
            sns.barplot,
            metric,
            group_order=group_order,
            sort_by=sort_by,
            annotate=annotate,
            ax=ax,
            show=show,
            savedir=savedir,
            filename=filename,
            title=title,
            **kwargs,
        )

    def snsbox(
        self,
        metric,
        *,
        group_order: dict | None = None,
        sort_by: list | str | None = None,
        annotate=None,
        ax=None,
        show: bool = True,
        savedir: str | None = None,
        filename: str | None = None,
        title: str | None = None,
        **kwargs,
    ):
        """
        Box plot using seaborn.

        Parameters
        ----------
        metric : str or BatchResult
            Either a key from Summary.data, or a BatchResult from a batch method.
        group_order : dict[str, list] | None
            Control group display order.  See :meth:`_sns_plot_common`.
        sort_by : list[str] | str | None
            Override spatial sort priority.  See :meth:`_sns_plot_common`.
        ax, show, savedir, filename, title
            Save/display options.
        **kwargs
            Passed to seaborn.boxplot (e.g., width, palette, fliersize).

        Returns
        -------
        tuple[Figure, Axes, DataFrame]
        """
        import seaborn as sns

        return self._sns_plot_common(
            sns.boxplot,
            metric,
            group_order=group_order,
            sort_by=sort_by,
            annotate=annotate,
            ax=ax,
            show=show,
            savedir=savedir,
            filename=filename,
            title=title,
            **kwargs,
        )

    def snsviolin(
        self,
        metric,
        *,
        group_order: dict | None = None,
        sort_by: list | str | None = None,
        annotate=None,
        ax=None,
        show: bool = True,
        savedir: str | None = None,
        filename: str | None = None,
        title: str | None = None,
        **kwargs,
    ):
        """
        Violin plot using seaborn.

        Parameters
        ----------
        metric : str or BatchResult
            Either a key from Summary.data, or a BatchResult from a batch method.
        group_order : dict[str, list] | None
            Control group display order.  See :meth:`_sns_plot_common`.
        sort_by : list[str] | str | None
            Override spatial sort priority.  See :meth:`_sns_plot_common`.
        ax, show, savedir, filename, title
            Save/display options.
        **kwargs
            Passed to seaborn.violinplot (e.g., inner, split, palette).

        Returns
        -------
        tuple[Figure, Axes, DataFrame]
        """
        import seaborn as sns

        defaults = {"inner": "box"}
        for k, v in defaults.items():
            kwargs.setdefault(k, v)

        return self._sns_plot_common(
            sns.violinplot,
            metric,
            group_order=group_order,
            sort_by=sort_by,
            annotate=annotate,
            ax=ax,
            show=show,
            savedir=savedir,
            filename=filename,
            title=title,
            **kwargs,
        )

    def snspoint(
        self,
        metric,
        *,
        group_order: dict | None = None,
        sort_by: list | str | None = None,
        annotate=None,
        ax=None,
        show: bool = True,
        savedir: str | None = None,
        filename: str | None = None,
        title: str | None = None,
        **kwargs,
    ):
        """
        Point plot (mean + CI) using seaborn.

        Parameters
        ----------
        metric : str or BatchResult
            Either a key from Summary.data, or a BatchResult from a batch method.
        group_order : dict[str, list] | None
            Control group display order.  See :meth:`_sns_plot_common`.
        sort_by : list[str] | str | None
            Override spatial sort priority.  See :meth:`_sns_plot_common`.
        ax, show, savedir, filename, title
            Save/display options.
        **kwargs
            Passed to seaborn.pointplot (e.g., errorbar, markers, linestyles).

        Returns
        -------
        tuple[Figure, Axes, DataFrame]
        """
        import seaborn as sns

        defaults = {"errorbar": "se", "capsize": 0.1, "join": False}
        for k, v in defaults.items():
            kwargs.setdefault(k, v)

        return self._sns_plot_common(
            sns.pointplot,
            metric,
            group_order=group_order,
            sort_by=sort_by,
            annotate=annotate,
            ax=ax,
            show=show,
            savedir=savedir,
            filename=filename,
            title=title,
            **kwargs,
        )

    def snssuperplot(
        self,
        metric,
        *,
        group_order: dict | None = None,
        sort_by: list | str | None = None,
        annotate=None,
        ax=None,
        show: bool = True,
        savedir: str | None = None,
        filename: str | None = None,
        title: str | None = None,
        ylabel: str | None = None,
        bar_kwargs: dict | None = None,
        strip_kwargs: dict | None = None,
        **kwargs,
    ):
        """
        Superplot: bar plot (mean) with strip plot (individual dots) overlay.

        This is the "publication-ready" visualization showing mean bars with
        individual data points scattered on top, commonly used in scientific
        papers.  The dots are constrained within the bar width by default.

        Parameters
        ----------
        metric : str or BatchResult
            Either a key from Summary.data, or a BatchResult from a batch method.
        group_order : dict[str, list] | None
            Control group display order.  See :meth:`_sns_plot_common`.
        sort_by : list[str] | str | None
            Override spatial sort priority.  See :meth:`_sns_plot_common`.
        annotate : str or dict or None
            Statistical annotations.  See :meth:`_sns_plot_common`.
        ax : matplotlib.axes.Axes, optional
            Axes to plot on. If None, creates new figure.
        show : bool
            Display the plot. Default True.
        savedir : str | None
            Directory to save figure.
        filename : str | None
            Custom filename.
        title : str | None
            Plot title.
        ylabel : str | None
            Y-axis label. Auto-detected from metric when *None*.
        bar_kwargs : dict | None
            Extra kwargs for barplot (e.g., errorbar, capsize, saturation).
        strip_kwargs : dict | None
            Extra kwargs for stripplot (e.g., alpha, size, jitter).
        **kwargs
            Common kwargs passed to both plots (e.g., palette, dodge).

        Returns
        -------
        tuple[Figure, Axes, DataFrame]

        Examples
        --------
        ```pycon
        >>> import tempfile, shutil
        >>> from pathlib import Path
        >>> import pandas as pd
        >>> from py3r.behaviour.util.docdata import data_path
        >>> from py3r.behaviour.tracking.tracking_collection import TrackingCollection
        >>> from py3r.behaviour.features.features_collection import FeaturesCollection
        >>> from py3r.behaviour.summary.summary_collection import SummaryCollection
        >>> with tempfile.TemporaryDirectory() as d:
        ...     d = Path(d)
        ...     with data_path('py3r.behaviour.tracking._data', 'dlc_single.csv') as p:
        ...         _ = shutil.copy(p, d / 'A.csv'); _ = shutil.copy(p, d / 'B.csv')
        ...     tc = TrackingCollection.from_dlc({'A': str(d/'A.csv'), 'B': str(d/'B.csv')}, fps=30)
        >>> fc = FeaturesCollection.from_tracking_collection(tc)
        >>> for f in fc.values():
        ...     n = len(f.tracking.data)
        ...     states = pd.Series((['A', 'B', 'A'] * (n // 3 + 1))[:n],
        ...                        index=f.tracking.data.index)
        ...     f.store(states, 'zone', meta={})
        >>> sc = SummaryCollection.from_features_collection(fc)
        >>> fig, ax, df = sc.snssuperplot(sc.each.time_in_state('zone'), show=False)
        >>> isinstance(df, pd.DataFrame)
        True

        ```
        """
        import seaborn as sns

        spec = self.prepare_plot(
            metric,
            group_order=group_order,
            sort_by=sort_by,
            ax=ax,
            figsize=kwargs.pop("figsize", None),
        )

        # Bar plot (mean + error bars) — base layer
        bar_defaults = {"errorbar": None, "capsize": 0.1, "alpha": 0.7, "zorder": 1}
        bar_kw = {**spec.sns_kwargs, **bar_defaults, **(bar_kwargs or {}), **kwargs}
        sns.barplot(**bar_kw, legend=False)

        # Strip plot (individual dots) — overlay
        strip_defaults = {"alpha": 0.8, "jitter": True, "size": 4, "zorder": 2}
        strip_kw = {**spec.sns_kwargs, **strip_defaults, **(strip_kwargs or {}), **kwargs}
        sns.stripplot(**strip_kw)

        # Build seaborn params dict for statannotations passthrough
        sns_kw = {
            k: v
            for k, v in spec.sns_kwargs.items()
            if k in ("data", "x", "y", "hue", "order", "hue_order")
        }

        # Apply statistical annotations (before save/show)
        self._apply_annotations(spec.ax, spec.df, annotate, sns_kw)

        self._sns_post_plot(
            spec.fig,
            spec.ax,
            metric_name=spec.metric_name,
            title=title,
            ylabel=ylabel or spec.ylabel,
            filename_prefix=spec.filename_prefix,
            n_components=spec.n_components,
            n_groups=spec.n_groups,
            hide_legend=spec.hide_legend,
            legend_n_entries=spec.n_groups,
            savedir=savedir,
            filename=filename,
            default_suffix="superplot",
            show=show,
            created_fig=spec.created_fig,
        )

        return spec.fig, spec.ax, spec.df
