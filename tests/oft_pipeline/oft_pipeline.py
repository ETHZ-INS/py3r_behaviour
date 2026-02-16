# %% [markdown]
# # Open Field Test (OFT) — Full Analysis Pipeline
#
# This notebook demonstrates a complete OFT analysis workflow using
# `py3r.behaviour`, from raw DeepLabCut tracking CSVs through to
# publication-ready figures and behaviour flow analysis (BFA).
#
# **Sections**
#
# 1. Setup
# 2. Load & Preprocess
# 3. Compute Features
# 4. Summarise
# 5. Visualise
# 6. Behaviour Flow Analysis

# %% [markdown]
# ## 1. Setup

# %%
TEST_MODE = True

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

import py3r.behaviour as p3b

# Skip heavy visualisation deps (pycirclize, umap-learn) in CI
SKIP_HEAVY_VIZ = os.environ.get("CI", "").lower() in ("true", "1", "yes")

# Paths — point these at your own data outside test mode
if TEST_MODE:
    DATA_DIR = Path("data/tracking")
    TAGS_CSV = Path("data/tags.csv")
else:
    raise NotImplementedError("Example dataset artifact not yet bundled with package")

OUT_DIR = Path(os.environ.get("NB_OUT_DIR", Path.cwd() / "_artifacts"))
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Constants
FPS = 25
N_CLUSTERS = 25

# %% [markdown]
# ## 2. Load & Preprocess

# %% [markdown]
# ### 2.1 Load tracking data
#
# Load a `TrackingCollection` from a folder of DeepLabCut CSV files.
# Each CSV becomes one `Tracking` object keyed by its filename stem.

# %%
tc = p3b.TrackingCollection.from_dlc_folder(
    folder_path=DATA_DIR,
    fps=FPS,
    aspectratio_correction=0.75,
)
print(tc)

# %%
# norender
if TEST_MODE:
    # Structural checks on raw data (before preprocessing alters values)
    expected_handles = {p.stem for p in Path(DATA_DIR).glob("*.csv")}
    assert len(expected_handles) > 0, "No CSVs found in DATA_DIR"
    assert set(tc.keys()) == expected_handles
    assert len(tc) == len(expected_handles)

    for handle, t in tc.items():
        assert t.handle == handle
        assert "fps" in t.meta and float(t.meta["fps"]) == FPS
        df = t.data
        assert isinstance(df, pd.DataFrame)
        assert df.index.name == "frame"
        assert df.index.is_monotonic_increasing
        assert np.issubdtype(df.index.dtype, np.integer)
        cols = df.columns.astype(str)
        assert any(c.endswith(".x") for c in cols), f"{handle} missing .x columns"
        assert any(c.endswith(".y") for c in cols), f"{handle} missing .y columns"
        like_cols = [c for c in cols if c.endswith(".likelihood")]
        if like_cols:
            lk = df[like_cols].to_numpy()
            assert np.isfinite(lk).all()
            assert (lk >= 0).all() and (lk <= 1).all()

    print("Loading tests passed.")

# %% [markdown]
# ### 2.2 Add experimental tags
#
# A tags CSV maps recording handles to experimental metadata.
# It must contain a `handle` column matching filenames (without extension);
# every other column becomes a tag key–value pair.

# %%
tc.add_tags_from_csv(csv_path=TAGS_CSV)
tc.tags_info()

# %% [markdown]
# ### 2.3 Preprocess
#
# Standard preprocessing chain: remove low-confidence detections,
# interpolate short gaps, smooth trajectories, and rescale coordinates
# to real-world units.

# %%
tc.filter_likelihood(threshold=0.5)
tc.interpolate(limit=5)
tc.smooth_all(window=3, method="mean")
tc.rescale_by_known_distance(point1="tl", point2="br", distance_in_metres=0.64)
# %% [markdown]
# ### 2.4 Quality check — trajectory plots
#
# Save trajectory plots for every recording and display one inline for QC.

# %%
trajectories = ["bodycentre"]
static = ["tl", "tr", "bl", "br"]
lines = [("tr", "tl"), ("tl", "bl"), ("bl", "br"), ("br", "tr")]

tc.plot(trajectories=trajectories, static=static, lines=lines, show=False, savedir=OUT_DIR)

# Single inline plot for visual QC
tc[0].plot(trajectories=trajectories, static=static, lines=lines, show=True)

# %%
# norender
if TEST_MODE:
    # --- Tag checks ---
    tags_df = pd.read_csv(TAGS_CSV)
    present = tags_df[tags_df["handle"].isin(list(tc.keys()))]
    assert not present.empty, "tags.csv has no handles matching loaded data"
    required_cols = [c for c in tags_df.columns if c != "handle"]
    assert "treatment" in required_cols
    assert "timepoint" in required_cols
    for _, row in present.iterrows():
        h = row["handle"]
        tags = tc[h].tags
        for col in required_cols:
            assert col in tags, f"missing tag {col} for {h}"
            assert str(tags[col]) == str(row[col])
    expected_treatments = set(present["treatment"].unique().tolist())
    got_treatments = {tc[h].tags.get("treatment") for h in present["handle"]}
    assert got_treatments == expected_treatments

    # --- Preprocessing metadata ---
    for handle, t in tc.items():
        meta = t.meta
        assert float(meta["fps"]) == FPS
        assert "interpolation" in meta, f"{handle}: missing interpolation meta"
        assert "smoothing" in meta, f"{handle}: missing smoothing meta"
        assert meta.get("distance_units") == "m"

    # --- Golden coordinate values (post-preprocessing) ---
    H1 = "USSOFT1_1DeepCut_resnet50_Blockcourse1May9shuffle1_1030000"
    H2 = "USSOFT2_11DeepCut_resnet50_Blockcourse1May9shuffle1_1030000"
    GOLDEN_COORDS = {
        H1: {
            (10, "bodycentre.x"): 0.42322453052345843,
            (10, "bodycentre.y"): 0.47276440015430876,
            (50, "bodycentre.x"): 0.4075528138290211,
            (50, "bodycentre.y"): 0.46311350683685765,
        },
        H2: {
            (10, "bodycentre.x"): 0.6458406837205221,
            (10, "bodycentre.y"): 0.6093473239614059,
        },
    }
    for handle, coords in GOLDEN_COORDS.items():
        if handle in tc:
            for (frame, col), expected in coords.items():
                got = float(tc[handle].data.loc[frame, col])
                assert np.isclose(got, expected, rtol=1e-5), (
                    f"{handle} {col}@{frame}: {got} != {expected}"
                )

    # --- Trajectory plot files ---
    has_plot = any(p.suffix.lower() in {".png", ".jpg", ".svg"} for p in OUT_DIR.glob("*"))
    assert has_plot, f"No plot files in {OUT_DIR}"

    print("Tags, preprocessing, and trajectory plot tests passed.")

# %% [markdown]
# ## 3. Compute Features

# %% [markdown]
# ### 3.1 Create FeaturesCollection
#
# A `FeaturesCollection` wraps every recording's tracking data with
# methods for computing time-series features.

# %%
fc = p3b.FeaturesCollection.from_tracking_collection(tc)

# %% [markdown]
# ### 3.2 Spatial features — center zone
#
# Define the center boundary from the arena corners and detect
# when the mouse (bodycentre) is inside it.

# %%
center_boundary = fc.define_boundary(["tl", "tr", "bl", "br"], scaling=0.5)
in_center = fc.within_boundary_static(
    point="bodycentre", boundary=center_boundary, boundary_name="center"
)
in_center.store()

dist_change = fc.distance_change("bodycentre")
dist_change_in_center = in_center.astype("Int64") * dist_change
dist_change_in_center.store(name="dist_change_bodycentre_in_center")

# %% [markdown]
# ### 3.3 Kinematic features for BFA
#
# Speeds, angle deviations, inter-keypoint distances, body-part areas,
# and distance to the arena boundary — the standard feature set for
# behavioural flow analysis clustering.

# %%
# Speeds
for pt in ["nose", "neck", "earr", "earl", "bodycentre", "hipl", "hipr", "tailbase"]:
    fc.speed(pt).store()

# Angle deviations
fc.azimuth_deviation("tailbase", "hipr", "hipl").store()
fc.azimuth_deviation("bodycentre", "tailbase", "neck").store()
fc.azimuth_deviation("neck", "bodycentre", "headcentre").store()
fc.azimuth_deviation("headcentre", "earr", "earl").store()

# Inter-keypoint distances
for p1, p2 in [
    ("nose", "headcentre"),
    ("neck", "headcentre"),
    ("neck", "bodycentre"),
    ("bcr", "bodycentre"),
    ("bcl", "bodycentre"),
    ("tailbase", "bodycentre"),
    ("tailbase", "hipr"),
    ("tailbase", "hipl"),
    ("bcr", "hipr"),
    ("bcl", "hipl"),
    ("bcl", "earl"),
    ("bcr", "earr"),
    ("nose", "earr"),
    ("nose", "earl"),
]:
    fc.distance_between(p1, p2).store()

# Body-part areas
fc.area_of_boundary(["tailbase", "hipr", "hipl"], median=False).store()
fc.area_of_boundary(["hipr", "hipl", "bcl", "bcr"], median=False).store()
fc.area_of_boundary(["bcr", "earr", "earl", "bcl"], median=False).store()
fc.area_of_boundary(["earr", "nose", "earl"], median=False).store()

# Distance to arena boundary
bdry = fc.define_boundary(["tl", "tr", "br", "bl"], scaling=1.0)
for pt in ["nose", "neck", "bodycentre", "tailbase"]:
    fc.distance_to_boundary_static(pt, bdry, boundary_name="oft").store()

# %% [markdown]
# ### 3.4 K-means clustering
#
# Embed the feature time-series with temporal offsets, then cluster
# the embedded space with k-means.

# %%
features = fc[0].data.columns
offset = list(np.arange(-15, 16, 1))
embedding_dict = {f: offset for f in features}

cluster_labels, centroids, _ = fc.cluster_embedding(
    embedding_dict=embedding_dict, n_clusters=N_CLUSTERS
)
cluster_labels.store("kmeans_25", overwrite=True)

# %% [markdown]
# ### 3.5 Save features to disk

# %%
fc.save(f"{OUT_DIR}/features", data_format="csv", overwrite=True)

# %%
# norender
if TEST_MODE:
    # --- Center boundary golden values ---
    H1 = "USSOFT1_1DeepCut_resnet50_Blockcourse1May9shuffle1_1030000"
    H2 = "USSOFT2_11DeepCut_resnet50_Blockcourse1May9shuffle1_1030000"
    GOLDEN_IN_CENTER = {H1: 1, H2: 5}
    for handle, expected in GOLDEN_IN_CENTER.items():
        if handle in fc:
            got = int(in_center[handle].sum())
            assert got == expected, f"{handle} in_center: {got} != {expected}"

    # --- Feature columns exist ---
    first_handle = list(fc.keys())[0]
    cols = fc[first_handle].data.columns.tolist()
    expected_speed_cols = [
        "speed_of_nose_in_xy",
        "speed_of_neck_in_xy",
        "speed_of_earr_in_xy",
        "speed_of_earl_in_xy",
        "speed_of_bodycentre_in_xy",
        "speed_of_hipl_in_xy",
        "speed_of_hipr_in_xy",
        "speed_of_tailbase_in_xy",
    ]
    expected_azimuth_cols = [
        "azimuth_deviation_tailbase_to_hipr_and_hipl",
        "azimuth_deviation_bodycentre_to_tailbase_and_neck",
        "azimuth_deviation_neck_to_bodycentre_and_headcentre",
        "azimuth_deviation_headcentre_to_earr_and_earl",
    ]
    expected_boundary_cols = [
        "distance_to_boundary_static_nose_in_oft",
        "distance_to_boundary_static_neck_in_oft",
        "distance_to_boundary_static_bodycentre_in_oft",
        "distance_to_boundary_static_tailbase_in_oft",
    ]
    for col in expected_speed_cols + expected_azimuth_cols + expected_boundary_cols:
        assert col in cols, f"Missing column: {col}"
    distance_cols = [c for c in cols if c.startswith("distance_between_")]
    assert len(distance_cols) >= 14, f"Expected >= 14 distance columns, got {len(distance_cols)}"
    area_cols = [c for c in cols if "area_of_boundary" in c]
    assert len(area_cols) >= 4, f"Expected >= 4 area columns, got {len(area_cols)}"

    # --- Speed golden values ---
    GOLDEN_SPEED_SUM = {
        H1: 8.59331427489955,
        H2: 53.870505758473946,
    }
    for handle, expected_sum in GOLDEN_SPEED_SUM.items():
        if handle in fc:
            got_sum = float(fc[handle].data["speed_of_bodycentre_in_xy"].sum())
            assert np.isclose(got_sum, expected_sum, rtol=1e-5), (
                f"{handle} speed sum: {got_sum} != {expected_sum}"
            )

    GOLDEN_FRAME_VALUES = {
        H1: {
            (10, "speed_of_bodycentre_in_xy"): 0.0554176299795388,
            (50, "speed_of_bodycentre_in_xy"): 0.299759367213119,
            (10, "distance_between_nose_and_headcentre_in_xy"): 0.013693473948303931,
        },
    }
    for handle, frame_vals in GOLDEN_FRAME_VALUES.items():
        if handle in fc:
            for (frame, col), expected in frame_vals.items():
                got = float(fc[handle].data.loc[frame, col])
                assert np.isclose(got, expected, rtol=1e-5), (
                    f"{handle} {col}@{frame}: {got} != {expected}"
                )

    # --- Clustering ---
    assert set(cluster_labels.keys()) == set(fc.keys())
    for handle in fc.keys():
        assert "kmeans_25" in fc[handle].data.columns, f"{handle}: kmeans_25 not stored"
        valid = cluster_labels[handle].dropna()
        if len(valid) > 0:
            assert valid.min() >= 0, f"{handle}: negative cluster label"
            assert valid.max() < N_CLUSTERS, f"{handle}: label >= {N_CLUSTERS}"
    assert centroids.shape[0] == N_CLUSTERS, (
        f"Expected {N_CLUSTERS} centroids, got {centroids.shape[0]}"
    )

    # --- Save / manifest ---
    features_dir = Path(OUT_DIR) / "features"
    manifest_path = features_dir / "manifest.json"
    assert manifest_path.exists(), f"manifest.json not found in {features_dir}"
    with open(manifest_path) as f:
        manifest = json.load(f)
    assert set(manifest["elements_index"].keys()) == set(fc.keys())
    elements_dir = features_dir / "elements"
    for handle in fc.keys():
        assert (elements_dir / handle / "data.csv").exists(), f"Missing data.csv for {handle}"

    print("Feature computation, clustering, and save tests passed.")

# %% [markdown]
# ## 4. Summarise

# %% [markdown]
# ### 4.1 Create SummaryCollection
#
# Each `Summary` object holds scalar (or Series) metrics computed from
# a single recording's features.

# %%
sc = p3b.SummaryCollection.from_features_collection(fc)

# %% [markdown]
# ### 4.2 Compute summary measures
#
# Call summary methods and `.store()` the result to persist it.

# %%
sc.total_distance("bodycentre").store()
sc.time_true("within_boundary_static_bodycentre_in_center").store("time_in_center")
sc.sum_column("dist_change_bodycentre_in_center").store(name="distance_moved_in_center")

# %% [markdown]
# ### 4.3 Export results to CSV

# %%
summary_df = sc.to_df(include_tags=True)
summary_df.to_csv(f"{OUT_DIR}/OFT_results.csv")
summary_df.head()

# %%
# norender
if TEST_MODE:
    from py3r.behaviour.summary.summary import Summary

    # --- SummaryCollection structure ---
    assert set(sc.keys()) == set(fc.keys())
    assert len(sc) == len(fc)
    for handle, s in sc.items():
        assert isinstance(s, Summary), f"{handle} is not a Summary"
        assert s.handle == handle
        assert s.tags == fc[handle].tags

    # --- Stored summary metrics ---
    summary_stored = [
        "total_distance_bodycentre",
        "time_in_center",
        "distance_moved_in_center",
    ]
    for handle in sc.keys():
        for name in summary_stored:
            assert name in sc[handle].data, f"{handle}: missing '{name}'"
            val = sc[handle].data[name]
            assert isinstance(val, (int, float, bool, np.integer, np.floating)), (
                f"{handle}: '{name}' should be scalar, got {type(val)}"
            )

    # --- Golden summary values ---
    H1 = "USSOFT1_1DeepCut_resnet50_Blockcourse1May9shuffle1_1030000"
    H2 = "USSOFT2_11DeepCut_resnet50_Blockcourse1May9shuffle1_1030000"
    H3 = "USSOFT1_8DeepCut_resnet50_Blockcourse1May9shuffle1_1030000"
    GOLDEN_SUMMARY = {
        H1: {
            "total_distance_bodycentre": 0.3437325709959821,
            "time_in_center": 0.04,
            "distance_moved_in_center": 0.023400103796424813,
        },
        H2: {
            "total_distance_bodycentre": 2.154820230338958,
            "time_in_center": 0.2,
            "distance_moved_in_center": 0.383824548159979,
        },
        H3: {
            "total_distance_bodycentre": 0.7647580838539083,
            "time_in_center": 0.0,
            "distance_moved_in_center": 0.0,
        },
    }
    for handle, expected_vals in GOLDEN_SUMMARY.items():
        if handle in sc:
            for metric, expected in expected_vals.items():
                got = float(sc[handle].data[metric])
                assert np.isclose(got, expected, rtol=1e-5), (
                    f"{handle} {metric}: {got} != {expected}"
                )

    # --- Sanity checks ---
    for handle in sc.keys():
        s = sc[handle].data
        assert s["time_in_center"] >= 0
        assert s["total_distance_bodycentre"] >= 0
        assert s["distance_moved_in_center"] >= 0 or np.isnan(s["distance_moved_in_center"])

    # --- CSV round-trip ---
    csv_path = Path(OUT_DIR) / "OFT_results.csv"
    assert csv_path.exists()
    loaded = pd.read_csv(csv_path, index_col=0)
    assert len(loaded) == len(summary_df)
    for col in summary_stored:
        assert col in loaded.columns, f"CSV missing '{col}'"

    # --- summary_df structure ---
    assert isinstance(summary_df, pd.DataFrame)
    assert summary_df.index.name == "handle"
    assert set(summary_df.index) == set(sc.keys())

    print("Summary computation and CSV tests passed.")

# %% [markdown]
# ## 5. Visualise
#
# The `sns*` methods on `SummaryCollection` wrap seaborn categorical plots
# with sensible defaults — auto titles, y-labels, filenames, and colour
# palettes.

# %% [markdown]
# ### 5.1 Plot types compared (ungrouped)
#
# Three views of the same metric — `total_distance_bodycentre` — to
# compare what each plot type looks like.
#
# Also available: `snsbox`, `snsviolin`, `snspoint`, `snsswarm`.

# %%
sc.time_in_state("kmeans_25").store("time_in_cluster")
fig, ax, df_strip = sc.snsstrip(
    "time_in_cluster",
    show=True,
    savedir=OUT_DIR,
)
fig, ax, df_bar = sc.snsbar(
    "time_in_cluster",
    show=True,
    savedir=OUT_DIR,
)
fig, ax, df_super = sc.snssuperplot(
    "time_in_cluster",
    show=True,
    savedir=OUT_DIR,
)

# %% [markdown]
# ### 5.2 Single Summary delegation
#
# Individual `Summary` objects can call the same `sns*` methods.
# They delegate to a 1-item `SummaryCollection` internally.
# The auto filename is prefixed with the recording handle.

# %%
single = sc[list(sc.keys())[0]]
fig, ax, df_single = single.snsbar(
    single.time_in_state("within_boundary_static_bodycentre_in_center"),
    show=True,
    savedir=OUT_DIR,
)

# %% [markdown]
# ### 5.3 Grouped plots
#
# Group by experimental tags with `groupby()`.
# Use `group_order` to control how groups are arranged on the x-axis.

# %%
sc_grouped = sc.groupby(tags=["treatment", "timepoint"])

# Keys = tag names (must match groupby tags), values = desired display order
GROUP_ORDER = {"treatment": ["control", "FST"], "timepoint": ["45m", "1d"]}

# %%
# Scalar metric — grouped superplot
fig, ax, df_gsup = sc_grouped.snssuperplot(
    "total_distance_bodycentre",
    group_order=GROUP_ORDER,
    show=True,
    savedir=str(OUT_DIR),
)

# %%
# Multi-component metric — 25 clusters × 4 groups
fig, ax, df_gbar = sc_grouped.snsbar(
    sc_grouped.time_in_state("kmeans_25"),
    group_order=GROUP_ORDER,
    show=True,
    savedir=str(OUT_DIR),
)

# %% [markdown]
# Even though the summary metric 'time_in_cluster' was created before grouping, the grouped plots
# work as expected with this summary metric (but the auto-generated title is different, because we
# stored it with a manual name).

# %%
# Multi-component metric — 25 clusters × 4 groups
fig, ax, df_gbar = sc_grouped.snsbar(
    "time_in_cluster",
    group_order=GROUP_ORDER,
    show=True,
    savedir=str(OUT_DIR),
)

# %% [markdown]
# ### 5.3b sort_by — independent spatial ordering
#
# `sort_by` overrides the spatial arrangement on the x-axis without changing
# colour assignment.  Here `groupby(tags=["treatment", "timepoint"])` means
# treatment drives the base colour (control=blue, FST=orange).  Adding
# `sort_by="timepoint"` interleaves control/FST within each timepoint.

# %%
# Interleaved superplot — timepoint as primary spatial axis, colours by treatment
fig, ax, df_interleaved = sc_grouped.snssuperplot(
    "total_distance_bodycentre",
    group_order=GROUP_ORDER,
    sort_by="timepoint",
    show=True,
    savedir=str(OUT_DIR),
    filename="total_distance_interleaved_superplot.png",
)

# %%
# Power-user workflow with prepare_plot — full seaborn control
import seaborn as sns

spec = sc_grouped.prepare_plot(
    "total_distance_bodycentre",
    group_order=GROUP_ORDER,
    sort_by=["timepoint", "treatment"],
)
sns.boxplot(**spec.sns_kwargs, width=0.6)
spec.ax.set_ylabel(spec.ylabel)
spec.ax.set_title("Custom: prepare_plot + boxplot")
import matplotlib.pyplot as plt

plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

# %% [markdown]
# ### 5.4 Statistical annotations
#
# Use `annotate="help"` to discover available tests, corrections, and the
# group labels in your data. Then pass `annotate={...}` with actual pairs.

# %%
# Discover labels and options (no annotation applied, just prints a guide)
fig_ann, ax_ann, df_ann = sc_grouped.snssuperplot(
    "total_distance_bodycentre",
    group_order=GROUP_ORDER,
    annotate="help",
    show=False,
)

# %%
# Apply annotations
fig_ann, ax_ann, df_ann = sc_grouped.snsbox(
    "total_distance_bodycentre",
    group_order=GROUP_ORDER,
    annotate={
        "pairs": [("control, 45m", "FST, 45m"), ("control, 1d", "FST, 1d")],
        "test": "Mann-Whitney",
        "correction": None,
        "text_format": "star",
        "headroom": 0.0,  # add extra space for annotations if needed
    },
    savedir=str(OUT_DIR),
    filename="total_distance_annotated_superplot.png",
    show=True,
)

# %% [markdown]
# ### 5.5 Metric input options
#
# Two ways to pass a metric to any `sns*` method:
#
# 1. **String key** — a previously stored metric name
# 2. **SummaryResult** object — inline computation (not stored)
# (Both of these may be either single component,  or multi-component)

# %%
# 1. String key
fig, ax, _ = sc.snsstrip("total_distance_bodycentre", show=False)

# 2. SummaryResult object (inline)
fig, ax, df_mc = sc.snsbar(
    sc.time_in_state("within_boundary_static_bodycentre_in_center"),
    show=False,
)

# %%
# norender
if TEST_MODE:
    # --- Flat tidy DataFrame structure ---
    # The y-column is renamed from "value" to the ylabel by prepare_plot,
    # so we check for the structural columns only.
    for label, df_check in [
        ("strip", df_strip),
        ("bar", df_bar),
        ("super", df_super),
    ]:
        assert {"component", "_handle"} <= set(df_check.columns), (
            f"{label}: missing required columns"
        )
        assert len(df_check) > 0, f"{label}: empty DataFrame"

    # --- Single Summary delegation ---
    assert {"component", "_handle"} <= set(df_single.columns)

    # --- Grouped tidy DataFrame structure ---
    for label, df_check in [("gsup", df_gsup), ("gbar", df_gbar)]:
        assert {"component", "_handle", "_group"} <= set(df_check.columns), (
            f"{label}: missing required columns"
        )
        assert df_check["_group"].nunique() > 1, f"{label}: expected multiple groups"
        assert len(df_check) > 0, f"{label}: empty DataFrame"

    # Multi-component grouped bar: 25 clusters × 4 groups
    assert df_gbar["component"].nunique() == N_CLUSTERS, (
        f"Expected {N_CLUSTERS} components, got {df_gbar['component'].nunique()}"
    )

    # --- Auto-named plot files ---
    assert len(list(OUT_DIR.glob("*stripplot.png"))) >= 1
    assert len(list(OUT_DIR.glob("*superplot.png"))) >= 1
    assert len(list(OUT_DIR.glob("*barplot.png"))) >= 1

    print("Visualisation tests passed.")

# %% [markdown]
# ## 6. Behaviour Flow Analysis (BFA)

# %% [markdown]
# ### 6.1 Compute BFA results and statistics

# %%
bfa_results = sc_grouped.bfa(column="kmeans_25", all_states=np.arange(0, N_CLUSTERS))
bfa_stats = p3b.SummaryCollection.bfa_stats(bfa_results)

with open(f"{OUT_DIR}/bfa_results.json", "w") as f:
    json.dump(bfa_results, f, indent=4)
with open(f"{OUT_DIR}/bfa_stats.json", "w") as f:
    json.dump(bfa_stats, f, indent=4)

# %% [markdown]
# ### 6.2 BFA histograms
#
# Distribution of shuffled transition values vs observed, per group comparison.

# %%
p3b.SummaryCollection.plot_bfa_results(
    bfa_results,
    add_stats=True,
    stats=bfa_stats,
    bins=20,
    figsize=(4, 3),
    save_dir=OUT_DIR,
    show=True,
)

# %% [markdown]
# ### 6.3 Chord diagrams
#
# Requires `pycirclize` — install with `pip install py3r-behaviour[viz]`.

# %%
if not SKIP_HEAVY_VIZ:
    sc_grouped.plot_chord(
        column="kmeans_25",
        all_states=np.arange(0, N_CLUSTERS),
        save_dir=OUT_DIR,
        show=True,
        start=-265,
        end=95,
        space=5,
        r_lim=(93, 100),
        label_kws=dict(r=94, size=12, color="white"),
        link_kws=dict(ec="black", lw=0.5),
    )

# %% [markdown]
# ### 6.4 UMAP embedding of transition matrices
#
# Requires `umap-learn` — install with `pip install py3r-behaviour[viz]`.

# %%
if not SKIP_HEAVY_VIZ:
    fig, ax = sc_grouped.plot_transition_umap(
        column="kmeans_25",
        all_states=np.arange(0, N_CLUSTERS),
        n_neighbors=15,
        min_dist=0.1,
        random_state=42,
        figsize=(6, 5),
        show=True,
        save_dir=str(OUT_DIR),
    )

# %%
# norender
if TEST_MODE:
    # --- BFA results structure ---
    assert isinstance(bfa_results, dict), "BFA results should be a dict"
    assert Path(f"{OUT_DIR}/bfa_results.json").exists()
    assert isinstance(bfa_stats, dict), "BFA stats should be a dict"
    assert Path(f"{OUT_DIR}/bfa_stats.json").exists()

    # --- Heavy viz files (if run) ---
    if not SKIP_HEAVY_VIZ:
        chord_files = list(OUT_DIR.glob("*chord*")) + list(OUT_DIR.glob("*bfa*"))
        assert any(p.suffix.lower() in {".png", ".jpg", ".svg", ".pdf"} for p in chord_files), (
            "No chord plot files found"
        )
        assert Path(f"{OUT_DIR}/transition_umap.png").exists()

    print("BFA tests passed.")

# %% [markdown]
# ## Done

# %%
# norender
if TEST_MODE:
    print("\n" + "=" * 60)
    print("ALL OFT PIPELINE TESTS PASSED")
    if SKIP_HEAVY_VIZ:
        print("(Skipped: chord diagrams, UMAP — requires pycirclize/umap-learn)")
    print("=" * 60)
    print(f"\nOutputs saved to: {OUT_DIR}")
