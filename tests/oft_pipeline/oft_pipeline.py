# %% [markdown]
# # Open Field Test (OFT) — Full analysis pipeline example

# %% [markdown]
# ## Setup

# %%
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

import py3r.behaviour as p3b

try:
    from IPython.display import display
except ImportError:

    def display(x):
        print(x)


# Skip heavy visualisation deps (pycirclize, umap-learn) in CI
SKIP_HEAVY_VIZ = os.environ.get("CI", "").lower() in ("true", "1", "yes")

# Paths
DATA_DIR = Path("data/tracking")
TAGS_CSV = Path("data/tags.csv")

OUT_DIR = Path(os.environ.get("NB_OUT_DIR", Path.cwd() / "_artifacts"))
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Constants
FPS = 30
N_CLUSTERS = 25

# %% [markdown]
# ## Load & Preprocess

# %% [markdown]
# ### Load tracking data
#
# Load a `TrackingCollection` from a folder of DeepLabCut CSV files.
# Each CSV becomes one `Tracking` object keyed by its filename stem.
# The provided `fps` is written into each leaf's metadata for downstream methods.
# Return type here is `TrackingCollection`.
# Alternative loaders with the same pattern: `from_yolo3r_folder`, `from_dlcma_folder`.

# %%
tc = p3b.TrackingCollection.from_dlc_folder(
    folder_path=DATA_DIR,
    fps=FPS,
)
print(tc)
# Main object types in `py3r.behaviour` implement `.copy()`.
# We'll keep an untouched copy for didactic examples in this notebook.
tc_raw_for_demo = tc.copy()

# %%
# norender
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
# ### Add experimental tags
#
# A tags CSV maps recording handles to experimental metadata.
# It must contain a `handle` column matching filename stems;
# every other column becomes a tag key–value pair.
# `tags_info()` is a quick schema check: coverage and cardinality per tag.
# `add_tags_from_csv(...)` mutates each `Tracking` in-place and returns `None`.

# %%
tc.add_tags_from_csv(csv_path=TAGS_CSV)
tc.tags_info()
# %% [markdown]
# ### Didactic: batch processing
#
# With a `TrackingCollection`, `.each` delegates calls to each `Tracking`.
# Think "batch call the same `Tracking` method for all recordings".
# This `.each` batch processing pattern also applies to `FeaturesCollection`
# and `SummaryCollection`, as we will see later.
#
# Methods on `Tracking` are `inplace=True` by default, so `.each` returns a
# `BatchResult`. If `inplace=False`, `.each` returns a `TrackingCollection`.
#
# Passing a `BatchResult` back into `.each` maps values by handle.

# %%
demo_inplace = tc_raw_for_demo.copy().each.filter_likelihood(threshold=0.9)
demo_new_collection = tc_raw_for_demo.copy().each.filter_likelihood(
    threshold=0.9,
    inplace=False,
)
print(type(demo_inplace).__name__)  # expected: BatchResult
print(type(demo_new_collection).__name__)  # expected: TrackingCollection

# %% [markdown]
# ### Preprocess
#
# Standard preprocessing chain: remove low-confidence detections,
# interpolate short gaps, smooth trajectories, and rescale coordinates
# to real-world units.
# This order is intentional: filter -> interpolate -> smooth -> rescale.
#
# In this main path we use in-place behavior (typical analysis workflow).
# Equivalent non-in-place variants are shown above in the didactic batch section.

# %%
tc.each.filter_likelihood(threshold=0.9)
tc.each.interpolate(limit=5)
tc.each.smooth_all(window=3, method="mean")
tc.each.rescale_by_known_distance(
    point1="tl",
    point2="br",
    distance_in_metres=0.64,
)

# %% [markdown]
# ### Re-running preprocessing
#
# Most preprocessing methods guard against re-application. For parameter tuning,
# prefer `inplace=False` and work on a copy.

# %%
try:
    tc.each.interpolate(limit=5)
except Exception as e:
    print(e)


# %% [markdown]
# ### Quality check — trajectory plots
#
# Save trajectory plots for every recording and display one inline for QC.
# Pattern used here:
# - batch save all (`tc.each.plot(..., savedir=...)`)
# - inspect one representative recording inline (`tc[0].plot(...)`)

# %%
trajectories = ["bodycentre"]
static = ["tl", "tr", "bl", "br"]
lines = [("tr", "tl"), ("tl", "bl"), ("bl", "br"), ("br", "tr")]

tc.each.plot(trajectories=trajectories, static=static, lines=lines, show=False, savedir=OUT_DIR)

# Single inline plot for visual QC
tc[0].plot(trajectories=trajectories, static=static, lines=lines, show=True)

# %%
# norender
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
H1 = "OFT1_1"
H2 = "OFT1_10"
GOLDEN_COORDS = {
    H1: {
        (10, "bodycentre.x"): 0.5627381131700598,
        (10, "bodycentre.y"): 0.2774778306233837,
        (50, "bodycentre.x"): 0.700039705397047,
        (50, "bodycentre.y"): 0.41430397632776833,
    },
    H2: {
        (10, "bodycentre.x"): 0.6185524853964406,
        (10, "bodycentre.y"): 0.24807079148985997,
        (50, "bodycentre.x"): 0.7445399009882802,
        (50, "bodycentre.y"): 0.24343088410270214,
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
# ## Compute Features

# %% [markdown]
# ### Create FeaturesCollection
#
# A `FeaturesCollection` wraps every recording's tracking data with
# methods for computing time-series features.
# Most feature methods return `FeaturesResult`; call `.store()` to persist
# to `Features.data` and register metadata in `Features.meta`.

# %%
fc = p3b.FeaturesCollection.from_tracking_collection(tc)

# %% [markdown]
# ### Spatial features — boundaries
#
# Define/store named boundaries on each `Features` leaf, then use either:
# - mapped `BatchResult` boundary objects (smart per-handle passthrough), or
# - boundary names (resolved from stored per-recording assets).
# Here we use both and assert they match.

# %%
ordered_oft_corners = ["tl", "tr", "br", "bl"]

# %% [markdown]
# Define and store a center boundary for each recording.

# %%
center_boundary = fc.each.define_static_boundary(
    ordered_oft_corners,
    scale_dim1=0.5,
    scale_dim2=0.5,
    name="center",
)

# %% [markdown]
# Compare boundary usage styles: pass boundary objects vs stored boundary names.

# %%
in_center = fc.each.within_boundary(point="bodycentre", boundary=center_boundary)
in_center_by_name = fc.each.within_boundary(point="bodycentre", boundary="center")
for handle in fc.keys():
    assert in_center[handle].equals(in_center_by_name[handle])

# %% [markdown]
# Store the result. Without a manual name, an automatic descriptive name is used.
# `.store` always returns the stored name

# %%
in_center.store()

# %% [markdown]
# `BatchResult` supports logical composition (for example, arena periphery).

# %%
_ = fc.each.define_static_boundary(
    ordered_oft_corners,
    scale_dim1=0.8,
    scale_dim2=0.8,
    name="not_periphery",
)
_ = fc.each.define_static_boundary(
    ordered_oft_corners,
    name="oft",
)
(
    fc.each.within_boundary("bodycentre", "oft")
    & (~fc.each.within_boundary("bodycentre", "not_periphery"))
).store("in_periphery")

# %% [markdown]
# Corner occupancy can be represented as a single state feature instead of many
# independent booleans.

# %%
in_corners = dict()
for c in ordered_oft_corners:
    _ = fc.each.define_static_boundary(
        ordered_oft_corners,
        scale_dim1=0.2,
        scale_dim2=0.2,
        name=f"{c}_corner",
        anchor=c,
    )
    in_corners[c] = fc.each.within_boundary("bodycentre", boundary=f"{c}_corner")

# %%
# Store a convenience boolean for "in any corner".
(in_corners["tl"] | in_corners["tr"] | in_corners["bl"] | in_corners["br"]).store("in_corner")

# %%
# Store a categorical corner-state feature for state-based analyses.
fc.each.compose_state_from_booleans(in_corners).store("corner_state")

# %%
# Keep these existing columns out of clustering feature selection.
non_bfa_feats = fc[0].data.columns

# %% [markdown]
# `BatchResult` also supports element-wise arithmetic across handles.

# %%
dist_change = fc.each.distance_change("bodycentre")
dist_change_in_center = in_center.astype("Int64") * dist_change
dist_change_in_center.store(name="dist_change_bodycentre_in_center")

# %%
# `BatchResult` also supports general binary operations.
fast_outside_center = ~in_center & ((fc.each.speed("bodycentre") * 100) > 10.0)
# This is an example only; we do not store it.

# %% [markdown]
# ### Kinematic features for BFA
#
# Speeds, angle deviations, inter-keypoint distances, body-part areas,
# and distance to the arena boundary — the standard feature set for
# behavioural flow analysis clustering.
# The loop pattern below intentionally stores each feature as a named column,
# so later clustering/summary code can reference columns deterministically.
#
# Specific choices used here:
# - For kinematic polygons, we define named dynamic boundaries, then compute
#   dynamic area (`median=False`) over their ordered points.
# - For arena distance, we define named static boundaries and loop over points.

# %%
# Speeds
for pt in ["nose", "neck", "earr", "earl", "bodycentre", "hipl", "hipr", "tailbase"]:
    fc.each.speed(pt).store()

# %% [markdown]
# Compute angular features.

# %%
# Angle deviations
for basepoint, pointdirection1, pointdirection2 in [
    ("tailbase", "hipr", "hipl"),
    ("bodycentre", "tailbase", "neck"),
    ("neck", "bodycentre", "headcentre"),
    ("headcentre", "earr", "earl"),
]:
    fc.each.azimuth_deviation(basepoint, pointdirection1, pointdirection2).store()

# %% [markdown]
# Compute inter-keypoint distances.

# %%
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
    fc.each.distance_between(p1, p2).store()

# %% [markdown]
# Define dynamic body boundaries and store per-boundary area features.

# %%
DYNAMIC_BODY_BOUNDARIES = [
    ("mouse_rear", ["tailbase", "hipr", "hipl"]),
    ("mouse_mid", ["hipr", "hipl", "bcl", "bcr"]),
    ("mouse_front", ["bcr", "earr", "earl", "bcl"]),
    ("mouse_face", ["earr", "nose", "earl"]),
]

for boundary_name, boundary_points in DYNAMIC_BODY_BOUNDARIES:
    fc.each.define_dynamic_boundary(boundary_points, name=boundary_name)
    fc.each.area_of_boundary(boundary_name).store()

# %% [markdown]
# Compute distance-to-boundary features for selected points.

# %%
STATIC_DISTANCE_TO_BOUNDARY_POINTS = ["nose", "neck", "bodycentre", "tailbase"]

for pt in STATIC_DISTANCE_TO_BOUNDARY_POINTS:
    fc.each.distance_to_boundary(pt, "oft").store()

# %% [markdown]
# Inspect stored boundary assets on one recording.

# %%
fc[0].list_boundaries()

# %% [markdown]
# ### K-means clustering
#
# Embed the feature time-series with temporal offsets, then cluster
# the embedded space with k-means.
# Returns `(cluster_labels, centroids, scaling_factors)`, where:
# - `cluster_labels` is a per-handle `BatchResult` of label series
# - `centroids` is a DataFrame with `n_clusters` rows
#
# Option notes:
# - `offset` controls temporal context window.
# - `cluster_embedding` also supports weighting/normalization knobs for advanced runs.

# %%
cluster_features = list(set(fc[0].data.columns) - set(non_bfa_feats))
offset = list(np.arange(-15, 16, 1))
embedding_dict = {f: offset for f in cluster_features}

cluster_labels, centroids, _ = fc.cluster_embedding_stream(
    embedding_dict=embedding_dict, n_clusters=N_CLUSTERS
)
cluster_labels.store("kmeans_25", overwrite=True)

# %% [markdown]
# ### Save features to disk
#
# `save()` writes a collection manifest plus per-handle element folders.
# This makes downstream loading deterministic and auditable.
# Later you can reconstruct with `p3b.FeaturesCollection.load(path)`.

# %%
fc.save(f"{OUT_DIR}/features", data_format="csv", overwrite=True)

# %%
# norender
# --- Center boundary golden values ---
H1 = "OFT1_1"
H2 = "OFT1_10"
GOLDEN_IN_CENTER = {H1: 11, H2: 0}
for handle, expected in GOLDEN_IN_CENTER.items():
    if handle in fc:
        got = int(in_center[handle].sum())
        assert got == expected, f"{handle} in_center: {got} != {expected}"

# --- Feature columns exist ---
first_handle = list(fc.keys())[0]
cols = fc[first_handle].data.columns.tolist()
btab = fc[first_handle].list_boundaries()
assert {"center", "oft"} <= set(btab.index), f"Missing stored boundaries on {first_handle}"
assert (btab.loc[["center", "oft"], "kind"] == "static").all()

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
    H1: 13.430283546187345,
    H2: 7.708798105663596,
}
for handle, expected_sum in GOLDEN_SPEED_SUM.items():
    if handle in fc:
        got_sum = float(fc[handle].data["speed_of_bodycentre_in_xy"].sum())
        assert np.isclose(got_sum, expected_sum, rtol=1e-5), (
            f"{handle} speed sum: {got_sum} != {expected_sum}"
        )

GOLDEN_FRAME_VALUES = {
    H1: {
        (10, "speed_of_bodycentre_in_xy"): 0.053635888585183436,
        (50, "speed_of_bodycentre_in_xy"): 0.029256103508145354,
        (10, "distance_between_nose_and_headcentre_in_xy"): 0.01306663873926128,
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
# ## Summarise

# %% [markdown]
# ### Create SummaryCollection
#
# Each `Summary` object holds scalar (or Series) metrics computed from
# a single recording's features.
# Return type here is `SummaryCollection`.

# %%
sc = p3b.SummaryCollection.from_features_collection(fc)

# %% [markdown]
# ### Compute summary measures
#
# Call summary methods and `.store()` the result to persist it.
# Stored summary metrics become scalar columns in each `Summary.data` record.
#
# Same pattern as features:
# - compute result (`sc.total_distance(...)`, etc.)
# - then `.store(...)` to persist by metric name.

# %%
sc.each.total_distance("bodycentre").store()
sc.each.time_true("within_boundary_static_bodycentre_in_center").store("time_in_center")
sc.each.sum_column("dist_change_bodycentre_in_center").store(name="distance_moved_in_center")

# by_state API example: average speed by composed spatial zone.
sc.each.by_state(
    "corner_state",
    all_states=ordered_oft_corners,
).mean_column("speed_of_bodycentre_in_xy").store("mean_speed_corners")

# by_state + all_states API example: force explicit cluster domain (0-9),
# including states absent in a recording.
sc.each.by_state("kmeans_25", all_states=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]).mean_column(
    "speed_of_bodycentre_in_xy"
).store("mean_speed_bodycentre_by_kmeans_25")

# %% [markdown]
# ### Export results to CSV
#
# `to_df(include_tags=True)` flattens summary metrics + selected tag columns
# into one analysis-ready table (indexed by handle).
# By default, series metrics, like time_in_state, are ignored (`series="ignore"`).
# If `series="separate"` then each series metric will be output as its own df over the collection.

# %%
summary_df, series_dfs = sc.to_df(include_tags=True, series="separate")
summary_df.to_csv(f"{OUT_DIR}/OFT_results.csv")

display(summary_df.head())
for key, val in series_dfs.items():
    print(key)
    display(val.head())
# %%
# norender
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

# --- Stored by_state metrics ---
for handle in sc.keys():
    zone_speed = sc[handle].data["mean_speed_corners"]
    cluster_speed = sc[handle].data["mean_speed_bodycentre_by_kmeans_25"]

    assert isinstance(zone_speed, pd.Series), (
        f"{handle}: mean_speed_bodycentre_by_zone should be a Series, got {type(zone_speed)}"
    )
    assert isinstance(cluster_speed, pd.Series), (
        f"{handle}: mean_speed_bodycentre_by_cluster_0_9 should be a Series, "
        f"got {type(cluster_speed)}"
    )
    assert sc[handle].meta["mean_speed_corners"].get("_ylabel") == "Speed (m/s)", (
        f"{handle}: mean_speed_corners should preserve speed ylabel"
    )
    assert sc[handle].meta["mean_speed_bodycentre_by_kmeans_25"].get("_ylabel") == "Speed (m/s)", (
        f"{handle}: mean_speed_bodycentre_by_kmeans_25 should preserve speed ylabel"
    )

    expected_zone_states = {"tl", "tr", "br", "bl", "none"}
    assert set(zone_speed.index.astype(str)).issubset(expected_zone_states), (
        f"{handle}: unexpected zone states {set(zone_speed.index.astype(str))}"
    )

# --- Golden summary values ---
H1 = "OFT1_1"
H2 = "OFT1_10"
H3 = "OFT1_11"
GOLDEN_SUMMARY = {
    H1: {
        "total_distance_bodycentre": 0.44767611820624487,
        "time_in_center": 0.36666666666666664,
        "distance_moved_in_center": 0.07746912799395118,
    },
    H2: {
        "total_distance_bodycentre": 0.25695993685545326,
        "time_in_center": 0.0,
        "distance_moved_in_center": 0.0,
    },
    H3: {
        "total_distance_bodycentre": 0.16675110816606706,
        "time_in_center": 0.0,
        "distance_moved_in_center": 0.0,
    },
}
for handle, expected_vals in GOLDEN_SUMMARY.items():
    if handle in sc:
        for metric, expected in expected_vals.items():
            got = float(sc[handle].data[metric])
            assert np.isclose(got, expected, rtol=1e-5), f"{handle} {metric}: {got} != {expected}"

# --- Sanity checks ---
for handle in sc.keys():
    s = sc[handle].data
    assert s["time_in_center"] >= 0
    assert s["total_distance_bodycentre"] >= 0
    assert s["distance_moved_in_center"] >= 0 or np.isnan(s["distance_moved_in_center"])

    zone_state = fc[handle].data["corner_state"]
    assert zone_state.notna().all(), f"{handle}: zone_state contains NaN"
    zone_counts = zone_state.value_counts()
    assert int(zone_counts.sum()) == len(zone_state), f"{handle}: zone_state count mismatch"

    # by_state over zone_state should include "tl" and match center occupancy in seconds.
    zone_speed = s["mean_speed_corners"]
    assert "tl" in zone_speed.index, f"{handle}: tl missing in zone by_state index"

    center_time = s["time_in_center"]
    center_frames = int(fc[handle].data["within_boundary_static_bodycentre_in_center"].sum())
    assert np.isclose(center_time, center_frames / FPS), (
        f"{handle}: time_in_center does not match center boolean frames"
    )

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
# ## Visualise
#
# The `sns*` methods on `SummaryCollection` wrap seaborn categorical plots
# with sensible defaults — auto titles, y-labels, filenames, and colour
# palettes.
# All `sns*` helpers return `(fig, ax, tidy_df)` to support both quick plotting
# and explicit downstream checks/customization.
#
# In practice:
# - pass a stored metric name (`"total_distance_bodycentre"`) for reuse
# - or pass a live `SummaryResult` for one-off plotting.

# %% [markdown]
# ### Plot types compared (ungrouped)
#
# Three views of the same metric — `total_distance_bodycentre` — to
# compare what each plot type looks like.
#
# Also available: `snsbox`, `snsviolin`, `snspoint`, `snsswarm`.

# %%
sc.each.time_in_state("kmeans_25").store("time_in_cluster")
fig, ax, df_strip = sc.snsstrip(
    "time_in_cluster",
    random_state=42,  # optional, for point jitter
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
    random_state=42,  # optional, for point jitter
    show=True,
    savedir=OUT_DIR,
)

# %% [markdown]
# ### Single Summary delegation
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
# ### Grouped plots
#
# Group by experimental tags with `groupby()` to compare conditions directly.
# Use `group_order` to control x-axis arrangement.
# `groupby(...)` returns a grouped `SummaryCollection` with the same plotting API.

# %%
sc_grouped = sc.groupby(tags=["treatment", "timepoint"])

# Keys = tag names (must match groupby tags), values = desired display order
GROUP_ORDER = {"treatment": ["control", "stressor"], "timepoint": ["pre", "post"]}

# %%
# Scalar metric — grouped superplot
fig, ax, df_gsup = sc_grouped.snssuperplot(
    "total_distance_bodycentre",
    group_order=GROUP_ORDER,
    random_state=42,  # optional, for point jitter
    show=True,
    savedir=str(OUT_DIR),
)

# %%
# Multi-component metric — 25 clusters × 4 groups
fig, ax, df_gbar = sc_grouped.snsbar(
    sc_grouped.each.time_in_state("kmeans_25"),
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
# ### sort_by — independent spatial ordering
#
# `sort_by` overrides the spatial arrangement on the x-axis without changing
# color assignment. Here `groupby(tags=["treatment", "timepoint"])` means
# treatment drives the base color (control=blue, stressor=orange). Adding
# `sort_by="timepoint"` interleaves control/stressor within each timepoint.

# %%
# Interleaved superplot — timepoint as primary spatial axis, colours by treatment
fig, ax, df_interleaved = sc_grouped.snssuperplot(
    "total_distance_bodycentre",
    group_order=GROUP_ORDER,
    sort_by="timepoint",
    random_state=42,  # optional, for point jitter
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
# ### Statistical annotations
#
# Use `annotate="help"` to discover available tests, corrections, and the
# group labels in your data. Then pass `annotate={...}` with actual pairs.

# %%
# Discover labels and options (no annotation applied, just prints a guide)
fig_ann, ax_ann, df_ann = sc_grouped.snssuperplot(
    "total_distance_bodycentre",
    group_order=GROUP_ORDER,
    annotate="help",
    random_state=42,  # optional, for point jitter
    show=False,
)

# %%
# Apply annotations
fig_ann, ax_ann, df_ann = sc_grouped.snsbox(
    "total_distance_bodycentre",
    group_order=GROUP_ORDER,
    annotate={
        "pairs": [("control, pre", "stressor, pre"), ("control, post", "stressor, post")],
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
# ### Metric input options
#
# Two ways to pass a metric to any `sns*` method:
#
# 1. **String key** — a previously stored metric name
# 2. **SummaryResult** object — inline computation (not stored)
# Both options can represent single- or multi-component metrics.

# %%
# 1. String key
fig, ax, _ = sc.snsstrip(
    "total_distance_bodycentre",
    random_state=42,  # optional, for point jitter
    show=False,
)

# 2. SummaryResult object (inline)
fig, ax, df_mc = sc.snsbar(
    sc.each.time_in_state("within_boundary_static_bodycentre_in_center"),
    show=False,
)

# %% [markdown]
# ### Multi-metric plotting
#
# `sns*` methods can accept multiple metrics via list input, or alias maps via dict input.
# `merge_by` controls how metrics are combined (default: `"metric"`).
# When plotting multiple metrics together, they must share a common y-axis label.

# %%


# Ungrouped multi-metric demo combining two by_state metrics with the same y-axis
# (mean speed of bodycentre):
# - composed spatial zones (corners + center + outer)
# - kmeans clusters with explicit all_states=[0..9]
fig, ax, df_multi_flat = sc.snsbar(
    {
        "corners": "mean_speed_corners",
        "kmeans_0_to_9": "mean_speed_bodycentre_by_kmeans_25",
    },
    show=True,
    savedir=OUT_DIR,
    filename="demo_multi_metric_by_state_speed_barplot.png",
)

# %%
# Grouped multi-metric demo
fig, ax, df_multi_grouped = sc_grouped.snsbar(
    ["time_in_center", "time_in_cluster"],
    merge_by=None,
    group_order=GROUP_ORDER,
    show=True,
    savedir=OUT_DIR,
    filename="demo_multi_metric_grouped_barplot.png",
)

# %%
# norender
# --- Flat tidy DataFrame structure ---
for label, df_check in [
    ("strip", df_strip),
    ("bar", df_bar),
    ("super", df_super),
]:
    cols = set(df_check.columns)
    assert {"component", "_handle"} <= cols, f"{label}: missing required id columns"
    y_cols = [c for c in df_check.columns if c not in {"component", "_handle", "_group"}]
    assert len(y_cols) >= 1, f"{label}: missing y-value column(s)"
    assert len(df_check) > 0, f"{label}: empty DataFrame"

# --- Single Summary delegation ---
cols_single = set(df_single.columns)
assert {"component", "_handle"} <= cols_single
y_cols_single = [c for c in df_single.columns if c not in {"component", "_handle", "_group"}]
assert len(y_cols_single) >= 1

# --- Grouped tidy DataFrame structure ---
for label, df_check in [("gsup", df_gsup), ("gbar", df_gbar)]:
    cols = set(df_check.columns)
    assert {"component", "_handle", "_group"} <= cols, f"{label}: missing required columns"
    y_cols = [c for c in df_check.columns if c not in {"component", "_handle", "_group"}]
    assert len(y_cols) >= 1, f"{label}: missing y-value column(s)"
    assert df_check["_group"].nunique() > 1, f"{label}: expected multiple groups"
    assert len(df_check) > 0, f"{label}: empty DataFrame"

# Multi-component grouped bar: 25 clusters × 4 groups
assert df_gbar["component"].nunique() == N_CLUSTERS, (
    f"Expected {N_CLUSTERS} components, got {df_gbar['component'].nunique()}"
)

# --- Multi-metric demos ---
# Flat case uses explicit aliases in the dict input:
# - "corners" (state-composed speed means)
# - "cluster_0_9" (kmeans speed means with explicit all_states)
components_multi_flat = set(df_multi_flat["component"].astype(str).tolist())
assert any(c.startswith("corners") for c in components_multi_flat)

# Grouped case should still return one grouped tidy DataFrame via standard path.
assert "_group" in df_multi_grouped.columns
components_multi_grouped = set(df_multi_grouped["component"].astype(str).tolist())
assert any("time_in_cluster" in c for c in components_multi_grouped)
assert "time_in_center" in components_multi_grouped

# --- Auto-named plot files ---
assert len(list(OUT_DIR.glob("*stripplot.png"))) >= 1
assert len(list(OUT_DIR.glob("*superplot.png"))) >= 1
assert len(list(OUT_DIR.glob("*barplot.png"))) >= 1

print("Visualisation tests passed.")

# %% [markdown]
# ## Behaviour Flow Analysis (BFA)

# %% [markdown]
# ### Compute BFA results and statistics
#
# `bfa()` returns a nested dict of observed/shuffled transition statistics,
# and `bfa_stats()` derives effect-size-style summaries for reporting.
# `all_states=np.arange(0, N_CLUSTERS)` makes the state space explicit.

# %%
bfa_results = sc_grouped.bfa(
    column="kmeans_25",
    all_states=np.arange(0, N_CLUSTERS),
    random_state=42,
)
bfa_stats = p3b.SummaryCollection.bfa_stats(bfa_results)

with open(f"{OUT_DIR}/bfa_results.json", "w") as f:
    json.dump(bfa_results, f, indent=4)
with open(f"{OUT_DIR}/bfa_stats.json", "w") as f:
    json.dump(bfa_stats, f, indent=4)

# %% [markdown]
# ### BFA histograms
#
# Distribution of shuffled transition values vs observed, per group comparison.
# Useful as a quick sanity check before interpreting chord/UMAP views.

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
# ### Chord diagrams
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
# ### UMAP embedding of transition matrices
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
print("\n" + "=" * 60)
print("ALL OFT PIPELINE TESTS PASSED")
if SKIP_HEAVY_VIZ:
    print("(Skipped: chord diagrams, UMAP — requires pycirclize/umap-learn)")
print("=" * 60)
print(f"\nOutputs saved to: {OUT_DIR}")

# %%
