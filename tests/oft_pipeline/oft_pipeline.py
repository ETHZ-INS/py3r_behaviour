# %% [markdown]
# specify whether we're running in test mode or not

# %%
import os  # noqa: E402

TEST_MODE = True
# Skip heavy visualization dependencies (pycirclize, umap) in CI environments
SKIP_HEAVY_VIZ = os.environ.get("CI") == "true"

# %% [markdown]
# set (local) paths

# %%
import json  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

# Inputs (bundled with the notebook)
if TEST_MODE:
    DATA_DIR = Path("data/tracking")
    TAGS_CSV = Path("data/tags.csv")
else:
    raise NotImplementedError("Example dataset artifact not yet bundled with package")

# Outputs (isolated, overridable)
OUT_DIR = Path(os.environ.get("NB_OUT_DIR", Path.cwd() / "_artifacts"))
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Recording parameters
FPS = 25


# %% [markdown]
# load a tracking collection from a folder (dlc format)

# %%
import py3r.behaviour as p3b  # noqa: E402

tc = p3b.TrackingCollection.from_dlc_folder(
    folder_path=DATA_DIR,
    fps=FPS,
    aspectratio_correction=0.75,
)
print(tc)

# %%
if TEST_MODE:
    # Expected handles detected from files on disk
    expected_handles = {p.stem for p in Path(DATA_DIR).glob("*.csv")}
    assert len(expected_handles) > 0, "No CSVs found in DATA_DIR"

    # Collection-level invariants
    assert set(tc.keys()) == expected_handles
    assert len(tc) == len(expected_handles)

    # Element-level invariants
    for handle, t in tc.items():
        # key/handle consistency
        assert t.handle == handle
        # fps metadata
        assert "fps" in t.meta and float(t.meta["fps"]) == FPS
        # dataframe basics
        df = t.data
        assert isinstance(df, pd.DataFrame)
        assert df.index.name == "frame"
        assert df.index.is_monotonic_increasing
        assert np.issubdtype(df.index.dtype, np.integer)
        # minimal schema: at least one x/y point
        cols = df.columns.astype(str)
        assert any(c.endswith(".x") for c in cols), f"{handle} missing .x columns"
        assert any(c.endswith(".y") for c in cols), f"{handle} missing .y columns"
        # likelihood columns (if present) should be finite and in [0,1]
        like_cols = [c for c in cols if c.endswith(".likelihood")]
        if like_cols:
            lk = df[like_cols].to_numpy()
            assert np.isfinite(lk).all()
            assert (lk >= 0).all() and (lk <= 1).all()

    print("TrackingCollection loading tests passed.")

# %% [markdown]
# Add tags from a CSV for grouping/analysis.
#
# CSV must contain a 'handle' column matching filenames (without extension)
# other column names are the tag names, and those column values are the tag values.
# See example file `tags.csv`

# %%
tc.add_tags_from_csv(csv_path=TAGS_CSV)
print(tc.tags_info())

# %%
if TEST_MODE:
    # Validate tags loaded from CSV are attached to each matching handle
    tags_df = pd.read_csv(TAGS_CSV)
    present = tags_df[tags_df["handle"].isin(list(tc.keys()))]
    assert not present.empty, "tags.csv has no handles matching loaded data"

    required_cols = [c for c in tags_df.columns if c != "handle"]
    for _, row in present.iterrows():
        h = row["handle"]
        tags = tc[h].tags
        for col in required_cols:
            assert col in tags, f"missing tag {col} for {h}"
            assert str(tags[col]) == str(row[col]), (
                f"tag {col} mismatch for {h}: {tags[col]} != {row[col]}"
            )

    # Check expected tags exist
    assert "treatment" in required_cols, "Expected 'treatment' tag column"
    assert "timepoint" in required_cols, "Expected 'timepoint' tag column"

    # Category set sanity checks
    expected_treatments = set(present["treatment"].unique().tolist())
    got_treatments = set(tc[h].tags.get("treatment") for h in present["handle"])
    assert got_treatments == expected_treatments

    print("Tags loading tests passed.")

# %% [markdown]
# basic preprocessing

# %%
# Remove low-confidence detections (thresholds depend on your tracking software/data)
tc.filter_likelihood(threshold=0.5)

# Interpolate missing points before smoothing
tc.interpolate(limit=5)

# Smooth all points with mean centre window 3
tc.smooth_all(window=3, method="mean")

# Rescale distance to metres according to two corners of the OFT, here named 'tl' and 'br'
tc.rescale_by_known_distance(point1="tl", point2="br", distance_in_metres=0.64)

# %%
if TEST_MODE:
    # Meta and structural invariants after preprocessing
    for handle, t in tc.items():
        meta = t.meta
        # fps preserved
        assert "fps" in meta and float(meta["fps"]) == FPS
        # preprocessing recorded
        assert "interpolation" in meta, f"missing interpolation meta for {handle}"
        assert "smoothing" in meta, f"missing smoothing meta for {handle}"
        # units set by rescale_by_known_distance
        assert meta.get("distance_units") == "m"

        # coordinates present and finite (at least some should be valid after preprocessing)
        df = t.data
        coord_cols = [c for c in df.columns if str(c).endswith(".x") or str(c).endswith(".y")]
        assert len(coord_cols) >= 2, f"no coordinate columns found for {handle}"

    # Golden checks for preprocessed coordinate values
    HANDLE_1 = "USSOFT1_1DeepCut_resnet50_Blockcourse1May9shuffle1_1030000"
    HANDLE_2 = "USSOFT2_11DeepCut_resnet50_Blockcourse1May9shuffle1_1030000"

    GOLDEN_COORDS = {
        HANDLE_1: {
            (10, "bodycentre.x"): 0.42322453052345843,
            (10, "bodycentre.y"): 0.47276440015430876,
            (50, "bodycentre.x"): 0.4075528138290211,
            (50, "bodycentre.y"): 0.46311350683685765,
        },
        HANDLE_2: {
            (10, "bodycentre.x"): 0.6458406837205221,
            (10, "bodycentre.y"): 0.6093473239614059,
        },
    }

    for handle, coords in GOLDEN_COORDS.items():
        if handle in tc:
            for (frame, col), expected in coords.items():
                got = float(tc[handle].data.loc[frame, col])
                assert np.isclose(got, expected, rtol=1e-5), (
                    f"{handle} {col} at frame {frame}: got {got}, expected {expected}"
                )

    print("Preprocessing tests passed.")

# %% [markdown]
# basic plots

# %%
# Plot trajectories (per recording, using 'bodycentre'
# for trajectory of mouse and corners of OFT as static frame)
trajectories = ["bodycentre"]
static = ["tl", "tr", "bl", "br"]
lines = [("tr", "tl"), ("tl", "bl"), ("bl", "br"), ("br", "tr")]

tc.plot(
    trajectories=trajectories,
    static=static,
    lines=lines,
    show=False,
    savedir=OUT_DIR,
)

# single inline display plot for QC:
tc[0].plot(trajectories=trajectories, static=static, lines=lines, show=True)

# %%
if TEST_MODE:
    # Plots saved by the earlier tc.plot call should exist in OUT_DIR
    has_plot = any(
        p.suffix.lower() in {".png", ".jpg", ".jpeg", ".svg"} for p in Path(OUT_DIR).glob("*")
    )
    assert has_plot, f"No plot artifacts found in {OUT_DIR}"

    print("Plotting tests passed.")

# %% [markdown]
# create a `FeaturesCollection` object from the `TrackingCollection`

# %%
fc = p3b.FeaturesCollection.from_tracking_collection(tc)

# %%
# FeaturesCollection creation checks
if TEST_MODE:
    from py3r.behaviour.features.features import Features

    # Keys and sizes mirror TrackingCollection
    assert set(fc.keys()) == set(tc.keys())
    assert len(fc) == len(tc)

    for handle, f in fc.items():
        # Type and handle
        assert isinstance(f, Features), f"{handle} is not a Features instance"
        assert f.handle == handle
        # Tags propagated from tracking
        assert f.tags == tc[handle].tags
        # Tracking carried through and aligned
        assert hasattr(f, "tracking")
        assert len(f.tracking.data) == len(tc[handle].data)
        assert f.tracking.handle == handle
        # DataFrame exists (may be empty right after creation)
        assert hasattr(f, "data")
        assert hasattr(f, "meta")

    print("FeaturesCollection creation tests passed.")

# %% [markdown]
# compute basic OFT features (time in center)

# %%
# Define boundary of center area and check if mouse
# (defined by 'bodycentre') is inside defined boundary
center_boundary = fc.define_boundary(["tl", "tr", "bl", "br"], scaling=0.5)
in_center = fc.within_boundary_static(
    point="bodycentre", boundary=center_boundary, boundary_name="center"
)
in_center.store()

# Distance change for center calculations
dist_change = fc.distance_change("bodycentre")
dist_change_in_center = in_center.astype("Int64") * dist_change
dist_change_in_center.store(name="dist_change_bodycentre_in_center")

# %%
if TEST_MODE:
    # Check center boundary feature was computed and stored
    stored_cols = [
        "within_boundary_static_bodycentre_in_center",
        "dist_change_bodycentre_in_center",
    ]
    for handle in fc.keys():
        for col in stored_cols:
            assert col in fc[handle].data.columns, f"{handle}: missing column {col}"

    # BatchResult structure checks
    assert set(in_center.keys()) == set(fc.keys())
    n_frames = len(tc[list(tc.keys())[0]].data)
    for handle in fc.keys():
        assert len(in_center[handle]) == n_frames, f"{handle}: in_center length mismatch"

    # Golden checks for center boundary feature
    HANDLE_1 = "USSOFT1_1DeepCut_resnet50_Blockcourse1May9shuffle1_1030000"
    HANDLE_2 = "USSOFT2_11DeepCut_resnet50_Blockcourse1May9shuffle1_1030000"

    GOLDEN_IN_CENTER_COUNT = {
        HANDLE_1: 1,
        HANDLE_2: 5,
    }

    for handle, expected_count in GOLDEN_IN_CENTER_COUNT.items():
        if handle in fc:
            got_count = int(in_center[handle].sum())
            assert got_count == expected_count, (
                f"{handle} in_center count: got {got_count}, expected {expected_count}"
            )

    print("Basic OFT features (center) tests passed.")

# %% [markdown]
# compute features for BFA clustering

# %%
# Speed of different keypoints
fc.speed("nose").store()
fc.speed("neck").store()
fc.speed("earr").store()
fc.speed("earl").store()
fc.speed("bodycentre").store()
fc.speed("hipl").store()
fc.speed("hipr").store()
fc.speed("tailbase").store()

# Angle deviations
fc.azimuth_deviation("tailbase", "hipr", "hipl").store()
fc.azimuth_deviation("bodycentre", "tailbase", "neck").store()
fc.azimuth_deviation("neck", "bodycentre", "headcentre").store()
fc.azimuth_deviation("headcentre", "earr", "earl").store()

# Distance between two keypoints
fc.distance_between("nose", "headcentre").store()
fc.distance_between("neck", "headcentre").store()
fc.distance_between("neck", "bodycentre").store()
fc.distance_between("bcr", "bodycentre").store()
fc.distance_between("bcl", "bodycentre").store()
fc.distance_between("tailbase", "bodycentre").store()
fc.distance_between("tailbase", "hipr").store()
fc.distance_between("tailbase", "hipl").store()
fc.distance_between("bcr", "hipr").store()
fc.distance_between("bcl", "hipl").store()
fc.distance_between("bcl", "earl").store()
fc.distance_between("bcr", "earr").store()
fc.distance_between("nose", "earr").store()
fc.distance_between("nose", "earl").store()

# Area spanned by three or four keypoints
fc.area_of_boundary(["tailbase", "hipr", "hipl"], median=False).store()
fc.area_of_boundary(["hipr", "hipl", "bcl", "bcr"], median=False).store()
fc.area_of_boundary(["bcr", "earr", "earl", "bcl"], median=False).store()
fc.area_of_boundary(["earr", "nose", "earl"], median=False).store()

# Distance to OFT boundary
bdry = fc.define_boundary(["tl", "tr", "br", "bl"], scaling=1.0)
fc.distance_to_boundary_static("nose", bdry, boundary_name="oft").store()
fc.distance_to_boundary_static("neck", bdry, boundary_name="oft").store()
fc.distance_to_boundary_static("bodycentre", bdry, boundary_name="oft").store()
fc.distance_to_boundary_static("tailbase", bdry, boundary_name="oft").store()

# %%
if TEST_MODE:
    # Check that BFA-related features were computed and stored
    # Naming convention: speed_of_{point}_in_xy
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
    # Naming convention: azimuth_deviation_{basepoint}_to_{point1}_and_{point2}
    expected_azimuth_cols = [
        "azimuth_deviation_tailbase_to_hipr_and_hipl",
        "azimuth_deviation_bodycentre_to_tailbase_and_neck",
        "azimuth_deviation_neck_to_bodycentre_and_headcentre",
        "azimuth_deviation_headcentre_to_earr_and_earl",
    ]
    # Naming convention: distance_between_{point1}_and_{point2}_in_xy
    expected_distance_cols = [
        "distance_between_nose_and_headcentre_in_xy",
        "distance_between_neck_and_headcentre_in_xy",
        "distance_between_neck_and_bodycentre_in_xy",
    ]
    # Naming convention: distance_to_boundary_static_{point}_in_{boundary_name}
    expected_boundary_cols = [
        "distance_to_boundary_static_nose_in_oft",
        "distance_to_boundary_static_neck_in_oft",
        "distance_to_boundary_static_bodycentre_in_oft",
        "distance_to_boundary_static_tailbase_in_oft",
    ]

    first_handle = list(fc.keys())[0]
    cols = fc[first_handle].data.columns.tolist()

    for col in expected_speed_cols:
        assert col in cols, f"Missing speed column: {col}"
    for col in expected_azimuth_cols:
        assert col in cols, f"Missing azimuth column: {col}"
    for col in expected_distance_cols:
        assert col in cols, f"Missing distance column: {col}"
    for col in expected_boundary_cols:
        assert col in cols, f"Missing boundary distance column: {col}"

    # Check area columns exist (naming may vary)
    area_cols = [c for c in cols if "area_of_boundary" in c]
    assert len(area_cols) >= 4, f"Expected at least 4 area columns, got {len(area_cols)}"

    # Golden checks for specific feature values
    HANDLE_1 = "USSOFT1_1DeepCut_resnet50_Blockcourse1May9shuffle1_1030000"
    HANDLE_2 = "USSOFT2_11DeepCut_resnet50_Blockcourse1May9shuffle1_1030000"

    # Sum of bodycentre speed across all frames
    GOLDEN_SPEED_SUM = {
        HANDLE_1: 8.59331427489955,
        HANDLE_2: 53.870505758473946,
    }
    for handle, expected_sum in GOLDEN_SPEED_SUM.items():
        if handle in fc:
            got_sum = float(fc[handle].data["speed_of_bodycentre_in_xy"].sum())
            assert np.isclose(got_sum, expected_sum, rtol=1e-5), (
                f"{handle} speed sum: got {got_sum}, expected {expected_sum}"
            )

    # Specific frame values
    GOLDEN_FRAME_VALUES = {
        HANDLE_1: {
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
                    f"{handle} {col} at frame {frame}: got {got}, expected {expected}"
                )

    print("BFA feature computation tests passed.")

# %% [markdown]
# save features to csv

# %%
fc.save(f"{OUT_DIR}/features", data_format="csv", overwrite=True)

# %%
if TEST_MODE:
    features_dir = Path(OUT_DIR) / "features"
    assert features_dir.exists(), f"Features directory not created: {features_dir}"

    # Check manifest.json exists and contains correct handles
    manifest_path = features_dir / "manifest.json"
    assert manifest_path.exists(), f"manifest.json not found in {features_dir}"

    with open(manifest_path) as f:
        manifest = json.load(f)

    assert "elements_index" in manifest, "manifest.json missing 'elements_index'"
    saved_handles = set(manifest["elements_index"].keys())
    expected_handles = set(fc.keys())
    assert saved_handles == expected_handles, (
        f"Handles mismatch: saved {len(saved_handles)}, expected {len(expected_handles)}"
    )

    # Check that element folders exist for each handle
    elements_dir = features_dir / "elements"
    assert elements_dir.exists(), f"elements/ directory not found in {features_dir}"

    for handle in fc.keys():
        handle_dir = elements_dir / handle
        assert handle_dir.exists(), f"Missing element folder for handle: {handle}"
        assert (handle_dir / "data.csv").exists(), f"Missing data.csv for handle: {handle}"

    print("Feature saving tests passed.")

# %% [markdown]
# create dictionary for feature embedding and perform k-means clustering

# %%
# Create dictionary for feature embedding
features = fc[0].data.columns
offset = list(np.arange(-15, 16, 1))
embedding_dict = {f: offset for f in features}

# Cluster the embedded feature space using k-means clustering
# The keyword n_clusters defines the number of clusters used.
N_CLUSTERS = 25
cluster_labels, centroids, _ = fc.cluster_embedding(
    embedding_dict=embedding_dict, n_clusters=N_CLUSTERS
)
cluster_labels.store("kmeans_25", overwrite=True)

# %%
if TEST_MODE:
    # Check clustering results
    assert set(cluster_labels.keys()) == set(fc.keys()), "Cluster labels keys mismatch"

    for handle in fc.keys():
        labels = cluster_labels[handle]
        # Check that cluster labels are stored
        assert "kmeans_25" in fc[handle].data.columns, f"{handle}: kmeans_25 not stored"

        # Check that labels are within expected range (0 to N_CLUSTERS-1, plus potential NaN)
        valid_labels = labels.dropna()
        if len(valid_labels) > 0:
            assert valid_labels.min() >= 0, f"{handle}: negative cluster label"
            assert valid_labels.max() < N_CLUSTERS, f"{handle}: cluster label >= N_CLUSTERS"

    # Check centroids shape
    assert centroids.shape[0] == N_CLUSTERS, (
        f"Expected {N_CLUSTERS} centroids, got {centroids.shape[1]}"
    )

    print("Clustering tests passed.")

# %% [markdown]
# create a `SummaryCollection` object from the `FeaturesCollection`

# %%
sc = p3b.SummaryCollection.from_features_collection(fc)

# %%
if TEST_MODE:
    from py3r.behaviour.summary.summary import Summary

    # Keys and sizes mirror FeaturesCollection
    assert set(sc.keys()) == set(fc.keys())
    assert len(sc) == len(fc)

    for handle, s in sc.items():
        assert isinstance(s, Summary), f"{handle} is not a Summary instance"
        assert s.handle == handle
        assert s.tags == fc[handle].tags

    print("SummaryCollection creation tests passed.")

# %% [markdown]
# compute summary measures per recording

# %%
# Total distance moved
sc.total_distance("bodycentre").store()

# Time in center
sc.time_true("within_boundary_static_bodycentre_in_center").store("time_in_center")

# Distance moved in center
sc.sum_column("dist_change_bodycentre_in_center").store(name="distance_moved_in_center")

# %%
if TEST_MODE:
    summary_stored = [
        "total_distance_bodycentre",
        "time_in_center",
        "distance_moved_in_center",
    ]
    for handle in sc.keys():
        for name in summary_stored:
            assert name in sc[handle].data, f"{handle}: missing summary '{name}'"
            # values are scalars (number or bool)
            val = sc[handle].data[name]
            assert isinstance(val, (int, float, bool, np.integer, np.floating)), (
                f"{handle}: '{name}' should be scalar, got {type(val)}"
            )

    # Sanity checks
    for handle in sc.keys():
        s = sc[handle].data
        assert s["time_in_center"] >= 0, f"{handle}: time_in_center < 0"
        assert s["distance_moved_in_center"] >= 0 or np.isnan(s["distance_moved_in_center"]), (
            f"{handle}: distance_moved_in_center < 0"
        )
        assert s["total_distance_bodycentre"] >= 0, f"{handle}: total_distance_bodycentre < 0"

    # Golden checks for summary values
    HANDLE_1 = "USSOFT1_1DeepCut_resnet50_Blockcourse1May9shuffle1_1030000"
    HANDLE_2 = "USSOFT2_11DeepCut_resnet50_Blockcourse1May9shuffle1_1030000"
    HANDLE_3 = "USSOFT1_8DeepCut_resnet50_Blockcourse1May9shuffle1_1030000"

    GOLDEN_SUMMARY = {
        HANDLE_1: {
            "total_distance_bodycentre": 0.3437325709959821,
            "time_in_center": 0.04,
            "distance_moved_in_center": 0.023400103796424813,
        },
        HANDLE_2: {
            "total_distance_bodycentre": 2.154820230338958,
            "time_in_center": 0.2,
            "distance_moved_in_center": 0.383824548159979,
        },
        HANDLE_3: {
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
                    f"{handle} {metric}: got {got}, expected {expected}"
                )

    print("Summary computation tests passed.")

# %% [markdown]
# collate scalar outputs into DataFrame and save results in CSV

# %%
summary_df = sc.to_df(include_tags=True)
summary_df.to_csv(f"{OUT_DIR}/OFT_results.csv")
print(summary_df)

# %%
if TEST_MODE:
    # summary_df structure checks
    assert isinstance(summary_df, pd.DataFrame)
    assert summary_df.index.name == "handle"
    assert len(summary_df) == len(sc)
    assert set(summary_df.index) == set(sc.keys())

    for col in [
        "total_distance_bodycentre",
        "time_in_center",
        "distance_moved_in_center",
    ]:
        assert col in summary_df.columns, f"summary_df missing column '{col}'"

    # CSV written and round-trip check
    csv_path = Path(OUT_DIR) / "OFT_results.csv"
    assert csv_path.exists(), f"Expected CSV at {csv_path}"
    loaded = pd.read_csv(csv_path, index_col=0)
    assert len(loaded) == len(summary_df)

    print("Summary DataFrame and CSV tests passed.")

# %% [markdown]
# seaborn plotting wrappers
#
# Demonstrates the sns* plotting API on flat and grouped SummaryCollections.
# All plots use auto-generated titles, ylabels, and filenames.

# %%
import matplotlib  # noqa: E402

matplotlib.use("Agg")  # non-interactive backend for test mode

# --- 1. Single Summary: delegation to SummaryCollection ---
# A single Summary delegates plotting to a 1-item SummaryCollection.
# Auto-filename is prefixed with the recording handle.
single_summary = sc[list(sc.keys())[0]]
fig_single, ax_single, df_single = single_summary.snsbar(
    single_summary.time_in_state("within_boundary_static_bodycentre_in_center"),
    show=False,
    savedir=OUT_DIR,
)
print(f"Single Summary snsbar: {len(df_single)} rows")

# --- 2. Flat SC strip (stored scalar metric, string key) ---
# Uses a previously stored metric by name. Auto ylabel comes from stored meta.
fig_strip, ax_strip, df_strip = sc.snsstrip(
    "total_distance_bodycentre",
    show=False,
    savedir=OUT_DIR,
)
print(f"snsstrip scalar: {len(df_strip)} rows")

# --- 3. Flat SC superplot (multi-component SummaryResult) ---
# Passing a SummaryResult directly; auto ylabel from _ylabel attribute.
fig_super, ax_super, df_super = sc.snssuperplot(
    sc.time_in_state("within_boundary_static_bodycentre_in_center"),
    show=False,
    savedir=OUT_DIR,
)
print(f"snssuperplot time_in_state: {len(df_super)} rows")

# %%
if TEST_MODE:
    # Tidy DataFrame structure checks (flat)
    for label, df_check in [("single", df_single), ("strip", df_strip), ("super", df_super)]:
        assert {"component", "value", "_handle"} <= set(df_check.columns), (
            f"{label}: missing required columns"
        )
        assert len(df_check) > 0, f"{label}: tidy DataFrame is empty"

    # Auto-generated files should exist (auto naming = no explicit filename)
    flat_pngs = list(Path(OUT_DIR).glob("*stripplot.png")) + list(
        Path(OUT_DIR).glob("*superplot.png")
    )
    assert len(flat_pngs) >= 2, f"Expected at least 2 auto-named plot PNGs, got {len(flat_pngs)}"

    # Single-summary file should be prefixed with the handle slug
    single_pngs = list(Path(OUT_DIR).glob("*barplot.png"))
    assert len(single_pngs) >= 1, "Single Summary barplot not saved"

    print("Flat seaborn plotting tests passed.")

# %% [markdown]
# group by tags for grouped analyses and plotting

# %%
sc_grouped = sc.groupby(tags=["treatment", "timepoint"])

# group_order controls display ordering on plots:
# keys = tag names (matching groupby tags), values = desired value order
GROUP_ORDER = {"treatment": ["control", "FST"], "timepoint": ["45m", "1d"]}

# --- 4. Grouped SC superplot (scalar stored metric) ---
# Scalar metric with group_order; auto palette, auto ylabel from stored meta.
fig_gsup, ax_gsup, df_gsup = sc_grouped.snssuperplot(
    "total_distance_bodycentre",
    group_order=GROUP_ORDER,
    show=False,
    savedir=str(OUT_DIR),
)
print(f"Grouped superplot: {len(df_gsup)} rows, groups: {list(df_gsup['_group'].unique())}")

# --- 5. Grouped SC bar (multi-component SummaryResult + group_order) ---
# 25 clusters × 4 groups, component-major ordering with groups dodged per cluster.
fig_gbar, ax_gbar, df_gbar = sc_grouped.snsbar(
    sc_grouped.time_in_state("kmeans_25"),
    group_order=GROUP_ORDER,
    show=False,
    savedir=str(OUT_DIR),
)
print(
    f"Grouped bar kmeans: {len(df_gbar)} rows, "
    f"{df_gbar['component'].nunique()} components, "
    f"{df_gbar['_group'].nunique()} groups"
)

# %%
if TEST_MODE:
    # Tidy DataFrame structure checks (grouped)
    for label, df_check in [("gsup", df_gsup), ("gbar", df_gbar)]:
        assert {"component", "value", "_handle", "_group"} <= set(df_check.columns), (
            f"{label}: missing required columns"
        )
        assert df_check["_group"].nunique() > 1, f"{label}: expected multiple groups"
        assert len(df_check) > 0, f"{label}: tidy DataFrame is empty"

    # Multi-component grouped bar should have 25 components × 4 groups
    assert df_gbar["component"].nunique() == N_CLUSTERS, (
        f"Expected {N_CLUSTERS} components, got {df_gbar['component'].nunique()}"
    )

    # Auto-generated grouped plot files should exist
    grouped_pngs = list(Path(OUT_DIR).glob("*superplot.png")) + list(
        Path(OUT_DIR).glob("*barplot.png")
    )
    assert len(grouped_pngs) >= 3, (
        f"Expected at least 3 auto-named plot PNGs (1 flat + 2 grouped), got {len(grouped_pngs)}"
    )

    print("Grouped seaborn plotting tests passed.")

# %% [markdown]
# perform behavior flow analysis (BFA)

# %%
# Perform behavior flow analysis on clustering results
bfa_results = sc_grouped.bfa(column="kmeans_25", all_states=np.arange(0, N_CLUSTERS))
print("BFA Results:")
print(bfa_results)

# Save BFA results
with open(f"{OUT_DIR}/bfa_results.json", "w") as f:
    json.dump(bfa_results, f, indent=4)

# %%
if TEST_MODE:
    # Check BFA results structure
    assert isinstance(bfa_results, dict), "BFA results should be a dict"

    # Check JSON was saved
    bfa_json_path = Path(OUT_DIR) / "bfa_results.json"
    assert bfa_json_path.exists(), f"BFA results JSON not saved at {bfa_json_path}"

    # Load and verify JSON
    with open(bfa_json_path) as f:
        loaded_bfa = json.load(f)
    assert isinstance(loaded_bfa, dict), "Loaded BFA results should be a dict"

    print("BFA results tests passed.")

# %% [markdown]
# compute BFA statistics

# %%
# Compute the statistics and save the results
bfa_stats = p3b.SummaryCollection.bfa_stats(bfa_results)
print("BFA Statistics:")
print(bfa_stats)

with open(f"{OUT_DIR}/bfa_stats.json", "w") as f:
    json.dump(bfa_stats, f, indent=4)

# %%
if TEST_MODE:
    # Check BFA stats structure
    assert isinstance(bfa_stats, dict), "BFA stats should be a dict"

    # Check JSON was saved
    bfa_stats_path = Path(OUT_DIR) / "bfa_stats.json"
    assert bfa_stats_path.exists(), f"BFA stats JSON not saved at {bfa_stats_path}"

    print("BFA stats tests passed.")

# %% [markdown]
# plot BFA chord diagrams (requires pycirclize - skipped in CI)

# %%
if not SKIP_HEAVY_VIZ:
    sc_grouped.plot_chord(
        column="kmeans_25",
        all_states=np.arange(0, N_CLUSTERS),
        save_dir=OUT_DIR,
        show=False,
        start=-265,
        end=95,
        space=5,
        r_lim=(93, 100),
        label_kws=dict(r=94, size=12, color="white"),
        link_kws=dict(ec="black", lw=0.5),
    )

# %%
if TEST_MODE and not SKIP_HEAVY_VIZ:
    # Check that chord plots were saved
    chord_plots = list(Path(OUT_DIR).glob("*chord*")) + list(Path(OUT_DIR).glob("*bfa*"))
    # The plot may have different naming conventions, so just check output dir has more files
    has_new_plots = any(
        p.suffix.lower() in {".png", ".jpg", ".jpeg", ".svg", ".pdf"}
        for p in Path(OUT_DIR).glob("*")
    )
    assert has_new_plots, f"No plot artifacts found in {OUT_DIR} after BFA chord plotting"

    print("BFA chord plot tests passed.")

# %% [markdown]
# plot BFA histograms

# %%
# Plot BFA result histograms showing distribution of shuffled values vs observed
p3b.SummaryCollection.plot_bfa_results(
    bfa_results,
    add_stats=True,
    stats=bfa_stats,
    bins=20,
    figsize=(4, 3),
    save_dir=OUT_DIR,
    show=False,
)

# %%
if TEST_MODE:
    # Check that histogram plots were saved (one per comparison)
    # Files are named after comparisons, e.g., "control_1d_vs_FST_1d.png"
    plot_files_before = len(list(Path(OUT_DIR).glob("*.png")))
    assert plot_files_before > 0, "No PNG files found after BFA histogram plotting"

    print("BFA histogram tests passed.")

# %% [markdown]
# plot UMAP embeddings of transition matrices (requires umap-learn - skipped in CI)

# %%
if not SKIP_HEAVY_VIZ:
    # Plot UMAP embedding of per-subject transition matrices
    # Groups can be specified as a list of group names or as sequential groups for gradient coloring
    fig, ax = sc_grouped.plot_transition_umap(
        column="kmeans_25",
        all_states=np.arange(0, N_CLUSTERS),
        n_neighbors=15,
        min_dist=0.1,
        random_state=42,
        figsize=(6, 5),
        show=False,
        save_dir=OUT_DIR,
    )

# %%
if TEST_MODE and not SKIP_HEAVY_VIZ:
    # Check that UMAP plot was saved
    umap_path = Path(OUT_DIR) / "transition_umap.png"
    assert umap_path.exists(), f"UMAP plot not saved at {umap_path}"

    print("UMAP embedding tests passed.")

# %% [markdown]
# final summary

# %%
if TEST_MODE:
    print("\n" + "=" * 60)
    print("ALL OFT PIPELINE TESTS PASSED SUCCESSFULLY!")
    if SKIP_HEAVY_VIZ:
        print("(Skipped: chord diagrams, UMAP - requires pycirclize/umap-learn)")
    print("=" * 60)
    print(f"\nOutputs saved to: {OUT_DIR}")
    print("  - OFT_results.csv")
    print("  - features/")
    print("  - bfa_results.json, bfa_stats.json")
    print("  - Trajectory plots")
    print("  - Seaborn summary plots (auto-named)")
    print("  - BFA histograms")
    if not SKIP_HEAVY_VIZ:
        print("  - BFA chord diagrams")
        print("  - transition_umap.png")

# %%
