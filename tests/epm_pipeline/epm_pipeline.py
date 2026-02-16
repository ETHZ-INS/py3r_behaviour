# %% [markdown]
# norender
# specify whether we're running in test mode or not

# %%
TEST_MODE = True

# %% [markdown]
# set (local) paths

# %%
import os  # noqa: E402
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

# %% [markdown]
# load a tracking collection from a folder (dlc format)

# %%
import py3r.behaviour as p3b  # noqa: E402

tc = p3b.TrackingCollection.from_dlc_folder(folder_path=DATA_DIR, fps=25)
print(tc)

# %%
# norender
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
        assert "fps" in t.meta and float(t.meta["fps"]) == 25.0
        # dataframe basics
        df = t.data
        assert isinstance(df, pd.DataFrame)
        assert len(df.index) == 100  # tiny test data should be 100 frames each
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

# %% [markdown]
# Add tags from a CSV for grouping/analysis.
#
# CSV must contain a 'handle' column matching filenames (without extension)
# other column names are the tag names, and those column values are the tag values.
# See example file `tags.csv`
#

# %%
tc.add_tags_from_csv(csv_path=TAGS_CSV)
print(tc.tags_info())

# %%
# norender
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

    # Category set sanity checks (example for 'treatment' if present)
    if "treatment" in required_cols:
        expected_treatments = set(present["treatment"].unique().tolist())
        got_treatments = set(tc[h].tags.get("treatment") for h in present["handle"])
        assert got_treatments == expected_treatments

# %% [markdown]
# basic preprocessing

# %%
# Remove low-confidence detections (thresholds depend on your tracking software/data)
tc.filter_likelihood(threshold=0.5)

# Interpolate missing points before smoothing
tc.interpolate(limit=5)

# Smooth all points with mean centre window 3
tc.smooth_all(window=3, method="mean")

# Rescale distance to metres according to two corners of the EPM, here named 'tl' and 'br'
tc.rescale_by_known_distance(point1="tl", point2="br", distance_in_metres=0.655)

# %%
# norender
if TEST_MODE:
    # Meta and structural invariants after preprocessing
    for handle, t in tc.items():
        meta = t.meta
        # fps preserved
        assert "fps" in meta and float(meta["fps"]) == 25.0
        # preprocessing recorded
        assert "interpolation" in meta, f"missing interpolation meta for {handle}"
        assert "smoothing" in meta, f"missing smoothing meta for {handle}"
        # units set by rescale_by_known_distance
        assert meta.get("distance_units") == "m"

        # coordinates present and finite
        df = t.data
        coord_cols = [c for c in df.columns if str(c).endswith(".x") or str(c).endswith(".y")]
        assert len(coord_cols) >= 2, f"no coordinate columns found for {handle}"

        # golden checks
        EXPECTED_TENTH_X = {
            "CRS5EPM_3DeepCut_resnet50_Official EPM Network 2020Jul16shuffle1_300000": 0.488097,
            "CRS5EPM_5DeepCut_resnet50_Official EPM Network 2020Jul16shuffle1_300000": 0.489231,
        }
        for h, expected in EXPECTED_TENTH_X.items():
            if h in tc and expected is not None:
                # choose the first available .x column consistently
                xcol = next((c for c in tc[h].data.columns if str(c).endswith(".x")), None)
                assert xcol is not None, f"no .x column for {h}"
                got = float(tc[h].data[xcol].iloc[9])
                assert np.isclose(got, float(expected), atol=1e-6), (
                    f"{h} {xcol} first value {got} != {expected}"
                )

# %% [markdown]
# basic plots

# %%
# Plot trajectories (per recording, using 'bodycentre'
# for trajectory of mouse and corners of EPM as static frame)
trajectories = ["bodycentre"]
static = ["tl", "tr", "ctr", "rt", "rb", "cbr", "br", "bl", "cbl", "lb", "lt", "ctl"]
lines = [
    ("tl", "tr"),
    ("tr", "ctr"),
    ("ctr", "rt"),
    ("rt", "rb"),
    ("rb", "cbr"),
    ("cbr", "br"),
    ("br", "bl"),
    ("bl", "cbl"),
    ("cbl", "lb"),
    ("lb", "lt"),
    ("lt", "ctl"),
    ("ctl", "tl"),
]

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
# norender
if TEST_MODE:
    # Plots saved by the earlier tc.plot call should exist in OUT_DIR
    has_plot = any(
        p.suffix.lower() in {".png", ".jpg", ".jpeg", ".svg"} for p in Path(OUT_DIR).glob("*")
    )
    assert has_plot, f"No plot artifacts found in {OUT_DIR}"

# %% [markdown]
# create a `FeaturesCollection` object from the `TrackingCollection`

# %%
fc = p3b.FeaturesCollection.from_tracking_collection(tc)

# %%
# norender
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

# %% [markdown]
# compute features

# %%
# Define different boundaries (open arms, closed arms) and check if
# mouse (defined by 'bodycentre') is inside defined boundary
# Adjust boundaries so they match orientation of your EPM.
# Open arms
_oa_boundary = fc.define_boundary(["tl", "tr", "ctr", "ctl"], scaling=1.1, centre=["ctr", "ctl"])
on_oa1 = fc.within_boundary_static(point="bodycentre", boundary=_oa_boundary)
_oa_boundary = fc.define_boundary(["cbl", "cbr", "br", "bl"], scaling=1.1, centre=["cbr", "cbl"])
on_oa2 = fc.within_boundary_static(point="bodycentre", boundary=_oa_boundary)
on_oa = on_oa1 | on_oa2
on_oa.store(name="bodycentre_on_open_arms")

dist_change_on_oa = on_oa.astype("Int64") * fc.distance_change("bodycentre")
dist_change_on_oa.store(name="dist_change_bodycentre_on_oa")

# Closed arms
_ca_boundary = fc.define_boundary(["ctr", "rt", "rb", "cbr"], scaling=1.1, centre=["ctr", "cbr"])
on_ca1 = fc.within_boundary_static(point="bodycentre", boundary=_ca_boundary)
_ca_boundary = fc.define_boundary(["lt", "ctl", "cbl", "lb"], scaling=1.1, centre=["ctl", "cbl"])
on_ca2 = fc.within_boundary_static(point="bodycentre", boundary=_ca_boundary)
# you can apply binary operators to BatchResult objects
on_ca = on_ca1 | on_ca2
on_oa.store(name="bodycentre_on_closed_arms")

dist_change_on_ca = on_ca.astype("Int64") * fc.distance_change("bodycentre")
dist_change_on_ca.store(name="dist_change_bodycentre_on_ca")

# 7) (Optional) Save features to csv
fc.save(f"{OUT_DIR}/features", data_format="csv", overwrite=True)

# %%
# norender
if TEST_MODE:
    import numpy as np
    import pandas as pd

    # 1) Stored feature names
    stored = [
        "bodycentre_on_open_arms",
        "dist_change_bodycentre_on_oa",
        "bodycentre_on_closed_arms",
        "dist_change_bodycentre_on_ca",
    ]
    for handle in fc.keys():
        for col in stored:
            assert col in fc[handle].data.columns, f"{handle}: missing column {col}"

    # 2) BatchResult structure and dtypes
    assert set(on_oa.keys()) == set(fc.keys())
    assert set(on_ca.keys()) == set(fc.keys())
    n_frames = len(tc[list(tc.keys())[0]].data)
    for handle in fc.keys():
        assert len(on_oa[handle]) == n_frames, f"{handle}: on_oa length"
        assert len(on_ca[handle]) == n_frames, f"{handle}: on_ca length"

    # 3) Readout values (fill EXPECTED_* after first run from printed output)
    # Frames in open/closed arms (count True), total distance on each arm type
    readouts = {}
    for handle in fc.keys():
        readouts[handle] = {
            "frames_on_open_arms": int(on_oa[handle].sum()),
            "frames_on_closed_arms": int(on_ca[handle].sum()),
            "dist_change_on_open_arms": float(dist_change_on_oa[handle].sum()),
            "dist_change_on_closed_arms": float(dist_change_on_ca[handle].sum()),
        }

    golden_readouts = {
        "CRS5EPM_3DeepCut_resnet50_Official EPM Network 2020Jul16shuffle1_300000": {
            "frames_on_open_arms": 0,
            "frames_on_closed_arms": 75,
            "dist_change_on_open_arms": 0.0,
            "dist_change_on_closed_arms": 0.1176029233306571,
        },
        "CRS5EPM_5DeepCut_resnet50_Official EPM Network 2020Jul16shuffle1_300000": {
            "frames_on_open_arms": 59,
            "frames_on_closed_arms": 21,
            "dist_change_on_open_arms": 0.1521520034852008,
            "dist_change_on_closed_arms": 0.02520351400470445,
        },
        "CRS5EPM_2DeepCut_resnet50_Official EPM Network 2020Jul16shuffle1_300000": {
            "frames_on_open_arms": 0,
            "frames_on_closed_arms": 98,
            "dist_change_on_open_arms": 0.0,
            "dist_change_on_closed_arms": 0.10859539481613446,
        },
        "CRS5EPM_4DeepCut_resnet50_Official EPM Network 2020Jul16shuffle1_300000": {
            "frames_on_open_arms": 6,
            "frames_on_closed_arms": 45,
            "dist_change_on_open_arms": 0.007535566866832588,
            "dist_change_on_closed_arms": 0.11650001436309684,
        },
        "CRS5EPM_1DeepCut_resnet50_Official EPM Network 2020Jul16shuffle1_300000": {
            "frames_on_open_arms": 0,
            "frames_on_closed_arms": 98,
            "dist_change_on_open_arms": 0.0,
            "dist_change_on_closed_arms": 0.032020908105939834,
        },
        "CRS5EPM_6DeepCut_resnet50_Official EPM Network 2020Jul16shuffle1_300000": {
            "frames_on_open_arms": 88,
            "frames_on_closed_arms": 0,
            "dist_change_on_open_arms": 0.4734916975469136,
            "dist_change_on_closed_arms": 0.0,
        },
    }

    for handle in fc.keys():
        if handle in golden_readouts:
            assert (
                readouts[handle]["frames_on_open_arms"]
                == golden_readouts[handle]["frames_on_open_arms"]
            ), f"{handle} frames_on_open_arms"
            assert (
                readouts[handle]["frames_on_closed_arms"]
                == golden_readouts[handle]["frames_on_closed_arms"]
            ), f"{handle} frames_on_closed_arms"
            assert np.isclose(
                readouts[handle]["dist_change_on_open_arms"],
                golden_readouts[handle]["dist_change_on_open_arms"],
                rtol=1e-5,
            ), f"{handle} dist_change_on_open_arms"
            assert np.isclose(
                readouts[handle]["dist_change_on_closed_arms"],
                golden_readouts[handle]["dist_change_on_closed_arms"],
                rtol=1e-5,
            ), f"{handle} dist_change_on_closed_arms"

    # 4) Stored columns match BatchResult series (first handle)
    first_handle = list(fc.keys())[0]
    np.testing.assert_allclose(
        fc[first_handle].data["dist_change_bodycentre_on_oa"].values,
        dist_change_on_oa[first_handle].values,
        equal_nan=True,
    )
    np.testing.assert_allclose(
        fc[first_handle].data["dist_change_bodycentre_on_ca"].values,
        dist_change_on_ca[first_handle].values,
        equal_nan=True,
    )

    # 5) Sanity: distance-on-arms is non-negative; open+closed frames <= n_frames
    for handle in fc.keys():
        assert (dist_change_on_oa[handle] >= 0).all() | (dist_change_on_oa[handle].isna().all()), (
            f"{handle}: negative dist_change_on_oa"
        )
        assert (dist_change_on_ca[handle] >= 0).all() | (dist_change_on_ca[handle].isna().all()), (
            f"{handle}: negative dist_change_on_ca"
        )
        # Open and closed can both be False (centre); so sum of frames can be < n_frames
        assert on_oa[handle].sum() + on_ca[handle].sum() <= n_frames + 1, (
            f"{handle}: open+closed frame count sanity"
        )

    print("EPM open/closed arms and distance-on-arms tests passed.")

# %%
# 8) Create SummaryCollection object
sc = p3b.SummaryCollection.from_features_collection(fc)

# 9) Compute summary measures per recording
# Total distance moved
sc.total_distance("bodycentre").store()

# Time on open arms
sc.time_true("bodycentre_on_open_arms").store("time_on_open_arms")

# Distance moved on open arms
sc.sum_column("dist_change_bodycentre_on_oa").store(name="distance_moved_on_open_arms")

# Time on closed arms
sc.time_true("bodycentre_on_closed_arms").store("time_on_closed_arms")

# Distance moved on closed arms
sc.sum_column("dist_change_bodycentre_on_ca").store(name="distance_moved_on_closed_arms")

# 10) Collate scalar outputs into DataFrame and save results in CSV
summary_df = sc.to_df(include_tags=True)
summary_df.to_csv(f"{OUT_DIR}/EPM_results.csv")

# %%
# norender
if TEST_MODE:
    from pathlib import Path

    import numpy as np
    import pandas as pd

    # 1) SummaryCollection creation and keys
    assert set(sc.keys()) == set(fc.keys()), "sc keys should match fc"
    assert len(sc) == len(fc)

    # Stored summary column names (as in section 9)
    summary_stored = [
        "total_distance_bodycentre",
        "time_on_open_arms",
        "distance_moved_on_open_arms",
        "time_on_closed_arms",
        "distance_moved_on_closed_arms",
    ]
    for handle in sc.keys():
        for name in summary_stored:
            assert name in sc[handle].data, f"{handle}: missing summary '{name}'"
            # values are scalars (number or bool)
            val = sc[handle].data[name]
            assert isinstance(val, (int, float, bool)), (
                f"{handle}: '{name}' should be scalar, got {type(val)}"
            )

    # 2) summary_df structure
    assert isinstance(summary_df, pd.DataFrame)
    assert summary_df.index.name == "handle"
    assert len(summary_df) == len(sc)
    assert set(summary_df.index) == set(sc.keys())
    for col in summary_stored:
        assert col in summary_df.columns, f"summary_df missing column '{col}'"

    # Golden values from EPM_results.csv (fps=25)
    HANDLE_1 = "CRS5EPM_1DeepCut_resnet50_Official EPM Network 2020Jul16shuffle1_300000"
    HANDLE_2 = "CRS5EPM_2DeepCut_resnet50_Official EPM Network 2020Jul16shuffle1_300000"
    HANDLE_3 = "CRS5EPM_3DeepCut_resnet50_Official EPM Network 2020Jul16shuffle1_300000"
    HANDLE_4 = "CRS5EPM_4DeepCut_resnet50_Official EPM Network 2020Jul16shuffle1_300000"
    HANDLE_5 = "CRS5EPM_5DeepCut_resnet50_Official EPM Network 2020Jul16shuffle1_300000"
    HANDLE_6 = "CRS5EPM_6DeepCut_resnet50_Official EPM Network 2020Jul16shuffle1_300000"
    EXPECTED_TOTAL_DIST = {
        HANDLE_1: 0.032020908105939834,
        HANDLE_2: 0.10859539481613446,
        HANDLE_3: 0.16380038244236952,
        HANDLE_4: 0.26863732936398277,
        HANDLE_5: 0.2221842800357452,
        HANDLE_6: 0.5264480412838328,
    }
    EXPECTED_TIME_OPEN = {
        HANDLE_1: 0.0,
        HANDLE_2: 0.0,
        HANDLE_3: 0.0,
        HANDLE_4: 0.24,
        HANDLE_5: 2.36,
        HANDLE_6: 3.52,
    }
    EXPECTED_TIME_CLOSED = {
        HANDLE_1: 0.0,
        HANDLE_2: 0.0,
        HANDLE_3: 0.0,
        HANDLE_4: 0.24,
        HANDLE_5: 2.36,
        HANDLE_6: 3.52,
    }
    EXPECTED_DIST_OPEN = {
        HANDLE_1: 0.0,
        HANDLE_2: 0.0,
        HANDLE_3: 0.0,
        HANDLE_4: 0.007535566866832588,
        HANDLE_5: 0.1521520034852008,
        HANDLE_6: 0.4734916975469136,
    }
    EXPECTED_DIST_CLOSED = {
        HANDLE_1: 0.032020908105939834,
        HANDLE_2: 0.10859539481613446,
        HANDLE_3: 0.1176029233306571,
        HANDLE_4: 0.11650001436309684,
        HANDLE_5: 0.02520351400470445,
        HANDLE_6: 0.0,
    }
    for handle in sc.keys():
        if handle in EXPECTED_TOTAL_DIST:
            assert np.isclose(
                sc[handle].data["total_distance_bodycentre"],
                EXPECTED_TOTAL_DIST[handle],
                rtol=1e-5,
            ), f"{handle} total_distance_bodycentre"
        if handle in EXPECTED_TIME_OPEN:
            assert np.isclose(
                sc[handle].data["time_on_open_arms"],
                EXPECTED_TIME_OPEN[handle],
                rtol=1e-5,
            ), f"{handle} time_on_open_arms"
        if handle in EXPECTED_TIME_CLOSED:
            assert np.isclose(
                sc[handle].data["time_on_closed_arms"],
                EXPECTED_TIME_CLOSED[handle],
                rtol=1e-5,
            ), f"{handle} time_on_closed_arms"
        if handle in EXPECTED_DIST_OPEN:
            assert np.isclose(
                sc[handle].data["distance_moved_on_open_arms"],
                EXPECTED_DIST_OPEN[handle],
                rtol=1e-5,
            ), f"{handle} distance_moved_on_open_arms"
        if handle in EXPECTED_DIST_CLOSED:
            assert np.isclose(
                sc[handle].data["distance_moved_on_closed_arms"],
                EXPECTED_DIST_CLOSED[handle],
                rtol=1e-5,
            ), f"{handle} distance_moved_on_closed_arms"

    # 4) CSV written and round-trip
    csv_path = Path(f"{OUT_DIR}/EPM_results.csv")
    assert csv_path.exists(), f"Expected CSV at {csv_path}"
    loaded = pd.read_csv(csv_path, index_col=0)
    assert loaded.index.name == "handle" or "handle" in loaded.columns
    for col in summary_stored:
        assert col in loaded.columns, f"CSV missing column '{col}'"
    assert len(loaded) == len(summary_df)

    # 5) Sanity: times and distances non-negative
    for handle in sc.keys():
        s = sc[handle].data
        assert s["time_on_open_arms"] >= 0, f"{handle}: time_on_open_arms < 0"
        assert s["time_on_closed_arms"] >= 0, f"{handle}: time_on_closed_arms < 0"
        assert s["distance_moved_on_open_arms"] >= 0, f"{handle}: distance_moved_on_open_arms < 0"
        assert s["distance_moved_on_closed_arms"] >= 0, (
            f"{handle}: distance_moved_on_closed_arms < 0"
        )
        assert s["total_distance_bodycentre"] >= 0, f"{handle}: total_distance_bodycentre < 0"

    print("SummaryCollection + EPM results + CSV tests passed.")

# %% [markdown]
# seaborn plotting wrappers (quick smoke test)

# %%
# norender
import matplotlib  # noqa: E402

matplotlib.use("Agg")

# Flat superplot of total distance (auto title, ylabel, filename)
fig, ax, df_tidy = sc.snssuperplot(
    sc.total_distance("bodycentre"),
    show=False,
    savedir=OUT_DIR,
)
print(f"EPM snssuperplot: {len(df_tidy)} rows")

# Grouped bar of time on open arms by treatment (auto everything)
sc_grouped = sc.groupby(tags=["treatment"])
fig, ax, df_grouped = sc_grouped.snsbar(
    "time_on_open_arms",
    show=False,
    savedir=OUT_DIR,
)
print(f"EPM grouped snsbar: {len(df_grouped)} rows, groups: {list(df_grouped['_group'].unique())}")

# %%
# norender
if TEST_MODE:
    # Tidy DataFrame structure (y-column is renamed to the ylabel by prepare_plot)
    assert {"component", "_handle"} <= set(df_tidy.columns)
    assert len(df_tidy) > 0

    # Grouped DataFrame structure
    assert {"component", "_handle", "_group"} <= set(df_grouped.columns)
    assert df_grouped["_group"].nunique() == 3, "Expected 3 treatment groups"

    # At least one auto-named plot saved
    sns_pngs = list(Path(OUT_DIR).glob("*superplot.png")) + list(Path(OUT_DIR).glob("*barplot.png"))
    assert len(sns_pngs) >= 2, f"Expected at least 2 sns plot PNGs, got {len(sns_pngs)}"

    print("EPM seaborn plotting tests passed.")
