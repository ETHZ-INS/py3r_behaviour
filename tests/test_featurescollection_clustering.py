"""
Tests for the clustering API on FeaturesCollection.

Checks:
  - cluster_embedding_stream is the canonical path; cluster_embedding is gone.
  - normalize, feature_weights, normalize_details all work correctly.
  - CentroidsDf wrapper: delegation, scaling_recipe, save/load roundtrip.
  - Edge cases: NaNs in features, constant features, impute_weight policy.
  - build_column_weights utility correctness.
  - assign_clusters_by_centroids allow_missing_features subspace assignment.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from py3r.behaviour.features.centroids_df import CentroidsDf
from py3r.behaviour.features.features import Features
from py3r.behaviour.features.features_collection import FeaturesCollection
from py3r.behaviour.tracking.tracking import Tracking
from py3r.behaviour.util.series_utils import build_column_weights

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_tracking(handle: str, n_frames: int = 200, *, seed: int = 0) -> Tracking:
    """Minimal Tracking with a dummy bodypoint so Features.__init__ is happy."""
    rng = np.random.default_rng(seed)
    data = pd.DataFrame(
        {
            "bp.x": rng.standard_normal(n_frames).cumsum() + 100,
            "bp.y": rng.standard_normal(n_frames).cumsum() + 100,
        },
        index=pd.RangeIndex(n_frames, name="frame"),
    )
    meta = {"fps": 30.0, "rescale_distance_method": "dummy"}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return Tracking(data, meta, handle=handle)


def _make_fc(
    n_objects: int = 3,
    n_frames: int = 200,
    *,
    seed: int = 42,
    inject_nans: bool = True,
    add_constant_feature: bool = True,
) -> tuple[FeaturesCollection, dict[str, list[int]]]:
    """
    Build a FeaturesCollection with synthetic features stored in .data.

    Features:
      speed  — random walk speed-like values
      accel  — random acceleration-like values
      const  — constant value (tests zero-std normalisation)

    Some NaN values are injected to test missing-value handling.

    Returns (fc, embedding_dict).
    """
    rng = np.random.default_rng(seed)
    feats_list: list[Features] = []
    for i in range(n_objects):
        t = _make_tracking(f"rec_{i}", n_frames, seed=seed + i)
        f = Features(t)
        speed = pd.Series(
            rng.standard_normal(n_frames) * 0.1 + 1.0,
            index=t.data.index,
        )
        accel = pd.Series(
            rng.standard_normal(n_frames) * 0.05 + 0.5,
            index=t.data.index,
        )
        if inject_nans:
            for s in (speed, accel):
                mask = rng.random(n_frames) < 0.05
                s[mask] = np.nan
        f.store(speed, "speed")
        f.store(accel, "accel")
        if add_constant_feature:
            const = pd.Series(7.0, index=t.data.index)
            f.store(const, "const")
        feats_list.append(f)

    fc = FeaturesCollection.from_list(feats_list)
    embedding_dict: dict[str, list[int]] = {"speed": [0, 1], "accel": [0, 1]}
    if add_constant_feature:
        embedding_dict["const"] = [0]
    return fc, embedding_dict


def _sort_centroids(cents) -> pd.DataFrame:
    """Sort centroid rows for order-independent comparison."""
    df = cents.to_df() if isinstance(cents, CentroidsDf) else cents
    return df.sort_values(by=list(df.columns)).reset_index(drop=True)


# ---------------------------------------------------------------------------
# build_column_weights
# ---------------------------------------------------------------------------


class TestBuildColumnWeights:
    def test_basic_substring_matching(self):
        cols = ["speed_t0", "speed_t+1", "accel_t0", "dist_t0"]
        w = build_column_weights(cols, {"speed": 4.0, "accel": 2.0})
        assert w == {"speed_t0": 4.0, "speed_t+1": 4.0, "accel_t0": 2.0, "dist_t0": 1.0}

    def test_empty_rules_gives_all_ones(self):
        cols = ["a_t0", "b_t0"]
        w = build_column_weights(cols, {})
        assert all(v == 1.0 for v in w.values())

    def test_ambiguous_match_raises(self):
        cols = ["speed_fast_t0"]
        with pytest.raises(ValueError, match="multiple rules"):
            build_column_weights(cols, {"speed": 2.0, "fast": 3.0})

    def test_unused_rule_raises(self):
        cols = ["speed_t0", "accel_t0"]
        with pytest.raises(ValueError, match="matched no columns"):
            build_column_weights(cols, {"speed": 2.0, "azimtuh": 3.0})


# ---------------------------------------------------------------------------
# feature_weights convenience parameter
# ---------------------------------------------------------------------------


class TestFeatureWeights:
    @pytest.fixture()
    def fc_and_embedding(self):
        return _make_fc(
            n_objects=2, n_frames=100, inject_nans=False, add_constant_feature=False, seed=44
        )

    def test_feature_weights_equivalent_to_explicit_column_dict(self, fc_and_embedding):
        """feature_weights should also accept explicit per-column dicts."""
        fc, emb = fc_and_embedding
        rules = {"speed": 3.0}

        batch_rules, cents_rules = fc.cluster_embedding_stream(
            emb, n_clusters=2, random_state=0, feature_weights=rules
        )

        first_feat = next(iter(fc.features_dict.values()))
        cols = first_feat.embedding_df(emb).columns
        explicit = build_column_weights(cols, rules)
        batch_cw, cents_cw = fc.cluster_embedding_stream(
            emb, n_clusters=2, random_state=0, feature_weights=explicit
        )

        pd.testing.assert_frame_equal(
            _sort_centroids(cents_rules).astype(np.float64),
            _sort_centroids(cents_cw).astype(np.float64),
        )
        for key in batch_rules.keys():
            pd.testing.assert_series_equal(
                pd.Series(batch_rules[key]),
                pd.Series(batch_cw[key]),
            )

    def test_feature_weights_small_chunks(self, fc_and_embedding):
        """feature_weights should work correctly with small chunk/batch sizes."""
        fc, emb = fc_and_embedding
        batch, cents = fc.cluster_embedding_stream(
            emb,
            n_clusters=2,
            random_state=0,
            feature_weights={"speed": 3.0},
            chunk_size=30,
            n_epochs=5,
            batch_size=16,
        )
        assert cents.shape[0] == 2
        for key in batch.keys():
            assert pd.Series(batch[key]).notna().any()

    def test_typo_in_rule_raises(self, fc_and_embedding):
        fc, emb = fc_and_embedding
        with pytest.raises(ValueError, match="matched no columns"):
            fc.cluster_embedding_stream(emb, n_clusters=2, feature_weights={"speeed": 3.0})


# ---------------------------------------------------------------------------
# cluster_embedding removed
# ---------------------------------------------------------------------------


class TestRemovedClusterEmbedding:
    @pytest.fixture()
    def fc_and_embedding(self):
        return _make_fc(
            n_objects=3, n_frames=200, inject_nans=False, add_constant_feature=False, seed=99
        )

    def test_cluster_embedding_removed(self, fc_and_embedding):
        """cluster_embedding was removed in 3.3.0 and always raises."""
        fc, emb = fc_and_embedding
        with pytest.raises(NotImplementedError, match="3.3.0"):
            fc.cluster_embedding(emb, n_clusters=3)

    def test_cluster_embedding_removed_regardless_of_args(self, fc_and_embedding):
        """Passes any combination of args — should still raise immediately."""
        fc, emb = fc_and_embedding
        with pytest.raises(NotImplementedError):
            fc.cluster_embedding(emb, n_clusters=3, normalize=True)


# ---------------------------------------------------------------------------
# assign_clusters_by_centroids
# ---------------------------------------------------------------------------


class TestAssignClusters:
    @pytest.fixture()
    def setup(self):
        fc, emb = _make_fc(
            n_objects=1, n_frames=100, inject_nans=False, add_constant_feature=False, seed=11
        )
        feat = fc[list(fc.keys())[0]]
        _, cents = fc.cluster_embedding_stream(emb, n_clusters=3, random_state=0, normalize=True)
        return feat, emb, cents

    def test_centroidsdf_no_embedding_needed(self, setup):
        """CentroidsDf carries embedding in recipe — no need to pass it."""
        feat, emb, cents = setup
        result = feat.assign_clusters_by_centroids(cents)
        assert len(result) == len(feat.data)
        assert pd.Series(result).notna().sum() > 0

    def test_legacy_scaling_factors_path(self, setup):
        """Plain DataFrame + scaling_factors from recipe should still work."""
        feat, emb, cents = setup
        sf = cents.scaling_recipe.get("constant_factors") or {}
        result = feat.assign_clusters_by_centroids(cents.to_df(), emb, scaling_factors=sf or None)
        assert len(result) == len(feat.data)
        assert pd.Series(result).notna().sum() > 0

    def test_plain_df_requires_embedding(self, setup):
        feat, emb, cents = setup
        with pytest.raises(ValueError, match="embedding is required"):
            feat.assign_clusters_by_centroids(cents.to_df())

    def test_removed_rescale_factors_raises(self, setup):
        feat, emb, cents = setup
        embed_cols = feat.embedding_df(emb).columns
        rf = {c: 1.0 for c in embed_cols}
        with pytest.raises(NotImplementedError, match="rescale_factors"):
            feat.assign_clusters_by_centroids(cents, emb, rescale_factors=rf)

    def test_removed_custom_scaling_raises(self, setup):
        feat, emb, cents = setup
        with pytest.raises(NotImplementedError, match="custom_scaling"):
            feat.assign_clusters_by_centroids(
                cents,
                emb,
                custom_scaling={"speed": {"scale": 1.0}, "accel": {"scale": 1.0}},
            )


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_nans_in_features(self):
        """NaN rows should propagate as pd.NA labels, not crash."""
        fc, emb = _make_fc(
            n_objects=2, n_frames=100, inject_nans=True, add_constant_feature=False, seed=55
        )
        batch, cents = fc.cluster_embedding_stream(
            emb, n_clusters=2, random_state=0, chunk_size=30, n_epochs=3, batch_size=16
        )
        for key in batch.keys():
            labels = pd.Series(batch[key])
            assert labels.isna().any(), "Expected some NA labels from NaN rows"
            assert labels.notna().any(), "Expected some valid labels"

    def test_constant_feature_normalize(self):
        """A constant feature has std=0; normalize should not produce inf."""
        fc, emb = _make_fc(
            n_objects=2, n_frames=100, inject_nans=False, add_constant_feature=True, seed=33
        )
        batch, cents = fc.cluster_embedding_stream(
            emb,
            n_clusters=2,
            random_state=0,
            normalize=True,
            chunk_size=30,
            n_epochs=3,
            batch_size=16,
        )
        recipe = cents.scaling_recipe
        const_cols = [c for c in recipe["constant_factors"] if c.startswith("const")]
        for c in const_cols:
            assert recipe["constant_factors"][c] == pytest.approx(1.0), f"Expected 1.0 for {c}"
        assert np.all(np.isfinite(cents.values))

    def test_impute_weight_policy(self):
        """impute_weight should produce labels for all rows, even NaN ones."""
        fc, emb = _make_fc(
            n_objects=2, n_frames=100, inject_nans=True, add_constant_feature=False, seed=66
        )
        batch, cents = fc.cluster_embedding_stream(
            emb,
            n_clusters=2,
            random_state=0,
            missing_policy="impute_weight",
            chunk_size=30,
            n_epochs=3,
            batch_size=16,
        )
        for key in batch.keys():
            labels = pd.Series(batch[key])
            assert labels.notna().all(), "impute_weight should fill all labels"

    def test_single_features_object(self):
        """Features.cluster_embedding_stream should work on a single recording."""
        fc, emb = _make_fc(
            n_objects=1, n_frames=80, inject_nans=False, add_constant_feature=False, seed=88
        )
        feat = fc[list(fc.keys())[0]]
        result, cents = feat.cluster_embedding_stream(
            emb, n_clusters=2, random_state=0, chunk_size=20, n_epochs=5, batch_size=16
        )
        assert len(result) == len(feat.data)


# ---------------------------------------------------------------------------
# CentroidsDf wrapper
# ---------------------------------------------------------------------------


class TestCentroidsDf:
    @pytest.fixture()
    def setup(self):
        fc, emb = _make_fc(
            n_objects=2, n_frames=100, inject_nans=False, add_constant_feature=False, seed=21
        )
        batch, centroids = fc.cluster_embedding_stream(
            emb, n_clusters=3, random_state=0, normalize=True
        )
        return fc, emb, batch, centroids

    def test_centroids_is_centroidsdf(self, setup):
        _, _, _, centroids = setup
        assert isinstance(centroids, CentroidsDf)

    def test_dataframe_delegation(self, setup):
        _, _, _, centroids = setup
        assert hasattr(centroids, "columns")
        assert hasattr(centroids, "shape")
        assert centroids.shape[0] == 3
        assert np.all(np.isfinite(centroids.values))

    def test_scaling_recipe_present(self, setup):
        _, emb, _, centroids = setup
        recipe = centroids.scaling_recipe
        assert "version" in recipe
        assert "embedding_dict" in recipe
        assert "columns" in recipe
        assert "normalize_individual_base" in recipe
        assert "constant_factors" in recipe
        assert recipe["embedding_dict"] == emb

    def test_save_load_roundtrip(self, setup, tmp_path):
        _, emb, _, centroids = setup
        centroids.save(tmp_path / "run/")
        loaded = CentroidsDf.load(tmp_path / "run/")
        pd.testing.assert_frame_equal(centroids.to_df(), loaded.to_df())
        assert loaded.scaling_recipe == centroids.scaling_recipe

    def test_assign_via_recipe_matches_legacy_scaling_factors(self, setup):
        """Recipe-based assign should match the legacy scaling_factors path."""
        fc, emb, _, centroids = setup
        feat = fc[list(fc.keys())[0]]
        sf = centroids.scaling_recipe.get("constant_factors") or {}
        result_recipe = feat.assign_clusters_by_centroids(centroids)
        result_sf = feat.assign_clusters_by_centroids(
            centroids.to_df(), emb, scaling_factors=sf or None
        )
        pd.testing.assert_series_equal(
            pd.Series(result_recipe).astype("Int64"),
            pd.Series(result_sf).astype("Int64"),
        )


# ---------------------------------------------------------------------------
# normalize_details parameter
# ---------------------------------------------------------------------------


class TestNormalizeDetails:
    @pytest.fixture()
    def fc_and_emb(self):
        return _make_fc(
            n_objects=3, n_frames=150, inject_nans=False, add_constant_feature=False, seed=55
        )

    def test_all_global_matches_normalize_true(self, fc_and_emb):
        """normalize_details={'speed':'global','accel':'global'} == normalize=True."""
        fc, emb = fc_and_emb
        _, cents_norm = fc.cluster_embedding_stream(
            emb, n_clusters=2, random_state=0, normalize=True
        )
        _, cents_det = fc.cluster_embedding_stream(
            emb,
            n_clusters=2,
            random_state=0,
            normalize_details={"speed": "global", "accel": "global"},
        )
        pd.testing.assert_frame_equal(
            _sort_centroids(cents_norm).astype(np.float64),
            _sort_centroids(cents_det).astype(np.float64),
        )

    def test_all_none_matches_no_normalization(self, fc_and_emb):
        """normalize_details={'speed':'none','accel':'none'} == no normalization."""
        fc, emb = fc_and_emb
        _, cents_plain = fc.cluster_embedding_stream(emb, n_clusters=2, random_state=0)
        _, cents_none = fc.cluster_embedding_stream(
            emb, n_clusters=2, random_state=0, normalize_details={"speed": "none", "accel": "none"}
        )
        pd.testing.assert_frame_equal(
            _sort_centroids(cents_plain).astype(np.float64),
            _sort_centroids(cents_none).astype(np.float64),
        )

    def test_individual_mode_recipe_captured(self, fc_and_emb):
        """Individual mode columns appear in normalize_individual_base=True in recipe."""
        fc, emb = fc_and_emb
        _, centroids = fc.cluster_embedding_stream(
            emb,
            n_clusters=2,
            random_state=0,
            normalize_details={"speed": "individual", "accel": "global"},
        )
        recipe = centroids.scaling_recipe
        assert recipe["normalize_individual_base"]["speed"] is True
        assert recipe["normalize_individual_base"]["accel"] is False

    def test_individual_mode_constant_factors_excludes_individual_std(self, fc_and_emb):
        """constant_factors should be 1.0 for individual columns (no global std applied)."""
        fc, emb = fc_and_emb
        _, centroids = fc.cluster_embedding_stream(
            emb,
            n_clusters=2,
            random_state=0,
            normalize_details={"speed": "individual", "accel": "global"},
        )
        cf = centroids.scaling_recipe.get("constant_factors") or {}
        for col, factor in cf.items():
            if col.startswith("speed"):
                assert factor == pytest.approx(1.0), (
                    f"Expected constant=1.0 for individual column {col}, got {factor}"
                )

    def test_overlap_rule_raises(self, fc_and_emb):
        fc, emb = fc_and_emb
        with pytest.raises(ValueError, match="overlap"):
            fc.cluster_embedding_stream(
                emb, n_clusters=2, normalize_details={"speed": "global", "peed": "individual"}
            )

    def test_unmatched_rule_raises(self, fc_and_emb):
        fc, emb = fc_and_emb
        with pytest.raises(ValueError, match="matched no columns"):
            fc.cluster_embedding_stream(emb, n_clusters=2, normalize_details={"typo_col": "global"})

    def test_assign_via_recipe_individual(self, fc_and_emb):
        """After fitting with individual mode, recipe-based assign works on new Features."""
        fc, emb = fc_and_emb
        _, centroids = fc.cluster_embedding_stream(
            emb,
            n_clusters=2,
            random_state=0,
            normalize_details={"speed": "individual", "accel": "global"},
        )
        feat = fc[list(fc.keys())[0]]
        result = feat.assign_clusters_by_centroids(centroids)
        labels = pd.Series(result)
        assert len(labels) == len(feat.data)
        assert labels.notna().any()


# ---------------------------------------------------------------------------
# allow_missing_features subspace assignment
# ---------------------------------------------------------------------------


def _make_features_with(t, feature_names: list[str], *, seed: int) -> Features:
    """Return a Features built on *t* with the given named features stored."""
    rng = np.random.default_rng(seed)
    n = len(t.data)
    feat = Features(t)
    for name in feature_names:
        feat.store(pd.Series(rng.standard_normal(n) + 1.0, index=t.data.index), name)
    return feat


def _plain_cents(columns: list[str], n_clusters: int = 2) -> pd.DataFrame:
    """Return a trivial centroid DataFrame with the given columns."""
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        rng.standard_normal((n_clusters, len(columns))),
        columns=columns,
    )


class TestAssignClustersAllowMissingFeatures:
    """
    Tests for allow_missing_features in assign_clusters_by_centroids.

    Topology used throughout
    ------------------------
    * feat_full   – Features with {speed, accel}
    * feat_no_accel – Features with {speed} only (accel base column absent)
    * emb_full    – {"speed": [0, 1], "accel": [0, 1]}
    * emb_speed   – {"speed": [0, 1]}
    * cents_full  – plain DataFrame with speed_t0/t+1 + accel_t0/t+1 columns
    * cents_speed – plain DataFrame with speed_t0/t+1 columns only
    """

    @pytest.fixture()
    def ctx(self):
        t = _make_tracking("r", n_frames=80, seed=9)
        feat_full = _make_features_with(t, ["speed", "accel"], seed=9)
        feat_no_accel = _make_features_with(t, ["speed"], seed=9)

        emb_full = {"speed": [0, 1], "accel": [0, 1]}
        emb_speed = {"speed": [0, 1]}

        cols_full = feat_full.embedding_df(emb_full).columns.tolist()
        cols_speed = feat_full.embedding_df(emb_speed).columns.tolist()

        cents_full = _plain_cents(cols_full)
        cents_speed = _plain_cents(cols_speed)

        return {
            "t": t,
            "feat_full": feat_full,
            "feat_no_accel": feat_no_accel,
            "emb_full": emb_full,
            "emb_speed": emb_speed,
            "cols_full": cols_full,
            "cols_speed": cols_speed,
            "cents_full": cents_full,
            "cents_speed": cents_speed,
        }

    # --- "self" mode --------------------------------------------------------

    def test_self_mode_warns_missing_base_and_assigns(self, ctx):
        """
        allow_missing_features='self': self is missing 'accel'.

        Expected warnings
        -----------------
        1. Pre-filter: 'accel' base feature absent from self → embedding trimmed.
        2. Reconciliation (only_in_centroids): accel_t0 / accel_t+1 present in
           centroids but not in the reduced embed_df → dropped from centroids.

        Assignment must succeed in the speed-only subspace.
        """
        c = ctx
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = c["feat_no_accel"].assign_clusters_by_centroids(
                c["cents_full"],
                c["emb_full"],
                allow_missing_features="self",
            )

        messages = [str(w.message) for w in caught if issubclass(w.category, UserWarning)]
        # Warning 1: missing base feature
        assert any("absent from self" in m and "accel" in m for m in messages), messages
        # Warning 2: centroid columns dropped
        assert any("centroid column(s)" in m and "accel" in m for m in messages), messages

        labels = pd.Series(result)
        assert len(labels) == len(c["feat_no_accel"].data)
        assert labels.notna().any()
        assert set(labels.dropna().unique()).issubset({0, 1})

    def test_self_mode_meta_records_flag(self, ctx):
        """allow_missing_features is stored in _params."""
        c = ctx
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = c["feat_no_accel"].assign_clusters_by_centroids(
                c["cents_full"],
                c["emb_full"],
                allow_missing_features="self",
            )
        assert result._params["allow_missing_features"] == "self"

    def test_self_mode_raises_when_self_has_extra_cols_not_in_centroids(self, ctx):
        """
        allow_missing_features='self' only covers self having *fewer* features.
        If self produces columns the centroids don't have, that is a 'centroids'
        problem and must raise with a helpful suggestion.
        """
        c = ctx
        # feat_full has both speed + accel; cents_speed only has speed → only_in_self non-empty
        with pytest.raises(ValueError, match="allow_missing_features='centroids' or 'both'"):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                c["feat_full"].assign_clusters_by_centroids(
                    c["cents_speed"],
                    c["emb_full"],
                    allow_missing_features="self",
                )

    # --- "centroids" mode ---------------------------------------------------

    def test_centroids_mode_warns_extra_self_cols_and_assigns(self, ctx):
        """
        allow_missing_features='centroids': centroids fitted on speed only, self
        has speed + accel.

        Expected warning
        ----------------
        Reconciliation (only_in_self): accel_t0 / accel_t+1 produced by self
        have no counterpart in the speed-only centroids → dropped.

        Assignment must succeed in the speed-only subspace.
        """
        c = ctx
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = c["feat_full"].assign_clusters_by_centroids(
                c["cents_speed"],
                c["emb_full"],
                allow_missing_features="centroids",
            )

        messages = [str(w.message) for w in caught if issubclass(w.category, UserWarning)]
        assert any(
            "embedding column(s) produced by self" in m and "accel" in m for m in messages
        ), messages

        labels = pd.Series(result)
        assert len(labels) == len(c["feat_full"].data)
        assert labels.notna().any()
        assert set(labels.dropna().unique()).issubset({0, 1})

    def test_centroids_mode_raises_when_self_missing_base(self, ctx):
        """
        allow_missing_features='centroids' does NOT tolerate self missing base
        features — the pre-filter only applies to 'self' / 'both'.
        """
        c = ctx
        # feat_no_accel missing 'accel'; embedding_df would raise internally
        with pytest.raises(ValueError, match="not present in self.data"):
            c["feat_no_accel"].assign_clusters_by_centroids(
                c["cents_full"],
                c["emb_full"],
                allow_missing_features="centroids",
            )

    # --- "both" mode --------------------------------------------------------

    def test_both_mode_warns_all_sides_and_assigns(self, ctx):
        """
        allow_missing_features='both': self missing 'extra_feat', centroids built
        on speed + extra_feat, embedding also asks for accel (present in self but
        not in centroids).

        Expected warnings
        -----------------
        1. Pre-filter: 'extra_feat' absent from self.
        2. Reconciliation (only_in_self): accel_t0/t+1 produced by self but not
           in the extra_feat centroids → dropped.
        3. Reconciliation (only_in_centroids): extra_feat_t0/t+1 in centroids
           but not produced by self → dropped.

        Assignment succeeds in the speed-only subspace.
        """
        c = ctx
        # Build centroids that have speed + an 'extra_feat' column
        extra_cols = ["extra_feat_t0", "extra_feat_t+1"]
        cols_speed_extra = c["cols_speed"] + extra_cols
        cents_speed_extra = _plain_cents(cols_speed_extra)

        # Embedding that requests speed + accel (both in self) + extra_feat (absent)
        emb_three = {"speed": [0, 1], "accel": [0, 1], "extra_feat": [0, 1]}

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = c["feat_full"].assign_clusters_by_centroids(
                cents_speed_extra,
                emb_three,
                allow_missing_features="both",
            )

        messages = [str(w.message) for w in caught if issubclass(w.category, UserWarning)]
        # Warning 1: 'extra_feat' missing from self (pre-filter)
        assert any("absent from self" in m and "extra_feat" in m for m in messages), messages
        # Warning 2: accel columns dropped from self side
        assert any(
            "embedding column(s) produced by self" in m and "accel" in m for m in messages
        ), messages
        # Warning 3: extra_feat columns dropped from centroid side
        assert any("centroid column(s)" in m and "extra_feat" in m for m in messages), messages

        labels = pd.Series(result)
        assert len(labels) == len(c["feat_full"].data)
        assert labels.notna().any()
        assert set(labels.dropna().unique()).issubset({0, 1})

    # --- no shared columns --------------------------------------------------

    def test_empty_intersection_raises_for_self_and_both_modes(self, ctx):
        """
        "self" and "both" reach the intersection check; with no columns in common
        a ValueError is raised.

        Scenario: self only has 'speed'; embedding asks for 'accel' only;
        centroids also have only 'accel' columns.  After the pre-filter strips
        'accel' from the embedding (absent from self) the embed_df is empty,
        so the shared subspace is empty.
        """
        c = ctx
        cols_accel = [col for col in c["cols_full"] if col.startswith("accel")]
        cents_accel_only = _plain_cents(cols_accel)
        emb_accel = {"accel": [0, 1]}

        for mode in ("self", "both"):
            with pytest.raises(ValueError, match="No columns remain in common"):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    c["feat_no_accel"].assign_clusters_by_centroids(
                        cents_accel_only,
                        emb_accel,
                        allow_missing_features=mode,
                    )

    def test_centroids_mode_raises_from_embedding_df_when_self_missing_base(self, ctx):
        """
        "centroids" mode does not pre-filter the embedding, so if self is missing
        a base feature embedding_df raises before the intersection check is reached.
        """
        c = ctx
        cols_accel = [col for col in c["cols_full"] if col.startswith("accel")]
        cents_accel_only = _plain_cents(cols_accel)
        emb_accel = {"accel": [0, 1]}

        with pytest.raises(ValueError, match="not present in self.data"):
            c["feat_no_accel"].assign_clusters_by_centroids(
                cents_accel_only,
                emb_accel,
                allow_missing_features="centroids",
            )

    # --- default (None) still strict ----------------------------------------

    def test_default_none_still_raises_on_mismatch(self, ctx):
        """Without allow_missing_features the strict column-equality check is preserved."""
        c = ctx
        with pytest.raises(ValueError, match="Columns in embedding and centroids do not match"):
            c["feat_full"].assign_clusters_by_centroids(
                c["cents_speed"],
                c["emb_full"],
            )
