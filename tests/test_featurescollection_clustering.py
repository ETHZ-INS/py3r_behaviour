"""
Tests for the clustering API on FeaturesCollection.

Checks:
  - cluster_embedding_stream is the canonical path; cluster_embedding is a
    deprecated shim that delegates to it.
  - normalize, feature_weights, normalize_details all work correctly.
  - CentroidsDf wrapper: delegation, scaling_recipe, save/load roundtrip.
  - Edge cases: NaNs in features, constant features, impute_weight policy.
  - build_column_weights utility correctness.
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


def _label_agreement_permutation_invariant(lab_a: pd.Series, lab_b: pd.Series) -> float:
    """
    Best-case label agreement over all permutations of cluster IDs.

    For k clusters this is O(k!) — fine for small k in tests.
    """
    from itertools import permutations

    common = lab_a.index.intersection(lab_b.index)
    a, b = lab_a[common].values, lab_b[common].values
    unique_ids = np.unique(np.concatenate([a, b]))
    best = 0.0
    for perm in permutations(unique_ids):
        mapping = dict(zip(unique_ids, perm, strict=True))
        remapped = np.array([mapping[v] for v in b])
        agreement = (a == remapped).mean()
        best = max(best, agreement)
    return best


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

        batch_rules, cents_rules, sf_rules = fc.cluster_embedding_stream(
            emb,
            n_clusters=2,
            random_state=0,
            feature_weights=rules,
        )

        first_feat = next(iter(fc.features_dict.values()))
        cols = first_feat.embedding_df(emb).columns
        explicit = build_column_weights(cols, rules)
        batch_cw, cents_cw, sf_cw = fc.cluster_embedding_stream(
            emb,
            n_clusters=2,
            random_state=0,
            feature_weights=explicit,
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

    def test_feature_weights_stream(self, fc_and_embedding):
        """feature_weights should work correctly with small chunk/batch sizes."""
        fc, emb = fc_and_embedding
        batch, cents, sf = fc.cluster_embedding_stream(
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
            fc.cluster_embedding_stream(
                emb,
                n_clusters=2,
                feature_weights={"speeed": 3.0},
            )


# ---------------------------------------------------------------------------
# Deprecated / removed params
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
# assign_clusters_by_centroids: new vs deprecated
# ---------------------------------------------------------------------------


class TestAssignClusters:
    @pytest.fixture()
    def setup(self):
        fc, emb = _make_fc(
            n_objects=1, n_frames=100, inject_nans=False, add_constant_feature=False, seed=11
        )
        feat = fc[list(fc.keys())[0]]
        _, cents, sf = fc.cluster_embedding_stream(
            emb,
            n_clusters=3,
            random_state=0,
            normalize=True,
        )
        return feat, emb, cents, sf

    def test_new_scaling_factors_param(self, setup):
        feat, emb, cents, sf = setup
        result = feat.assign_clusters_by_centroids(cents, emb, scaling_factors=sf)
        assert len(result) == len(feat.data)
        assert pd.Series(result).notna().sum() > 0

    def test_centroidsdf_no_embedding_needed(self, setup):
        feat, emb, cents, _ = setup
        # CentroidsDf carries embedding in recipe — no need to pass it
        result = feat.assign_clusters_by_centroids(cents)
        assert len(result) == len(feat.data)
        assert pd.Series(result).notna().sum() > 0

    def test_plain_df_requires_embedding(self, setup):
        feat, emb, cents, _ = setup
        with pytest.raises(ValueError, match="embedding is required"):
            feat.assign_clusters_by_centroids(cents.to_df())

    def test_removed_rescale_factors_raises(self, setup):
        feat, emb, cents, sf = setup
        embed_cols = feat.embedding_df(emb).columns
        rf = {c: 1.0 for c in embed_cols}
        with pytest.raises(NotImplementedError, match="rescale_factors"):
            feat.assign_clusters_by_centroids(cents, emb, rescale_factors=rf)

    def test_removed_custom_scaling_raises(self, setup):
        feat, emb, cents, _ = setup
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
        batch, cents, _ = fc.cluster_embedding_stream(
            emb,
            n_clusters=2,
            random_state=0,
            chunk_size=30,
            n_epochs=3,
            batch_size=16,
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
        batch, cents, sf = fc.cluster_embedding_stream(
            emb,
            n_clusters=2,
            random_state=0,
            normalize=True,
            chunk_size=30,
            n_epochs=3,
            batch_size=16,
        )
        assert sf is not None
        const_cols = [c for c in sf if c.startswith("const")]
        for c in const_cols:
            assert sf[c] == pytest.approx(1.0), f"Expected 1.0 for {c}, got {sf[c]}"
        assert np.all(np.isfinite(cents.values))

    def test_impute_weight_policy(self):
        """impute_weight should produce labels for all rows, even NaN ones."""
        fc, emb = _make_fc(
            n_objects=2, n_frames=100, inject_nans=True, add_constant_feature=False, seed=66
        )
        batch, cents, _ = fc.cluster_embedding_stream(
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
        result, cents, sf = feat.cluster_embedding_stream(
            emb,
            n_clusters=2,
            random_state=0,
            chunk_size=20,
            n_epochs=5,
            batch_size=16,
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
        batch, centroids, sf = fc.cluster_embedding_stream(
            emb, n_clusters=3, random_state=0, normalize=True
        )
        return fc, emb, batch, centroids, sf

    def test_centroids_is_centroidsdf(self, setup):
        _, _, _, centroids, _ = setup
        assert isinstance(centroids, CentroidsDf)

    def test_dataframe_delegation(self, setup):
        _, _, _, centroids, _ = setup
        assert hasattr(centroids, "columns")
        assert hasattr(centroids, "shape")
        assert centroids.shape[0] == 3
        assert np.all(np.isfinite(centroids.values))

    def test_scaling_recipe_present(self, setup):
        _, emb, _, centroids, _ = setup
        recipe = centroids.scaling_recipe
        assert "version" in recipe
        assert "embedding_dict" in recipe
        assert "columns" in recipe
        assert "normalize_individual_base" in recipe
        assert "constant_factors" in recipe
        assert recipe["embedding_dict"] == emb

    def test_save_load_roundtrip(self, setup, tmp_path):
        _, emb, _, centroids, _ = setup
        centroids.save(tmp_path / "run/")
        loaded = CentroidsDf.load(tmp_path / "run/")
        pd.testing.assert_frame_equal(centroids.to_df(), loaded.to_df())
        assert loaded.scaling_recipe == centroids.scaling_recipe

    def test_assign_via_recipe_matches_scaling_factors(self, setup):
        """Recipe-based assign on the same data should match scaling_factors assign."""
        fc, emb, _, centroids, sf = setup
        feat = fc[list(fc.keys())[0]]
        # Via recipe (CentroidsDf) — embedding inferred from recipe
        result_recipe = feat.assign_clusters_by_centroids(centroids)
        # Via legacy scaling_factors (plain DF) — embedding must be provided
        result_sf = feat.assign_clusters_by_centroids(centroids.to_df(), emb, scaling_factors=sf)
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
        _, cents_norm, sf_norm = fc.cluster_embedding_stream(
            emb, n_clusters=2, random_state=0, normalize=True
        )
        _, cents_det, sf_det = fc.cluster_embedding_stream(
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
        _, cents_plain, _ = fc.cluster_embedding_stream(emb, n_clusters=2, random_state=0)
        _, cents_none, _ = fc.cluster_embedding_stream(
            emb,
            n_clusters=2,
            random_state=0,
            normalize_details={"speed": "none", "accel": "none"},
        )
        pd.testing.assert_frame_equal(
            _sort_centroids(cents_plain).astype(np.float64),
            _sort_centroids(cents_none).astype(np.float64),
        )

    def test_individual_mode_recipe_captured(self, fc_and_emb):
        """Individual mode columns appear in normalize_individual_base=True in recipe."""
        fc, emb = fc_and_emb
        _, centroids, _ = fc.cluster_embedding_stream(
            emb,
            n_clusters=2,
            random_state=0,
            normalize_details={"speed": "individual", "accel": "global"},
        )
        recipe = centroids.scaling_recipe
        assert recipe["normalize_individual_base"]["speed"] is True
        assert recipe["normalize_individual_base"]["accel"] is False

    def test_individual_mode_constant_factors_excludes_individual_std(self, fc_and_emb):
        """scaling_factors should not include the per-recording std for individual cols."""
        fc, emb = fc_and_emb
        _, _, sf = fc.cluster_embedding_stream(
            emb,
            n_clusters=2,
            random_state=0,
            normalize_details={"speed": "individual", "accel": "global"},
        )
        # Speed columns get weight 1.0 (no weight) and no global std applied,
        # so their constant factor should be 1.0.
        for col in sf or {}:
            if col.startswith("speed"):
                assert sf[col] == pytest.approx(1.0), (
                    f"Expected constant=1.0 for individual column {col}, got {sf[col]}"
                )

    def test_overlap_rule_raises(self, fc_and_emb):
        fc, emb = fc_and_emb
        with pytest.raises(ValueError, match="overlap"):
            fc.cluster_embedding_stream(
                emb,
                n_clusters=2,
                normalize_details={"speed": "global", "peed": "individual"},
            )

    def test_unmatched_rule_raises(self, fc_and_emb):
        fc, emb = fc_and_emb
        with pytest.raises(ValueError, match="matched no columns"):
            fc.cluster_embedding_stream(
                emb,
                n_clusters=2,
                normalize_details={"typo_col": "global"},
            )

    def test_individual_mode_recipe_captured_stream(self, fc_and_emb):
        """normalize_details with individual mode is captured correctly in recipe."""
        fc, emb = fc_and_emb
        _, centroids, sf = fc.cluster_embedding_stream(
            emb,
            n_clusters=2,
            random_state=0,
            normalize_details={"speed": "individual", "accel": "global"},
            n_epochs=3,
        )
        assert isinstance(centroids, CentroidsDf)
        recipe = centroids.scaling_recipe
        assert recipe["normalize_individual_base"]["speed"] is True
        assert recipe["normalize_individual_base"]["accel"] is False

    def test_assign_via_recipe_individual(self, fc_and_emb):
        """After fitting with individual mode, recipe-based assign works on new Features."""
        fc, emb = fc_and_emb
        _, centroids, _ = fc.cluster_embedding_stream(
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
