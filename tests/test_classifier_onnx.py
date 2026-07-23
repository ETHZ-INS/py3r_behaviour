"""Tests for OnnxClassifier (src/py3r/behaviour/classifier.py).

Covers:
  - single-output models -> pd.Series, indexed like the embedding
  - multi-output models -> pd.DataFrame
  - NaN propagation from embedding_df shifts (edge frames)
  - fit() is explicitly unimplemented
  - missing onnxruntime raises a clear ImportError
"""

import warnings

import numpy as np
import pandas as pd
import pytest

onnx = pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")
from onnx import TensorProto, helper  # noqa: E402

from py3r.behaviour.classifier import OnnxClassifier  # noqa: E402
from py3r.behaviour.features.features import Features  # noqa: E402
from py3r.behaviour.tracking.tracking import Tracking  # noqa: E402

# ---------------------------------------------------------------------------
# ONNX model builders (no training framework required)
# ---------------------------------------------------------------------------


def _build_sum_model(path):
    """Input (N,2) -> output (N,) = col0 + col1, via MatMul with weight [1,1]."""
    X = helper.make_tensor_value_info("input", TensorProto.FLOAT, [None, 2])
    Y = helper.make_tensor_value_info("output", TensorProto.FLOAT, [None])
    W = helper.make_tensor("W", TensorProto.FLOAT, [2], [1.0, 1.0])
    node = helper.make_node("MatMul", ["input", "W"], ["output"])
    graph = helper.make_graph([node], "sum_model", [X], [Y], initializer=[W])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    onnx.checker.check_model(model)
    onnx.save(model, str(path))
    return path


def _build_multi_output_model(path):
    """Input (N,2) -> output (N,2) = [col0+col1, col0-col1], via MatMul with a 2x2 weight."""
    X = helper.make_tensor_value_info("input", TensorProto.FLOAT, [None, 2])
    Y = helper.make_tensor_value_info("output", TensorProto.FLOAT, [None, 2])
    # W columns: [1, 1] and [1, -1] -> out[:,0]=x0+x1, out[:,1]=x0-x1
    W = helper.make_tensor("W", TensorProto.FLOAT, [2, 2], [1.0, 1.0, 1.0, -1.0])
    node = helper.make_node("MatMul", ["input", "W"], ["output"])
    graph = helper.make_graph([node], "multi_model", [X], [Y], initializer=[W])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    onnx.checker.check_model(model)
    onnx.save(model, str(path))
    return path


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def features():
    data = pd.DataFrame(
        {
            "a.x": [0.0, 1.0, 2.0, 3.0, 4.0],
            "a.y": [0.0, 0.0, 0.0, 0.0, 0.0],
        },
        index=pd.RangeIndex(5, name="frame"),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        t = Tracking(data, {"fps": 30.0}, handle="test")
        f = Features(t)
    counter = pd.Series(range(len(t.data)), index=t.data.index, dtype=float)
    f.store(counter, "counter", meta={})
    return f


# ---------------------------------------------------------------------------
# predict: single output -> Series
# ---------------------------------------------------------------------------


class TestPredictSingleOutput:
    def test_returns_series(self, features, tmp_path):
        model_path = _build_sum_model(tmp_path / "sum.onnx")
        clf = OnnxClassifier(model_path, {"counter": [0, -1]})
        result = clf.predict(features)
        assert isinstance(result, pd.Series)

    def test_values_match_expected_sum(self, features, tmp_path):
        # embedding: counter_t0 = [0,1,2,3,4]; counter_t-1 shifts backward -> [nan,0,1,2,3]
        model_path = _build_sum_model(tmp_path / "sum.onnx")
        clf = OnnxClassifier(model_path, {"counter": [0, -1]})
        result = clf.predict(features)
        assert result.iloc[1] == pytest.approx(1.0)  # 1 + 0
        assert result.iloc[4] == pytest.approx(7.0)  # 4 + 3

    def test_index_matches_features_data(self, features, tmp_path):
        model_path = _build_sum_model(tmp_path / "sum.onnx")
        clf = OnnxClassifier(model_path, {"counter": [0, -1]})
        result = clf.predict(features)
        assert list(result.index) == list(features.data.index)

    def test_nan_at_shifted_edge(self, features, tmp_path):
        model_path = _build_sum_model(tmp_path / "sum.onnx")
        clf = OnnxClassifier(model_path, {"counter": [0, -1]})
        result = clf.predict(features)
        # counter_t-1 is NaN at frame 0 -> model output NaN (matmul propagates NaN)
        assert np.isnan(result.iloc[0])


# ---------------------------------------------------------------------------
# predict: multi output -> DataFrame
# ---------------------------------------------------------------------------


class TestPredictMultiOutput:
    def test_returns_dataframe(self, features, tmp_path):
        model_path = _build_multi_output_model(tmp_path / "multi.onnx")
        clf = OnnxClassifier(model_path, {"counter": [0, -1]})
        result = clf.predict(features)
        assert isinstance(result, pd.DataFrame)
        assert result.shape == (len(features.data), 2)

    def test_values_match_expected(self, features, tmp_path):
        model_path = _build_multi_output_model(tmp_path / "multi.onnx")
        clf = OnnxClassifier(model_path, {"counter": [0, -1]})
        result = clf.predict(features)
        # frame 4: counter_t0=4, counter_t-1=3 -> [4+3, 4-3] = [7, 1]
        assert result.iloc[4, 0] == pytest.approx(7.0)
        assert result.iloc[4, 1] == pytest.approx(1.0)

    def test_index_matches_features_data(self, features, tmp_path):
        model_path = _build_multi_output_model(tmp_path / "multi.onnx")
        clf = OnnxClassifier(model_path, {"counter": [0, -1]})
        result = clf.predict(features)
        assert list(result.index) == list(features.data.index)


# ---------------------------------------------------------------------------
# embedding_dict validation is delegated to Features.embedding_df
# ---------------------------------------------------------------------------


class TestEmbeddingValidation:
    def test_missing_column_raises(self, features, tmp_path):
        model_path = _build_sum_model(tmp_path / "sum.onnx")
        clf = OnnxClassifier(model_path, {"does_not_exist": [0, -1]})
        with pytest.raises(ValueError, match="not present"):
            clf.predict(features)


# ---------------------------------------------------------------------------
# fit() is explicitly unimplemented
# ---------------------------------------------------------------------------


class TestFitNotImplemented:
    def test_fit_raises(self, features, tmp_path):
        model_path = _build_sum_model(tmp_path / "sum.onnx")
        clf = OnnxClassifier(model_path, {"counter": [0, -1]})
        with pytest.raises(NotImplementedError, match="inference-only"):
            clf.fit(features)


# ---------------------------------------------------------------------------
# Missing onnxruntime dependency
# ---------------------------------------------------------------------------


class TestMissingOnnxruntime:
    def test_import_error_message(self, tmp_path, monkeypatch):
        import builtins

        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "onnxruntime":
                raise ImportError("no onnxruntime")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        model_path = tmp_path / "unused.onnx"
        with pytest.raises(ImportError, match="pip install onnxruntime"):
            OnnxClassifier(model_path, {"counter": [0]})
