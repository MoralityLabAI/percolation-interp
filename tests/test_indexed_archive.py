"""NumPy array interchange and pickle-free graph archive regression tests."""
from __future__ import annotations

import numpy as np
import pytest

from percinterp import from_attribution_arrays, from_attribution_edges, profile, rank_entry
from percinterp.archive import load_graph_npz, save_graph_npz


def fixture():
    names = ["output:answer", "layer0.gate:12", "token:input", "layer0.up:8", "layer1.down:3"]
    stages = [3, 1, 0, 1, 2]
    src = np.array([2, 2, 1, 3, 4, 2], dtype=np.int64)
    dst = np.array([1, 3, 4, 4, 0, 0], dtype=np.int64)
    weights = np.array([.9, -.3, .4, -.8, .6, -.12])
    return names, stages, src, dst, weights


def test_indexed_matches_record_import():
    names, stages, src, dst, weights = fixture()
    rows = [(names[int(a)], names[int(b)], float(w)) for a, b, w in zip(src, dst, weights)]
    a = from_attribution_arrays(names, stages, src, dst, weights)
    b = from_attribution_edges(names, stages, rows)
    assert a.node_ids == b.node_ids
    assert a.signed_edges() == b.signed_edges()
    entry = rank_entry(a.graph.weight, a.graph.edge_layer("dst"))
    assert 0 <= profile(a.graph, entry).p_span <= 1


def test_pickle_free_archive_roundtrip(tmp_path):
    data = from_attribution_arrays(*fixture())
    dest = tmp_path / "graph.npz"
    save_graph_npz(dest, data, provenance={"producer": "synthetic_fixture", "method": "direct"})
    with np.load(dest, allow_pickle=False) as z:
        assert all(z[k].dtype.kind != "O" for k in z.files)
    restored, provenance = load_graph_npz(dest)
    assert restored.node_ids == data.node_ids
    assert restored.signed_edges() == data.signed_edges()
    assert np.array_equal(restored.graph.sources, data.graph.sources)
    assert provenance == {"producer": "synthetic_fixture", "method": "direct"}


def test_zero_weight_dropped_and_archive_empty_graph(tmp_path):
    names, stages, src, dst, weights = fixture()
    weights[-1] = 0
    data = from_attribution_arrays(names, stages, src, dst, weights)
    assert data.dropped_zero_edges == 1
    save_graph_npz(tmp_path / "without_zero.npz", data)
    restored, _ = load_graph_npz(tmp_path / "without_zero.npz")
    assert restored.graph.n_edges == len(weights) - 1
    empty = from_attribution_arrays(["a", "b"], [0, 1], np.array([], dtype=int),
                                    np.array([], dtype=int), np.array([], dtype=float))
    save_graph_npz(tmp_path / "empty.npz", empty)
    assert load_graph_npz(tmp_path / "empty.npz")[0].graph.n_edges == 0


@pytest.mark.parametrize("kind", ["bad_type", "unknown_id", "reverse", "duplicate", "nan", "inf"])
def test_invalid_edge_input(kind):
    names, stages, src, dst, w = fixture()
    src, dst, w = src.copy(), dst.copy(), w.copy()
    if kind == "bad_type":
        src = src.astype(float)
    elif kind == "unknown_id":
        src[0] = 17
    elif kind == "reverse":
        src[0], dst[0] = 0, 2
    elif kind == "duplicate":
        src, dst, w = (np.append(src, src[0]), np.append(dst, dst[0]),
                       np.append(w, 0.0))
    elif kind == "nan":
        w[0] = np.nan
    elif kind == "inf":
        w[0] = np.inf
    with pytest.raises(ValueError):
        from_attribution_arrays(names, stages, src, dst, w)


def test_corrupt_schema_and_provenance_rejected(tmp_path):
    p = tmp_path / "bad.npz"
    np.savez_compressed(p, schema=np.asarray("bad"))
    with pytest.raises(ValueError):
        load_graph_npz(p)
    with pytest.raises(TypeError):
        save_graph_npz(p, from_attribution_arrays(*fixture()), provenance={"x": 1})
