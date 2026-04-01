from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy import sparse
from scipy.sparse import csgraph
from sklearn.cluster import SpectralClustering
from sklearn.metrics import adjusted_rand_score

# ============================================================
# Core graph container
# ============================================================


@dataclass
class NSGGraph:
    n_nodes: int
    k: int
    indptr: np.ndarray  # (n+1,)
    indices: np.ndarray  # (n*k,)
    distances: np.ndarray  # (n*k,)
    ks_index: np.ndarray  # (n*k,)
    ks_stat: np.ndarray  # (n*k,)
    shared_count: np.ndarray  # (n*k,)
    jaccard: np.ndarray  # (n*k,)
    weight: np.ndarray  # (n*k,)

    def adjacency_directed(self, weighted: bool = False) -> sparse.csr_matrix:
        data = self.weight if weighted else np.ones_like(self.weight, dtype=np.float32)
        return sparse.csr_matrix(
            (data, self.indices, self.indptr), shape=(self.n_nodes, self.n_nodes)
        )

    def thresholded_directed(self, threshold: float, weighted: bool = False) -> sparse.csr_matrix:
        row_counts = np.diff(self.indptr)
        rows = np.repeat(np.arange(self.n_nodes, dtype=np.int32), row_counts)
        mask = self.weight >= threshold
        data = self.weight[mask] if weighted else np.ones(mask.sum(), dtype=np.float32)
        return sparse.csr_matrix(
            (data, (rows[mask], self.indices[mask])), shape=(self.n_nodes, self.n_nodes)
        )

    def thresholded_symmetric(
        self,
        threshold: float,
        weighted: bool = True,
        sym_rule: str = "mean",
        mutual_only: bool = False,
    ) -> sparse.csr_matrix:
        """
        Symmetrize the thresholded graph.

        sym_rule:
            "mean" -> (A + A.T) / 2
            "max"  -> max(A, A.T)
            "min"  -> min(A, A.T)
        mutual_only:
            if True, keep only edges present in both directions after thresholding
        """
        A = self.thresholded_directed(threshold, weighted=weighted).tocsr()

        if mutual_only:
            B = A.minimum(A.T)
            B.eliminate_zeros()
            return B

        AT = A.T.tocsr()
        if sym_rule == "mean":
            B = (A + AT) * 0.5
        elif sym_rule == "max":
            B = A.maximum(AT)
        elif sym_rule == "min":
            B = A.minimum(AT)
        else:
            raise ValueError("sym_rule must be one of {'mean', 'max', 'min'}")

        B.eliminate_zeros()
        return B


@dataclass
class AutoClusterResult:
    labels: np.ndarray
    threshold: float
    graph: NSGGraph
    partition_history: list[np.ndarray]
    threshold_scores: list[dict[str, float]]
    component_tree: dict[int, dict]


# ============================================================
# Approximate kNN backends
# ============================================================


def _knn_pynndescent(
    X: np.ndarray,
    k: int,
    metric: str = "euclidean",
    random_state: int = 0,
    n_trees: int | None = None,
    n_iters: int | None = None,
    low_memory: bool = True,
    diversify_prob: float = 0.0,
    pruning_degree_multiplier: float = 1.5,
    compressed: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    from pynndescent import NNDescent

    index = NNDescent(
        X,
        n_neighbors=k + 1,
        metric=metric,
        random_state=random_state,
        n_trees=n_trees,
        n_iters=n_iters,
        low_memory=low_memory,
        diversify_prob=diversify_prob,
        pruning_degree_multiplier=pruning_degree_multiplier,
        compressed=compressed,
    )
    indices, distances = index.neighbor_graph
    n = X.shape[0]

    if indices.shape[1] >= k + 1 and np.all(indices[:, 0] == np.arange(n)):
        indices = indices[:, 1 : k + 1]
        distances = distances[:, 1 : k + 1]
    else:
        indices = indices[:, :k]
        distances = distances[:, :k]

    return indices.astype(np.int32, copy=False), distances.astype(np.float32, copy=False)


def _knn_hnswlib(
    X: np.ndarray,
    k: int,
    metric: str = "l2",
    ef_construction: int = 200,
    M: int = 16,
    ef_search: int = 100,
    num_threads: int = -1,
) -> tuple[np.ndarray, np.ndarray]:
    import hnswlib

    X = np.asarray(X, dtype=np.float32, order="C")
    n, dim = X.shape

    index = hnswlib.Index(space=metric, dim=dim)
    index.init_index(max_elements=n, ef_construction=ef_construction, M=M)
    index.add_items(X, np.arange(n), num_threads=num_threads)
    index.set_ef(max(ef_search, k + 1))

    labels, distances = index.knn_query(X, k=k + 1, num_threads=num_threads)

    if labels.shape[1] >= k + 1 and np.all(labels[:, 0] == np.arange(n)):
        labels = labels[:, 1 : k + 1]
        distances = distances[:, 1 : k + 1]
    else:
        labels = labels[:, :k]
        distances = distances[:, :k]

    return labels.astype(np.int32, copy=False), distances.astype(np.float32, copy=False)


# ============================================================
# Edge weighting from the paper
# ============================================================


def _ks_discrete_index_and_stat(sorted_a: np.ndarray, sorted_b: np.ndarray) -> tuple[int, float]:
    """
    Exact KS D for two sorted arrays of equal length k, plus discrete s_K.
    """
    k = sorted_a.shape[0]
    i = j = 0
    cdf_a = cdf_b = 0.0
    d = 0.0

    while i < k and j < k:
        va = sorted_a[i]
        vb = sorted_b[j]

        if va <= vb:
            while i < k and sorted_a[i] == va:
                i += 1
            cdf_a = i / k

        if vb <= va:
            while j < k and sorted_b[j] == vb:
                j += 1
            cdf_b = j / k

        diff = abs(cdf_a - cdf_b)
        if diff > d:
            d = diff

    while i < k:
        va = sorted_a[i]
        while i < k and sorted_a[i] == va:
            i += 1
        cdf_a = i / k
        d = max(d, abs(cdf_a - cdf_b))

    while j < k:
        vb = sorted_b[j]
        while j < k and sorted_b[j] == vb:
            j += 1
        cdf_b = j / k
        d = max(d, abs(cdf_a - cdf_b))

    # Natural discretization onto {0, ..., k-1}
    s_k = int(np.clip(np.rint(d * k), 0, k - 1))
    return s_k, float(d)


def _harmonic_similarity(shared_count: int, ks_index: int, k: int) -> float:
    """
    Paper's combined score:
      s_A = 2 * [ ((s_J+1)/k) * (1 - s_K/(k+1)) ] /
                [ ((s_J+1)/k) + (1 - s_K/(k+1)) ]
    """
    a = (shared_count + 1.0) / k
    b = 1.0 - (ks_index / (k + 1.0))
    return 2.0 * a * b / (a + b)


def build_nsg(
    X: np.ndarray,
    k: int = 20,
    backend: str = "pynndescent",
    metric: str = "euclidean",
    batch_size: int = 2048,
    random_state: int = 0,
    ann_kwargs: dict | None = None,
) -> NSGGraph:
    """
    Build the directed neighborhood-similarity graph.
    """
    X = np.asarray(X)
    n = X.shape[0]
    if not (1 <= k < n):
        raise ValueError(f"k must satisfy 1 <= k < n; got k={k}, n={n}")

    ann_kwargs = {} if ann_kwargs is None else dict(ann_kwargs)

    if backend == "pynndescent":
        nbr_idx, nbr_dist = _knn_pynndescent(
            X, k=k, metric=metric, random_state=random_state, **ann_kwargs
        )
    elif backend == "hnswlib":
        hnsw_metric = {"euclidean": "l2", "l2": "l2", "cosine": "cosine", "ip": "ip"}.get(
            metric, metric
        )
        nbr_idx, nbr_dist = _knn_hnswlib(X, k=k, metric=hnsw_metric, **ann_kwargs)
    else:
        raise ValueError("backend must be 'pynndescent' or 'hnswlib'")

    # Sort each row by increasing neighbor distance.
    order = np.argsort(nbr_dist, axis=1)
    row = np.arange(n)[:, None]
    nbr_idx = nbr_idx[row, order].astype(np.int32, copy=False)
    nbr_dist = nbr_dist[row, order].astype(np.float32, copy=False)

    neighbor_ids_sorted = np.sort(nbr_idx, axis=1)
    indptr = np.arange(0, n * k + 1, k, dtype=np.int64)

    m = n * k
    ks_index = np.empty(m, dtype=np.int16)
    ks_stat = np.empty(m, dtype=np.float32)
    shared_count = np.empty(m, dtype=np.int16)
    jaccard = np.empty(m, dtype=np.float32)
    weight = np.empty(m, dtype=np.float32)

    # Marker array for fast shared-neighbor counts
    marks = np.full(n, -1, dtype=np.int32)

    pos = 0
    for start in range(0, n, batch_size):
        stop = min(n, start + batch_size)
        for i in range(start, stop):
            di = nbr_dist[i]
            ni_sorted = neighbor_ids_sorted[i]

            marks[ni_sorted] = i
            for j in nbr_idx[i]:
                nj_sorted = neighbor_ids_sorted[j]

                s_j = int(np.count_nonzero(marks[nj_sorted] == i))
                union = 2 * k - s_j
                jac = s_j / union if union > 0 else 0.0

                s_k, d_ks = _ks_discrete_index_and_stat(di, nbr_dist[j])
                s_a = _harmonic_similarity(s_j, s_k, k)

                ks_index[pos] = s_k
                ks_stat[pos] = d_ks
                shared_count[pos] = s_j
                jaccard[pos] = jac
                weight[pos] = s_a
                pos += 1

            marks[ni_sorted] = -1

    return NSGGraph(
        n_nodes=n,
        k=k,
        indptr=indptr,
        indices=nbr_idx.reshape(-1),
        distances=nbr_dist.reshape(-1),
        ks_index=ks_index,
        ks_stat=ks_stat,
        shared_count=shared_count,
        jaccard=jaccard,
        weight=weight,
    )


# ============================================================
# Threshold sweep + automatic threshold choice
# ============================================================


def _connected_components_labels(A: sparse.csr_matrix) -> np.ndarray:
    _, labels = csgraph.connected_components(A, directed=False, connection="weak")
    return labels.astype(np.int32, copy=False)


def _partition_stats(labels: np.ndarray) -> dict[str, float]:
    counts = np.bincount(labels)
    n = labels.size
    n_clusters = counts.size
    frac_singletons = float(np.mean(counts == 1))
    frac_tiny = float(np.mean(counts < max(3, int(0.001 * n))))
    largest_frac = float(counts.max() / n)
    entropy = float(-(counts / n * np.log((counts / n) + 1e-12)).sum())
    return {
        "n_clusters": float(n_clusters),
        "frac_singletons": frac_singletons,
        "frac_tiny": frac_tiny,
        "largest_frac": largest_frac,
        "entropy": entropy,
    }


def _partition_persistence_score(
    partitions: list[np.ndarray],
    i: int,
) -> float:
    """
    Stability against nearby thresholds using ARI.
    """
    cur = partitions[i]
    vals = []

    if i - 1 >= 0:
        vals.append(adjusted_rand_score(cur, partitions[i - 1]))
    if i + 1 < len(partitions):
        vals.append(adjusted_rand_score(cur, partitions[i + 1]))
    if i - 2 >= 0:
        vals.append(adjusted_rand_score(cur, partitions[i - 2]))
    if i + 2 < len(partitions):
        vals.append(adjusted_rand_score(cur, partitions[i + 2]))

    return float(np.mean(vals)) if vals else 0.0


def choose_threshold(
    graph: NSGGraph,
    thresholds: Sequence[float] | None = None,
    n_steps: int = 48,
    sym_rule: str = "mean",
    mutual_only: bool = False,
    min_clusters: int = 2,
    max_clusters: int | None = None,
) -> tuple[float, list[np.ndarray], list[dict[str, float]]]:
    n = graph.n_nodes
    if thresholds is None:
        thresholds = np.linspace(0.98, 0.05, n_steps, dtype=np.float32)
    else:
        thresholds = np.asarray(thresholds, dtype=np.float32)

    if max_clusters is None:
        max_clusters = max(10, int(np.sqrt(n)))

    partitions: list[np.ndarray] = []
    stats_list: list[dict[str, float]] = []

    for t in thresholds:
        A = graph.thresholded_symmetric(
            float(t),
            weighted=True,
            sym_rule=sym_rule,
            mutual_only=mutual_only,
        )
        labels = _connected_components_labels(A)
        stats = _partition_stats(labels)
        partitions.append(labels)
        stats_list.append(stats)

    scored: list[dict[str, float]] = []
    for i, t in enumerate(thresholds):
        stats = stats_list[i]
        persistence = _partition_persistence_score(partitions, i)

        n_clusters = stats["n_clusters"]
        largest_frac = stats["largest_frac"]
        frac_singletons = stats["frac_singletons"]
        frac_tiny = stats["frac_tiny"]

        valid = (
            n_clusters >= min_clusters
            and n_clusters <= max_clusters
            and largest_frac < 0.98
            and frac_singletons < 0.90
        )

        # Heuristic objective: stable partitions with non-degenerate sizes.
        score = (
            2.5 * persistence
            + 0.4 * stats["entropy"]
            - 1.2 * frac_singletons
            - 0.8 * frac_tiny
            - 0.5 * abs(largest_frac - 0.35)
        )

        scored.append(
            {
                "threshold": float(t),
                "score": float(score if valid else -np.inf),
                "persistence": float(persistence),
                **stats,
            }
        )

    best_idx = int(np.argmax([d["score"] for d in scored]))
    return float(thresholds[best_idx]), partitions, scored


# ============================================================
# Recursive bottleneck splitting with normalized cut
# ============================================================


def _mean_cut_weight_between(
    A: sparse.csr_matrix,
    mask_left: np.ndarray,
    mask_right: np.ndarray,
) -> float:
    left_idx = np.flatnonzero(mask_left)
    right_idx = np.flatnonzero(mask_right)
    if left_idx.size == 0 or right_idx.size == 0:
        return 0.0
    sub = A[left_idx][:, right_idx]
    if sub.nnz == 0:
        return 0.0
    return float(sub.data.mean())


def _component_internal_density(A_sub: sparse.csr_matrix) -> float:
    if A_sub.shape[0] <= 1:
        return 0.0
    deg = np.asarray(A_sub.sum(axis=1)).ravel()
    return float(deg.mean() / max(A_sub.shape[0] - 1, 1))


def _should_split_component(
    A_sub: sparse.csr_matrix,
    labels2: np.ndarray,
    min_component_size: int,
    min_split_fraction: float,
    max_bridge_weight_ratio: float,
) -> bool:
    n = A_sub.shape[0]
    if n < 2 * min_component_size:
        return False

    counts = np.bincount(labels2)
    if counts.size != 2:
        return False

    small = counts.min()
    if small < min_component_size or small / n < min_split_fraction:
        return False

    left = labels2 == 0
    right = ~left

    cut_mean = _mean_cut_weight_between(A_sub, left, right)
    within_left = _component_internal_density(A_sub[left][:, left])
    within_right = _component_internal_density(A_sub[right][:, right])
    within = 0.5 * (within_left + within_right)

    if within <= 1e-12:
        return False

    return cut_mean <= max_bridge_weight_ratio * within


def _spectral_bisect(
    A_sub: sparse.csr_matrix,
    random_state: int = 0,
) -> np.ndarray:
    # SpectralClustering wants a dense-like affinity input conceptually,
    # but accepts sparse affinity matrices.
    model = SpectralClustering(
        n_clusters=2,
        affinity="precomputed",
        assign_labels="kmeans",
        random_state=random_state,
        n_init=10,
    )
    labels = model.fit_predict(A_sub)
    return labels.astype(np.int32, copy=False)


def _recursive_split_component(
    A: sparse.csr_matrix,
    nodes: np.ndarray,
    next_label: int,
    labels_out: np.ndarray,
    tree: dict[int, dict],
    random_state: int,
    min_component_size: int,
    min_split_fraction: float,
    max_bridge_weight_ratio: float,
    max_recursion_depth: int,
    depth: int,
) -> int:
    node_id = int(next_label)

    if depth >= max_recursion_depth or nodes.size < 2 * min_component_size:
        labels_out[nodes] = next_label
        tree[node_id] = {"nodes": nodes, "split": False, "children": []}
        return next_label + 1

    A_sub = A[nodes][:, nodes].tocsr()

    try:
        split = _spectral_bisect(A_sub, random_state=random_state + depth)
    except Exception:
        labels_out[nodes] = next_label
        tree[node_id] = {"nodes": nodes, "split": False, "children": []}
        return next_label + 1

    if not _should_split_component(
        A_sub=A_sub,
        labels2=split,
        min_component_size=min_component_size,
        min_split_fraction=min_split_fraction,
        max_bridge_weight_ratio=max_bridge_weight_ratio,
    ):
        labels_out[nodes] = next_label
        tree[node_id] = {"nodes": nodes, "split": False, "children": []}
        return next_label + 1

    left_nodes = nodes[split == 0]
    right_nodes = nodes[split == 1]

    tree[node_id] = {"nodes": nodes, "split": True, "children": []}
    cur = next_label

    child_start = cur
    cur = _recursive_split_component(
        A,
        left_nodes,
        cur,
        labels_out,
        tree,
        random_state,
        min_component_size,
        min_split_fraction,
        max_bridge_weight_ratio,
        max_recursion_depth,
        depth + 1,
    )
    tree[node_id]["children"].append((child_start, left_nodes.size))

    child_start = cur
    cur = _recursive_split_component(
        A,
        right_nodes,
        cur,
        labels_out,
        tree,
        random_state,
        min_component_size,
        min_split_fraction,
        max_bridge_weight_ratio,
        max_recursion_depth,
        depth + 1,
    )
    tree[node_id]["children"].append((child_start, right_nodes.size))

    return cur


# ============================================================
# Tiny-cluster reassignment
# ============================================================


def _reassign_small_clusters(
    graph: NSGGraph,
    labels: np.ndarray,
    min_cluster_size: int,
) -> np.ndarray:
    counts = np.bincount(labels)
    large_ids = np.flatnonzero(counts >= min_cluster_size)
    if large_ids.size == 0:
        return labels

    small_ids = np.flatnonzero(counts < min_cluster_size)
    if small_ids.size == 0:
        return labels

    new_labels = labels.copy()
    row_counts = np.diff(graph.indptr)
    rows = np.repeat(np.arange(graph.n_nodes, dtype=np.int32), row_counts)

    # Directed edge list
    src = rows
    dst = graph.indices
    w = graph.weight

    large_set = set(int(x) for x in large_ids)

    for cid in small_ids:
        nodes = np.flatnonzero(labels == cid)
        if nodes.size == 0:
            continue

        score_by_cluster: dict[int, float] = {}
        mask = np.isin(src, nodes)
        for j, wij in zip(dst[mask], w[mask], strict=True):
            target_cluster = int(labels[j])
            if target_cluster in large_set and target_cluster != cid:
                score_by_cluster[target_cluster] = score_by_cluster.get(
                    target_cluster, 0.0
                ) + float(wij)

        if score_by_cluster:
            best = max(score_by_cluster.items(), key=lambda kv: kv[1])[0]
        else:
            # Fallback: nearest large cluster by outgoing graph weights unavailable
            best = int(large_ids[0])

        new_labels[nodes] = best

    # Relabel consecutively
    _, relabeled = np.unique(new_labels, return_inverse=True)
    return relabeled.astype(np.int32, copy=False)


# ============================================================
# Public API
# ============================================================


def auto_cluster_nsg(
    X: np.ndarray,
    k: int = 20,
    backend: str = "pynndescent",
    metric: str = "euclidean",
    ann_kwargs: dict | None = None,
    thresholds: Sequence[float] | None = None,
    n_steps: int = 48,
    sym_rule: str = "mean",
    mutual_only: bool = False,
    min_component_size: int = 25,
    min_split_fraction: float = 0.05,
    max_bridge_weight_ratio: float = 0.35,
    max_recursion_depth: int = 4,
    reassign_small: bool = True,
    final_min_cluster_size: int | None = None,
    batch_size: int = 2048,
    random_state: int = 0,
) -> AutoClusterResult:
    """
    Plug-and-play clustering wrapper around the neighborhood-similarity graph.

    Strategy:
      1. build NSG
      2. sweep thresholds and choose one by partition persistence
      3. get connected components on symmetrized filtered graph
      4. recursively split bottlenecked large components with spectral bisection
      5. optionally reassign tiny leftovers to nearest large cluster
    """
    if final_min_cluster_size is None:
        final_min_cluster_size = min_component_size

    graph = build_nsg(
        X=X,
        k=k,
        backend=backend,
        metric=metric,
        batch_size=batch_size,
        random_state=random_state,
        ann_kwargs=ann_kwargs,
    )

    threshold, partition_history, threshold_scores = choose_threshold(
        graph=graph,
        thresholds=thresholds,
        n_steps=n_steps,
        sym_rule=sym_rule,
        mutual_only=mutual_only,
        min_clusters=2,
    )

    A = graph.thresholded_symmetric(
        threshold=threshold,
        weighted=True,
        sym_rule=sym_rule,
        mutual_only=mutual_only,
    )

    base_labels = _connected_components_labels(A)
    base_counts = np.bincount(base_labels)

    labels = np.full(graph.n_nodes, -1, dtype=np.int32)
    tree: dict[int, dict] = {}
    next_label = 0

    for cid in np.argsort(-base_counts):
        nodes = np.flatnonzero(base_labels == cid)

        if nodes.size < 2 * min_component_size:
            labels[nodes] = next_label
            tree[next_label] = {"nodes": nodes, "split": False, "children": []}
            next_label += 1
            continue

        next_label = _recursive_split_component(
            A=A,
            nodes=nodes,
            next_label=next_label,
            labels_out=labels,
            tree=tree,
            random_state=random_state,
            min_component_size=min_component_size,
            min_split_fraction=min_split_fraction,
            max_bridge_weight_ratio=max_bridge_weight_ratio,
            max_recursion_depth=max_recursion_depth,
            depth=0,
        )

    # Consecutive relabel
    _, labels = np.unique(labels, return_inverse=True)
    labels = labels.astype(np.int32, copy=False)

    if reassign_small:
        labels = _reassign_small_clusters(
            graph=graph,
            labels=labels,
            min_cluster_size=final_min_cluster_size,
        )

    return AutoClusterResult(
        labels=labels,
        threshold=float(threshold),
        graph=graph,
        partition_history=partition_history,
        threshold_scores=threshold_scores,
        component_tree=tree,
    )
