"""Deterministic average-linkage partition with a region capacity."""
from __future__ import annotations

import heapq

import numpy as np


def partition(graph: np.ndarray, cap: int, threshold: float) -> np.ndarray:
    """Merge supported regions greedily and return contiguous hard labels."""
    size = len(graph)
    weights = np.array((graph + graph.T) * 0.5, dtype=np.float64)
    np.fill_diagonal(weights, 0)
    counts = np.ones(size, dtype=np.int32)
    members = [[i] for i in range(size)]
    versions = [0] * size
    left, right = np.where(np.triu((weights >= threshold) & (weights > 0), 1))
    queue = [(-weights[i, j], int(i), int(j), 0, 0) for i, j in zip(left, right)] if cap > 1 else []
    heapq.heapify(queue)
    while queue:
        _, i, j, vi, vj = heapq.heappop(queue)
        if vi != versions[i] or vj != versions[j] or not counts[i] or not counts[j]:
            continue
        merged = int(counts[i] + counts[j])
        if merged > cap:
            continue
        members[i].extend(members[j])
        members[j].clear()
        versions[i] += 1
        versions[j] += 1
        counts[i], counts[j] = merged, 0
        weights[i] += weights[j]
        weights[:, i] = weights[i]
        weights[i, i] = 0.0
        candidates = np.flatnonzero((counts > 0) & (counts + merged <= cap))
        support = weights[i, candidates] / (counts[candidates] * merged)
        for k, score in zip(candidates, support):
            if k != i and score > 0 and score >= threshold:
                a, b = sorted((i, int(k)))
                heapq.heappush(queue, (-score, a, b, versions[a], versions[b]))
    labels = np.empty(size, dtype=np.int16)
    for region, nodes in enumerate(group for group in members if group):
        labels[nodes] = region
    return labels


def partition_sweep(graph: np.ndarray, caps: list[int], threshold: float) -> dict[int, np.ndarray]:
    """Reuse a larger-cap solution if none of its merges exceed the lower cap."""
    out = {}
    max_size = float("inf")
    for cap in sorted(caps, reverse=True):
        if max_size > cap:
            labels = partition(graph, cap, threshold)
            max_size = np.bincount(labels).max()
        out[cap] = labels
    return out


def membership_changes(labels: np.ndarray, baseline: np.ndarray) -> np.ndarray:
    """Flag each node whose set of regional peers changed, ignoring region IDs."""
    pairs, counts = np.unique(np.stack((labels, baseline), axis=1),
                             axis=0, return_inverse=False, return_counts=True)
    shared = np.zeros((int(labels.max()) + 1, int(baseline.max()) + 1), dtype=int)
    shared[pairs[:, 0], pairs[:, 1]] = counts
    together = shared[labels, baseline]
    return (together != np.bincount(labels)[labels]) | (together != np.bincount(baseline)[baseline])
