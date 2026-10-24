#!/usr/bin/env python3
"""Calculate betweenness, closeness, and eigenvector centrality from graph CSVs."""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from array import array
from pathlib import Path

from reports import export_top_rankings

try:
    import igraph as ig
    import numpy as np
except ImportError:
    ig = np = None


def log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


class Progress:
    def __init__(self, label: str, total: int):
        self.label, self.total = label, total
        self.started = self.last = time.monotonic()
        log(f"{label}: 0/{total}")

    def update(self, done: int) -> None:
        now = time.monotonic()
        if done != self.total and now - self.last < 1:
            return
        elapsed = now - self.started
        eta = elapsed * (self.total - done) / done if done else 0
        log(f"{self.label}: {done}/{self.total} ({done / max(1, self.total):.1%}), elapsed {elapsed:.0f}s, ETA ~{eta:.0f}s")
        self.last = now


def load_native_graph(nodes_path: Path, edges_path: Path):
    """Keep IDs and edges compact; avoid millions of Python adjacency sets."""
    ids = array("q")
    lookup = {}
    with nodes_path.open(newline="", encoding="utf-8-sig") as file:
        for row in csv.DictReader(file):
            user = int(row["id"])
            if user not in lookup:
                lookup[user] = len(ids)
                ids.append(user)
            if len(ids) % 250000 == 0:
                log(f"Loading nodes: {len(ids):,}")
    edges = array("q")
    with edges_path.open(newline="", encoding="utf-8-sig") as file:
        for index, row in enumerate(csv.DictReader(file), 1):
            pair = []
            for column in ("source", "target"):
                user = int(row[column])
                if user not in lookup:
                    lookup[user] = len(ids)
                    ids.append(user)
                pair.append(lookup[user])
            edges.extend(pair)
            if index % 500000 == 0:
                log(f"Loading edges: {index:,}")
    graph = ig.Graph(n=len(ids), edges=np.frombuffer(edges, dtype=np.int64).reshape(-1, 2), directed=False)
    graph.simplify()
    return graph, np.frombuffer(ids, dtype=np.int64)


def native_closeness(graph, groups: list[list[int]], samples: int, seed: int):
    """WF closeness, estimated from within-component mean pivot distances."""
    n = graph.vcount()
    scores = np.zeros(n)
    total = sum(min(samples, len(group)) for group in groups if len(group) > 1)
    progress = Progress("Closeness source traversals", total)
    done = 0
    for component_index, vertices in enumerate(groups):
        size = len(vertices)
        if size <= 1:
            continue
        subgraph = graph if size == n else graph.induced_subgraph(vertices)
        correction = (size - 1) / (n - 1)
        count = min(samples, size)
        if count == size:
            for start in range(0, size, 16):
                end = min(start + 16, size)
                scores[vertices[start:end]] = np.asarray(subgraph.closeness(vertices=range(start, end), normalized=True)) * correction
                done += end - start
                progress.update(done)
            continue
        pivots = random.Random(seed + component_index).sample(range(size), count)
        sums = np.zeros(size)
        counts = np.full(size, count, dtype=np.int32)
        for pivot in pivots:
            sums += np.asarray(subgraph.distances(source=[pivot])[0], dtype=np.float64)
            counts[pivot] -= 1  # Exclude distance to oneself, as in exact closeness.
            done += 1
            progress.update(done)
        scores[vertices] = correction * np.divide(counts, sums, out=np.zeros(size), where=sums > 0)
    progress.update(total)
    return scores


def native_eigenvector(graph, groups: list[list[int]]):
    scores = np.zeros(graph.vcount())
    progress = Progress("Eigenvector components", len(groups))
    for index, vertices in enumerate(groups, 1):
        if len(vertices) == 1:
            scores[vertices[0]] = 1.0
        else:
            subgraph = graph if len(vertices) == graph.vcount() else graph.induced_subgraph(vertices)
            vector = np.asarray(subgraph.eigenvector_centrality(), dtype=np.float64)
            if vector.sum() < 0:
                vector = -vector
            norm = np.linalg.norm(vector)
            if not np.isfinite(norm) or norm == 0:
                raise RuntimeError("eigenvector calculation returned invalid scores")
            scores[vertices] = vector / norm
        progress.update(index)
    return scores


def native_betweenness(graph, samples: int, seed: int):
    n = graph.vcount()
    if n <= 2:
        return np.zeros(n)
    count = min(n, samples)
    sources = range(n) if count == n else random.Random(seed).sample(range(n), count)
    raw = np.zeros(n)
    progress = Progress("Betweenness source traversals", count)
    for start in range(0, count, 4):
        end = min(start + 4, count)
        raw += np.asarray(graph.betweenness(directed=False, sources=sources[start:end]))
        progress.update(end)
    return raw * (n / count) * 2.0 / ((n - 1) * (n - 2))


def file_signature(path: Path) -> dict:
    stat = path.stat()
    return {"path": str(path.resolve()), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nodes", type=Path, default=Path("data/nodes.csv"))
    parser.add_argument("--edges", type=Path, default=Path("data/edges.csv"))
    parser.add_argument("--out", type=Path, default=Path("data/centrality.csv"))
    parser.add_argument("--samples", type=int, default=128, help="sampled sources; closeness samples per component (default: 128)")
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--exact", action="store_true", help="use all sources; very slow on large graphs")
    parser.add_argument("--top", type=int, default=10, help="ranking size for each metric")
    parser.add_argument("--rank-only", action="store_true", help="export tops from the existing --out CSV")
    args = parser.parse_args()
    if args.top < 1 or args.samples < 2:
        parser.error("--top must be positive; --samples must be at least 2")
    if args.rank_only:
        if args.exact:
            parser.error("--rank-only cannot recalculate scores with --exact")
        print(f"Rankings: {export_top_rankings(args.out, args.top).resolve()}")
        return 0
    if ig is None or np is None:
        parser.error("Install requirements.txt and run with .venv/bin/python")
    inputs = {"nodes": file_signature(args.nodes), "edges": file_signature(args.edges)}
    log("Loading graph with igraph...")
    graph, ids = load_native_graph(args.nodes, args.edges)
    n = graph.vcount()
    if n == 0:
        parser.error("The graph contains no nodes")
    groups = list(graph.connected_components())
    samples = n if args.exact else args.samples
    log(f"Graph: {n:,} nodes; {graph.ecount():,} edges; {len(groups):,} components.")
    log(f"Mode: {'EXACT' if samples >= n else f'APPROXIMATE, sources={samples}, seed={args.random_seed}'}")
    values = {
        "closeness": native_closeness(graph, groups, samples, args.random_seed),
        "eigenvector": native_eigenvector(graph, groups),
        "betweenness": native_betweenness(graph, samples, args.random_seed),
    }
    if inputs != {"nodes": file_signature(args.nodes), "edges": file_signature(args.edges)}:
        raise RuntimeError("Graph files changed during analysis; rerun on a stable snapshot")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_suffix(args.out.suffix + ".tmp")
    progress = Progress("Writing centrality CSV", n)
    columns = ("betweenness", "closeness", "eigenvector")
    with temporary.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["id", *columns])
        for index, user in enumerate(ids):
            writer.writerow([int(user), *(float(values[name][index]) for name in columns)])
            if (index + 1) % 50000 == 0 or index + 1 == n:
                progress.update(index + 1)
    metadata = {
        **inputs, "vertex_count": n, "edge_count": graph.ecount(), "component_count": len(groups),
        "exact": samples >= n, "samples": samples, "random_seed": args.random_seed,
        "igraph_version": ig.__version__,
        "betweenness_method": "exact" if samples >= n else "uniform source sampling, scaled by n/k",
        "closeness_method": "exact WF" if samples >= max(map(len, groups)) else "WF correction times reciprocal sampled mean distance within each component",
        "eigenvector_method": "native eigensolver, L2 normalized separately per component; singleton=1",
        "betweenness_sources": min(samples, n),
        "graph_coverage": "observed graph; missing friend lists and boundary edges can affect scores",
    }
    metadata_path = args.out.with_suffix(".meta.json")
    metadata_temporary = metadata_path.with_suffix(".json.tmp")
    metadata_temporary.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(args.out)
    metadata_temporary.replace(metadata_path)
    report = export_top_rankings(args.out, args.top)
    print(f"Results: {args.out.resolve()}\nMethods: {metadata_path.resolve()}\nRankings: {report.resolve()}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        log("\nInterrupted. Rerun to calculate the metrics from the beginning.")
        raise SystemExit(130)
    except (OSError, ValueError, RuntimeError) as exc:
        log(f"Analysis failed: {exc}")
        raise SystemExit(2)


