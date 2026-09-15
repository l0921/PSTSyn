import os
import pickle
import argparse
from typing import Dict, List, Tuple

import numpy as np
import torch
from tqdm import tqdm


# ============================================================
# Drug-specific static PPI subgraph extraction
# ------------------------------------------------------------
# This script only builds drug-specific PPI subgraphs from DTI seeds.
# Cell-line specificity is NOT handled here. It is handled in the main
# model by cell-conditioned node/edge gating and attention.
# ============================================================


def seed_key_from_array(seed: np.ndarray) -> Tuple[int, ...]:
    """
    Use non-zero DTI seed node indices as the cache key.
    Drugs with the same target set share one cached PPI subgraph.
    """
    idx = np.where(np.asarray(seed) > 0)[0].astype(int).tolist()
    return tuple(sorted(set(idx)))


def build_adj(
    num_nodes: int,
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
) -> List[List[Tuple[int, float]]]:
    """
    Build a CPU adjacency list:
        adj[u] = [(v, weight), ...]

    Neighbor lists are sorted by descending edge weight for deterministic
    high-confidence expansion.
    """
    adj: List[List[Tuple[int, float]]] = [[] for _ in range(num_nodes)]

    src_all = edge_index[0].cpu().tolist()
    dst_all = edge_index[1].cpu().tolist()
    w_all = edge_weight.cpu().tolist()

    for u, v, w in zip(src_all, dst_all, w_all):
        adj[int(u)].append((int(v), float(w)))

    for u in range(num_nodes):
        adj[u].sort(key=lambda x: (-x[1], x[0]))

    return adj


def _top_items_by_score(
    items: Dict[int, float],
    limit: int,
) -> Dict[int, float]:
    """
    Keep only the strongest candidates to prevent 2-hop expansion from
    exploding on hub proteins.
    """
    if limit <= 0 or len(items) <= limit:
        return items

    top = sorted(items.items(), key=lambda x: (-x[1], x[0]))[:limit]
    return dict(top)


def extract_khop_nodes(
    seed_indices: List[int],
    adj: List[List[Tuple[int, float]]],
    k_hop: int = 2,
    max_nodes: int = 192,
    frontier_limit: int = 1536,
    candidate_pool_factor: int = 6,
) -> Tuple[List[int], Dict[int, int], Dict[int, float]]:
    """
    Extract a drug-specific k-hop PPI subgraph around DTI seed targets.

    Design:
        1. Seed target proteins are always kept.
        2. Default neighborhood is 2-hop.
        3. Nodes closer to seeds are preferred.
        4. Within the same hop distance, nodes with stronger PPI support
           are preferred.
        5. Expansion is capped by a frontier limit and candidate pool limit
           to reduce noisy high-degree hub expansion.

    Returns:
        nodes:
            Selected global PPI node indices.
        dist:
            Global node index -> shortest distance from any seed.
        score:
            Global node index -> accumulated path-support score.

    Notes:
        This is a static drug-specific subgraph. Cell-line-specific node
        activation and edge reweighting should remain in the main model.
    """
    if len(seed_indices) == 0:
        return [], {}, {}

    if k_hop < 0:
        raise ValueError(f"k_hop must be non-negative, got {k_hop}")

    if max_nodes <= 0:
        raise ValueError(f"max_nodes must be positive, got {max_nodes}")

    seed_indices = sorted(set(int(i) for i in seed_indices))
    seed_set = set(seed_indices)

    # dist stores shortest hop distance from any seed.
    dist: Dict[int, int] = {u: 0 for u in seed_indices}

    # score stores accumulated support. For seeds, use a large value so
    # they are always kept and ranked first among distance-0 nodes.
    score: Dict[int, float] = {u: 1e9 for u in seed_indices}

    # path_strength approximates the strongest known seed-to-node path.
    # For direct neighbors, strength is edge weight. For 2-hop nodes,
    # strength is roughly product of weights along the strongest path.
    path_strength: Dict[int, float] = {u: 1.0 for u in seed_indices}

    frontier = set(seed_indices)
    candidate_pool_limit = max(max_nodes * candidate_pool_factor, max_nodes)

    for hop in range(1, k_hop + 1):
        next_scores: Dict[int, float] = {}
        next_strength: Dict[int, float] = {}

        for u in frontier:
            base_strength = path_strength.get(u, 1.0)

            for v, w in adj[u]:
                v = int(v)
                w = float(w)

                # Do not re-expand already fixed closer nodes.
                if v in dist and dist[v] < hop:
                    continue

                # Use path product to avoid promoting weak 2-hop paths
                # connected through noisy hubs.
                candidate_strength = base_strength * w

                if v not in dist:
                    dist[v] = hop

                # Accumulate support from multiple parents at the same hop.
                next_scores[v] = next_scores.get(v, 0.0) + candidate_strength
                next_strength[v] = max(next_strength.get(v, 0.0), candidate_strength)

        # Remove seed nodes from expansion candidates. They are already kept.
        for s in seed_set:
            next_scores.pop(s, None)
            next_strength.pop(s, None)

        # Keep strongest frontier nodes only. This prevents huge 2-hop
        # neighborhoods from being dominated by generic hub proteins.
        next_scores = _top_items_by_score(next_scores, frontier_limit)

        for v, s in next_scores.items():
            score[v] = score.get(v, 0.0) + s
            path_strength[v] = max(path_strength.get(v, 0.0), next_strength.get(v, 0.0))

        frontier = set(next_scores.keys())
        if len(frontier) == 0:
            break

        # Keep the candidate pool bounded. Final selection still prefers
        # lower hop distance first.
        if len(score) >= candidate_pool_limit:
            break

    nodes = list(seed_indices)

    candidates = [n for n in score.keys() if n not in seed_set]
    candidates = sorted(
        candidates,
        key=lambda n: (dist.get(n, k_hop + 1), -score.get(n, 0.0), n),
    )

    remaining = max_nodes - len(nodes)
    if remaining > 0:
        nodes.extend(candidates[:remaining])

    # If a drug has more seed targets than max_nodes, keep all seed nodes.
    # This preserves the biological target set instead of silently dropping
    # drug targets; in normal use, target counts are far below max_nodes.

    return nodes, dist, score


def build_local_subgraph(
    nodes: List[int],
    seed: np.ndarray,
    adj: List[List[Tuple[int, float]]],
    dist: Dict[int, int],
    score: Dict[int, float],
    max_edges: int = 1536,
) -> Dict[str, torch.Tensor]:
    """
    Build the local PPI subgraph induced by selected nodes.

    Improvements over early-stop traversal:
        1. First collect all internal edges among selected nodes.
        2. If too many edges exist, keep the strongest biologically more
           reliable edges by priority.
        3. Normalize edge weights by target-node incoming weight sum for
           stable message passing.

    The returned fields are backward-compatible with the main model:
        nodes, local_seed, edge_index, edge_weight

    Extra metadata fields are also saved for analysis/debugging:
        seed_mask, node_distance, node_score, raw_edge_weight
    """
    if len(nodes) == 0:
        return {
            "nodes": torch.zeros(1, dtype=torch.long),
            "local_seed": torch.zeros(1, dtype=torch.float32),
            "seed_mask": torch.zeros(1, dtype=torch.float32),
            "node_distance": torch.full((1,), -1, dtype=torch.long),
            "node_score": torch.zeros(1, dtype=torch.float32),
            "edge_index": torch.zeros((2, 0), dtype=torch.long),
            "edge_weight": torch.zeros(0, dtype=torch.float32),
            "raw_edge_weight": torch.zeros(0, dtype=torch.float32),
        }

    node_to_local = {int(n): i for i, n in enumerate(nodes)}
    seed = np.asarray(seed, dtype=np.float32)

    edge_dict: Dict[Tuple[int, int], float] = {}

    for u in nodes:
        u = int(u)
        u_local = node_to_local[u]

        for v, w in adj[u]:
            v = int(v)
            if v not in node_to_local:
                continue

            v_local = node_to_local[v]
            key = (u_local, v_local)
            # If duplicate edges exist, keep the strongest one.
            edge_dict[key] = max(edge_dict.get(key, 0.0), float(w))

    edges = [(src, dst, w) for (src, dst), w in edge_dict.items()]

    if len(edges) > max_edges:
        local_seed_values = {node_to_local[n]: float(seed[n]) for n in nodes}

        def edge_priority(edge):
            src, dst, w = edge
            seed_bonus = int(local_seed_values.get(src, 0.0) > 0) + int(local_seed_values.get(dst, 0.0) > 0)
            src_global = nodes[src]
            dst_global = nodes[dst]
            hop_priority = min(dist.get(src_global, 10**9), dist.get(dst_global, 10**9))
            return (-seed_bonus, hop_priority, -float(w), src, dst)

        edges = sorted(edges, key=edge_priority)[:max_edges]
    else:
        edges = sorted(edges, key=lambda e: (e[0], e[1]))

    nodes_tensor = torch.tensor(nodes, dtype=torch.long)
    local_seed = torch.tensor(seed[nodes], dtype=torch.float32)
    seed_mask = (local_seed > 0).float()
    node_distance = torch.tensor([dist.get(int(n), -1) for n in nodes], dtype=torch.long)
    node_score = torch.tensor([score.get(int(n), 0.0) for n in nodes], dtype=torch.float32)

    if len(edges) == 0:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_weight = torch.zeros(0, dtype=torch.float32)
        raw_edge_weight = torch.zeros(0, dtype=torch.float32)
    else:
        src_list = [e[0] for e in edges]
        dst_list = [e[1] for e in edges]
        w_list = [e[2] for e in edges]

        edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
        raw_edge_weight = torch.tensor(w_list, dtype=torch.float32)

        # Normalize by target-node incoming weighted degree.
        dst = edge_index[1]
        deg = torch.zeros(len(nodes), dtype=torch.float32)
        deg.index_add_(0, dst, raw_edge_weight)
        deg = deg.clamp(min=1e-6)
        edge_weight = raw_edge_weight / deg[dst]

    return {
        "nodes": nodes_tensor,
        "local_seed": local_seed,
        "seed_mask": seed_mask,
        "node_distance": node_distance,
        "node_score": node_score,
        "edge_index": edge_index,
        "edge_weight": edge_weight,
        "raw_edge_weight": raw_edge_weight,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ppi_graph", type=str, default="data/PPI_flow/ppi_graph.pt")
    parser.add_argument("--drug_seed", type=str, default="data/PPI_flow/drug_seed.pkl")
    parser.add_argument("--out", type=str, default="data/PPI_flow/drug_ppi_subgraphs.pt")
    parser.add_argument("--k_hop", type=int, default=2)
    parser.add_argument("--max_nodes", type=int, default=192)
    parser.add_argument("--max_edges", type=int, default=1536)
    parser.add_argument(
        "--frontier_limit",
        type=int,
        default=1536,
        help="Max number of expansion frontier nodes kept at each hop.",
    )
    parser.add_argument(
        "--candidate_pool_factor",
        type=int,
        default=6,
        help="Candidate pool cap = max_nodes * candidate_pool_factor.",
    )
    args = parser.parse_args()

    if not os.path.exists(args.ppi_graph):
        raise FileNotFoundError(args.ppi_graph)

    if not os.path.exists(args.drug_seed):
        raise FileNotFoundError(args.drug_seed)

    print("Loading PPI graph:", args.ppi_graph)
    ppi_graph = torch.load(args.ppi_graph, map_location="cpu")

    num_nodes = int(ppi_graph["num_nodes"])
    edge_index = ppi_graph["edge_index"].long().cpu()
    edge_weight = ppi_graph["edge_weight"].float().cpu()

    print("PPI nodes:", num_nodes)
    print("PPI edges:", edge_index.size(1))

    print("Building adjacency list...")
    adj = build_adj(num_nodes, edge_index, edge_weight)

    print("Loading drug seed:", args.drug_seed)
    with open(args.drug_seed, "rb") as f:
        drug_seed = pickle.load(f)

    subgraphs_by_key = {}
    smiles_to_key = {}

    print(f"Extracting {args.k_hop}-hop drug-specific static PPI subgraphs...")
    for smiles, seed in tqdm(drug_seed.items()):
        seed = np.asarray(seed, dtype=np.float32)
        key = seed_key_from_array(seed)

        smiles_to_key[smiles] = key

        # Drugs sharing the same mapped target set reuse one cached subgraph.
        if key in subgraphs_by_key:
            continue

        seed_indices = list(key)

        nodes, dist, score = extract_khop_nodes(
            seed_indices=seed_indices,
            adj=adj,
            k_hop=args.k_hop,
            max_nodes=args.max_nodes,
            frontier_limit=args.frontier_limit,
            candidate_pool_factor=args.candidate_pool_factor,
        )

        subgraph = build_local_subgraph(
            nodes=nodes,
            seed=seed,
            adj=adj,
            dist=dist,
            score=score,
            max_edges=args.max_edges,
        )

        subgraphs_by_key[key] = subgraph

    out_obj = {
        "subgraphs_by_key": subgraphs_by_key,
        "smiles_to_key": smiles_to_key,
        "num_nodes": num_nodes,
        "k_hop": args.k_hop,
        "max_nodes": args.max_nodes,
        "max_edges": args.max_edges,
        "frontier_limit": args.frontier_limit,
        "candidate_pool_factor": args.candidate_pool_factor,
        "note": (
            "Static drug-specific PPI subgraphs. Cell-line-specific node/edge "
            "modulation is implemented in the main PSTSyn model."
        ),
    }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save(out_obj, args.out)

    print("Saved:", args.out)
    print("Unique drugs:", len(drug_seed))
    print("Unique subgraphs:", len(subgraphs_by_key))

    sizes = []
    edge_sizes = []
    seed_counts = []
    hop0_counts = []
    hop1_counts = []
    hop2_counts = []

    for sg in subgraphs_by_key.values():
        sizes.append(int(sg["nodes"].numel()))
        edge_sizes.append(int(sg["edge_index"].size(1)))
        seed_counts.append(int(sg["seed_mask"].sum().item()))
        distances = sg["node_distance"]
        hop0_counts.append(int((distances == 0).sum().item()))
        hop1_counts.append(int((distances == 1).sum().item()))
        hop2_counts.append(int((distances == 2).sum().item()))

    if len(sizes) > 0:
        print("Subgraph node size:")
        print("  min:", min(sizes))
        print("  max:", max(sizes))
        print("  mean:", sum(sizes) / len(sizes))

        print("Subgraph edge size:")
        print("  min:", min(edge_sizes))
        print("  max:", max(edge_sizes))
        print("  mean:", sum(edge_sizes) / len(edge_sizes))

        print("Seed count:")
        print("  min:", min(seed_counts))
        print("  max:", max(seed_counts))
        print("  mean:", sum(seed_counts) / len(seed_counts))

        print("Hop composition mean:")
        print("  hop0:", sum(hop0_counts) / len(hop0_counts))
        print("  hop1:", sum(hop1_counts) / len(hop1_counts))
        print("  hop2:", sum(hop2_counts) / len(hop2_counts))


if __name__ == "__main__":
    main()
