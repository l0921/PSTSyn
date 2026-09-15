
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GlobalAttention, HypergraphConv, TransformerConv
from torch_geometric.nn import global_mean_pool as gap


# Folded Morgan fingerprint width. Must match the value used when fingerprints
# are attached in the training script.
FINGERPRINT_BITS = int(os.environ.get('FINGERPRINT_BITS', '256'))
# Number of local PPI message-passing steps (kept small: 1-2).
LOCAL_STEPS = int(os.environ.get('LOCAL_STEPS', '2'))
class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class CellFeatureExtractorMLP(nn.Module):
    def __init__(self, input_dim=954, output_dim=128, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(256, output_dim),
        )

    def forward(self, x):
        if x.dim() == 3:
            x = x.squeeze(1)
        return self.net(x)


class MolecularDrugEncoder(nn.Module):

    def __init__(self, num_features_xd=78, output_dim=128, heads=4, dropout=0.2):
        super().__init__()
        hidden = num_features_xd  # 78
        self.conv1 = TransformerConv(num_features_xd, hidden, heads=heads)
        self.res1 = nn.Linear(num_features_xd, hidden * heads)
        self.bn1 = nn.BatchNorm1d(hidden * heads)
        self.ln1 = nn.LayerNorm(hidden * heads)

        self.conv2 = TransformerConv(hidden * heads, hidden, heads=heads)
        self.res2 = nn.Linear(hidden * heads, hidden * heads)
        self.bn2 = nn.BatchNorm1d(hidden * heads)
        self.ln2 = nn.LayerNorm(hidden * heads)

        final_dim = hidden * heads
        gate_nn = nn.Sequential(
            nn.Linear(final_dim, final_dim),
            nn.SiLU(),
            nn.Linear(final_dim, 1),
        )
        self.attn_pool = GlobalAttention(gate_nn)
        self.fc = nn.Linear(final_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.silu = nn.SiLU()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.ln1(self.silu(self.bn1(self.conv1(x, edge_index) + self.res1(x))))
        x = self.ln2(self.silu(self.bn2(self.conv2(x, edge_index) + self.res2(x))))
        pooled = self.attn_pool(x, data.batch)
        return self.silu(self.fc(self.dropout(pooled)))


class FingerprintEncoder(nn.Module):


    def __init__(self, n_bits=FINGERPRINT_BITS, output_dim=128, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_bits, 256),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(256, output_dim),
        )

    def forward(self, x):
        if x.dim() == 3:
            x = x.squeeze(1)
        return self.net(x)


class DualViewDrugFusion(nn.Module):

    def __init__(self, dim=128, dropout=0.2):
        super().__init__()
        self.gate = nn.Sequential(nn.Linear(dim * 2, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, graph_embed, fingerprint_embed):
        gate = torch.sigmoid(self.gate(torch.cat([graph_embed, fingerprint_embed], dim=-1)))
        return self.norm(graph_embed + self.dropout(gate * fingerprint_embed))


class DrugPerturbationToken(nn.Module):

    def __init__(self, dim=128, dropout=0.2):
        super().__init__()
        self.flow_proj = nn.Sequential(
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
        )
        self.gate_mlp = nn.Sequential(
            nn.Linear(dim * 3, dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, mol, seed, protein_embedding, flow):
        # seed: [B, num_nodes]; protein_embedding: [num_nodes, dim]; flow: [B, dim].
        denom = seed.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        seed_repr = (seed @ protein_embedding) / denom
        flow_repr = self.flow_proj(flow)
        pert = seed_repr + flow_repr
        gate = torch.sigmoid(self.gate_mlp(torch.cat([mol, seed_repr, flow_repr], dim=-1)))
        return self.norm(mol + gate * pert)


class SeedExpressionVulnerability(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 16),
            nn.SiLU(),
            nn.Linear(16, 1),
        )

    def forward(self, seed_pair, node_expression, observed):
        # seed_pair: [N, 2], node_expression/observed: [N].
        feats = torch.stack([node_expression, observed.to(node_expression.dtype)], dim=-1)
        vuln = 0.5 + torch.sigmoid(self.net(feats))  # (0.5, 1.5)
        return seed_pair * vuln


class PathwayAnchoredPerturbationNetwork(nn.Module):

    def __init__(
        self,
        num_nodes,
        emb_dim=128,
        dropout=0.2,
        subgraph_cache_path='data/PPI_flow/drug_ppi_subgraphs.pt',
        pathway_graph_path='data/PPI_flow/pathway_hypergraph.pt',
        local_steps=LOCAL_STEPS,
    ):
        super().__init__()
        self.local_steps = int(local_steps)
        self.num_nodes = int(num_nodes)
        self.emb_dim = emb_dim

        cache = torch.load(subgraph_cache_path, map_location='cpu', weights_only=False)
        self.subgraphs = self._prepare_subgraph_cache(cache['subgraphs_by_key'])
        self._pair_graph_cache = {}
        self._pair_graph_cache_max = int(os.environ.get('PAIR_GRAPH_CACHE_MAX', '20000'))

        pathway = self._load_pathways(pathway_graph_path)
        self.num_pathways = int(pathway['num_pathways'])
        self.register_buffer('ppi_to_cell_gene', self._invert_cell_mapping(pathway, self.num_nodes))
        protein_to_pathway_mask = torch.zeros(self.num_nodes, self.num_pathways, dtype=torch.bool)
        ppi_index = pathway['protein_pathway_index'][0].long()
        path_index = pathway['protein_pathway_index'][1].long()
        protein_to_pathway_mask[ppi_index, path_index] = True
        self.register_buffer('protein_to_pathway_mask', protein_to_pathway_mask)

        self.protein_emb = nn.Parameter(torch.randn(self.num_nodes, emb_dim) * 0.02)
        self.protein_input_norm = nn.LayerNorm(emb_dim)
        self.expression_proj = MLP(2, emb_dim, emb_dim, dropout)
        self.seed_proj = nn.Linear(1, emb_dim)
        self.seed_vulnerability = SeedExpressionVulnerability()
        # Expression-conditioned edge gate: lightweight MLP, no attention.
        edge_gate_in = emb_dim * 2 + 3  # src/dst tokens, raw weight, src/dst expression
        self.edge_gate_mlp = nn.Sequential(
            nn.Linear(edge_gate_in, emb_dim // 2),
            nn.SiLU(),
            nn.Linear(emb_dim // 2, 1),
        )
        self.dt = nn.Parameter(torch.tensor(0.5))
        self.prop_norm = nn.LayerNorm(emb_dim)
        self.pathway_base = nn.Parameter(torch.randn(1, self.num_pathways, emb_dim) * 0.02)
        self.pathway_norm = nn.LayerNorm(emb_dim)

    # ---- static data preparation -------------------------------------------------
    @staticmethod
    def _load_pathways(path):
        if not path or not os.path.exists(path):
            raise FileNotFoundError(f'Missing real Reactome pathway hypergraph: {path}')
        obj = torch.load(path, map_location='cpu', weights_only=False)
        required = {'num_pathways', 'protein_pathway_index', 'pathway_hyperedge_index'}
        missing = required - set(obj)
        if missing:
            raise KeyError(f'pathway_hypergraph.pt missing fields: {sorted(missing)}')
        return obj

    @staticmethod
    def _invert_cell_mapping(pathway, num_nodes):
        mapping = pathway.get('cell_gene_to_ppi')
        if mapping is None:
            raise KeyError('Rebuild pathway_hypergraph.pt to include real cell_gene_to_ppi mapping')
        inverse = torch.full((num_nodes,), -1, dtype=torch.long)
        mapping = torch.as_tensor(mapping, dtype=torch.long)
        valid = mapping >= 0
        inverse[mapping[valid]] = torch.arange(mapping.numel(), dtype=torch.long)[valid]
        return inverse

    @staticmethod
    def _prepare_subgraph_cache(subgraphs):
        prepared = {}
        for key, sg in subgraphs.items():
            prepared[tuple(key)] = {
                'nodes': sg['nodes'].long().cpu().contiguous(),
                'edge_index': sg['edge_index'].long().cpu().contiguous(),
                'raw_edge_weight': sg.get('raw_edge_weight', sg['edge_weight']).float().cpu().contiguous(),
            }
        return prepared

    @staticmethod
    def _seed_key(seed):
        if seed.dim() > 1:
            seed = seed.view(-1)
        return tuple(torch.nonzero(seed > 0, as_tuple=False).view(-1).detach().cpu().tolist())

    @staticmethod
    def _merge_sample_subgraphs(sg1, sg2):
        chunks = [sg for sg in (sg1, sg2) if sg is not None]
        if not chunks:
            return torch.zeros(1, dtype=torch.long), torch.zeros((2, 0), dtype=torch.long), torch.zeros(0)
        nodes = torch.unique(torch.cat([sg['nodes'] for sg in chunks]), sorted=True)
        n_nodes = int(nodes.numel())
        edge_src, edge_dst, raw_chunks = [], [], []
        for sg in chunks:
            local_nodes = sg['nodes']
            edge_index = sg['edge_index']
            if edge_index.numel() == 0:
                continue
            src_global = local_nodes[edge_index[0]]
            dst_global = local_nodes[edge_index[1]]
            edge_src.append(torch.searchsorted(nodes, src_global))
            edge_dst.append(torch.searchsorted(nodes, dst_global))
            raw_chunks.append(sg['raw_edge_weight'])
        if not edge_src:
            return nodes, torch.zeros((2, 0), dtype=torch.long), torch.zeros(0)
        src = torch.cat(edge_src).long()
        dst = torch.cat(edge_dst).long()
        raw = torch.cat(raw_chunks).float()
        code = src * n_nodes + dst
        unique_code, inverse = torch.unique(code, sorted=True, return_inverse=True)
        raw_weight = raw.new_full((unique_code.numel(),), float('-inf'))
        raw_weight.scatter_reduce_(0, inverse, raw, reduce='amax', include_self=True)
        edge_index = torch.stack([unique_code // n_nodes, unique_code % n_nodes], dim=0).long().contiguous()
        return nodes, edge_index, raw_weight.contiguous()

    def _get_pair_graph(self, key1, key2):
        merge_key = (key1, key2) if key1 <= key2 else (key2, key1)
        cached = self._pair_graph_cache.get(merge_key)
        if cached is not None:
            return cached
        merged = self._merge_sample_subgraphs(
            self.subgraphs.get(key1), self.subgraphs.get(key2)
        )
        if self._pair_graph_cache_max > 0:
            if len(self._pair_graph_cache) >= self._pair_graph_cache_max:
                self._pair_graph_cache.pop(next(iter(self._pair_graph_cache)))
            self._pair_graph_cache[merge_key] = merged
        return merged

    def _build_pair_local_graph(self, d1_seed, d2_seed, device):
        d1_seed_cpu = d1_seed.detach().cpu() if d1_seed.is_cuda else d1_seed.detach()
        d2_seed_cpu = d2_seed.detach().cpu() if d2_seed.is_cuda else d2_seed.detach()
        nodes_all, seeds_all, edges_all, raw_all, batch_all = [], [], [], [], []
        offset = 0
        for batch_id in range(d1_seed_cpu.size(0)):
            key1 = self._seed_key(d1_seed_cpu[batch_id])
            key2 = self._seed_key(d2_seed_cpu[batch_id])
            nodes, edge_index, raw_weight = self._get_pair_graph(key1, key2)
            n_local = nodes.numel()
            nodes_all.append(nodes.to(device, non_blocking=True))
            seed_pair = torch.stack(
                [d1_seed_cpu[batch_id, nodes], d2_seed_cpu[batch_id, nodes]], dim=-1
            ).to(device, non_blocking=True)
            seeds_all.append(seed_pair)
            batch_all.append(torch.full((n_local,), batch_id, dtype=torch.long, device=device))
            if edge_index.numel():
                edges_all.append(edge_index.to(device, non_blocking=True) + offset)
                raw_all.append(raw_weight.to(device, non_blocking=True))
            offset += n_local
        nodes = torch.cat(nodes_all)
        seed_pair = torch.cat(seeds_all)
        batch_vec = torch.cat(batch_all)
        if not edges_all:
            return (
                nodes, seed_pair,
                torch.zeros((2, 0), dtype=torch.long, device=device),
                torch.zeros(0, device=device),
                torch.zeros(0, device=device),
                batch_vec,
            )
        edge_index = torch.cat(edges_all, dim=1)
        raw_weight = torch.cat(raw_all)
        degree = torch.zeros(nodes.numel(), dtype=raw_weight.dtype, device=device)
        degree.index_add_(0, edge_index[1], raw_weight)
        edge_weight = raw_weight / degree[edge_index[1]].clamp_min(1e-6)
        return nodes, seed_pair, edge_index, edge_weight, raw_weight, batch_vec

    def _node_expression_values(self, cell_features, nodes, batch_vec):
        expression = torch.log1p(cell_features.clamp_min(0.0))
        expression = (expression - expression.mean(dim=-1, keepdim=True)) / expression.std(
            dim=-1, keepdim=True
        ).clamp_min(1e-5)
        gene_index = self.ppi_to_cell_gene[nodes]
        observed = gene_index >= 0
        safe_index = gene_index.clamp_min(0)
        node_expression = expression[batch_vec, safe_index] * observed.to(expression.dtype)
        return node_expression, observed

    def _local_propagation(self, h, signal, edge_index, edge_weight, raw_weight, node_expression):
        """Cell-expression-modulated PPI diffusion of node states and per-drug signal."""
        if edge_index.numel() == 0:
            return h, signal
        src, dst = edge_index
        dt_scale = torch.sigmoid(self.dt)
        for _ in range(self.local_steps):
            gate_input = torch.cat([
                h[src], h[dst],
                raw_weight.unsqueeze(-1),
                node_expression[src].unsqueeze(-1),
                node_expression[dst].unsqueeze(-1),
            ], dim=-1)
            edge_gate = torch.sigmoid(self.edge_gate_mlp(gate_input).squeeze(-1))
            combined_weight = (edge_weight * edge_gate).unsqueeze(-1)

            aggregated = torch.zeros_like(h)
            aggregated.index_add_(0, dst, h[src] * combined_weight)
            h = self.prop_norm(h + dt_scale * aggregated)

            incoming = torch.zeros_like(signal)
            incoming.index_add_(0, dst, signal[src] * combined_weight)
            signal = signal + dt_scale * incoming
        return h, signal

    def forward(self, d1_seed, d2_seed, cell_features, has_ppi_pair):
        if d1_seed.dim() == 3:
            d1_seed = d1_seed.squeeze(1)
            d2_seed = d2_seed.squeeze(1)
        device = cell_features.device
        batch_size = d1_seed.size(0)
        nodes, seed_pair, edge_index, edge_weight, raw_weight, batch_vec = self._build_pair_local_graph(
            d1_seed, d2_seed, device
        )
        node_expression, observed = self._node_expression_values(cell_features, nodes, batch_vec)
        # Innovation 1: expression-conditioned target vulnerability.
        eff_seed = self.seed_vulnerability(seed_pair, node_expression, observed)

        # Symmetric protein-token seed injection (sum of both drug channels).
        expr_context = self.expression_proj(
            torch.stack([node_expression, observed.to(node_expression.dtype)], dim=-1)
        )
        seed_context = self.seed_proj(eff_seed.sum(dim=-1, keepdim=True))
        protein_tokens = self.protein_input_norm(self.protein_emb[nodes] + expr_context + seed_context)

        h, signal = self._local_propagation(
            protein_tokens, eff_seed, edge_index, edge_weight, raw_weight, node_expression
        )

        # Node -> pathway membership over the local subgraph (sparse aggregation).
        node_pathway = self.protein_to_pathway_mask[nodes.clamp(0, self.num_nodes - 1)]  # [N, P]
        node_idx, path_idx = torch.nonzero(node_pathway, as_tuple=True)  # membership edges
        flat_slot = batch_vec[node_idx] * self.num_pathways + path_idx  # into [B*P]

        n_slots = batch_size * self.num_pathways
        path_token_sum = torch.zeros(n_slots, self.emb_dim, device=device)
        path_count = torch.zeros(n_slots, device=device)
        path_act = torch.zeros(n_slots, 2, device=device)
        path_token_sum.index_add_(0, flat_slot, h[node_idx])
        path_count.index_add_(0, flat_slot, torch.ones_like(flat_slot, dtype=path_count.dtype))
        path_act.index_add_(0, flat_slot, signal[node_idx])

        path_token_sum = path_token_sum.view(batch_size, self.num_pathways, self.emb_dim)
        path_count = path_count.view(batch_size, self.num_pathways)
        path_act = path_act.view(batch_size, self.num_pathways, 2)

        denom = path_count.clamp_min(1.0).unsqueeze(-1)
        pathway_tokens = path_token_sum / denom + self.pathway_base
        pathway_tokens = self.pathway_norm(pathway_tokens)
        drug_activation = path_act / denom  # [B, P, 2]
        active_mask = path_count > 0  # [B, P]

        has_ppi_pair = has_ppi_pair.view(-1, 1).to(pathway_tokens.dtype)
        pathway_tokens = pathway_tokens * has_ppi_pair.unsqueeze(-1)
        drug_activation = drug_activation * has_ppi_pair.unsqueeze(-1)
        active_mask = active_mask & has_ppi_pair.bool()

        # Per-drug PPI-flow summary: seed-signal-weighted pool of diffused protein tokens.
        # This closes the gap where Innovation 1 only fed pathway_context before.
        w1 = signal[:, 0].clamp_min(0.0)
        w2 = signal[:, 1].clamp_min(0.0)
        flow1 = torch.zeros(batch_size, self.emb_dim, device=device)
        flow2 = torch.zeros(batch_size, self.emb_dim, device=device)
        den1 = torch.zeros(batch_size, 1, device=device)
        den2 = torch.zeros(batch_size, 1, device=device)
        flow1.index_add_(0, batch_vec, h * w1.unsqueeze(-1))
        flow2.index_add_(0, batch_vec, h * w2.unsqueeze(-1))
        den1.index_add_(0, batch_vec, w1.unsqueeze(-1))
        den2.index_add_(0, batch_vec, w2.unsqueeze(-1))
        flow1 = (flow1 / den1.clamp_min(1e-6)) * has_ppi_pair
        flow2 = (flow2 / den2.clamp_min(1e-6)) * has_ppi_pair
        return pathway_tokens, drug_activation, active_mask, flow1, flow2


class PairPathwayCrosstalk(nn.Module):

    def __init__(self, pathway_hyperedge_index, num_pathways, emb_dim=128, dropout=0.2):
        super().__init__()
        self.num_pathways = int(num_pathways)
        self.emb_dim = emb_dim
        hyperedge_index = pathway_hyperedge_index.long()
        self.register_buffer('pathway_hyperedge_index', hyperedge_index)
        self.num_hyperedges = (
            int(hyperedge_index[1].max().item()) + 1 if hyperedge_index.numel() else 0
        )
        self._batched_hyperedge_cache = {}

        self.gate_mlp = nn.Sequential(
            nn.Linear(emb_dim + 3, emb_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(emb_dim, emb_dim),
        )
        self.hyper = HypergraphConv(emb_dim, emb_dim, heads=1, concat=False, dropout=dropout)
        self.norm = nn.LayerNorm(emb_dim)
        self.score = nn.Sequential(nn.LayerNorm(emb_dim), nn.Linear(emb_dim, 1))

    def _get_batched_hyperedge_index(self, batch_size, device):
        if self.pathway_hyperedge_index.numel() == 0:
            return self.pathway_hyperedge_index
        key = (batch_size, device.type, device.index)
        cached = self._batched_hyperedge_cache.get(key)
        if cached is not None and cached.device == device:
            return cached
        base = self.pathway_hyperedge_index.to(device)
        edge_count = base.size(1)
        repeated = base.repeat(1, batch_size).clone()
        offsets = torch.arange(batch_size, device=device).repeat_interleave(edge_count)
        repeated[0] += offsets * self.num_pathways
        repeated[1] += offsets * self.num_hyperedges
        self._batched_hyperedge_cache[key] = repeated
        return repeated

    def forward(self, pathway_tokens, drug_activation, active_mask):
        batch_size = pathway_tokens.size(0)
        device = pathway_tokens.device
        a1 = drug_activation[..., 0]
        a2 = drug_activation[..., 1]
        shared = torch.minimum(a1, a2)
        exclusive = (a1 - a2).abs()
        complementarity = exclusive / (a1 + a2 + 1e-6)  # symmetric in a1,a2
        pair_feats = torch.stack([shared, exclusive, complementarity], dim=-1)  # [B, P, 3]

        gate = torch.sigmoid(self.gate_mlp(torch.cat([pathway_tokens, pair_feats], dim=-1)))
        updated = pathway_tokens * gate
        updated = updated * active_mask.unsqueeze(-1).to(updated.dtype)

        if self.pathway_hyperedge_index.numel():
            flat = updated.reshape(batch_size * self.num_pathways, self.emb_dim)
            batched_index = self._get_batched_hyperedge_index(batch_size, device)
            crosstalk = self.hyper(flat, batched_index).view(batch_size, self.num_pathways, self.emb_dim)
        else:
            crosstalk = torch.zeros_like(updated)
        tokens = self.norm(updated + crosstalk) * active_mask.unsqueeze(-1).to(updated.dtype)

        mask = active_mask.bool()
        valid = mask.any(dim=-1)
        score = self.score(tokens).squeeze(-1)
        safe_mask = mask.clone()
        safe_mask[:, 0] |= ~valid
        score = score.masked_fill(~safe_mask, torch.finfo(score.dtype).min)
        weight = torch.softmax(score, dim=-1) * mask.to(score.dtype)
        weight = weight / weight.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        pathway_context = torch.einsum('bp,bpd->bd', weight, tokens)
        pathway_context = pathway_context * valid.unsqueeze(-1).to(pathway_context.dtype)
        return pathway_context


class StateTransitionBlock(nn.Module):
    """Single-drug perturbation-driven residual cell-state update."""

    def __init__(self, emb_dim=128, dropout=0.2):
        super().__init__()
        in_dim = emb_dim * 5
        self.gate_mlp = nn.Sequential(
            nn.Linear(in_dim, emb_dim), nn.SiLU(), nn.Dropout(dropout), nn.Linear(emb_dim, emb_dim)
        )
        self.delta_mlp = nn.Sequential(
            nn.Linear(in_dim, emb_dim), nn.SiLU(), nn.Dropout(dropout), nn.Linear(emb_dim, emb_dim)
        )
        self.norm = nn.LayerNorm(emb_dim)

    def forward(self, c0, p, pathway_context):
        features = torch.cat([c0, p, c0 * p, (c0 - p).abs(), pathway_context], dim=-1)
        gate = torch.sigmoid(self.gate_mlp(features))
        return self.norm(c0 + gate * self.delta_mlp(features))


class PairStateTransitionBlock(nn.Module):
    """Order-invariant pair-drug perturbation-driven cell-state update."""

    def __init__(self, emb_dim=128, dropout=0.2):
        super().__init__()
        in_dim = emb_dim * 8
        self.gate_mlp = nn.Sequential(
            nn.Linear(in_dim, emb_dim), nn.SiLU(), nn.Dropout(dropout), nn.Linear(emb_dim, emb_dim)
        )
        self.delta_mlp = nn.Sequential(
            nn.Linear(in_dim, emb_dim), nn.SiLU(), nn.Dropout(dropout), nn.Linear(emb_dim, emb_dim)
        )
        self.norm = nn.LayerNorm(emb_dim)

    def forward(self, c0, p1, p2, pathway_context):
        pair_mean = (p1 + p2) / 2
        pair_prod = p1 * p2
        pair_diff = (p1 - p2).abs()
        pair_max = torch.maximum(p1, p2)
        features = torch.cat(
            [c0, pair_mean, pair_prod, pair_diff, pair_max, c0 * pair_mean, (c0 - pair_mean).abs(), pathway_context],
            dim=-1,
        )
        gate = torch.sigmoid(self.gate_mlp(features))
        return self.norm(c0 + gate * self.delta_mlp(features))


class PSTSyn(nn.Module):

    def __init__(
        self,
        n_output=2,
        num_features_xd=78,
        num_features_xt=954,
        output_dim=128,
        dropout=0.2,
        ppi_graph_path='data/PPI_flow/ppi_graph.pt',
        ppi_subgraph_path='data/PPI_flow/drug_ppi_subgraphs.pt',
        pathway_graph_path='data/PPI_flow/pathway_hypergraph.pt',
        fingerprint_bits=FINGERPRINT_BITS,
    ):
        super().__init__()
        # Base drug representation: molecular graph and Morgan fingerprint fusion.
        self.drug_encoder = MolecularDrugEncoder(num_features_xd, output_dim, dropout=dropout)
        self.fingerprint_bits = fingerprint_bits
        self.fingerprint_encoder = FingerprintEncoder(fingerprint_bits, output_dim, dropout)
        self.mol_fusion = DualViewDrugFusion(output_dim, dropout)
        self.cell_encoder = CellFeatureExtractorMLP(num_features_xt, output_dim, dropout)

        ppi_graph = torch.load(ppi_graph_path, map_location='cpu', weights_only=False)
        self.ppi_num_nodes = int(ppi_graph['num_nodes'])

        # Innovation 1.
        self.perturbation = PathwayAnchoredPerturbationNetwork(
            self.ppi_num_nodes, output_dim, dropout, ppi_subgraph_path, pathway_graph_path,
        )
        self.drug_token = DrugPerturbationToken(output_dim, dropout)
        # Innovation 2.
        pathway = torch.load(pathway_graph_path, map_location='cpu', weights_only=False)
        self.pair_pathway = PairPathwayCrosstalk(
            pathway['pathway_hyperedge_index'], int(pathway['num_pathways']), output_dim, dropout,
        )

        self.single_state_transition = StateTransitionBlock(output_dim, dropout)
        self.pair_state_transition = PairStateTransitionBlock(output_dim, dropout)

        # Single readout head. Features: counterfactual states + pathway + cell
        # + order-invariant drug-pair descriptors (mean / product / abs-diff).
        self.classifier = nn.Sequential(
            nn.Linear(output_dim * 9, output_dim * 2),
            nn.LayerNorm(output_dim * 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim * 2, n_output),
        )

    @staticmethod
    def _cell_features(data):
        cell = data.cell.float()
        return cell.squeeze(1) if cell.dim() == 3 else cell

    @staticmethod
    def _ppi_seed(data):
        seed = data.ppi_seed.float()
        return seed.squeeze(1) if seed.dim() == 3 else seed

    def _fingerprint_features(self, data, batch_size, device):
        fp = getattr(data, 'fingerprint', None)
        if fp is None:
            return torch.zeros(batch_size, self.fingerprint_bits, device=device)
        return fp.float().to(device).view(batch_size, -1)

    def forward(self, data1, data2, return_aux=False):
        # Base drug representation: molecular graph and Morgan fingerprint fusion.
        d1_graph = self.drug_encoder(data1)
        d2_graph = self.drug_encoder(data2)
        batch_size = d1_graph.size(0)
        d1_fp = self.fingerprint_encoder(self._fingerprint_features(data1, batch_size, d1_graph.device))
        d2_fp = self.fingerprint_encoder(self._fingerprint_features(data2, batch_size, d2_graph.device))
        d1_mol = self.mol_fusion(d1_graph, d1_fp)
        d2_mol = self.mol_fusion(d2_graph, d2_fp)

        cell_features = self._cell_features(data1)
        c0 = self.cell_encoder(cell_features)
        has_d1 = data1.has_ppi.view(-1, 1).to(c0.dtype)
        has_d2 = data2.has_ppi.view(-1, 1).to(c0.dtype)
        has_pair = torch.maximum(has_d1, has_d2)

        # Innovation 1: DTI seed -> cell-conditioned PPI perturbation -> pathway tokens.
        d1_seed = self._ppi_seed(data1) * has_d1
        d2_seed = self._ppi_seed(data2) * has_d2
        pathway_tokens, drug_activation, active_mask, flow1, flow2 = self.perturbation(
            d1_seed, d2_seed, cell_features, has_pair.squeeze(-1)
        )
        # Innovation 2: drug-pair pathway complementarity + crosstalk -> one context.
        pathway_context = self.pair_pathway(pathway_tokens, drug_activation, active_mask)

        # Drug perturbation tokens: mol + DTI seed + PPI-flow (shared parameters).
        protein_embedding = self.perturbation.protein_emb
        d1_token = self.drug_token(d1_mol, d1_seed, protein_embedding, flow1)
        d2_token = self.drug_token(d2_mol, d2_seed, protein_embedding, flow2)

        # Additive counterfactual state transition.
        c_a = self.single_state_transition(c0, d1_token, pathway_context)
        c_b = self.single_state_transition(c0, d2_token, pathway_context)
        c_ab = self.pair_state_transition(c0, d1_token, d2_token, pathway_context)
        delta_a = c_a - c0
        delta_b = c_b - c0
        c_add = c0 + delta_a + delta_b
        r_syn = c_ab - c_add

        # Order-invariant pair descriptors (no d1||d2 concatenation).
        pair_mean = 0.5 * (d1_token + d2_token)
        pair_prod = d1_token * d2_token
        pair_diff = (d1_token - d2_token).abs()
        features = torch.cat([
            c_ab, c_add, r_syn, torch.abs(r_syn), pathway_context,
            c0, pair_mean, pair_prod, pair_diff,
        ], dim=-1)
        logits = self.classifier(features)

        if return_aux:
            aux = {
                'c0': c0,
                'c_add': c_add,
                'c_ab': c_ab,
                'r_syn': r_syn,
                'pathway_context': pathway_context,
                'active_pathway_mask': active_mask,
            }
            return logits, aux
        return logits
