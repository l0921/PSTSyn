import os
import math
import pickle
import zipfile
import urllib.request

import pandas as pd
import torch

DATA_DIR = 'data'
PPI_FLOW_DIR = os.path.join(DATA_DIR, 'PPI_flow')
STRING_DIR = os.path.join(DATA_DIR, 'STRING_PPI')

PPI_GRAPH = os.path.join(PPI_FLOW_DIR, 'ppi_graph.pt')
DRUG_SEED = os.path.join(PPI_FLOW_DIR, 'drug_seed.pkl')
PPI_INFO = os.path.join(STRING_DIR, '9606.protein.info.v12.0.txt')
PPI_ALIASES = os.path.join(STRING_DIR, '9606.protein.aliases.v12.0.txt')
OUT_FILE = os.path.join(PPI_FLOW_DIR, 'pathway_hypergraph.pt')
REACTOME_ZIP = os.path.join(PPI_FLOW_DIR, 'ReactomePathways.gmt.zip')
REACTOME_URL = 'https://reactome.org/download/current/ReactomePathways.gmt.zip'

CELL_FEATURE_FILE = os.path.join(DATA_DIR, 'new_cell_features_954.csv')

MIN_PROTEINS = 8
MAX_PROTEINS = 400
MAX_PATHWAYS = 128
MIN_OVERLAP = 4
MIN_JACCARD = 0.035
MAX_NEIGHBORS = 8
MAX_REDUNDANCY = 0.85


def norm(x):
    if pd.isna(x):
        return ''
    return str(x).strip().upper()


def download_reactome():
    if os.path.exists(REACTOME_ZIP) and os.path.getsize(REACTOME_ZIP) > 0:
        return
    print('Downloading Reactome pathway GMT...')
    req = urllib.request.Request(REACTOME_URL, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req, timeout=60) as r:
        data = r.read()
    with open(REACTOME_ZIP, 'wb') as f:
        f.write(data)
    print('Saved:', REACTOME_ZIP, len(data), 'bytes')


def iter_reactome_gmt():
    with zipfile.ZipFile(REACTOME_ZIP, 'r') as zf:
        names = [n for n in zf.namelist() if n.endswith('.gmt')]
        if not names:
            raise RuntimeError('No .gmt file found in Reactome zip')
        with zf.open(names[0]) as f:
            for raw in f:
                line = raw.decode('utf-8', 'replace').strip()
                if not line:
                    continue
                parts = line.split('\t')
                if len(parts) < 3:
                    continue
                name, stable_id, genes = parts[0], parts[1], parts[2:]
                if 'R-HSA-' not in stable_id and 'Homo sapiens' not in name:
                    continue
                yield name, stable_id, [norm(g) for g in genes if norm(g)]


def load_reactome_pathways():
    pathways = []
    reactome_genes = set()
    for name, stable_id, genes in iter_reactome_gmt():
        genes = sorted(set(genes))
        reactome_genes.update(genes)
        pathways.append({'name': name, 'stable_id': stable_id, 'genes': genes})
    print('Reactome HSA pathways:', len(pathways))
    print('Reactome unique genes:', len(reactome_genes))
    return pathways, reactome_genes


def load_gene_to_ppi_idx(ppi_graph, reactome_genes):
    protein_ids = ppi_graph['protein_ids']
    protein_to_idx = {sid: i for i, sid in enumerate(protein_ids)}
    ppi_sid_set = set(protein_to_idx)
    gene_to_idx = {}

    info = pd.read_csv(PPI_INFO, sep='\t', usecols=['#string_protein_id', 'preferred_name'])
    info = info[info['#string_protein_id'].isin(ppi_sid_set)]
    for sid, gene in zip(info['#string_protein_id'], info['preferred_name']):
        gene = norm(gene)
        if gene in reactome_genes:
            gene_to_idx.setdefault(gene, set()).add(protein_to_idx[sid])

    print('Mapped by STRING preferred_name:', len(gene_to_idx))

    alias_cols = ['#string_protein_id', 'alias']
    for chunk in pd.read_csv(PPI_ALIASES, sep='\t', usecols=alias_cols, chunksize=500000):
        chunk = chunk[chunk['#string_protein_id'].isin(ppi_sid_set)]
        if chunk.empty:
            continue
        chunk['alias_norm'] = chunk['alias'].astype(str).str.strip().str.upper()
        chunk = chunk[chunk['alias_norm'].isin(reactome_genes)]
        for sid, alias in zip(chunk['#string_protein_id'], chunk['alias_norm']):
            gene_to_idx.setdefault(alias, set()).add(protein_to_idx[sid])

    print('Mapped after STRING aliases:', len(gene_to_idx))
    return gene_to_idx


def load_cell_gene_to_ppi_idx(ppi_graph):
    """Map the 954 measured Ensembl genes to STRING proteins without aliases guessing."""
    cell_gene_ids = list(pd.read_csv(CELL_FEATURE_FILE, nrows=0).columns[1:])
    gene_set = set(cell_gene_ids)
    protein_to_idx = {sid: i for i, sid in enumerate(ppi_graph['protein_ids'])}
    mapping = {gene: set() for gene in cell_gene_ids}

    alias_cols = ['#string_protein_id', 'alias', 'source']
    for chunk in pd.read_csv(PPI_ALIASES, sep='\t', usecols=alias_cols, chunksize=500000):
        chunk = chunk[
            chunk['alias'].isin(gene_set)
            & chunk['#string_protein_id'].isin(protein_to_idx)
            & chunk['source'].astype(str).str.contains('Ensembl_gene|HGNC_ensembl', regex=True)
        ]
        for sid, gene in zip(chunk['#string_protein_id'], chunk['alias']):
            mapping[gene].add(protein_to_idx[sid])

    ambiguous = [gene for gene, nodes in mapping.items() if len(nodes) > 1]
    if ambiguous:
        raise RuntimeError(f'Ambiguous cell-gene to STRING mappings: {ambiguous[:5]}')

    cell_gene_to_ppi = torch.full((len(cell_gene_ids),), -1, dtype=torch.long)
    for i, gene in enumerate(cell_gene_ids):
        if mapping[gene]:
            cell_gene_to_ppi[i] = next(iter(mapping[gene]))

    print('Cell-expression genes mapped to STRING PPI:', int((cell_gene_to_ppi >= 0).sum()))
    return cell_gene_ids, cell_gene_to_ppi


def target_frequency(num_nodes):
    freq = torch.zeros(num_nodes, dtype=torch.float32)
    if not os.path.exists(DRUG_SEED):
        return freq
    with open(DRUG_SEED, 'rb') as f:
        drug_seed = pickle.load(f)
    for seed in drug_seed.values():
        v = torch.as_tensor(seed, dtype=torch.float32).view(-1)
        if v.numel() == num_nodes:
            freq += (v > 0).float()
    return freq


def map_pathway_proteins(pathways, gene_to_idx):
    mapped = []
    for item in pathways:
        proteins = set()
        for gene in item['genes']:
            proteins.update(gene_to_idx.get(gene, set()))
        if proteins:
            mapped.append({
                'name': item['name'],
                'stable_id': item['stable_id'],
                'proteins': sorted(proteins),
                'gene_count': len(item['genes']),
            })
    return mapped


def select_pathways(pathways, freq):
    scored = []
    for item in pathways:
        proteins = item['proteins']
        size = len(proteins)
        if size < MIN_PROTEINS or size > MAX_PROTEINS:
            continue
        p = torch.tensor(proteins, dtype=torch.long)
        target_hits = float(freq[p].sum().item())
        if target_hits <= 0:
            continue
        # Prefer target-enriched, reasonably specific pathways instead of
        # selecting only the broadest Reactome parents.
        score = target_hits / math.sqrt(float(size)) + 0.05 * math.log1p(size)
        scored.append((score, target_hits, size, item))
    scored.sort(key=lambda x: (x[0], x[2]), reverse=True)

    selected = []
    selected_sets = []
    for _, _, _, item in scored:
        proteins = set(item['proteins'])
        max_jaccard = max(
            (len(proteins & old) / float(len(proteins | old)) for old in selected_sets),
            default=0.0,
        )
        if max_jaccard > MAX_REDUNDANCY:
            continue
        selected.append(item)
        selected_sets.append(proteins)
        if len(selected) == MAX_PATHWAYS:
            break
    if len(selected) < 16:
        raise RuntimeError(f'Too few Reactome pathways mapped to STRING PPI: {len(selected)}')
    return selected


def build_overlap_hyperedges(selected):
    sets = [set(x['proteins']) for x in selected]
    hyper_sets = []
    seen = set()
    for i, si in enumerate(sets):
        neigh = []
        for j, sj in enumerate(sets):
            if i == j:
                continue
            inter = len(si & sj)
            if inter < MIN_OVERLAP:
                continue
            jac = inter / float(len(si | sj))
            if jac >= MIN_JACCARD:
                neigh.append((jac, inter, j))
        neigh.sort(reverse=True)
        members = tuple(sorted([i] + [j for _, _, j in neigh[:MAX_NEIGHBORS]]))
        if len(members) >= 2 and members not in seen:
            seen.add(members)
            hyper_sets.append(members)

    if not hyper_sets:
        raise RuntimeError('No pathway overlap hyperedges built; relax overlap thresholds.')

    rows, cols = [], []
    for h, members in enumerate(hyper_sets):
        for p in members:
            rows.append(p)
            cols.append(h)
    return torch.tensor([rows, cols], dtype=torch.long), hyper_sets


def main():
    os.makedirs(PPI_FLOW_DIR, exist_ok=True)
    download_reactome()
    pathways, reactome_genes = load_reactome_pathways()
    ppi_graph = torch.load(PPI_GRAPH, map_location='cpu')
    num_nodes = int(ppi_graph['num_nodes'])
    gene_to_idx = load_gene_to_ppi_idx(ppi_graph, reactome_genes)
    mapped = map_pathway_proteins(pathways, gene_to_idx)
    cell_gene_ids, cell_gene_to_ppi = load_cell_gene_to_ppi_idx(ppi_graph)
    freq = target_frequency(num_nodes)
    selected = select_pathways(mapped, freq)

    protein_rows, pathway_rows = [], []
    for pid, item in enumerate(selected):
        for protein in item['proteins']:
            protein_rows.append(protein)
            pathway_rows.append(pid)
    protein_pathway_index = torch.tensor([protein_rows, pathway_rows], dtype=torch.long)
    pathway_hyperedge_index, hyper_sets = build_overlap_hyperedges(selected)

    out = {
        'source': 'ReactomePathways.gmt.zip current, Homo sapiens R-HSA pathways; mapped to STRING PPI via protein.info and aliases',
        'num_proteins': num_nodes,
        'num_pathways': len(selected),
        'protein_pathway_index': protein_pathway_index,
        'pathway_hyperedge_index': pathway_hyperedge_index,
        'pathway_names': [x['name'] for x in selected],
        'pathway_stable_ids': [x['stable_id'] for x in selected],
        'pathway_protein_counts': [len(x['proteins']) for x in selected],
        'cell_gene_ids': cell_gene_ids,
        'cell_gene_to_ppi': cell_gene_to_ppi,
        'hyperedge_members': [list(x) for x in hyper_sets],
        'config': {
            'min_proteins': MIN_PROTEINS,
            'max_proteins': MAX_PROTEINS,
            'max_pathways': MAX_PATHWAYS,
            'min_overlap': MIN_OVERLAP,
            'min_jaccard': MIN_JACCARD,
            'max_neighbors': MAX_NEIGHBORS,
            'max_redundancy': MAX_REDUNDANCY,
        }
    }
    torch.save(out, OUT_FILE)
    print('Saved:', OUT_FILE)
    print('Mapped Reactome pathways:', len(mapped))
    print('Selected pathways:', len(selected))
    print('Protein-pathway memberships:', protein_pathway_index.size(1))
    print('Pathway hyperedges:', len(hyper_sets))
    print('Top 10 selected pathways:')
    for i, item in enumerate(selected[:10]):
        print(i, item['stable_id'], len(item['proteins']), item['name'])


if __name__ == '__main__':
    main()
