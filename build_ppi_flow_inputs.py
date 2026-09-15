import os
import pickle
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm


# =========================
# 1. Path configuration
# =========================

DATA_DIR = "data"

STRING_DIR = os.path.join(DATA_DIR, "STRING_PPI")
DGIDB_DIR = os.path.join(DATA_DIR, "DGIdb")

PPI_LINKS = os.path.join(STRING_DIR, "9606.protein.links.v12.0.txt")
PPI_INFO = os.path.join(STRING_DIR, "9606.protein.info.v12.0.txt")
PPI_ALIASES = os.path.join(STRING_DIR, "9606.protein.aliases.v12.0.txt")

DGIDB_INTERACTIONS = os.path.join(DGIDB_DIR, "interactions.tsv")

LABEL_FILE = os.path.join(DATA_DIR, "new_labels_0_10.csv")
SMILES_FILE = os.path.join(DATA_DIR, "smiles_0_10.csv")

OUT_DIR = os.path.join(DATA_DIR, "PPI_flow")
os.makedirs(OUT_DIR, exist_ok=True)

OUT_GRAPH = os.path.join(OUT_DIR, "ppi_graph.pt")
OUT_SEED = os.path.join(OUT_DIR, "drug_seed.pkl")
OUT_MISSING = os.path.join(OUT_DIR, "missing_dti_mapping.csv")
OUT_TARGET_MAPPING = os.path.join(OUT_DIR, "drug_target_mapping.csv")


# =========================
# 2. Utility functions
# =========================

def norm(x):
    """Normalize names for matching."""
    if pd.isna(x):
        return ""
    return str(x).strip().upper()


def check_file_exists(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")


def normalize_drug_name(name):

    name = norm(name)

    alias_map = {
        "5-FU": "FLUOROURACIL",
        "ABT-888": "VELIPARIB",
        "AZD1775": "ADAVOSERTIB",
        "BEZ-235": "DACTOLISIB",
        "MK-8669": "RIDAFOROLIMUS",
        "ZOLINZA": "VORINOSTAT",
        "MITOMYCINE": "MITOMYCIN",
        "L778123": "L-778123",
        "L778,123": "L-778123",
    }

    return alias_map.get(name, name)


# =========================
# 3. Build PPI graph
# =========================

def build_ppi_graph(score_threshold=700):
    """
    Build STRING PPI graph from protein.links file.

    Output:
        proteins: list of STRING protein IDs
        protein_to_idx: dict, STRING protein ID -> node index
        ppi_graph.pt:
            {
                edge_index: Tensor[2, E],
                edge_weight: Tensor[E],
                num_nodes: int,
                protein_ids: list[str]
            }
    """
    print("Loading STRING PPI links...")

    check_file_exists(PPI_LINKS)

    ppi = pd.read_csv(PPI_LINKS, sep=r"\s+")

    required_cols = {"protein1", "protein2", "combined_score"}
    if not required_cols.issubset(set(ppi.columns)):
        raise ValueError(
            f"{PPI_LINKS} must contain columns: {required_cols}, "
            f"but got {list(ppi.columns)}"
        )

    ppi = ppi[ppi["combined_score"] >= score_threshold].copy()

    proteins = sorted(set(ppi["protein1"]).union(set(ppi["protein2"])))
    protein_to_idx = {p: i for i, p in enumerate(proteins)}

    src = ppi["protein1"].map(protein_to_idx).values
    dst = ppi["protein2"].map(protein_to_idx).values
    weight = ppi["combined_score"].values.astype(np.float32) / 1000.0

    # STRING links are treated as undirected.
    edge_src = np.concatenate([src, dst])
    edge_dst = np.concatenate([dst, src])
    edge_weight = np.concatenate([weight, weight])

    edge_index = torch.LongTensor(np.stack([edge_src, edge_dst], axis=0))
    edge_weight = torch.FloatTensor(edge_weight)

    graph = {
        "edge_index": edge_index,
        "edge_weight": edge_weight,
        "num_nodes": len(proteins),
        "protein_ids": proteins,
        "score_threshold": score_threshold,
    }

    torch.save(graph, OUT_GRAPH)

    print(f"PPI score threshold: {score_threshold}")
    print(f"PPI nodes: {len(proteins)}")
    print(f"PPI edges: {edge_index.shape[1]}")
    print(f"Saved PPI graph: {OUT_GRAPH}")

    return proteins, protein_to_idx


# =========================
# 4. Build gene/alias -> STRING ID mapping
# =========================

def build_gene_to_string_mapping(protein_to_idx):
    """
    Build mapping from gene symbols / aliases / UniProt / Ensembl IDs to STRING protein IDs.

    Uses:
        9606.protein.info.v12.0.txt
        9606.protein.aliases.v12.0.txt
    """
    print("Building gene/alias to STRING mapping...")

    check_file_exists(PPI_INFO)
    check_file_exists(PPI_ALIASES)

    gene_to_string = {}

    # 1) preferred_name from protein.info
    info = pd.read_csv(PPI_INFO, sep="\t")

    if "#string_protein_id" not in info.columns or "preferred_name" not in info.columns:
        raise ValueError(
            f"{PPI_INFO} must contain '#string_protein_id' and 'preferred_name'. "
            f"Got columns: {list(info.columns)}"
        )

    for _, row in info.iterrows():
        sid = row["#string_protein_id"]
        gene = norm(row["preferred_name"])

        if sid in protein_to_idx and gene:
            gene_to_string.setdefault(gene, set()).add(sid)

    # 2) aliases from protein.aliases
    aliases = pd.read_csv(PPI_ALIASES, sep="\t")

    if "#string_protein_id" not in aliases.columns or "alias" not in aliases.columns:
        raise ValueError(
            f"{PPI_ALIASES} must contain '#string_protein_id' and 'alias'. "
            f"Got columns: {list(aliases.columns)}"
        )

    for _, row in aliases.iterrows():
        sid = row["#string_protein_id"]
        alias = norm(row["alias"])

        if sid in protein_to_idx and alias:
            gene_to_string.setdefault(alias, set()).add(sid)

    print(f"gene/alias mapping size: {len(gene_to_string)}")

    return gene_to_string


# =========================
# 5. Load DGIdb DTI
# =========================

def load_dgidb_targets():
    """
    Load DGIdb drug-gene interactions.

    Returns:
        drug_to_genes:
            {
                "ERLOTINIB": {"EGFR", ...},
                ...
            }
    """
    print("Loading DGIdb interactions...")

    check_file_exists(DGIDB_INTERACTIONS)

    inter = pd.read_csv(DGIDB_INTERACTIONS, sep="\t")

    if "gene_name" not in inter.columns:
        raise ValueError(
            f"{DGIDB_INTERACTIONS} must contain 'gene_name'. "
            f"Got columns: {list(inter.columns)}"
        )

    drug_to_genes = {}

    for _, row in inter.iterrows():
        gene = norm(row.get("gene_name", ""))

        if not gene:
            continue

        # DGIdb usually contains both drug_name and drug_claim_name.
        drug_name = norm(row.get("drug_name", ""))
        drug_claim_name = norm(row.get("drug_claim_name", ""))

        if drug_name:
            drug_to_genes.setdefault(drug_name, set()).add(gene)

        if drug_claim_name:
            drug_to_genes.setdefault(drug_claim_name, set()).add(gene)

    print(f"DGIdb drugs: {len(drug_to_genes)}")

    return drug_to_genes


# =========================
# 6. Load SMILES -> drug name mapping
# =========================

def load_smiles_to_name():
    """
    Load your smiles_0_10.csv.

    Expected format:
        drug_name,smile

    No header.
    Example:
        5-FU,O=c1[nH]cc(F)c(=O)[nH]1
        ABT-888,CC1(c2nc3c(C(N)=O)cccc3[nH]2)CCCN1
    """
    print("Loading SMILES-drug name mapping...")

    check_file_exists(SMILES_FILE)

    df = pd.read_csv(SMILES_FILE, header=None, names=["drug_name", "smile"])

    df["drug_name"] = df["drug_name"].apply(normalize_drug_name)
    df["smile"] = df["smile"].astype(str).str.strip()

    smiles_to_name = {
        row["smile"]: row["drug_name"]
        for _, row in df.iterrows()
    }

    print(f"SMILES-drug mappings: {len(smiles_to_name)}")

    return smiles_to_name


# =========================
# 7. Build DTI seed vectors
# =========================

def build_drug_seed_vectors(proteins, protein_to_idx, gene_to_string, drug_to_genes, smiles_to_name):
    """
    For each SMILES in new_labels_0_10.csv, build a DTI seed vector over PPI nodes.

    seed[i] = 1 / num_targets if protein_i is a drug target.
    Otherwise 0.
    """
    print("Building drug DTI seed vectors...")

    check_file_exists(LABEL_FILE)

    labels = pd.read_csv(LABEL_FILE)

    if "drug1" not in labels.columns or "drug2" not in labels.columns:
        raise ValueError(
            f"{LABEL_FILE} must contain 'drug1' and 'drug2'. "
            f"Got columns: {list(labels.columns)}"
        )

    all_smiles = sorted(set(labels["drug1"]).union(set(labels["drug2"])))

    drug_seed = {}
    missing_records = []
    target_records = []

    num_nodes = len(proteins)

    for smile in tqdm(all_smiles, desc="Mapping drugs to PPI seeds"):
        smile = str(smile).strip()

        drug_name = smiles_to_name.get(smile, "")
        dgidb_genes = drug_to_genes.get(drug_name, set())

        seed = np.zeros(num_nodes, dtype=np.float32)

        mapped_string_ids = set()
        mapped_gene_records = []

        for gene in dgidb_genes:
            candidate_string_ids = gene_to_string.get(norm(gene), set())

            for sid in candidate_string_ids:
                if sid in protein_to_idx:
                    mapped_string_ids.add(sid)
                    mapped_gene_records.append((gene, sid))

        if mapped_string_ids:
            for sid in mapped_string_ids:
                seed[protein_to_idx[sid]] = 1.0

            seed = seed / seed.sum()

            for gene, sid in mapped_gene_records:
                target_records.append({
                    "smile": smile,
                    "drug_name": drug_name,
                    "target_gene": gene,
                    "string_protein_id": sid,
                    "ppi_node_index": protein_to_idx[sid],
                })
        else:
            missing_records.append({
                "smile": smile,
                "drug_name": drug_name,
                "num_dgidb_genes": len(dgidb_genes),
                "num_mapped_string_targets": 0,
                "reason": (
                    "drug name not found in smiles_0_10.csv"
                    if not drug_name else
                    "no DGIdb target or no target mapped to STRING PPI"
                )
            })

        drug_seed[smile] = seed

    with open(OUT_SEED, "wb") as f:
        pickle.dump(drug_seed, f)

    pd.DataFrame(missing_records).to_csv(OUT_MISSING, index=False)
    pd.DataFrame(target_records).drop_duplicates().to_csv(OUT_TARGET_MAPPING, index=False)

    print(f"Saved drug seed vectors: {OUT_SEED}")
    print(f"Saved missing mapping file: {OUT_MISSING}")
    print(f"Saved target mapping file: {OUT_TARGET_MAPPING}")

    total = len(all_smiles)
    missing = len(missing_records)
    mapped = total - missing

    print(f"Total unique SMILES in labels: {total}")
    print(f"Mapped to at least one PPI target: {mapped}")
    print(f"Missing or zero-seed drugs: {missing}")

    if total > 0:
        print(f"Mapping rate: {mapped / total:.2%}")

    return drug_seed


# =========================
# 8. Main
# =========================

def main():
    # You can change threshold to 400 or 900 for ablation.
    score_threshold = 700

    proteins, protein_to_idx = build_ppi_graph(score_threshold=score_threshold)

    gene_to_string = build_gene_to_string_mapping(protein_to_idx)

    drug_to_genes = load_dgidb_targets()

    smiles_to_name = load_smiles_to_name()

    build_drug_seed_vectors(
        proteins=proteins,
        protein_to_idx=protein_to_idx,
        gene_to_string=gene_to_string,
        drug_to_genes=drug_to_genes,
        smiles_to_name=smiles_to_name
    )


if __name__ == "__main__":
    main()
