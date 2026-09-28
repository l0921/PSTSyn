# PSTSyn

PSTSyn is a hierarchical biological modeling framework for drug synergy prediction by integrating protein perturbation propagation, drug-pair pathway crosstalk, and cellular state transition.

The model integrates drug molecular structures, cell-line gene expression, drug-target interactions, protein-protein interactions, and pathway information to predict drug synergy.

## Data availability

The datasets used in this study are available from the  
[PSTSyn v1.0.0 release](https://github.com/l0921/PSTSyn/releases/tag/v1.0.0).

To download the data, open the release page and select the dataset archive under **Assets**.

After downloading, extract the files and place them in the `data/` directory.

## Model architecture

![PSTSyn model architecture](./pstsyn_architecture.jpg)

PSTSyn contains three main components:

- **Protein Perturbation Propagation (PPP)** for modeling drug-induced perturbations on the protein interaction network.
- **Drug Pair Pathway Crosstalk (DPC)** for modeling pathway-level interactions between two drugs.
- **Cell State Transition (CST)** for modeling cellular state changes under individual and combined drug treatments.

## Requirements

The main dependencies include:

- Python
- PyTorch
- PyTorch Geometric
- RDKit
- NumPy
- pandas
- scikit-learn
- SciPy
- NetworkX
- tqdm

If `environment.yml` is provided, the environment can be created using:

```bash
conda env create -f environment.yml
```

## Reproduction

The experiment can be reproduced by running the preprocessing and training scripts in the following order.

### 1. Build the PPI network and drug-target inputs

```bash
python build_ppi_flow_inputs.py
```

This script constructs the STRING PPI network and maps DGIdb drug targets to proteins. The default STRING confidence threshold is 700. :chatgpt-content-reference{index="0"}

### 2. Build drug-specific PPI subgraphs

```bash
python build_ppi_subgraphs.py
```

The default setting uses two-hop PPI neighborhoods with at most 192 nodes and 1,536 edges for each drug-specific subgraph. :chatgpt-content-reference{index="1"}

### 3. Build the Reactome pathway hypergraph

```bash
python build_reactome_pathway_hypergraph.py
```

This script maps Reactome pathways to the STRING PPI network and constructs the pathway hypergraph used by PSTSyn. :chatgpt-content-reference{index="2"}

### 4. Generate the processed dataset

```bash
python creat_data_DC.py
```

This step constructs molecular graphs and PyTorch Geometric datasets using the drug, cell-line, and biological-prior information.

### 5. Train PSTSyn

```bash
python train_PSTSyn.py
```

The default implementation performs five-fold cross-validation using Adam optimization.

The main default settings include:

```text
Random seed: 0
Batch size: 256
Learning rate: 5e-4
Maximum epochs: 100
Dropout: 0.30
Latent dimension: 128
```

The complete training configuration is automatically saved during execution. :chatgpt-content-reference{index="3"}

## Quick reproduction

The complete workflow is:

```bash
python build_ppi_flow_inputs.py
python build_ppi_subgraphs.py
python build_reactome_pathway_hypergraph.py
python creat_data_DC.py
python train_PSTSyn.py
```

If the processed PPI, pathway, and dataset files are already included in the release archive, the preprocessing steps can be skipped and the model can be trained directly using:

```bash
python train_PSTSyn.py
```

## Output

By default, the training results are saved in:

```text
data/result/new_labels_0_10_PSTSyn_slim_ordered_v8/
```

The final five-fold evaluation summary is saved as:

```text
5Fold_Summary.csv
```

The training script also saves the model checkpoints, fold splits, prediction results, and experimental configuration for reproducibility. :chatgpt-content-reference{index="4"}
