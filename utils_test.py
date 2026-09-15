import os
from itertools import islice
import sys
import numpy as np
from math import sqrt
from scipy import stats

import torch
from torch_geometric.data import InMemoryDataset
from torch_geometric import data as DATA
from torch_geometric.loader import DataLoader

class TestbedDataset(InMemoryDataset):
    def __init__(self, root='/tmp', dataset='_drug1',
                 xd=None, xt=None, y=None, xt_featrue=None,
                 transform=None, pre_transform=None,
                 smile_graph=None,
                 drug_seed=None,
                 ppi_num_nodes=0):

        """
        Parameters
        ----------
        root : str
            PyG processed data root.
        dataset : str
            Dataset name, e.g. new_labels_0_10_drug1.
        xd : list / np.ndarray
            Drug SMILES list.
        xt : list / np.ndarray
            Cell line id/name list.
        y : list / np.ndarray
            Labels.
        xt_featrue : np.ndarray
            Cell line feature matrix. Kept as original variable name for compatibility.
        smile_graph : dict
            SMILES -> molecular graph tuple.
        drug_seed : dict
            SMILES -> PPI seed vector, shape [ppi_num_nodes].
        ppi_num_nodes : int
            Number of PPI nodes. Used to create zero seed for missing DTI drugs.
        """

        self.dataset = dataset
        self.drug_seed = drug_seed
        self.ppi_num_nodes = int(ppi_num_nodes) if ppi_num_nodes is not None else 0

        super(TestbedDataset, self).__init__(root, transform, pre_transform)

        if os.path.isfile(self.processed_paths[0]):
            print('Pre-processed data found: {}, loading ...'.format(self.processed_paths[0]))
            try:
                self.data, self.slices = torch.load(self.processed_paths[0])
            except TypeError:
                # For older torch versions.
                self.data, self.slices = torch.load(self.processed_paths[0])
        else:
            print('Pre-processed data {} not found, doing pre-processing...'.format(self.processed_paths[0]))
            self.process(xd, xt, xt_featrue, y, smile_graph)
            try:
                self.data, self.slices = torch.load(self.processed_paths[0])
            except TypeError:
                self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return [self.dataset + '.pt']

    def download(self):
        pass

    def _download(self):
        pass

    def _process(self):
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)

    def get_cell_feature(self, cellId, cell_features):
        for row in islice(cell_features, 0, None):
            if cellId in row[0]:
                return row[1:]
        return False

    def _get_ppi_seed(self, smiles):
        """
        Return:
            seed: np.ndarray, shape [ppi_num_nodes]
            has_ppi: float, 1.0 if seed is non-zero else 0.0
        """
        if self.drug_seed is not None and smiles in self.drug_seed:
            seed = self.drug_seed[smiles]
            seed = np.asarray(seed, dtype=np.float32)
        else:
            seed = np.zeros(self.ppi_num_nodes, dtype=np.float32)

        # Defensive check: if seed length is inconsistent, replace with zero vector.
        if self.ppi_num_nodes > 0 and seed.shape[0] != self.ppi_num_nodes:
            print(
                f'Warning: PPI seed length mismatch for SMILES {smiles}. '
                f'Expected {self.ppi_num_nodes}, got {seed.shape[0]}. Use zero seed.'
            )
            seed = np.zeros(self.ppi_num_nodes, dtype=np.float32)

        has_ppi = 1.0 if float(np.sum(seed)) > 0 else 0.0
        return seed, has_ppi

    def process(self, xd, xt, xt_featrue, y, smile_graph):
        assert (len(xd) == len(xt) and len(xt) == len(y)), \
            "The three lists xd, xt, and y must have the same length!"

        data_list = []
        data_len = len(xd)
        print('number of data', data_len)

        for i in range(data_len):
            smiles = xd[i]
            target = xt[i]
            labels = y[i]

            if smiles not in smile_graph:
                print('SMILES not found in smile_graph:', smiles)
                sys.exit()

            # Molecular graph from RDKit-preprocessed graph dictionary.
            c_size, features, edge_index = smile_graph[smiles]

            GCNData = DATA.Data(
                x=torch.Tensor(features),
                edge_index=torch.LongTensor(edge_index).transpose(1, 0),
                y=torch.Tensor([labels])
            )

            # Cell line feature.
            cell = self.get_cell_feature(target, xt_featrue)

            if cell is False:
                print('cell feature not found:', target)
                sys.exit()

            new_cell = []
            for n in cell:
                new_cell.append(float(n))

            GCNData.cell = torch.FloatTensor([new_cell])
            GCNData.__setitem__('c_size', torch.LongTensor([c_size]))

            # =========================
            # New: DTI-guided PPI seed
            # =========================
            seed, has_ppi = self._get_ppi_seed(smiles)

            # Shape after batching:
            # each sample [1, N] -> batch [B, N]
            GCNData.ppi_seed = torch.FloatTensor([seed])

            # Shape after batching:
            # each sample [1] -> batch [B]
            GCNData.has_ppi = torch.FloatTensor([has_ppi])

            data_list.append(GCNData)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        print('Graph construction done. Saving to file.')
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


def rmse(y, f):
    return sqrt(((y - f) ** 2).mean(axis=0))


def save_AUCs(AUCs, filename):
    with open(filename, 'a') as f:
        f.write('\t'.join(map(str, AUCs)) + '\n')


def mse(y, f):
    return ((y - f) ** 2).mean(axis=0)


def pearson(y, f):
    return np.corrcoef(y, f)[0, 1]


def spearman(y, f):
    return stats.spearmanr(y, f)[0]


def ci(y, f):
    ind = np.argsort(y)
    y = y[ind]
    f = f[ind]
    i = len(y) - 1
    j = i - 1
    z = 0.0
    S = 0.0

    while i > 0:
        while j >= 0:
            if y[i] > y[j]:
                z = z + 1
                u = f[i] - f[j]
                if u > 0:
                    S = S + 1
                elif u == 0:
                    S = S + 0.5
            j = j - 1
        i = i - 1
        j = i - 1

    return S / z if z != 0 else 0.0

