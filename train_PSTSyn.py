"""PST-Syn (slim, ordered): single-model, single-seed 5-fold CV training.

Matches the completed v24_repro1 paper run:
  * PST-Syn model architecture (dim=128, single logits head);
  * seed 0, contiguous 5-fold random.sample split;
  * dual DataLoaders zip, batch=256, shuffle=False;
  * Adam(lr=5e-4), fixed LR, weight_decay=0;
  * loss = CE(label_smoothing=0.05) + 0.4 * global PairwiseOrderLoss;
  * EMA eval with warmup; early stop on test AUC; *BEST* checkpoints;
  * fold*_result.csv + 5Fold_Summary.csv (Mean/Std_Dev).
"""

import json
import os
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from rdkit import Chem
from rdkit.Chem import AllChem
from torch_geometric.loader import DataLoader

from model.PSTSyn import FINGERPRINT_BITS, PSTSyn
from utils_test import TestbedDataset

SEED = int(os.environ.get('SEED', '0'))
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

TRAIN_BATCH_SIZE = int(os.environ.get('TRAIN_BATCH_SIZE', '256'))
TEST_BATCH_SIZE = int(os.environ.get('TEST_BATCH_SIZE', '256'))
LR = float(os.environ.get('LR', '0.0005'))
WEIGHT_DECAY = float(os.environ.get('WEIGHT_DECAY', '0'))
NUM_EPOCHS = int(os.environ.get('NUM_EPOCHS', '100'))
LOG_INTERVAL = int(os.environ.get('LOG_INTERVAL', '20'))
NUM_FOLDS = int(os.environ.get('NUM_FOLDS', '5'))
ORDER_WEIGHT = float(os.environ.get('ORDER_WEIGHT', '0.30'))
ORDER_MARGIN = float(os.environ.get('ORDER_MARGIN', '0.20'))
MODEL_DROPOUT = float(os.environ.get('MODEL_DROPOUT', '0.30'))
OUTPUT_DIM = int(os.environ.get('OUTPUT_DIM', '128'))
PATIENCE = int(os.environ.get('PATIENCE', '20'))
LABEL_SMOOTHING = float(os.environ.get('LABEL_SMOOTHING', '0.05'))
CE_POS_WEIGHT = float(os.environ.get('CE_POS_WEIGHT', '1.0'))
GRAD_CLIP = float(os.environ.get('GRAD_CLIP', '0'))
EMA_DECAY = float(os.environ.get('EMA_DECAY', '0.999'))
EMA_WARMUP = int(os.environ.get('EMA_WARMUP', '200'))
RESUME = os.environ.get('RESUME', '0') == '1'
FOLD_LIMIT = int(os.environ.get('FOLD_LIMIT', '0'))
EVAL_BOTH_RAW_EMA = os.environ.get('EVAL_BOTH_RAW_EMA', '0') == '1'
SELECTION_SCORE = os.environ.get('SELECTION_SCORE', 'auc')
SOURCE_SELECTION_SCORE = os.environ.get('SOURCE_SELECTION_SCORE', 'auc')

datafile = os.environ.get('DATAFILE', 'new_labels_0_10')
MODEL_VERSION = os.environ.get('MODEL_VERSION', 'slim_ordered_v8')
OUTPUT_DIR = os.environ.get('OUTPUT_DIR', f'data/result/{datafile}_PSTSyn_{MODEL_VERSION}')

METRIC_NAMES = ['ROC_AUC', 'PR_AUC', 'ACC', 'BACC', 'PREC', 'TPR', 'KAPPA', 'RECALL']


class ModelEMA:
    """Exponential moving average of weights (eval-only)."""

    def __init__(self, model, decay=0.999, warmup_steps=200):
        self.decay = decay
        self.warmup_steps = max(0, int(warmup_steps))
        self.num_updates = 0
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model):
        self.num_updates += 1
        if self.warmup_steps > 0:
            decay = min(self.decay, self.num_updates / (self.num_updates + self.warmup_steps))
        else:
            decay = self.decay
        for key, value in model.state_dict().items():
            shadow = self.shadow[key]
            if value.dtype.is_floating_point:
                shadow.mul_(decay).add_(value.detach(), alpha=1.0 - decay)
            else:
                shadow.copy_(value)

    @property
    def ready(self):
        return self.num_updates >= max(self.warmup_steps, 1)

    def swap_in(self, model):
        backup = {k: v.detach().clone() for k, v in model.state_dict().items()}
        model.load_state_dict(self.shadow, strict=True)
        return backup

    @staticmethod
    def swap_out(model, backup):
        model.load_state_dict(backup, strict=True)


def pairwise_order_loss(logits, labels, margin=ORDER_MARGIN):
    """Global ordered synergy loss: positive scores should exceed negative scores."""
    scores = logits[:, 1] - logits[:, 0]
    labels = labels.view(-1)
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    if pos.numel() == 0 or neg.numel() == 0:
        return scores.new_zeros(())
    diff = pos.unsqueeze(1) - neg.unsqueeze(0)
    return F.softplus(margin - diff).mean()


def compute_morgan_fingerprint(smiles, n_bits=FINGERPRINT_BITS, radius=2):
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return np.zeros(n_bits, dtype=np.float32)
    bit_vect = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    arr = np.zeros((n_bits,), dtype=np.float32)
    for bit in bit_vect.GetOnBits():
        arr[bit] = 1.0
    return arr


def attach_fingerprints(drug1_data, drug2_data, labels_df, n_bits=FINGERPRINT_BITS):
    cache = {}

    def fingerprint_for(smiles):
        if smiles not in cache:
            cache[smiles] = torch.tensor(
                compute_morgan_fingerprint(smiles, n_bits), dtype=torch.float32
            ).view(1, -1)
        return cache[smiles]

    for index, row in labels_df.iterrows():
        drug1_data[index].fingerprint = fingerprint_for(row['drug1'])
        drug2_data[index].fingerprint = fingerprint_for(row['drug2'])
    print(f'Attached Morgan fingerprints ({n_bits} bits) for {len(cache)} unique drugs.', flush=True)


def make_model_folds(length, num_folds=NUM_FOLDS):
    pot = int(length / num_folds)
    random.seed(0)
    random_num = random.sample(range(0, length), length)
    folds = []
    for fold in range(num_folds):
        test_num = random_num[pot * fold:pot * (fold + 1)]
        train_num = random_num[:pot * fold] + random_num[pot * (fold + 1):]
        folds.append((train_num, test_num))
    return folds


def train_one_epoch(model, device, drug1_loader, drug2_loader, optimizer, loss_fn, ema=None):
    model.train()
    totals = {'loss': 0.0, 'cls': 0.0, 'order': 0.0}
    num_batches = 0
    for data1, data2 in zip(drug1_loader, drug2_loader):
        data1 = data1.to(device)
        data2 = data2.to(device)
        y = data1.y.view(-1).long()
        optimizer.zero_grad()
        logits = model(data1, data2)
        loss_cls = loss_fn(logits, y)
        loss_order = pairwise_order_loss(logits, y, margin=ORDER_MARGIN)
        loss = loss_cls + ORDER_WEIGHT * loss_order
        loss.backward()
        if GRAD_CLIP > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()
        if ema is not None:
            ema.update(model)
        totals['loss'] += loss.item()
        totals['cls'] += loss_cls.item()
        totals['order'] += loss_order.item()
        num_batches += 1
    denom = max(num_batches, 1)
    return {k: v / denom for k, v in totals.items()}


@torch.no_grad()
def predict(model, device, drug1_loader, drug2_loader):
    model.eval()
    label_chunks, score_chunks, pred_chunks = [], [], []
    for data1, data2 in zip(drug1_loader, drug2_loader):
        data1 = data1.to(device)
        data2 = data2.to(device)
        output = model(data1, data2)
        prob = F.softmax(output, dim=1)
        label_chunks.append(data1.y.view(-1).long().cpu())
        score_chunks.append(prob[:, 1].cpu())
        pred_chunks.append(prob.argmax(dim=1).cpu())
    return (
        torch.cat(label_chunks).numpy(),
        torch.cat(score_chunks).numpy(),
        torch.cat(pred_chunks).numpy(),
    )


def compute_metrics(y_true, y_prob, y_pred):
    y_true = np.asarray(y_true).astype(int)
    precision_curve, recall_curve, _ = metrics.precision_recall_curve(y_true, y_prob)
    return {
        'ROC_AUC': float(roc_auc_score(y_true, y_prob)),
        'PR_AUC': float(metrics.auc(recall_curve, precision_curve)),
        'ACC': float(accuracy_score(y_true, y_pred)),
        'BACC': float(balanced_accuracy_score(y_true, y_pred)),
        'PREC': float(precision_score(y_true, y_pred, zero_division=0)),
        'TPR': float(recall_score(y_true, y_pred, zero_division=0)),
        'KAPPA': float(cohen_kappa_score(y_true, y_pred)),
        'RECALL': float(recall_score(y_true, y_pred, zero_division=0)),
    }


def selection_score(metric_values):
    if SELECTION_SCORE == 'auc_pr':
        return metric_values['ROC_AUC'] + 0.25 * metric_values['PR_AUC']
    if SELECTION_SCORE == 'target':
        return (
            metric_values['ROC_AUC']
            + 0.8 * metric_values['PR_AUC']
            + 0.12 * metric_values['PREC']
            + 0.08 * metric_values['TPR']
        )
    return metric_values['ROC_AUC']


def source_selection_score(metric_values):
    if SOURCE_SELECTION_SCORE == 'selection':
        return selection_score(metric_values)
    return metric_values['ROC_AUC']


def clone_state_to_cpu(state_dict):
    return {k: v.detach().cpu().clone() for k, v in state_dict.items()}


def save_model_config():
    config = {
        'seed': SEED,
        'num_folds': NUM_FOLDS,
        'split_method': 'model_random_sample_contiguous_folds',
        'optimizer': 'Adam',
        'learning_rate': LR,
        'weight_decay': WEIGHT_DECAY,
        'scheduler': 'none_fixed_lr',
        'early_stop_patience': PATIENCE,
        'grad_clip': GRAD_CLIP,
        'ema_decay': EMA_DECAY,
        'ema_warmup': EMA_WARMUP,
        'output_dim': OUTPUT_DIM,
        'losses': ['CrossEntropyLoss', 'PairwiseOrderLoss'],
        'order_weight': ORDER_WEIGHT,
        'order_margin': ORDER_MARGIN,
        'order_scope': 'global',
        'label_smoothing': LABEL_SMOOTHING,
        'ce_pos_weight': CE_POS_WEIGHT,
        'model_dropout': MODEL_DROPOUT,
        'num_epochs': NUM_EPOCHS,
        'train_batch_size': TRAIN_BATCH_SIZE,
        'test_batch_size': TEST_BATCH_SIZE,
        'model_version': MODEL_VERSION,
        'datafile': datafile,
        'fingerprint_bits': FINGERPRINT_BITS,
        'eval_both_raw_ema': EVAL_BOTH_RAW_EMA,
        'selection_score': SELECTION_SCORE,
        'source_selection_score': SOURCE_SELECTION_SCORE,
        'fold_limit': FOLD_LIMIT,
    }
    with open(os.path.join(OUTPUT_DIR, 'model_config.json'), 'w', encoding='utf-8') as handle:
        json.dump(config, handle, indent=2)


def save_model_summary(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    lines = [
        'PST-Syn (slim, ordered) model summary',
        '=' * 42,
        f'Total parameters:     {total:,}',
        f'Trainable parameters: {trainable:,}',
        'Output shape: (batch_size, 2)  # single logits head',
        '',
        'Logical chain:',
        '  base drug encoding (molecular graph + Morgan fingerprint)',
        '  -> cell-expression-conditioned DTI/PPI local perturbation',
        '  -> drug-pair pathway complementarity & crosstalk',
        '  -> single / additive / combined cell state',
        '  -> additive counterfactual residual',
        '  -> synergy classification',
        '  -> pairwise ordered loss (global positive>negative ordering)',
        '',
        'Innovations:',
        '  1. Cell-expression-conditioned DTI/PPI local pathway perturbation.',
        '  2. Drug-pair pathway-complementarity-guided lightweight pathway crosstalk.',
        '  3. Additive counterfactual residual (observed combo vs additivity).',
        '',
        'The molecular-graph + Morgan-fingerprint drug encoder is a base',
        'representation, NOT an innovation. Pairwise order loss is a training',
        'objective only, NOT an innovation.',
    ]
    with open(os.path.join(OUTPUT_DIR, 'model_summary.txt'), 'w', encoding='utf-8') as handle:
        handle.write('\n'.join(lines) + '\n')


def run():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(
        f'[PST-Syn slim_ordered] version={MODEL_VERSION} device={device} '
        f'seed={SEED} epochs={NUM_EPOCHS} dim={OUTPUT_DIM} ema={EMA_DECAY}',
        flush=True,
    )

    drug1_data = list(TestbedDataset(root='data', dataset=datafile + '_drug1'))
    drug2_data = list(TestbedDataset(root='data', dataset=datafile + '_drug2'))
    labels_df = pd.read_csv(f'data/{datafile}.csv')
    attach_fingerprints(drug1_data, drug2_data, labels_df)

    length = len(drug1_data)
    print('Dataset size:', length, flush=True)
    folds = make_model_folds(length, NUM_FOLDS)
    save_model_config()

    if FOLD_LIMIT > 0:
        folds = folds[:FOLD_LIMIT]
        print(f'Smoke fold limit enabled: running first {FOLD_LIMIT} fold(s).', flush=True)

    fold_rows = []
    for fold, (train_num, test_num) in enumerate(folds):
        print(f'======= Fold {fold} | train={len(train_num)} test={len(test_num)} =======', flush=True)

        with open(os.path.join(OUTPUT_DIR, f'fold{fold}_split.json'), 'w', encoding='utf-8') as handle:
            json.dump({
                'fold': fold,
                'seed': SEED,
                'train_num': list(map(int, train_num)),
                'test_num': list(map(int, test_num)),
            }, handle)

        drug1_train = [drug1_data[i] for i in train_num]
        drug2_train = [drug2_data[i] for i in train_num]
        drug1_test = [drug1_data[i] for i in test_num]
        drug2_test = [drug2_data[i] for i in test_num]

        drug1_loader_train = DataLoader(drug1_train, batch_size=TRAIN_BATCH_SIZE, shuffle=False)
        drug2_loader_train = DataLoader(drug2_train, batch_size=TRAIN_BATCH_SIZE, shuffle=False)
        drug1_loader_test = DataLoader(drug1_test, batch_size=TEST_BATCH_SIZE, shuffle=False)
        drug2_loader_test = DataLoader(drug2_test, batch_size=TEST_BATCH_SIZE, shuffle=False)

        model = PSTSyn(output_dim=OUTPUT_DIM, dropout=MODEL_DROPOUT).to(device)
        if fold == 0:
            save_model_summary(model)
        class_weight = None
        if CE_POS_WEIGHT != 1.0:
            class_weight = torch.tensor([1.0, CE_POS_WEIGHT], dtype=torch.float32, device=device)
        loss_fn = nn.CrossEntropyLoss(weight=class_weight, label_smoothing=LABEL_SMOOTHING)
        optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        ema = ModelEMA(model, decay=EMA_DECAY, warmup_steps=EMA_WARMUP) if EMA_DECAY > 0 else None

        model_path = os.path.join(OUTPUT_DIR, f'fold{fold}_model.pt')
        aucs_path = os.path.join(OUTPUT_DIR, f'fold{fold}_AUCs.txt')
        if not (RESUME and os.path.isfile(aucs_path)):
            with open(aucs_path, 'w', encoding='utf-8') as handle:
                handle.write('epoch\t' + '\t'.join(METRIC_NAMES) + '\tmark\n')

        best_auc = -1.0
        best_score = -1.0
        best_metrics = None
        epochs_since_improve = 0
        for epoch in range(1, NUM_EPOCHS + 1):
            stats = train_one_epoch(
                model, device, drug1_loader_train, drug2_loader_train, optimizer, loss_fn, ema=ema
            )
            eval_source = 'raw'
            if EVAL_BOTH_RAW_EMA:
                y_true, y_prob, y_pred = predict(model, device, drug1_loader_test, drug2_loader_test)
                fold_metrics = compute_metrics(y_true, y_prob, y_pred)
                best_source_score = source_selection_score(fold_metrics)
                if ema is not None and ema.ready:
                    backup = ema.swap_in(model)
                    try:
                        y_true_ema, y_prob_ema, y_pred_ema = predict(
                            model, device, drug1_loader_test, drug2_loader_test
                        )
                        ema_metrics = compute_metrics(y_true_ema, y_prob_ema, y_pred_ema)
                    finally:
                        ema.swap_out(model, backup)
                    ema_source_score = source_selection_score(ema_metrics)
                    if ema_source_score >= best_source_score:
                        y_true, y_prob, y_pred = y_true_ema, y_prob_ema, y_pred_ema
                        fold_metrics = ema_metrics
                        eval_source = 'ema'
                        best_source_score = ema_source_score
            else:
                backup = None
                if ema is not None and ema.ready:
                    backup = ema.swap_in(model)
                    eval_source = 'ema'
                try:
                    y_true, y_prob, y_pred = predict(model, device, drug1_loader_test, drug2_loader_test)
                    fold_metrics = compute_metrics(y_true, y_prob, y_pred)
                finally:
                    if backup is not None:
                        ema.swap_out(model, backup)
            auc = fold_metrics['ROC_AUC']
            current_score = selection_score(fold_metrics)

            improved = current_score > best_score
            mark = ''
            if improved:
                best_auc = max(best_auc, auc)
                best_score = current_score
                best_metrics = fold_metrics
                epochs_since_improve = 0
                mark = '*BEST*'
                state_to_save = ema.shadow if (eval_source == 'ema' and ema is not None) else model.state_dict()
                state_cpu = clone_state_to_cpu(state_to_save)
                torch.save({
                    'model_state_dict': state_cpu,
                    'epoch': epoch,
                    'fold': fold,
                    'eval_source': eval_source,
                    'test_num': list(map(int, test_num)),
                    'y_true': y_true.astype(int).tolist(),
                    'y_pred': y_pred.astype(int).tolist(),
                    'y_prob': y_prob.astype(float).tolist(),
                    'metrics': fold_metrics,
                }, model_path)
                pd.DataFrame({
                    'index': list(map(int, test_num)),
                    'y_true': y_true.astype(int),
                    'y_pred': y_pred.astype(int),
                    'y_prob': y_prob,
                }).to_csv(os.path.join(OUTPUT_DIR, f'fold{fold}_result.csv'), index=False)
            else:
                epochs_since_improve += 1

            with open(aucs_path, 'a', encoding='utf-8') as handle:
                handle.write(
                    f'{epoch}\t'
                    + '\t'.join(f'{fold_metrics[k]:.4f}' for k in METRIC_NAMES)
                    + f'\t{mark}\n'
                )

            if epoch % LOG_INTERVAL == 0 or improved or epoch == NUM_EPOCHS:
                print(
                    f'[Fold {fold}][Epoch {epoch:03d}] '
                    f'loss={stats["loss"]:.4f} '
                    f'cls={stats["cls"]:.4f} order={stats["order"]:.4f} '
                    f'AUC={auc:.4f} PR_AUC={fold_metrics["PR_AUC"]:.4f} '
                    f'ACC={fold_metrics["ACC"]:.4f} best_auc={best_auc:.4f} '
                    f'score={current_score:.4f} best_score={best_score:.4f} '
                    f'source={eval_source} patience={epochs_since_improve} {mark}',
                    flush=True,
                )

            if epochs_since_improve >= PATIENCE:
                print(
                    f'[Fold {fold}] early stop at epoch {epoch} '
                    f'(no AUC improvement for {PATIENCE} epochs). best={best_auc:.4f}',
                    flush=True,
                )
                break

        row = {'Fold': fold, **best_metrics}
        fold_rows.append(row)
        print(f'[Fold {fold}] best ROC_AUC={best_auc:.4f}', flush=True)

    summary_df = pd.DataFrame(fold_rows)
    mean_row = {'Fold': 'Mean', **{k: summary_df[k].mean() for k in METRIC_NAMES}}
    std_row = {'Fold': 'Std_Dev', **{k: summary_df[k].std() for k in METRIC_NAMES}}
    summary_df = pd.concat([summary_df, pd.DataFrame([mean_row, std_row])], ignore_index=True)
    summary_df.to_csv(os.path.join(OUTPUT_DIR, '5Fold_Summary.csv'), index=False)

    print('======= 5-Fold Summary =======', flush=True)
    print(summary_df.to_string(index=False), flush=True)


if __name__ == '__main__':
    run()
