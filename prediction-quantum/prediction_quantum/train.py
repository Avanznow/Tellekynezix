"""One training loop for every torch model (the circuit and the MLPs): Adam, cross-entropy, per-epoch logging.

train_classifier() returns the per-epoch history (train / validation loss and accuracy, seconds), the parameters
at the epoch with the best validation accuracy and the timings. Training time is the time spent in forward,
backward and optimiser steps; wall time also includes the per-epoch validation pass.
"""
import copy
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


def to_tensor(X, dtype=torch.float32):
    return torch.as_tensor(np.asarray(X), dtype=dtype)


@torch.no_grad()
def logits_of(model, X, batch_size=512):
    model.eval()
    X = to_tensor(X)
    return torch.cat([model(X[i:i + batch_size]) for i in range(0, len(X), batch_size)])


@torch.no_grad()
def evaluate(model, X, y, batch_size=512):
    """-> (mean cross-entropy, accuracy, predictions)."""
    logits = logits_of(model, X, batch_size)
    y_t = to_tensor(y, torch.long)
    loss = F.cross_entropy(logits, y_t).item()
    pred = logits.argmax(1)
    return loss, (pred == y_t).float().mean().item(), pred.numpy()


def predict(model, X, batch_size=512):
    return logits_of(model, X, batch_size).argmax(1).numpy()


def train_classifier(model, Xtr, ytr, Xva, yva, epochs, lr, batch_size, seed, weight_decay=0.0, log=None):
    """Adam + cross-entropy for `epochs` epochs; returns (history DataFrame, best dict, timing dict)."""
    gen = torch.Generator().manual_seed(int(seed))
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    Xtr_t, ytr_t = to_tensor(Xtr), to_tensor(ytr, torch.long)
    n = len(ytr_t)
    rows, best = [], dict(val_acc=-1.0, epoch=0, state_dict=None)
    train_seconds, t_wall = 0.0, time.perf_counter()
    for epoch in range(1, int(epochs) + 1):
        model.train()
        perm = torch.randperm(n, generator=gen)
        loss_sum, correct, t_epoch = 0.0, 0, time.perf_counter()
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            logits = model(Xtr_t[idx])
            loss = F.cross_entropy(logits, ytr_t[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            loss_sum += loss.item() * len(idx)
            correct += int((logits.argmax(1) == ytr_t[idx]).sum())
        epoch_seconds = time.perf_counter() - t_epoch
        train_seconds += epoch_seconds
        val_loss, val_acc, _ = evaluate(model, Xva, yva)
        rows.append(dict(epoch=epoch, train_loss=loss_sum / n, train_acc=correct / n, val_loss=val_loss, val_acc=val_acc,
                         epoch_seconds=epoch_seconds, cumulative_train_seconds=train_seconds))
        if val_acc > best["val_acc"]:
            best = dict(val_acc=val_acc, epoch=epoch, state_dict=copy.deepcopy(model.state_dict()))
        if log:
            log(f"epoch {epoch:3d}/{epochs}: train loss {loss_sum / n:.4f} acc {correct / n:.4f} | "
                f"val loss {val_loss:.4f} acc {val_acc:.4f} | {epoch_seconds:.1f} s")
    timing = dict(train_seconds=train_seconds, wall_seconds=time.perf_counter() - t_wall, epochs=int(epochs),
                  steps=int(epochs) * int(np.ceil(n / batch_size)), windows_forwarded=int(epochs) * n)
    return pd.DataFrame(rows), best, timing
