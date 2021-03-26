import time

import numpy as np
import cvxpy as cp

import torch
from torch import nn, optim
from torch.nn import functional as F

### Utils

def read_csv(
    fname,
    delim=',',
    remove_header=False,
    dtype=None,
):
    lines = []
    with open(fname) as f:
        content = f.read()
    lines = content.split('\n')
    table = [line.split(delim) for line in lines if line]

    if remove_header:
        table = table[1:]

    table = np.array(table)
    if dtype is not None:
        table = table.astype(dtype)

    return table

def save_submission(y, fname):
    with open(fname, 'w') as f:
        f.write('Id,Bound\n')
        for i, c in enumerate(y):
            f.write(f'{i},{c}\n')

def get_data(label, chars=False):
    if chars:
        X = read_csv(f'X{label}.csv', remove_header=True)[:, 1]
    else:
        X = read_csv(f'X{label}_mat100.csv', delim=' ', dtype=float)
    y = (read_csv(f'Y{label}.csv', remove_header=True, dtype=int)[:, 1]
         if 'tr' in label
         else None)
    return X, y

def print_accuracy(*, X=None, fn=None, logits=None, y):
    if logits is None:
        logits = fn(X)

    y_pred = (logits > 0).astype(int)
    acc = np.sum(y_pred == y) / y.shape[0]
    print(f'acc: {acc:.2f}')
    # print('classes: ', np.sum(y_pred), len(y_pred) - np.sum(y_pred))

### Funcs

def sigmoid(x):
    return np.where(x > 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))

def sigmoid1(x):
    return 1 / (1 + np.exp(-x))

def gaussian_fn(x, sigma):
    return np.exp((-1) * x**2 / (2 * sigma**2))

def pairwise_dists(X, Y):
    return np.sqrt(np.sum((X[:, None, :] - Y[:, :])**2, axis=-1))

def gaussian_kernel(X, sigma=None):
    dists = pairwise_dists(X, X)
    if sigma is None:
        sigma = np.mean(dists) / 2

    return gaussian_fn(dists, sigma=sigma)

def get_gaussian_kernel_fn(X, sigma, alphas):
    def fn(Z):
        dists = pairwise_dists(Z, X)
        return gaussian_fn(dists, sigma) @ alphas

    return fn

### Algos

def logistic_regression(
    X,
    y,
    num_iter=100,
    lr=1e-2,
    *args,
    **kwargs,
):
    w = np.random.randn(X.shape[1])

    for i in range(num_iter):
        logits = X @ w
        probs = sigmoid(logits)

        grads = (-1) * X.T @ (y - probs)
        w -= lr * grads

        y_pred = (logits > 0).astype(int)
        acc = np.sum(y_pred == y) / y.shape[0]

        if i % (num_iter // 10) == 0:
            print(f'acc: {acc:.2f}')
    print(f'acc: {acc:.2f}')

    def logits_fn(Z):
        return Z @ w

    return logits_fn


def kernel_logistic_regression(
    X,
    y,
    num_iter=100,
    lr=1e-2,
    lam=0.1,
    sigma=0.05,
    *args,
    **kwargs,
):
    y_sg = 2*y - 1
    n = len(X)
    K = gaussian_kernel(X, sigma=sigma)

    alpha = np.random.randn(n)
    for i in range(num_iter):
        logits = K @ alpha

        loss_g = (-1) * sigmoid((-1) * y_sg * logits)
        grads = (1/n) * K @ (loss_g * y_sg) + lam * K @ alpha
        alpha -= lr * grads

        y_pred = (logits > 0).astype(int)
        acc = np.sum(y_pred == y) / y.shape[0]
        if i % (num_iter // 10) == 0:
            print(f'acc: {acc:.2f}')
    print(f'acc: {acc:.2f}')

    return get_gaussian_kernel_fn(X, sigma, alpha)

def svm(
    X,
    y,
    C=0.01,
    sigma=0.05,
    *args,
    **kwargs,
):
    y_sg = 2*y - 1
    n = len(X)
    K = gaussian_kernel(X, sigma=sigma)

    alpha = cp.Variable(n)
    obj = cp.Minimize((1/2)*cp.quad_form(alpha, K) - alpha @ y_sg)
    alpha_y = alpha @ y_sg
    constrs = [0 <= alpha_y, alpha_y <= C]
    prob = cp.Problem(obj, constrs)

    prob.solve()

    logits = K @ alpha.value
    print_accuracy(logits=logits, y=y)

    return get_gaussian_kernel_fn(X, sigma, alpha.value)

def convs(
    X,
    Y,
    C=0.01,
    sigma=0.05,
    *args,
    **kwargs,
):
    bases = ['A', 'C', 'G', 'T']
    base_to_idx = {c:i for i, c in enumerate(bases)}
    def dna_to_vec(x):
        idxs = [base_to_idx[c] for c in x]
        return np.eye(4)[idxs].T

    def transform_input(Z):
        return torch.Tensor([dna_to_vec(z) for z in Z])

    Xv = transform_input(X)

    model = nn.Sequential(
        nn.Conv1d(4, 16, 4),
        nn.MaxPool1d(2),
        nn.ReLU(),
        nn.Conv1d(16, 32, 4),
        nn.MaxPool1d(2),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(736, 1),
    )

    # print(model(Xv[:3]).shape)
    # breakpoint()

    num_epochs = 100
    batch_size = 50
    Y = torch.tensor(Y)

    num_batches = len(X) // batch_size
    def get_batch(i):
        bs = batch_size
        b_slice = slice(bs * i, bs * (i+1))
        x, y = Xv[b_slice], Y[b_slice]

        # crop = np.random.randint(101) * 4
        # x = x.roll(crop)
        # y = y.roll(crop)
        return x, y

    opt = optim.Adam(model.parameters())

    for epoch in range(num_epochs):
        num_correct = 0
        for i in range(num_batches):
            x, y = get_batch(i)

            logits = model(x).flatten()
            loss = F.binary_cross_entropy_with_logits(logits, y.float())

            loss.backward()
            opt.step()
            opt.zero_grad()

            y_pred = (logits > 0).int()
            num_correct += torch.sum(y_pred == y).item()

        acc = num_correct / (batch_size * num_batches)
        if epoch % (num_epochs // 10) == 0:
            print(f'acc: {acc:.3f}')
    print(f'acc: {acc:.3f}')


    def fn(Z):
        Z = transform_input(Z)
        return model(Z).flatten().detach().numpy()

    return fn


def nn2(
    X,
    Y,
    C=0.01,
    num_epochs=1000,
    sigma=0.05,
    *args,
    **kwargs,
):
    hs = 100
    model = nn.Sequential(
        nn.Linear(100, hs),
        nn.ReLU(),
        nn.Linear(hs, 1),
    )

    Xv = torch.Tensor(X)

    batch_size = 50
    Y = torch.tensor(Y)

    num_batches = len(X) // batch_size
    def get_batch(i):
        bs = batch_size
        b_slice = slice(bs * i, bs * (i+1))
        return Xv[b_slice], Y[b_slice]

    opt = optim.Adam(model.parameters())

    for epoch in range(num_epochs):
        num_correct = 0
        for i in range(num_batches):
            x, y = get_batch(i)

            logits = model(x).flatten()
            loss = F.binary_cross_entropy_with_logits(logits, y.float())

            loss.backward()
            opt.step()
            opt.zero_grad()

            y_pred = (logits > 0).int()
            num_correct += torch.sum(y_pred == y).item()

        acc = num_correct / (batch_size * num_batches)
        if epoch % (num_epochs // 10) == 0:
            print(f'acc: {acc:.3f}')
    print(f'acc: {acc:.3f}')


    def fn(Z):
        Z = torch.tensor(Z).float()
        return model(Z).flatten().detach().numpy()

    return fn


def main(
    fn='svm',
    num_iter=100,
    lr=1e-2,
    n_val=0,
    out=True,
):
    fns = {
        'svm': svm,
        'nn': convs,
        'nn2': nn2,
        'logreg': logistic_regression,
        'klogreg': kernel_logistic_regression,
    }
    train_fn = fns[fn]

    chars = fn == 'nn'

    y_tests = []

    for data_id in range(3):
        X, y = get_data(f'tr{data_id}', chars=chars)
        if n_val:
            n_train = len(X) - n_val
            X, X_val = X[:n_train], X[n_train:]
            y, y_val = y[:n_train], y[n_train:]

        print(f'---train {data_id}---')
        pred_fn = train_fn(
            X,
            y,
            num_iter=num_iter,
            lr=lr,
        )

        X_test, _ = get_data(f'te{data_id}', chars=chars)
        y_test = (pred_fn(X_test) >= 0).astype(int)
        y_tests.append(y_test)

        if n_val:
            print(f'---test  {data_id}---')
            print_accuracy(fn=pred_fn, X=X_val, y=y_val)

    if out:
        y_submit = np.concatenate(y_tests)
        save_submission(y_submit, 'Yte.csv')


import fire
if __name__ == '__main__':
    fire.Fire(main)
